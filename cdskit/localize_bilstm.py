import numpy as np

AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
PAD_TOKEN = '<PAD>'
PAD_INDEX = 0


def _build_aa_to_idx():
    aa_to_idx = {PAD_TOKEN: PAD_INDEX}
    for i, aa in enumerate(AA_ALPHABET):
        aa_to_idx[aa] = i + 1
    return aa_to_idx


DEFAULT_AA_TO_IDX = _build_aa_to_idx()


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        txt = (
            'PyTorch is required for --model_arch bilstm_attention. '
            'Install torch first. Original error: {}'
        )
        raise ImportError(txt.format(str(exc)))
    return torch, nn


def resolve_torch_device(device_text='auto'):
    torch, _ = require_torch()
    device_text = str(device_text).strip().lower()
    if device_text in ('', 'auto'):
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    if device_text == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA device was requested but CUDA is not available.')
        return 'cuda'
    if device_text == 'cpu':
        return 'cpu'
    raise ValueError('Unsupported --dl_device: {}'.format(device_text))


def encode_aa_sequence(aa_seq, seq_len, aa_to_idx=None):
    aa_to_idx = DEFAULT_AA_TO_IDX if aa_to_idx is None else aa_to_idx
    seq_len = int(seq_len)
    seq = str(aa_seq).upper()
    out = np.zeros((seq_len,), dtype=np.int64)
    max_len = min(len(seq), seq_len)
    for i in range(max_len):
        ch = seq[i]
        out[i] = int(aa_to_idx.get(ch, aa_to_idx.get('X', 1)))
    return out


def encode_aa_sequences(aa_sequences, seq_len, aa_to_idx=None):
    aa_to_idx = DEFAULT_AA_TO_IDX if aa_to_idx is None else aa_to_idx
    encoded = np.zeros((len(aa_sequences), int(seq_len)), dtype=np.int64)
    for i, aa_seq in enumerate(aa_sequences):
        encoded[i, :] = encode_aa_sequence(
            aa_seq=aa_seq,
            seq_len=seq_len,
            aa_to_idx=aa_to_idx,
        )
    return encoded


def _build_bilstm_attention_module(
    nn,
    vocab_size,
    embed_dim,
    hidden_dim,
    num_layers,
    dropout,
    num_class,
):
    class BilstmAttentionClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(
                num_embeddings=int(vocab_size),
                embedding_dim=int(embed_dim),
                padding_idx=PAD_INDEX,
            )
            lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
            self.encoder = nn.LSTM(
                input_size=int(embed_dim),
                hidden_size=int(hidden_dim),
                num_layers=int(num_layers),
                batch_first=True,
                bidirectional=True,
                dropout=lstm_dropout,
            )
            self.attention = nn.Linear(int(hidden_dim) * 2, 1)
            self.classifier = nn.Linear(int(hidden_dim) * 2, int(num_class))

        def forward(self, tokens, mask):
            emb = self.embedding(tokens)
            encoded, _ = self.encoder(emb)
            att_logits = self.attention(encoded).squeeze(-1)
            att_logits = att_logits.masked_fill(~mask, -1.0e9)
            att_w = att_logits.softmax(dim=1)
            context = (encoded * att_w.unsqueeze(-1)).sum(dim=1)
            logits = self.classifier(context)
            return logits

    return BilstmAttentionClassifier()


def _class_weights_from_labels(labels, class_order):
    counts = {class_name: 0 for class_name in class_order}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    n_total = float(len(labels))
    n_class = float(len(class_order))
    weights = list()
    for class_name in class_order:
        count = float(counts.get(class_name, 0))
        if count <= 0:
            weights.append(0.0)
        else:
            weights.append(n_total / (n_class * count))
    return np.asarray(weights, dtype=np.float32)


def _init_class_sampling_pools(y, rng):
    observed_classes = sorted(np.unique(y).tolist())
    pools = dict()
    cursors = dict()
    for class_idx in observed_classes:
        class_indices = np.where(y == class_idx)[0].astype(np.int64)
        rng.shuffle(class_indices)
        pools[class_idx] = class_indices
        cursors[class_idx] = 0
    return observed_classes, pools, cursors


def _draw_index_from_class(class_idx, pools, cursors, rng):
    pool = pools[class_idx]
    if pool.shape[0] == 0:
        raise ValueError('No sample is available for class index {}.'.format(class_idx))
    cursor = int(cursors[class_idx])
    if cursor >= pool.shape[0]:
        pool = pool.copy()
        rng.shuffle(pool)
        pools[class_idx] = pool
        cursor = 0
    out_index = int(pool[cursor])
    cursors[class_idx] = cursor + 1
    return out_index


def _sample_balanced_batch_indices(observed_classes, pools, cursors, batch_size, rng):
    batch = list()
    num_class = len(observed_classes)
    if batch_size >= num_class:
        class_order = np.asarray(observed_classes, dtype=np.int64)
        rng.shuffle(class_order)
        for class_idx in class_order.tolist():
            batch.append(
                _draw_index_from_class(
                    class_idx=class_idx,
                    pools=pools,
                    cursors=cursors,
                    rng=rng,
                )
            )
    while len(batch) < batch_size:
        class_idx = int(observed_classes[int(rng.integers(0, num_class))])
        batch.append(
            _draw_index_from_class(
                class_idx=class_idx,
                pools=pools,
                cursors=cursors,
                rng=rng,
            )
        )
    batch = np.asarray(batch, dtype=np.int64)
    rng.shuffle(batch)
    return batch


def _resolve_loss_function(torch, nn, weight_tensor, loss_name='ce', focal_gamma=2.0):
    loss_name = str(loss_name).strip().lower()
    if loss_name == 'ce':
        return nn.CrossEntropyLoss(weight=weight_tensor), loss_name
    if loss_name == 'focal':
        import torch.nn.functional as F
        focal_gamma = float(focal_gamma)
        if focal_gamma < 0:
            raise ValueError('focal_gamma should be >= 0.')

        def focal_loss(logits, targets):
            log_probs = F.log_softmax(logits, dim=1)
            probs = log_probs.exp()
            target_idx = targets.unsqueeze(1)
            target_prob = probs.gather(1, target_idx).squeeze(1)
            ce = F.nll_loss(
                log_probs,
                targets,
                reduction='none',
                weight=weight_tensor,
            )
            focal_factor = (1.0 - target_prob).pow(focal_gamma)
            return (focal_factor * ce).mean()

        return focal_loss, loss_name
    raise ValueError('Unsupported loss_name: {}'.format(loss_name))


def fit_bilstm_attention_classifier(
    aa_sequences,
    labels,
    class_order,
    seq_len,
    embed_dim,
    hidden_dim,
    num_layers,
    dropout,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    seed,
    use_class_weight,
    device,
    loss_name='ce',
    balanced_batch=False,
    focal_gamma=2.0,
):
    torch, nn = require_torch()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    seq_len = int(seq_len)
    batch_size = int(batch_size)
    epochs = int(epochs)
    aa_to_idx = dict(DEFAULT_AA_TO_IDX)
    class_order = list(class_order)
    label_to_idx = {class_name: i for i, class_name in enumerate(class_order)}

    x = encode_aa_sequences(
        aa_sequences=aa_sequences,
        seq_len=seq_len,
        aa_to_idx=aa_to_idx,
    )
    y = np.asarray([label_to_idx[v] for v in labels], dtype=np.int64)

    if x.shape[0] != y.shape[0]:
        raise ValueError('Input sequence count and label count mismatch.')
    if x.shape[0] == 0:
        raise ValueError('No training sequence for bilstm_attention.')
    if epochs < 1:
        raise ValueError('--dl_epochs should be >= 1.')
    if batch_size < 1:
        raise ValueError('--dl_batch_size should be >= 1.')
    loss_name = str(loss_name).strip().lower()
    if loss_name not in ['ce', 'focal']:
        raise ValueError('loss_name should be ce or focal.')
    balanced_batch = bool(balanced_batch)
    if float(focal_gamma) < 0:
        raise ValueError('focal_gamma should be >= 0.')

    resolved_device = resolve_torch_device(device_text=device)
    model = _build_bilstm_attention_module(
        nn=nn,
        vocab_size=len(aa_to_idx),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_class=len(class_order),
    )
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    if use_class_weight:
        weights = _class_weights_from_labels(
            labels=labels,
            class_order=class_order,
        )
        weight_tensor = torch.as_tensor(
            weights,
            dtype=torch.float32,
            device=resolved_device,
        )
    else:
        weight_tensor = None
    loss_fn, _ = _resolve_loss_function(
        torch=torch,
        nn=nn,
        weight_tensor=weight_tensor,
        loss_name=loss_name,
        focal_gamma=focal_gamma,
    )

    indices = np.arange(x.shape[0], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    n_batch = int(np.ceil(x.shape[0] / float(batch_size)))
    for _ in range(epochs):
        model.train()
        if balanced_batch:
            observed_classes, pools, cursors = _init_class_sampling_pools(y=y, rng=rng)
            for _ in range(n_batch):
                batch_idx = _sample_balanced_batch_indices(
                    observed_classes=observed_classes,
                    pools=pools,
                    cursors=cursors,
                    batch_size=batch_size,
                    rng=rng,
                )
                xb = torch.as_tensor(
                    x[batch_idx, :],
                    dtype=torch.long,
                    device=resolved_device,
                )
                yb = torch.as_tensor(
                    y[batch_idx],
                    dtype=torch.long,
                    device=resolved_device,
                )
                mask = (xb != PAD_INDEX)
                optimizer.zero_grad(set_to_none=True)
                logits = model(tokens=xb, mask=mask)
                loss = loss_fn(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            continue
        rng.shuffle(indices)
        for start in range(0, indices.shape[0], batch_size):
            batch_idx = indices[start:start + batch_size]
            xb = torch.as_tensor(
                x[batch_idx, :],
                dtype=torch.long,
                device=resolved_device,
            )
            yb = torch.as_tensor(
                y[batch_idx],
                dtype=torch.long,
                device=resolved_device,
            )
            mask = (xb != PAD_INDEX)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens=xb, mask=mask)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    return {
        'class_order': list(class_order),
        'seq_len': int(seq_len),
        'aa_to_idx': dict(aa_to_idx),
        'embed_dim': int(embed_dim),
        'hidden_dim': int(hidden_dim),
        'num_layers': int(num_layers),
        'dropout': float(dropout),
        'state_dict': state_dict,
        'device': str(resolved_device),
    }


def _get_runtime_bilstm_model(localization_model, device_text='cpu'):
    torch, nn = require_torch()
    if '_runtime_model_cache' not in localization_model:
        localization_model['_runtime_model_cache'] = dict()
    cache = localization_model['_runtime_model_cache']
    resolved_device = resolve_torch_device(device_text=device_text)
    cache_key = str(resolved_device)
    if cache_key in cache:
        return cache[cache_key], resolved_device

    model = _build_bilstm_attention_module(
        nn=nn,
        vocab_size=len(localization_model['aa_to_idx']),
        embed_dim=localization_model['embed_dim'],
        hidden_dim=localization_model['hidden_dim'],
        num_layers=localization_model['num_layers'],
        dropout=localization_model.get('dropout', 0.0),
        num_class=len(localization_model['class_order']),
    )
    model.load_state_dict(localization_model['state_dict'], strict=True)
    model.eval()
    model.to(resolved_device)
    cache[cache_key] = model
    return model, resolved_device


def predict_bilstm_attention_batch(
    aa_sequences,
    localization_model,
    device='cpu',
    batch_size=512,
):
    torch, _ = require_torch()
    model, resolved_device = _get_runtime_bilstm_model(
        localization_model=localization_model,
        device_text=device,
    )
    aa_to_idx = localization_model['aa_to_idx']
    seq_len = int(localization_model['seq_len'])
    x = encode_aa_sequences(
        aa_sequences=aa_sequences,
        seq_len=seq_len,
        aa_to_idx=aa_to_idx,
    )
    probs = list()
    with torch.no_grad():
        for start in range(0, x.shape[0], int(batch_size)):
            batch_x = x[start:start + int(batch_size), :]
            xb = torch.as_tensor(
                batch_x,
                dtype=torch.long,
                device=resolved_device,
            )
            mask = (xb != PAD_INDEX)
            logits = model(tokens=xb, mask=mask)
            batch_probs = logits.softmax(dim=1).detach().cpu().numpy()
            probs.append(batch_probs)
    if len(probs) == 0:
        return np.zeros((0, len(localization_model['class_order'])), dtype=np.float64)
    return np.vstack(probs).astype(np.float64)


def predict_bilstm_attention(aa_seq, localization_model, device='cpu'):
    probs = predict_bilstm_attention_batch(
        aa_sequences=[aa_seq],
        localization_model=localization_model,
        device=device,
        batch_size=1,
    )
    class_order = list(localization_model['class_order'])
    class_probs = {class_order[i]: float(probs[0, i]) for i in range(len(class_order))}
    pred_index = int(np.argmax(probs[0, :]))
    return class_order[pred_index], class_probs

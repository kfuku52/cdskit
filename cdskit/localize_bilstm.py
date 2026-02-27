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
    torch,
    nn,
    vocab_size,
    embed_dim,
    hidden_dim,
    num_layers,
    dropout,
    num_class,
    feature_dim=0,
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
            self.use_feature_fusion = int(feature_dim) > 0
            feature_proj_dim = int(hidden_dim)
            if self.use_feature_fusion:
                self.feature_mlp = nn.Sequential(
                    nn.Linear(int(feature_dim), feature_proj_dim),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                )
                classifier_in_dim = (int(hidden_dim) * 2) + feature_proj_dim
            else:
                self.feature_mlp = None
                classifier_in_dim = int(hidden_dim) * 2
            self.classifier = nn.Linear(classifier_in_dim, int(num_class))

        def forward(self, tokens, mask, feature_vec=None, return_representation=False):
            emb = self.embedding(tokens)
            lengths = mask.long().sum(dim=1)
            lengths = lengths.clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                emb,
                lengths=lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            encoded_packed, _ = self.encoder(packed)
            encoded, _ = nn.utils.rnn.pad_packed_sequence(
                encoded_packed,
                batch_first=True,
                total_length=tokens.shape[1],
            )
            att_logits = self.attention(encoded).squeeze(-1)
            att_logits = att_logits.masked_fill(~mask, -1.0e9)
            att_w = att_logits.softmax(dim=1)
            context = (encoded * att_w.unsqueeze(-1)).sum(dim=1)
            if self.use_feature_fusion:
                if feature_vec is None:
                    feature_vec = context.new_zeros((context.shape[0], int(feature_dim)))
                fused_feature = self.feature_mlp(feature_vec)
                classifier_input = torch.cat([context, fused_feature], dim=1)
            else:
                classifier_input = context
            logits = self.classifier(classifier_input)
            if return_representation:
                return logits, classifier_input
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


def _class_weights_from_binary_targets(targets, num_class=2):
    targets = np.asarray(targets, dtype=np.int64)
    weights = np.zeros((int(num_class),), dtype=np.float32)
    if targets.shape[0] <= 0:
        return weights
    n_total = float(targets.shape[0])
    n_class = float(num_class)
    for class_idx in range(int(num_class)):
        count = float(np.sum(targets == class_idx))
        if count > 0.0:
            weights[class_idx] = n_total / (n_class * count)
    return weights


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


def _prepare_feature_matrix(feature_matrix, n_row):
    if feature_matrix is None:
        return None, None, None
    feat = np.asarray(feature_matrix, dtype=np.float32)
    if feat.ndim != 2:
        raise ValueError('feature_matrix should be 2D array.')
    if int(feat.shape[0]) != int(n_row):
        txt = 'feature_matrix row count mismatch: expected {}, got {}.'
        raise ValueError(txt.format(int(n_row), int(feat.shape[0])))
    feat_mean = feat.mean(axis=0).astype(np.float32)
    feat_scale = feat.std(axis=0).astype(np.float32)
    feat_scale[feat_scale < 1.0e-6] = 1.0
    feat_norm = (feat - feat_mean[np.newaxis, :]) / feat_scale[np.newaxis, :]
    return feat_norm.astype(np.float32), feat_mean, feat_scale


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
    feature_matrix=None,
    aux_tp_weight=0.0,
    aux_ctp_ltp_weight=0.0,
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
    feature_x, feature_mean, feature_scale = _prepare_feature_matrix(
        feature_matrix=feature_matrix,
        n_row=x.shape[0],
    )
    feature_dim = 0
    if feature_x is not None:
        feature_dim = int(feature_x.shape[1])

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
    aux_tp_weight = float(aux_tp_weight)
    aux_ctp_ltp_weight = float(aux_ctp_ltp_weight)
    if aux_tp_weight < 0.0:
        raise ValueError('aux_tp_weight should be >= 0.')
    if aux_ctp_ltp_weight < 0.0:
        raise ValueError('aux_ctp_ltp_weight should be >= 0.')

    resolved_device = resolve_torch_device(device_text=device)
    model = _build_bilstm_attention_module(
        torch=torch,
        nn=nn,
        vocab_size=len(aa_to_idx),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_class=len(class_order),
        feature_dim=feature_dim,
    )
    model.to(resolved_device)
    representation_dim = int(model.classifier.in_features)
    has_no_tp = 'noTP' in label_to_idx
    has_ctp_ltp_pair = ('cTP' in label_to_idx) and ('lTP' in label_to_idx)
    no_tp_idx = int(label_to_idx.get('noTP', 0))
    ctp_idx = int(label_to_idx.get('cTP', -1))
    ltp_idx = int(label_to_idx.get('lTP', -1))
    y_tp = (y != no_tp_idx).astype(np.int64)
    y_ctp_ltp = np.full((y.shape[0],), -1, dtype=np.int64)
    if (ctp_idx >= 0) and (ltp_idx >= 0):
        y_ctp_ltp[y == ctp_idx] = 0
        y_ctp_ltp[y == ltp_idx] = 1

    use_aux_tp = (aux_tp_weight > 0.0) and has_no_tp
    use_aux_ctp_ltp = (aux_ctp_ltp_weight > 0.0) and has_ctp_ltp_pair
    aux_tp_head = None
    aux_tp_loss_fn = None
    aux_ctp_ltp_head = None
    aux_ctp_ltp_loss_fn = None
    if use_aux_tp:
        aux_tp_head = nn.Linear(representation_dim, 2)
        aux_tp_head.to(resolved_device)
        tp_w = _class_weights_from_binary_targets(targets=y_tp, num_class=2)
        tp_w_t = torch.as_tensor(tp_w, dtype=torch.float32, device=resolved_device)
        aux_tp_loss_fn = nn.CrossEntropyLoss(weight=tp_w_t)
    if use_aux_ctp_ltp:
        aux_ctp_ltp_head = nn.Linear(representation_dim, 2)
        aux_ctp_ltp_head.to(resolved_device)
        valid_targets = y_ctp_ltp[y_ctp_ltp >= 0]
        ctp_ltp_w = _class_weights_from_binary_targets(targets=valid_targets, num_class=2)
        ctp_ltp_w_t = torch.as_tensor(ctp_ltp_w, dtype=torch.float32, device=resolved_device)
        aux_ctp_ltp_loss_fn = nn.CrossEntropyLoss(weight=ctp_ltp_w_t)

    parameters = list(model.parameters())
    if aux_tp_head is not None:
        parameters.extend(list(aux_tp_head.parameters()))
    if aux_ctp_ltp_head is not None:
        parameters.extend(list(aux_ctp_ltp_head.parameters()))
    optimizer = torch.optim.AdamW(
        parameters,
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
    use_auxiliary = use_aux_tp or use_aux_ctp_ltp

    def _compute_total_loss(batch_idx, logits, yb, representation):
        loss = loss_fn(logits, yb)
        if use_aux_tp:
            tp_targets = torch.as_tensor(
                y_tp[batch_idx],
                dtype=torch.long,
                device=resolved_device,
            )
            tp_logits = aux_tp_head(representation)
            loss = loss + (aux_tp_weight * aux_tp_loss_fn(tp_logits, tp_targets))
        if use_aux_ctp_ltp:
            ctp_ltp_targets_np = y_ctp_ltp[batch_idx]
            keep_mask_np = (ctp_ltp_targets_np >= 0)
            if np.any(keep_mask_np):
                keep_mask = torch.as_tensor(
                    keep_mask_np,
                    dtype=torch.bool,
                    device=resolved_device,
                )
                ctp_ltp_targets = torch.as_tensor(
                    ctp_ltp_targets_np[keep_mask_np],
                    dtype=torch.long,
                    device=resolved_device,
                )
                ctp_ltp_logits = aux_ctp_ltp_head(representation[keep_mask, :])
                loss = loss + (aux_ctp_ltp_weight * aux_ctp_ltp_loss_fn(ctp_ltp_logits, ctp_ltp_targets))
        return loss

    for _ in range(epochs):
        model.train()
        if aux_tp_head is not None:
            aux_tp_head.train()
        if aux_ctp_ltp_head is not None:
            aux_ctp_ltp_head.train()
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
                fb = None
                if feature_x is not None:
                    fb = torch.as_tensor(
                        feature_x[batch_idx, :],
                        dtype=torch.float32,
                        device=resolved_device,
                    )
                mask = (xb != PAD_INDEX)
                optimizer.zero_grad(set_to_none=True)
                if use_auxiliary:
                    logits, rep = model(
                        tokens=xb,
                        mask=mask,
                        feature_vec=fb,
                        return_representation=True,
                    )
                    loss = _compute_total_loss(
                        batch_idx=batch_idx,
                        logits=logits,
                        yb=yb,
                        representation=rep,
                    )
                else:
                    logits = model(tokens=xb, mask=mask, feature_vec=fb)
                    loss = loss_fn(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
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
            fb = None
            if feature_x is not None:
                fb = torch.as_tensor(
                    feature_x[batch_idx, :],
                    dtype=torch.float32,
                    device=resolved_device,
                )
            mask = (xb != PAD_INDEX)
            optimizer.zero_grad(set_to_none=True)
            if use_auxiliary:
                logits, rep = model(
                    tokens=xb,
                    mask=mask,
                    feature_vec=fb,
                    return_representation=True,
                )
                loss = _compute_total_loss(
                    batch_idx=batch_idx,
                    logits=logits,
                    yb=yb,
                    representation=rep,
                )
            else:
                logits = model(tokens=xb, mask=mask, feature_vec=fb)
                loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
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
        'use_feature_fusion': bool(feature_dim > 0),
        'feature_dim': int(feature_dim),
        'feature_mean': None if feature_mean is None else feature_mean.astype(np.float32).tolist(),
        'feature_scale': None if feature_scale is None else feature_scale.astype(np.float32).tolist(),
        'aux_tp_weight': float(aux_tp_weight),
        'aux_ctp_ltp_weight': float(aux_ctp_ltp_weight),
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
        torch=torch,
        nn=nn,
        vocab_size=len(localization_model['aa_to_idx']),
        embed_dim=localization_model['embed_dim'],
        hidden_dim=localization_model['hidden_dim'],
        num_layers=localization_model['num_layers'],
        dropout=localization_model.get('dropout', 0.0),
        num_class=len(localization_model['class_order']),
        feature_dim=int(localization_model.get('feature_dim', 0)),
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
    feature_matrix=None,
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
    feature_x = None
    feature_dim = int(localization_model.get('feature_dim', 0))
    use_feature_fusion = bool(localization_model.get('use_feature_fusion', feature_dim > 0))
    if use_feature_fusion and (feature_dim > 0):
        if feature_matrix is None:
            feature_x = np.zeros((x.shape[0], feature_dim), dtype=np.float32)
        else:
            feature_x = np.asarray(feature_matrix, dtype=np.float32)
            if feature_x.ndim != 2:
                raise ValueError('feature_matrix should be 2D array.')
            if int(feature_x.shape[0]) != int(x.shape[0]):
                txt = 'feature_matrix row count mismatch: expected {}, got {}.'
                raise ValueError(txt.format(int(x.shape[0]), int(feature_x.shape[0])))
            if int(feature_x.shape[1]) != int(feature_dim):
                txt = 'feature_matrix column mismatch: expected {}, got {}.'
                raise ValueError(txt.format(int(feature_dim), int(feature_x.shape[1])))
        mean = localization_model.get('feature_mean', None)
        scale = localization_model.get('feature_scale', None)
        if (mean is not None) and (scale is not None):
            mean = np.asarray(mean, dtype=np.float32)
            scale = np.asarray(scale, dtype=np.float32)
            if mean.shape[0] == feature_dim and scale.shape[0] == feature_dim:
                safe_scale = scale.copy()
                safe_scale[safe_scale < 1.0e-6] = 1.0
                feature_x = (feature_x - mean[np.newaxis, :]) / safe_scale[np.newaxis, :]
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
            fb = None
            if feature_x is not None:
                batch_f = feature_x[start:start + int(batch_size), :]
                fb = torch.as_tensor(
                    batch_f,
                    dtype=torch.float32,
                    device=resolved_device,
                )
            logits = model(tokens=xb, mask=mask, feature_vec=fb)
            batch_probs = logits.softmax(dim=1).detach().cpu().numpy()
            probs.append(batch_probs)
    if len(probs) == 0:
        return np.zeros((0, len(localization_model['class_order'])), dtype=np.float64)
    return np.vstack(probs).astype(np.float64)


def predict_bilstm_attention(aa_seq, localization_model, device='cpu', feature_vec=None):
    feature_matrix = None
    if feature_vec is not None:
        feature_matrix = np.asarray([feature_vec], dtype=np.float32)
    probs = predict_bilstm_attention_batch(
        aa_sequences=[aa_seq],
        localization_model=localization_model,
        device=device,
        batch_size=1,
        feature_matrix=feature_matrix,
    )
    class_order = list(localization_model['class_order'])
    class_probs = {class_order[i]: float(probs[0, i]) for i in range(len(class_order))}
    pred_index = int(np.argmax(probs[0, :]))
    return class_order[pred_index], class_probs

import numpy as np

from cdskit.localize_bilstm import (
    DEFAULT_AA_TO_IDX,
    PAD_INDEX,
    require_torch,
    resolve_torch_device,
)


def encode_aa_sequence_termini(aa_seq, seq_len, aa_to_idx=None):
    aa_to_idx = DEFAULT_AA_TO_IDX if aa_to_idx is None else aa_to_idx
    seq_len = int(seq_len)
    seq = str(aa_seq).upper()
    if len(seq) > seq_len:
        left = int(seq_len // 2)
        right = int(seq_len - left)
        seq = seq[:left] + seq[-right:]
    out = np.zeros((seq_len,), dtype=np.int64)
    max_len = min(len(seq), seq_len)
    for i in range(max_len):
        out[i] = int(aa_to_idx.get(seq[i], aa_to_idx.get("X", 1)))
    return out


def encode_aa_sequences_termini(aa_sequences, seq_len, aa_to_idx=None):
    aa_to_idx = DEFAULT_AA_TO_IDX if aa_to_idx is None else aa_to_idx
    encoded = np.zeros((len(aa_sequences), int(seq_len)), dtype=np.int64)
    for i, aa_seq in enumerate(aa_sequences):
        encoded[i, :] = encode_aa_sequence_termini(
            aa_seq=aa_seq,
            seq_len=seq_len,
            aa_to_idx=aa_to_idx,
        )
    return encoded


def _encode_layout(sequences, seq_len, aa_to_idx, layout):
    if layout == "legacy":
        return encode_aa_sequences_termini(sequences, seq_len, aa_to_idx)
    if layout == "windows":
        parts = []
        for sequence in sequences:
            chunks = []
            for start in range(0, max(1, len(sequence)), max(1, seq_len // 2)):
                chunks.append(
                    encode_aa_sequence_termini(
                        sequence[start : start + seq_len], seq_len, aa_to_idx
                    )
                )
                if start + seq_len >= len(sequence):
                    break
            parts.append(chunks)
        result = np.zeros(
            (len(sequences), max((len(x) for x in parts), default=1), seq_len),
            dtype=np.int64,
        )
        for i, chunks in enumerate(parts):
            result[i, : len(chunks)] = chunks
        return result
    if layout != "separate_termini":
        raise ValueError("Unsupported sequence_layout: {}".format(layout))
    # Both termini are encoded independently; no convolution crosses a synthetic join.
    width = (seq_len + 1) // 2
    result = np.zeros((len(sequences), 2, width), dtype=np.int64)
    for i, sequence in enumerate(sequences):
        sequence = str(sequence).upper()
        for j, part in enumerate((sequence[:width], sequence[-width:])):
            result[i, j] = encode_aa_sequence_termini(part, width, aa_to_idx)
    return result


def _forward_layout(model, tokens, feature_vec):
    return model(tokens=tokens, feature_vec=feature_vec)


def _parse_kernel_sizes(kernel_sizes):
    if isinstance(kernel_sizes, str):
        out = [int(v.strip()) for v in kernel_sizes.split(",") if v.strip() != ""]
    else:
        out = [int(v) for v in kernel_sizes]
    if len(out) == 0:
        raise ValueError("At least one CNN kernel size is required.")
    for value in out:
        if value < 1:
            raise ValueError("CNN kernel sizes should be >= 1.")
    return out


def _build_multilabel_cnn_module(
    torch,
    nn,
    vocab_size,
    embed_dim,
    num_filters,
    kernel_sizes,
    dropout,
    num_class,
    feature_dim=0,
    mask_padding=False,
    terminal_concat=False,
):
    kernel_sizes = _parse_kernel_sizes(kernel_sizes=kernel_sizes)

    class MultilabelCnnClassifier(nn.Module):  # type: ignore[name-defined]
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(
                num_embeddings=int(vocab_size),
                embedding_dim=int(embed_dim),
                padding_idx=PAD_INDEX,
            )
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=int(embed_dim),
                        out_channels=int(num_filters),
                        kernel_size=int(k),
                        padding=int(k) // 2,
                    )
                    for k in kernel_sizes
                ]
            )
            self.use_feature_fusion = int(feature_dim) > 0
            if self.use_feature_fusion:
                self.feature_mlp = nn.Sequential(
                    nn.Linear(int(feature_dim), int(num_filters)),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                )
                classifier_in_dim = (
                    int(num_filters) * len(kernel_sizes) * (2 if terminal_concat else 1)
                ) + int(num_filters)
            else:
                self.feature_mlp = None
                classifier_in_dim = (
                    int(num_filters) * len(kernel_sizes) * (2 if terminal_concat else 1)
                )
            self.dropout = nn.Dropout(float(dropout))
            self.classifier = nn.Linear(classifier_in_dim, int(num_class))

        def forward(self, tokens, feature_vec=None):
            segmented = tokens.ndim == 3
            if segmented:
                batch, segments, length = tokens.shape
                tokens = tokens.reshape(batch * segments, length)
            emb = self.embedding(tokens).transpose(1, 2)
            pooled = list()
            for conv in self.convs:
                z = conv(emb)
                z = z.relu()
                if mask_padding:
                    valid = tokens.ne(PAD_INDEX)
                    # Even kernels produce one extra output with symmetric padding.
                    z = z[:, :, : tokens.shape[1]]
                    z = z.masked_fill(~valid[:, None, :], float("-inf"))
                z = torch.amax(z, dim=2)
                if mask_padding:
                    z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
                if segmented:
                    z = (
                        z.reshape(batch, -1)
                        if terminal_concat
                        else z.reshape(batch, segments, -1).amax(dim=1)
                    )
                pooled.append(z)
            if self.use_feature_fusion:
                if feature_vec is None:
                    feature_vec = pooled[0].new_zeros(
                        (pooled[0].shape[0], int(feature_dim))
                    )
                pooled.append(self.feature_mlp(feature_vec))
            x = torch.cat(pooled, dim=1)
            return self.classifier(self.dropout(x))

    return MultilabelCnnClassifier()


def _prepare_feature_matrix(feature_matrix, n_row):
    if feature_matrix is None:
        return None, None, None
    feat = np.asarray(feature_matrix, dtype=np.float32)
    if feat.ndim != 2:
        raise ValueError("feature_matrix should be 2D array.")
    if int(feat.shape[0]) != int(n_row):
        txt = "feature_matrix row count mismatch: expected {}, got {}."
        raise ValueError(txt.format(int(n_row), int(feat.shape[0])))
    feat_mean = feat.mean(axis=0).astype(np.float32)
    feat_scale = feat.std(axis=0).astype(np.float32)
    feat_scale[feat_scale < 1.0e-6] = 1.0
    feat_norm = (feat - feat_mean[np.newaxis, :]) / feat_scale[np.newaxis, :]
    return feat_norm.astype(np.float32), feat_mean, feat_scale


def _class_pos_weight(label_matrix):
    y = np.asarray(label_matrix, dtype=np.float32)
    pos = np.sum(y > 0.5, axis=0)
    neg = float(y.shape[0]) - pos
    return (neg / np.clip(pos, 1.0, None)).astype(np.float32)


def _row_sampling_probabilities(label_matrix, sample_weight_power):
    sample_weight_power = float(sample_weight_power)
    if sample_weight_power <= 0.0:
        return None
    y = np.asarray(label_matrix, dtype=np.float32)
    pos = np.sum(y > 0.5, axis=0)
    pos[pos < 1.0] = 1.0
    class_weight = float(y.shape[0]) / pos
    row_weight = np.ones((y.shape[0],), dtype=np.float64)
    positive = y > 0.5
    for row_i in range(y.shape[0]):
        active = class_weight[positive[row_i, :]]
        if active.shape[0] > 0:
            row_weight[row_i] = float(np.max(active))
    row_weight = np.power(row_weight, sample_weight_power)
    row_weight[~np.isfinite(row_weight)] = 1.0
    row_weight[row_weight <= 0.0] = 1.0
    return row_weight / np.sum(row_weight)


def fit_multilabel_cnn_classifier(
    aa_sequences,
    label_matrix,
    class_order,
    seq_len=512,
    embed_dim=32,
    num_filters=64,
    kernel_sizes=(3, 5, 9, 15),
    dropout=0.25,
    epochs=6,
    batch_size=256,
    learning_rate=1.0e-3,
    weight_decay=1.0e-4,
    seed=1,
    use_class_weight=True,
    device="auto",
    feature_matrix=None,
    sample_weight_power=0.0,
    threshold_grid=None,
    threshold_objective="f1",
    threshold_objective_by_class=None,
    ensure_one_label=True,
    sequence_layout="separate_termini",
    mask_padding=True,
    tune_thresholds=False,
    validation_sequences=None,
    validation_labels=None,
    validation_features=None,
    patience=3,
    teacher_probabilities=None,
    distillation_weight=0.5,
):
    if tune_thresholds and not validation_sequences:
        raise ValueError("Threshold tuning requires independent validation sequences.")
    if validation_sequences and set(str(s).upper() for s in aa_sequences).intersection(
        str(s).upper() for s in validation_sequences
    ):
        raise ValueError("Training and validation sequences overlap.")
    torch, nn = require_torch()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    seq_len = int(seq_len)
    if seq_len < 4:
        raise ValueError("seq_len should be >= 4.")
    epochs = int(epochs)
    batch_size = int(batch_size)
    if epochs < 1:
        raise ValueError("epochs should be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size should be >= 1.")
    kernel_sizes = _parse_kernel_sizes(kernel_sizes=kernel_sizes)
    class_order = list(class_order)
    y = np.asarray(label_matrix, dtype=np.float32)
    if y.ndim != 2:
        raise ValueError("label_matrix should be 2D array.")
    if y.shape[0] != len(aa_sequences):
        raise ValueError("Sequence count and label row count mismatch.")
    if y.shape[1] != len(class_order):
        raise ValueError("label_matrix column count does not match class_order.")
    if y.shape[0] == 0:
        raise ValueError("No training sequence for multilabel CNN.")

    training_targets = y
    if teacher_probabilities is not None:
        teacher = np.asarray(teacher_probabilities, dtype=np.float32)
        if (
            teacher.shape != y.shape
            or not np.isfinite(teacher).all()
            or np.any((teacher < 0) | (teacher > 1))
        ):
            raise ValueError(
                "Teacher probabilities must match labels and be finite values in [0, 1]."
            )
        if not 0 <= float(distillation_weight) <= 1:
            raise ValueError("distillation_weight must be in [0, 1].")
        training_targets = (1 - float(distillation_weight)) * y + float(
            distillation_weight
        ) * teacher

    if sequence_layout not in ("legacy", "separate_termini", "windows"):
        raise ValueError("Unsupported sequence_layout: {}".format(sequence_layout))
    aa_to_idx = dict(DEFAULT_AA_TO_IDX)
    feature_x, feature_mean, feature_scale = _prepare_feature_matrix(
        feature_matrix=feature_matrix,
        n_row=len(aa_sequences),
    )
    feature_dim = 0 if feature_x is None else int(feature_x.shape[1])
    resolved_device = resolve_torch_device(device_text=device)
    model = _build_multilabel_cnn_module(
        torch=torch,
        nn=nn,
        vocab_size=len(aa_to_idx),
        embed_dim=int(embed_dim),
        num_filters=int(num_filters),
        kernel_sizes=kernel_sizes,
        dropout=float(dropout),
        num_class=len(class_order),
        feature_dim=feature_dim,
        mask_padding=mask_padding,
        terminal_concat=sequence_layout == "separate_termini",
    )
    model.to(resolved_device)
    if use_class_weight:
        pos_weight = torch.as_tensor(
            _class_pos_weight(label_matrix=y),
            dtype=torch.float32,
            device=resolved_device,
        )
    else:
        pos_weight = None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(aa_sequences), dtype=np.int64)
    sample_prob = _row_sampling_probabilities(
        label_matrix=y,
        sample_weight_power=sample_weight_power,
    )
    validation_x, validation_target, validation_f = None, None, None
    if validation_sequences:
        validation_x = validation_sequences
        validation_target = np.asarray(validation_labels, dtype=np.float32)
        if validation_target.shape != (len(validation_sequences), len(class_order)):
            raise ValueError("Validation labels have incompatible shape.")
        if feature_dim:
            validation_f = _normalize_runtime_features(
                validation_features,
                {
                    "feature_dim": feature_dim,
                    "feature_mean": feature_mean,
                    "feature_scale": feature_scale,
                },
                len(validation_sequences),
            )
    if int(patience) < 1:
        raise ValueError("patience must be positive.")
    best_loss, best_state, stale, selected_epoch = float("inf"), None, 0, epochs
    for epoch in range(epochs):
        if sample_prob is None:
            rng.shuffle(indices)
            epoch_indices = indices
        else:
            epoch_indices = rng.choice(
                indices,
                size=indices.shape[0],
                replace=True,
                p=sample_prob,
            )
        model.train()
        for start in range(0, epoch_indices.shape[0], batch_size):
            batch_idx = epoch_indices[start : start + batch_size]
            encoded = _encode_layout(
                [aa_sequences[i] for i in batch_idx],
                seq_len,
                aa_to_idx,
                sequence_layout,
            )
            xb = torch.as_tensor(encoded, dtype=torch.long, device=resolved_device)
            yb = torch.as_tensor(
                training_targets[batch_idx, :],
                dtype=torch.float32,
                device=resolved_device,
            )
            fb = None
            if feature_x is not None:
                fb = torch.as_tensor(
                    feature_x[batch_idx, :],
                    dtype=torch.float32,
                    device=resolved_device,
                )
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(_forward_layout(model, xb, fb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if validation_x is not None:
            assert validation_target is not None
            model.eval()
            total = 0.0
            with torch.no_grad():
                for start in range(0, len(validation_x), batch_size):
                    encoded = _encode_layout(
                        validation_x[start : start + batch_size],
                        seq_len,
                        aa_to_idx,
                        sequence_layout,
                    )
                    xb = torch.as_tensor(
                        encoded, dtype=torch.long, device=resolved_device
                    )
                    yb = torch.as_tensor(
                        validation_target[start : start + batch_size],
                        device=resolved_device,
                    )
                    fb = (
                        None
                        if validation_f is None
                        else torch.as_tensor(
                            validation_f[start : start + batch_size],
                            device=resolved_device,
                        )
                    )
                    # Unweighted validation BCE is comparable across class-weight experiments.
                    total += len(xb) * float(
                        nn.functional.binary_cross_entropy_with_logits(
                            model(xb, fb), yb
                        ).item()
                    )
            score = total / len(validation_x)
            if score < best_loss:
                best_loss, stale, selected_epoch = score, 0, epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    result = {
        "mode": "multilabel_cnn",
        "sequence_layout": sequence_layout,
        "mask_padding": bool(mask_padding),
        "terminal_concat": sequence_layout == "separate_termini",
        "class_order": list(class_order),
        "seq_len": int(seq_len),
        "aa_to_idx": dict(aa_to_idx),
        "embed_dim": int(embed_dim),
        "num_filters": int(num_filters),
        "kernel_sizes": list(kernel_sizes),
        "dropout": float(dropout),
        "feature_dim": int(feature_dim),
        "feature_mean": None
        if feature_mean is None
        else feature_mean.astype(np.float32).tolist(),
        "feature_scale": None
        if feature_scale is None
        else feature_scale.astype(np.float32).tolist(),
        "class_thresholds": {name: 0.5 for name in class_order},
        "selected_epoch": selected_epoch,
        "distillation_weight": float(distillation_weight)
        if teacher_probabilities is not None
        else 0.0,
        "sample_weight_power": float(sample_weight_power),
        "ensure_one_label": bool(ensure_one_label),
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "device": str(resolved_device),
    }

    if tune_thresholds:
        assert validation_target is not None
        from cdskit.localize_model import _tune_binary_threshold

        prob = predict_multilabel_cnn_batch(
            validation_sequences,
            result,
            feature_matrix=validation_features,
            apply_thresholds=False,
        )["prob_matrix"]
        for i, name in enumerate(class_order):
            if len(np.unique(validation_target[:, i])) == 2:
                result["class_thresholds"][name] = _tune_binary_threshold(
                    prob[:, i],
                    validation_target[:, i],
                    threshold_grid=threshold_grid,
                    objective=dict(threshold_objective_by_class or {}).get(
                        name, threshold_objective
                    ),
                )
    return result


def _get_runtime_cnn_model(localization_model, device_text="cpu"):
    torch, nn = require_torch()
    if "_runtime_model_cache" not in localization_model:
        localization_model["_runtime_model_cache"] = dict()
    cache = localization_model["_runtime_model_cache"]
    resolved_device = resolve_torch_device(device_text=device_text)
    cache_key = str(resolved_device)
    if cache_key in cache:
        return cache[cache_key], resolved_device
    model = _build_multilabel_cnn_module(
        torch=torch,
        nn=nn,
        vocab_size=len(localization_model["aa_to_idx"]),
        embed_dim=int(localization_model["embed_dim"]),
        num_filters=int(localization_model["num_filters"]),
        kernel_sizes=localization_model["kernel_sizes"],
        dropout=float(localization_model.get("dropout", 0.0)),
        num_class=len(localization_model["class_order"]),
        feature_dim=int(localization_model.get("feature_dim", 0)),
        mask_padding=bool(localization_model.get("mask_padding", False)),
        terminal_concat=bool(localization_model.get("terminal_concat", False)),
    )
    model.load_state_dict(localization_model["state_dict"], strict=True)
    model.eval()
    model.to(resolved_device)
    cache[cache_key] = model
    return model, resolved_device


def _normalize_runtime_features(feature_matrix, localization_model, n_row):
    feature_dim = int(localization_model.get("feature_dim", 0))
    if feature_dim <= 0:
        return None
    if feature_matrix is None:
        return np.zeros((n_row, feature_dim), dtype=np.float32)
    feat = np.asarray(feature_matrix, dtype=np.float32)
    if feat.ndim != 2:
        raise ValueError("feature_matrix should be 2D array.")
    if int(feat.shape[0]) != int(n_row):
        txt = "feature_matrix row count mismatch: expected {}, got {}."
        raise ValueError(txt.format(int(n_row), int(feat.shape[0])))
    if int(feat.shape[1]) != int(feature_dim):
        txt = "feature_matrix column mismatch: expected {}, got {}."
        raise ValueError(txt.format(int(feature_dim), int(feat.shape[1])))
    mean = localization_model.get("feature_mean", None)
    scale = localization_model.get("feature_scale", None)
    if (mean is not None) and (scale is not None):
        mean = np.asarray(mean, dtype=np.float32)
        scale = np.asarray(scale, dtype=np.float32)
        if mean.shape[0] == feature_dim and scale.shape[0] == feature_dim:
            safe_scale = scale.copy()
            safe_scale[safe_scale < 1.0e-6] = 1.0
            feat = (feat - mean[np.newaxis, :]) / safe_scale[np.newaxis, :]
    return feat.astype(np.float32)


def predict_multilabel_cnn_batch(
    aa_sequences,
    localization_model,
    device="cpu",
    batch_size=512,
    feature_matrix=None,
    apply_thresholds=True,
):
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive.")
    from cdskit.localize_model import to_canonical_aa_sequence

    # Match CLI preprocessing before either the CNN or specialist sees the input.
    aa_sequences = [to_canonical_aa_sequence(seq) for seq in aa_sequences]
    torch, _ = require_torch()
    model, resolved_device = _get_runtime_cnn_model(
        localization_model=localization_model,
        device_text=device,
    )
    if localization_model.get("sequence_only_features", False):
        from cdskit.localize_specialists import sequence_features

        feature_matrix = sequence_features(aa_sequences)
    feature_x = _normalize_runtime_features(
        feature_matrix=feature_matrix,
        localization_model=localization_model,
        n_row=len(aa_sequences),
    )
    probs = list()
    with torch.no_grad():
        for start in range(0, len(aa_sequences), int(batch_size)):
            encoded = _encode_layout(
                aa_sequences[start : start + int(batch_size)],
                int(localization_model["seq_len"]),
                localization_model["aa_to_idx"],
                localization_model.get("sequence_layout", "legacy"),
            )
            xb = torch.as_tensor(
                encoded,
                dtype=torch.long,
                device=resolved_device,
            )
            fb = None
            if feature_x is not None:
                fb = torch.as_tensor(
                    feature_x[start : start + int(batch_size), :],
                    dtype=torch.float32,
                    device=resolved_device,
                )
            logits = _forward_layout(model, xb, fb)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    if len(probs) == 0:
        prob = np.zeros((0, len(localization_model["class_order"])), dtype=np.float64)
    else:
        prob = np.vstack(probs).astype(np.float64)
    from cdskit.localize_specialists import apply_specialists

    prob = apply_specialists(aa_sequences, localization_model, prob)
    if not apply_thresholds:
        return {"prob_matrix": prob}
    class_order = list(localization_model["class_order"])
    thresholds = localization_model.get("class_thresholds", {})
    threshold_vec = np.asarray(
        [float(thresholds.get(class_name, 0.5)) for class_name in class_order],
        dtype=np.float64,
    )
    threshold_vec[~np.isfinite(threshold_vec)] = 0.5
    threshold_vec[threshold_vec <= 0.0] = 0.5
    pred = (prob >= threshold_vec.reshape((1, -1))).astype(np.int64)
    if bool(localization_model.get("ensure_one_label", True)):
        empty = np.where(np.sum(pred, axis=1) == 0)[0]
        if empty.shape[0] > 0:
            scores = prob[empty, :] / threshold_vec.reshape((1, -1))
            best = np.argmax(scores, axis=1)
            pred[empty, best] = 1
    return {
        "prob_matrix": prob,
        "prediction_matrix": pred,
    }

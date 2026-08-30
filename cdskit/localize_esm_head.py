import numpy as np

from cdskit.esm_embeddings import encoder_cache_key
from cdskit.localize_runtime import offline_requested


DEFAULT_ESM_MODEL_REVISION = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"


def require_transformers():
    try:
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        txt = (
            "transformers + torch are required for --model_arch esm_head. "
            "Install them first. Original error: {}"
        )
        raise ImportError(txt.format(str(exc))) from exc
    return torch, nn, AutoTokenizer, AutoModel


def resolve_torch_device(device_text="auto"):
    torch, _, _, _ = require_transformers()
    device_text = str(device_text).strip().lower()
    if device_text in ["", "auto"]:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device_text == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device was requested but CUDA is not available.")
        return "cuda"
    if device_text == "mps":
        if (not hasattr(torch.backends, "mps")) or (
            not torch.backends.mps.is_available()
        ):
            raise ValueError("MPS device was requested but MPS is not available.")
        return "mps"
    if device_text == "cpu":
        return "cpu"
    raise ValueError("Unsupported --dl_device: {}".format(device_text))


def _pool_last_hidden(last_hidden_state, attention_mask, pooling, torch):
    pooling = str(pooling or "cls").strip().lower()
    if pooling == "cls":
        return last_hidden_state[:, 0, :]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        sum_h = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return sum_h / denom
    raise ValueError("Unsupported --esm_pooling: {}".format(pooling))


def _build_linear_head(nn, in_dim, num_class):
    return nn.Linear(int(in_dim), int(num_class))


def _resolve_model_source(model_name, model_local_dir=""):
    model_name = str(model_name or "").strip()
    model_local_dir = str(model_local_dir or "").strip()
    if model_local_dir != "":
        return model_local_dir, True, "local"
    if model_name == "":
        raise ValueError(
            "--esm_model_name should not be empty when --esm_model_local_dir is not set."
        )
    return model_name, False, "huggingface"


def _encode_frozen_sequences(
    aa_sequences,
    tokenizer,
    encoder,
    device,
    max_len,
    pooling,
    batch_size,
    cache,
    cache_key,
    torch,
):
    features = np.empty(
        (len(aa_sequences), int(encoder.config.hidden_size)), dtype=np.float32
    )
    missing: dict[str, list[int]] = {}
    for index, sequence in enumerate(aa_sequences):
        cached = cache.get(cache_key, sequence) if cache is not None else None
        if cached is None:
            missing.setdefault(sequence, []).append(index)
        else:
            features[index] = cached
    sequences = list(missing)
    with torch.no_grad():
        for start in range(0, len(sequences), int(batch_size)):
            batch = sequences[start : start + int(batch_size)]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(max_len),
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            output = encoder(**tokens)
            pooled = _pool_last_hidden(
                output.last_hidden_state, tokens["attention_mask"], pooling, torch
            )
            values = pooled.detach().cpu().numpy()
            for sequence, value in zip(batch, values, strict=True):
                features[missing[sequence]] = value
                if cache is not None:
                    cache.put(cache_key, sequence, value)
    return features


def fit_esm_head_classifier(
    aa_sequences,
    labels,
    class_order,
    model_name,
    model_local_dir,
    max_len,
    pooling,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    seed,
    use_class_weight,
    device,
    model_revision=DEFAULT_ESM_MODEL_REVISION,
    embedding_cache=None,
):
    torch, nn, AutoTokenizer, AutoModel = require_transformers()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    labels = list(labels)
    class_order = list(class_order)
    if len(labels) == 0:
        raise ValueError("No training sequence for esm_head.")
    if len(aa_sequences) != len(labels):
        raise ValueError("Sequence count and label count mismatch for esm_head.")
    if int(epochs) < 1:
        raise ValueError("--dl_epochs should be >= 1.")
    if int(batch_size) < 1:
        raise ValueError("--dl_batch_size should be >= 1.")
    if int(max_len) < 4:
        raise ValueError("--esm_max_len should be >= 4.")

    resolved_device = resolve_torch_device(device_text=device)
    model_source, local_files_only, source_type = _resolve_model_source(
        model_name=model_name,
        model_local_dir=model_local_dir,
    )
    revision = None if source_type == "local" else str(model_revision or "").strip()
    if source_type != "local" and revision == "":
        raise ValueError("--esm_model_revision is required for a remote ESM model.")
    common_kwargs: dict[str, object] = {
        "local_files_only": bool(local_files_only or offline_requested()),
        "trust_remote_code": False,
    }
    if revision is not None:
        common_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(str(model_source), **common_kwargs)
    encoder = AutoModel.from_pretrained(
        str(model_source),
        use_safetensors=True,
        **common_kwargs,
    )
    encoder.eval()
    encoder.to(resolved_device)
    for p in encoder.parameters():
        p.requires_grad = False

    label_to_idx = {name: i for i, name in enumerate(class_order)}
    y = np.asarray([label_to_idx[v] for v in labels], dtype=np.int64)
    hidden_size = int(encoder.config.hidden_size)
    head = _build_linear_head(
        nn=nn,
        in_dim=hidden_size,
        num_class=len(class_order),
    )
    head.to(resolved_device)

    pooled_features = _encode_frozen_sequences(
        aa_sequences,
        tokenizer,
        encoder,
        resolved_device,
        max_len,
        pooling,
        batch_size,
        embedding_cache,
        encoder_cache_key(str(model_source), revision, max_len, pooling),
        torch,
    )
    # The encoder is frozen; only the much smaller pooled matrix is needed below.
    del encoder, tokenizer

    weight_tensor = None
    if bool(use_class_weight):
        counts = {name: 0 for name in class_order}
        for v in labels:
            counts[v] += 1
        n_total = float(len(labels))
        n_class = float(len(class_order))
        weights = list()
        for name in class_order:
            c = float(counts.get(name, 0))
            if c <= 0:
                weights.append(0.0)
            else:
                weights.append(n_total / (n_class * c))
        weight_tensor = torch.as_tensor(
            np.asarray(weights, dtype=np.float32),
            dtype=torch.float32,
            device=resolved_device,
        )
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    indices = np.arange(len(aa_sequences), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    for _ in range(int(epochs)):
        rng.shuffle(indices)
        for start in range(0, indices.shape[0], int(batch_size)):
            batch_idx = indices[start : start + int(batch_size)]
            batch_y = torch.as_tensor(
                y[batch_idx],
                dtype=torch.long,
                device=resolved_device,
            )
            pooled = torch.as_tensor(pooled_features[batch_idx], device=resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(pooled)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()

    head_state_dict = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    return {
        "class_order": list(class_order),
        "model_name": str(model_name),
        "model_revision": str(revision or ""),
        "model_local_dir": str(model_local_dir),
        "model_source_type": str(source_type),
        "max_len": int(max_len),
        "pooling": str(pooling),
        "head_in_dim": int(hidden_size),
        "head_state_dict": head_state_dict,
        "device": str(resolved_device),
    }


def _get_runtime_esm_encoder_and_head(localization_model, device_text="cpu"):
    _torch, nn, AutoTokenizer, AutoModel = require_transformers()
    if "_runtime_model_cache" not in localization_model:
        localization_model["_runtime_model_cache"] = dict()
    cache = localization_model["_runtime_model_cache"]

    resolved_device = resolve_torch_device(device_text=device_text)
    cache_key = str(resolved_device)
    if cache_key in cache:
        return cache[cache_key], resolved_device

    model_name = str(localization_model.get("model_name", ""))
    model_revision = str(localization_model.get("model_revision", "")).strip()
    model_local_dir = str(localization_model.get("model_local_dir", ""))
    model_source, local_files_only, source_type = _resolve_model_source(
        model_name=model_name,
        model_local_dir=model_local_dir,
    )
    offline = (
        bool(localization_model.get("_runtime_offline", False)) or offline_requested()
    )
    common_kwargs: dict[str, object] = {
        "local_files_only": bool(local_files_only or offline),
        "trust_remote_code": False,
    }
    if source_type != "local":
        if model_revision == "":
            raise ValueError(
                "Remote ESM model revision is missing from the model artifact."
            )
        common_kwargs["revision"] = model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_source),
        **common_kwargs,
    )
    encoder = AutoModel.from_pretrained(
        str(model_source),
        use_safetensors=True,
        **common_kwargs,
    )
    encoder.eval()
    encoder.to(resolved_device)
    for p in encoder.parameters():
        p.requires_grad = False

    head = _build_linear_head(
        nn=nn,
        in_dim=int(localization_model["head_in_dim"]),
        num_class=len(localization_model["class_order"]),
    )
    head.load_state_dict(localization_model["head_state_dict"], strict=True)
    head.eval()
    head.to(resolved_device)
    cache[cache_key] = {
        "tokenizer": tokenizer,
        "encoder": encoder,
        "head": head,
    }
    return cache[cache_key], resolved_device


def predict_esm_head_batch(
    aa_sequences,
    localization_model,
    device="cpu",
    batch_size=128,
):
    torch, _, _, _ = require_transformers()
    runtime, resolved_device = _get_runtime_esm_encoder_and_head(
        localization_model=localization_model,
        device_text=device,
    )
    tokenizer = runtime["tokenizer"]
    encoder = runtime["encoder"]
    head = runtime["head"]
    max_len = int(localization_model["max_len"])
    pooling = str(localization_model.get("pooling", "cls"))

    probs = list()
    with torch.no_grad():
        for start in range(0, len(aa_sequences), int(batch_size)):
            batch_seq = aa_sequences[start : start + int(batch_size)]
            tokens = tokenizer(
                batch_seq,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            tokens = {k: v.to(resolved_device) for k, v in tokens.items()}
            out = encoder(**tokens)
            pooled = _pool_last_hidden(
                last_hidden_state=out.last_hidden_state,
                attention_mask=tokens["attention_mask"],
                pooling=pooling,
                torch=torch,
            )
            logits = head(pooled)
            batch_probs = logits.softmax(dim=1).detach().cpu().numpy()
            probs.append(batch_probs)
    if len(probs) == 0:
        return np.zeros((0, len(localization_model["class_order"])), dtype=np.float64)
    return np.vstack(probs).astype(np.float64)


def predict_esm_head(aa_seq, localization_model, device="cpu"):
    probs = predict_esm_head_batch(
        aa_sequences=[aa_seq],
        localization_model=localization_model,
        device=device,
        batch_size=1,
    )
    class_order = list(localization_model["class_order"])
    class_probs = {class_order[i]: float(probs[0, i]) for i in range(len(class_order))}
    pred_index = int(np.argmax(probs[0, :]))
    return class_order[pred_index], class_probs

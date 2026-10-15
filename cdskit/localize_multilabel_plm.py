"""Frozen ESM residue embeddings and small multilabel localization heads.

Embedding caches contain no fitted task parameters and can be shared across folds.
Long sequences use overlapping windows, averaged at their original coordinates.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cdskit.localize_schema import CURRENT_FEATURE_SCHEMA
from cdskit.localize_decision import guard_multilabel_inputs, threshold_decisions

import numpy as np

from cdskit.esm_embeddings import ESMEmbeddingCache
from cdskit.localize_labels import masked_bce, require_observed

from cdskit.localize_bilstm import require_torch, resolve_torch_device
from cdskit.localize_runtime import offline_requested
from cdskit.util import atomic_output_path


def _build_head(nn, dim, labels, pooling="light_attention", hidden=128, dropout=0.25):
    torch, _ = require_torch()
    if pooling not in (
        "mean",
        "light_attention",
        "label_attention",
        "terminal_attention",
    ):
        raise ValueError("Unsupported PLM pooling: {}".format(pooling))

    class ResidueHead(nn.Module):  # type: ignore[name-defined]
        def __init__(self):
            super().__init__()
            self.pooling = pooling
            if pooling == "mean":
                self.output = nn.Sequential(
                    nn.Linear(dim, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, labels),
                )
            else:
                self.features = nn.Conv1d(dim, hidden, 9, padding=4)
                self.attention = nn.Conv1d(
                    dim,
                    labels if pooling == "label_attention" else hidden,
                    9,
                    padding=4,
                )
                self.dropout = nn.Dropout(dropout)
                if pooling in ("light_attention", "terminal_attention"):
                    extra = 2 * dim if pooling == "terminal_attention" else 0
                    self.output = nn.Linear(hidden * 2 + extra, labels)
                else:
                    self.output_weight = nn.Parameter(torch.empty(labels, hidden))
                    self.output_bias = nn.Parameter(torch.zeros(labels))
                    nn.init.xavier_uniform_(self.output_weight)

        def forward(self, embeddings, mask):
            if self.pooling == "mean":
                pooled = (embeddings * mask[:, :, None]).sum(1) / mask.sum(1)[
                    :, None
                ].clamp_min(1)
                return self.output(pooled)
            x = (embeddings * mask[:, :, None]).transpose(1, 2)
            features = self.dropout(self.features(x).relu())
            attention = (
                self.attention(x)
                .masked_fill(~mask[:, None, :], float("-inf"))
                .softmax(-1)
            )
            if self.pooling == "label_attention":
                pooled = attention @ features.transpose(1, 2)
                return (pooled * self.output_weight[None, :, :]).sum(
                    -1
                ) + self.output_bias
            weighted = (features * attention).sum(-1)
            maximum = features.masked_fill(~mask[:, None, :], float("-inf")).amax(-1)
            parts = [weighted, maximum]
            if self.pooling == "terminal_attention":
                positions = torch.arange(mask.shape[1], device=mask.device)[None, :]
                lengths = mask.sum(1)[:, None]
                for region in (positions < 32, positions >= (lengths - 32)):
                    selected = mask & region
                    parts.append(
                        (embeddings * selected[:, :, None]).sum(1)
                        / selected.sum(1)[:, None].clamp_min(1)
                    )
            return self.output(torch.cat(parts, dim=1))

    return ResidueHead()


class ResidueEncoder:
    def __init__(self, config, device="cpu"):
        self.config = dict(config)
        self.device = resolve_torch_device(device)
        self.model: Any = None
        self.tokenizer: Any = None
        source = str(config.get("model_name", "")).strip()
        revision = str(config.get("revision", "")).strip()
        if not source:
            raise ValueError("PLM model_name is required.")
        if os.path.isdir(source):
            # Local model contents, not a mutable path, identify cached embeddings.
            digest = hashlib.sha256()
            for path in sorted(Path(source).rglob("*")):
                if path.is_file():
                    digest.update(str(path.relative_to(source)).encode())
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
            identity = digest.hexdigest()
        else:
            if len(revision) != 40 or any(
                c not in "0123456789abcdef" for c in revision.lower()
            ):
                raise ValueError(
                    "Remote PLM revision must be an immutable 40-character commit SHA."
                )
            identity = revision
        window = int(config.get("window", 1000))
        overlap = int(config.get("overlap", 128))
        if window < 4 or not 0 <= overlap < window:
            raise ValueError("Require window >= 4 and 0 <= overlap < window.")
        self.window, self.overlap = window, overlap
        self.identity = {
            "model": source,
            "revision": identity,
            "window": window,
            "overlap": overlap,
            "format": "esm_residues_v1",
        }
        self._cache_identity = json.dumps(self.identity, sort_keys=True)
        self._memory_cache = ESMEmbeddingCache(
            int(config.get("memory_cache_bytes", 256 * 1024 * 1024))
        )
        self._window_batch_size = int(config.get("window_batch_size", 8))
        if self._window_batch_size < 1:
            raise ValueError("PLM window_batch_size must be positive.")

    def _load(self):
        if self.model is not None:
            return
        from transformers import AutoTokenizer, EsmModel

        local = os.path.isdir(self.config["model_name"])
        kwargs: dict[str, Any] = {
            "trust_remote_code": False,
            "local_files_only": local
            or bool(self.config.get("local_files_only", False))
            or offline_requested(),
        }
        if not local:
            kwargs["revision"] = self.config["revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"], **kwargs
        )
        torch, _ = require_torch()
        # Loading a frozen encoder must not advance the teacher's RNG stream.
        # Public MLM checkpoints omit the sequence pooler, which we never use.
        with torch.random.fork_rng(devices=[]):
            self.model = EsmModel.from_pretrained(
                self.config["model_name"],
                use_safetensors=True,
                add_pooling_layer=False,
                **kwargs,
            ).eval()
        self.model = self.model.to(self.device)
        self.model.requires_grad_(False)
        if (
            self.window + 2 + int(self.model.config.pad_token_id)
            >= self.model.config.max_position_embeddings
        ):
            raise ValueError("PLM window exceeds model positional capacity.")

    def _cache_path(self, sequence):
        cache_dir = str(self.config.get("cache_dir", "")).strip()
        if not cache_dir:
            return None
        key = hashlib.sha256(
            (self._cache_identity + "\n" + sequence).encode()
        ).hexdigest()
        return Path(cache_dir) / (key + ".npy")

    def encode(self, sequence):
        return self.encode_many([sequence], batch_size=1)[0]

    def encode_many(self, sequences, batch_size=8):
        """Encode distinct misses in length buckets; reuse frozen features only.

        Equal-length windows need no padding. The caller's batch size, a maximum
        of eight windows by default, and an 8192-token budget bound each forward.
        Overlap contributions are still accumulated in original window order.
        """
        sequences = [str(sequence).upper() for sequence in sequences]
        if any(not sequence for sequence in sequences):
            raise ValueError("PLM requires nonempty protein sequences.")
        if batch_size < 1:
            raise ValueError("PLM embedding batch_size must be positive.")
        values = {}
        missing = []
        for sequence in dict.fromkeys(sequences):
            cached = self._memory_cache.get(self._cache_identity, sequence)
            if cached is not None:
                values[sequence] = cached
                continue
            path = self._cache_path(sequence)
            if path is not None and path.exists():
                value = np.load(path, allow_pickle=False)
                if (
                    value.ndim != 2
                    or value.shape[0] != len(sequence)
                    or not np.isfinite(value).all()
                ):
                    raise ValueError("Invalid residue embedding cache: {}".format(path))
                self._memory_cache.put(self._cache_identity, sequence, value)
                values[sequence] = value
            else:
                missing.append(sequence)
        if missing:
            values.update(self._encode_missing(missing, batch_size))
        # Independent writable results preserve encode's public ownership contract.
        return [values[sequence].copy() for sequence in sequences]

    def _encode_missing(self, sequences, batch_size):
        self._load()
        torch, _ = require_torch()
        summed = {
            sequence: np.zeros(
                (len(sequence), self.model.config.hidden_size), dtype=np.float32
            )
            for sequence in sequences
        }
        counts = {
            sequence: np.zeros((len(sequence), 1), dtype=np.float32)
            for sequence in sequences
        }
        windows: dict[int, list[tuple[str, int, str]]] = {}
        for sequence in sequences:
            for start in range(0, len(sequence), self.window - self.overlap):
                part = sequence[start : start + self.window]
                windows.setdefault(len(part), []).append((sequence, start, part))
                if start + len(part) == len(sequence):
                    break
        # Within each sequence all full windows precede its one shorter tail;
        # process longer buckets first to retain the scalar addition order.
        with torch.no_grad():
            for length in sorted(windows, reverse=True):
                bucket = windows[length]
                limit = min(
                    batch_size, self._window_batch_size, max(1, 8192 // (length + 2))
                )
                for offset in range(0, len(bucket), limit):
                    batch = bucket[offset : offset + limit]
                    parts = [part for _, _, part in batch]
                    tokens = self.tokenizer(
                        parts if len(parts) > 1 else parts[0],
                        return_tensors="pt",
                        return_special_tokens_mask=True,
                    )
                    special = tokens.pop("special_tokens_mask").bool()
                    if not torch.all((~special).sum(1) == length):
                        raise ValueError(
                            "Tokenizer must produce exactly one token per amino acid."
                        )
                    tokens = {
                        key: value.to(self.device) for key, value in tokens.items()
                    }
                    hidden = self.model(**tokens).last_hidden_state.cpu()
                    for i, (sequence, start, _part) in enumerate(batch):
                        summed[sequence][start : start + length] += hidden[i][
                            ~special[i]
                        ].numpy()
                        counts[sequence][start : start + length] += 1
        values = {}
        for sequence in sequences:
            value = summed[sequence] / counts[sequence]
            path = self._cache_path(sequence)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                with atomic_output_path(str(path)) as temporary:
                    with open(temporary, "wb") as stream:
                        np.save(stream, value, allow_pickle=False)
            self._memory_cache.put(self._cache_identity, sequence, value)
            values[sequence] = value
        return values


def _batch(encoder, sequences, torch, device):
    values = (
        encoder.encode_many(sequences, batch_size=len(sequences))
        if hasattr(encoder, "encode_many")
        else [encoder.encode(seq) for seq in sequences]
    )
    length = max(len(value) for value in values)
    shape = (len(values), length, values[0].shape[1])
    cuda = torch.device(device).type == "cuda"
    # Assemble CUDA batches on device, without a large, padded NumPy temporary.
    # NumPy's huge-page advice can stall allocation on a crowded NUMA node.
    x: Any = (
        torch.zeros(shape, dtype=torch.float32, device=device)
        if cuda
        else np.zeros(shape, dtype=np.float32)
    )
    mask = np.zeros((len(values), length), dtype=bool)
    for i, value in enumerate(values):
        if cuda:
            x[i, : len(value)].copy_(torch.as_tensor(value, device="cpu"))
        else:
            x[i, : len(value)] = value
        mask[i, : len(value)] = True
    return torch.as_tensor(x, device=device), torch.as_tensor(mask, device=device)


def _training_loss(torch, nn, name, y, device):
    """Weights use training labels only; ASL focusing weights are detached."""

    class MaskedBCE(nn.BCEWithLogitsLoss):  # type: ignore[name-defined]
        def forward(self, logits, target):
            return masked_bce(
                logits, target, pos_weight=self.pos_weight, reduction=self.reduction
            )

    if name == "bce":
        return MaskedBCE()
    if name == "weighted_bce":
        positive = (y == 1).sum(0)
        negative = (y == 0).sum(0)
        weights = np.ones(y.shape[1], dtype=np.float32)
        supported = (positive > 0) & (negative > 0)
        weights[supported] = np.sqrt(negative[supported] / positive[supported])
        pos_weight = torch.as_tensor(np.minimum(weights, 10), device=device)
        return MaskedBCE(pos_weight=pos_weight)
    if name != "asl":
        raise ValueError("Unsupported PLM loss: {}".format(name))

    def asymmetric(logits, target):
        observed = ~torch.isnan(target)
        target = torch.where(observed, target, torch.zeros_like(target))
        positive = logits.sigmoid()
        negative = (1 - positive + 0.05).clamp(max=1)
        log_positive = torch.nn.functional.logsigmoid(logits)
        log_negative = negative.clamp_min(torch.finfo(logits.dtype).tiny).log()
        probability = positive * target + negative * (1 - target)
        gamma = target + 4 * (1 - target)
        weight = ((1 - probability) ** gamma).detach()
        cells = -(weight * (target * log_positive + (1 - target) * log_negative))
        return torch.where(
            observed, cells, torch.zeros_like(cells)
        ).sum() / observed.sum().clamp_min(1)

    return asymmetric


def fit_multilabel_plm(
    sequences,
    y,
    labels,
    config,
    validation_sequences=None,
    validation_y=None,
    epochs=6,
    batch_size=8,
    learning_rate=1e-3,
    seed=1,
    device="auto",
    patience=3,
    loss="bce",
    selection_metric="bce",
):
    if validation_sequences and set(str(s).upper() for s in sequences).intersection(
        str(s).upper() for s in validation_sequences
    ):
        raise ValueError("Training and validation sequences overlap.")
    torch, nn = require_torch()
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size and patience must be positive.")
    y = np.asarray(y, dtype=np.float32)
    if y.shape != (len(sequences), len(labels)) or not len(sequences):
        raise ValueError("Invalid PLM training label dimensions.")
    require_observed(y, "PLM training data", each_label=True)
    if validation_sequences:
        target = np.asarray(validation_y, dtype=np.float32)
        if target.shape != (len(validation_sequences), len(labels)):
            raise ValueError("Invalid PLM validation label dimensions.")
        require_observed(target, "PLM validation data")
    if selection_metric not in ("bce", "macro_ap"):
        raise ValueError("Unsupported PLM selection_metric.")
    torch.manual_seed(seed)
    resolved = resolve_torch_device(device)
    encoder = ResidueEncoder(config, resolved)
    dim = encoder.encode(sequences[0]).shape[1]
    head = _build_head(
        nn, dim, len(labels), config.get("pooling", "light_attention")
    ).to(resolved)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    loss_fn = _training_loss(torch, nn, loss, y, resolved)
    rng = np.random.default_rng(seed)
    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    training_history = []
    for epoch in range(epochs):
        head.train()
        indices = rng.permutation(len(sequences))
        for start in range(0, len(indices), batch_size):
            ids = indices[start : start + batch_size]
            if not np.isfinite(y[ids]).any():
                continue
            xb, mask = _batch(encoder, [sequences[i] for i in ids], torch, resolved)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = loss_fn(
                head(xb, mask), torch.as_tensor(y[ids], device=resolved)
            )
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        head.eval()
        if validation_sequences:
            target = np.asarray(validation_y, dtype=np.float32)
            if target.shape != (len(validation_sequences), len(labels)):
                raise ValueError("Invalid PLM validation label dimensions.")
            total = 0.0
            probabilities = []
            with torch.no_grad():
                for start in range(0, len(validation_sequences), batch_size):
                    seqs = validation_sequences[start : start + batch_size]
                    xb, mask = _batch(encoder, seqs, torch, resolved)
                    logits = head(xb, mask)
                    probabilities.append(logits.sigmoid().cpu().numpy())
                    total += float(
                        masked_bce(
                            logits,
                            torch.as_tensor(
                                target[start : start + batch_size], device=resolved
                            ),
                            reduction="sum",
                        ).item()
                    )
            validation_bce = total / int(np.isfinite(target).sum())
            from cdskit.localize_evaluation import average_precision

            probability = np.concatenate(probabilities)
            aps = [
                average_precision(
                    target[np.isfinite(target[:, j]), j],
                    probability[np.isfinite(target[:, j]), j],
                )
                for j in range(len(labels))
            ]
            supported_ap = [value for value in aps if value is not None]
            validation_ap = float(np.mean(supported_ap)) if supported_ap else None
            if selection_metric == "bce":
                score = validation_bce
            elif validation_ap is None:
                raise ValueError(
                    "macro_ap selection requires positive validation labels."
                )
            else:
                score = -validation_ap
        else:
            score = -epoch  # Fixed training budget without a validation partition.
        training_history.append(
            {
                "epoch": epoch + 1,
                "validation_bce": validation_bce if validation_sequences else None,
                "validation_macro_ap": validation_ap if validation_sequences else None,
            }
        )
        if score < best:
            best, best_epoch, stale = score, epoch + 1, 0
            best_state = {
                key: val.detach().cpu().clone()
                for key, val in head.state_dict().items()
            }
        else:
            stale += 1
            if stale >= patience:
                break
    return {
        "mode": "multilabel_plm",
        "label_contract": "observed_binary_v1",
        "observed_counts": np.isfinite(y).sum(axis=0).tolist(),
        "class_order": list(labels),
        "encoder": dict(config),
        "encoder_identity": encoder.identity,
        "embedding_dim": dim,
        "pooling": config.get("pooling", "light_attention"),
        "state_dict": best_state,
        "selected_epoch": best_epoch,
        "training_loss": loss,
        "selection_metric": selection_metric,
        "training_history": training_history,
        "class_thresholds": {label: 0.5 for label in labels},
        "ensure_one_label": True,
        "feature_schema": CURRENT_FEATURE_SCHEMA,
    }


@guard_multilabel_inputs("sequences", "model")
def predict_multilabel_plm(
    sequences, model, device="cpu", batch_size=8, apply_thresholds=True
):
    torch, nn = require_torch()
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    resolved = resolve_torch_device(device)
    cache = model.setdefault("_runtime_model_cache", {})
    if resolved not in cache:
        encoder = ResidueEncoder(model["encoder"], resolved)
        if encoder.identity != model["encoder_identity"]:
            raise ValueError(
                "PLM encoder contents differ from the trained model identity."
            )
        head = _build_head(
            nn, model["embedding_dim"], len(model["class_order"]), model["pooling"]
        )
        head.load_state_dict(model["state_dict"], strict=True)
        cache[resolved] = encoder, head.eval().to(resolved)
    encoder, head = cache[resolved]
    values = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            xb, mask = _batch(
                encoder, sequences[start : start + batch_size], torch, resolved
            )
            values.append(head(xb, mask).sigmoid().cpu().numpy())
    prob = (
        np.concatenate(values) if values else np.zeros((0, len(model["class_order"])))
    )
    from cdskit.localize_specialists import apply_specialists

    prob = apply_specialists(sequences, model, prob)
    result = {"prob_matrix": prob}
    if apply_thresholds:
        result.update(
            threshold_decisions(
                prob,
                model.get("class_thresholds", {}),
                model["class_order"],
                model.get("ensure_one_label", True),
            )
        )
    return result

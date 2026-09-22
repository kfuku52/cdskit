"""Provenance-bound out-of-fold artifacts, independent of training and evaluation."""

import hashlib
import json
import os
import numpy as np
from cdskit import __version__


def _validate_oof_arrays(prob_matrix, true_idx, class_names, row_index=None):
    from cdskit.localize_probabilities import validate_scores

    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("OOF class names must be nonempty and unique.")
    validate_scores(prob_matrix, class_names)
    if (
        true_idx.shape != (len(prob_matrix),)
        or true_idx.dtype.kind not in "iu"
        or np.any((true_idx < 0) | (true_idx >= len(class_names)))
    ):
        raise ValueError("OOF target indices must match rows and class names.")
    if not np.allclose(prob_matrix.sum(axis=1), 1.0):
        raise ValueError("OOF probability rows must sum to one.")
    if row_index is not None and (
        row_index.shape != true_idx.shape
        or row_index.dtype.kind not in "iu"
        or np.any(row_index < 0)
        or len(np.unique(row_index)) != len(row_index)
    ):
        raise ValueError("OOF row indices must be nonnegative, unique and aligned.")


def _save_oof_npz(path, prob_matrix, true_idx, class_names, cache_key=""):
    out_dir = os.path.dirname(path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    from cdskit.atomicio import atomic_output_path

    with atomic_output_path(path) as temporary:
        with open(temporary, "wb") as out:
            np.savez_compressed(
                out,
                prob_matrix=np.asarray(prob_matrix, dtype=np.float64),
                true_idx=np.asarray(true_idx, dtype=np.int64),
                class_names=np.asarray(class_names),
                cache_key=str(cache_key),
            )


def _load_oof_npz(path, fallback_true_idx=None, cache_key=""):
    with np.load(path, allow_pickle=False) as data:
        prob_matrix = np.asarray(data["prob_matrix"], dtype=np.float64)
        if "true_idx" in data.files:
            true_idx = np.asarray(data["true_idx"])
        elif fallback_true_idx is not None:
            true_idx = np.asarray(fallback_true_idx)
        else:
            raise KeyError(
                "true_idx is not a file in the archive and no fallback_true_idx was provided."
            )
        class_names = [str(v) for v in data["class_names"].tolist()]
        if cache_key:
            if "cache_key" not in data.files:
                raise ValueError(
                    "OOF cache has no provenance metadata: {}".format(path)
                )
            loaded_key = str(np.asarray(data["cache_key"]).tolist())
            if loaded_key != str(cache_key):
                raise ValueError(
                    "OOF cache provenance does not match current run: {}".format(path)
                )
    _validate_oof_arrays(prob_matrix, true_idx, class_names)
    return prob_matrix, true_idx, class_names


def _safe_cache_name(value):
    text = str(value or "").strip()
    if text == "":
        text = "fold"
    out = list()
    for ch in text:
        if ch.isalnum() or ch in ["-", "_", "."]:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _oof_fold_cache_path(cache_dir, model_arch, fold_label):
    return os.path.join(
        str(cache_dir),
        "{}_{}.npz".format(_safe_cache_name(model_arch), _safe_cache_name(fold_label)),
    )


def _content_fingerprint(values):
    digest = hashlib.sha256()

    def update_part(part):
        digest.update(len(part).to_bytes(8, byteorder="big", signed=False))
        digest.update(part)

    for value in values:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            update_part(str(array.dtype).encode("utf-8"))
            update_part(str(array.shape).encode("utf-8"))
            update_part(array.tobytes())
        else:
            update_part(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def _training_file_cache_key(
    path, model_arch, localize_strategy, dl_train_params, cv_seed
):
    digest = hashlib.sha256()
    with open(path, "rb") as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b""):
            digest.update(chunk)
    return _content_fingerprint(
        [
            __version__,
            digest.hexdigest(),
            model_arch,
            localize_strategy,
            dict(sorted(dict(dl_train_params).items())),
            int(cv_seed),
        ]
    )


def _oof_fold_cache_key(
    model_arch,
    localize_strategy,
    dl_train_params,
    x,
    aa_sequences,
    class_labels,
    perox_labels,
    fold_ids,
    cv_seed,
):
    payload = {
        "cdskit_version": __version__,
        "model_arch": str(model_arch),
        "localize_strategy": str(localize_strategy),
        "dl_train_params": dict(sorted(dict(dl_train_params).items())),
        "cv_seed": int(cv_seed),
        "training_content_sha256": _content_fingerprint(
            [
                np.asarray(x),
                list(aa_sequences),
                list(class_labels),
                list(perox_labels),
                None if fold_ids is None else list(fold_ids),
            ]
        ),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _save_oof_fold_npz(
    path, row_index, prob_matrix, true_idx, class_names, fold_label, cache_key=""
):
    out_dir = os.path.dirname(path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    from cdskit.atomicio import atomic_output_path

    with atomic_output_path(path) as temporary:
        with open(temporary, "wb") as out:
            np.savez_compressed(
                out,
                row_index=np.asarray(row_index, dtype=np.int64),
                prob_matrix=np.asarray(prob_matrix, dtype=np.float64),
                true_idx=np.asarray(true_idx, dtype=np.int64),
                class_names=np.asarray(list(class_names)),
                fold_label=str(fold_label),
                cache_key=str(cache_key),
            )


def _load_oof_fold_npz(path, class_names, cache_key=""):
    with np.load(path, allow_pickle=False) as data:
        loaded_names = [str(v) for v in np.asarray(data["class_names"]).tolist()]
        if loaded_names != list(class_names):
            raise ValueError(
                "Class names in OOF fold cache do not match LOCALIZATION_CLASSES."
            )
        if str(cache_key or "") != "":
            if "cache_key" not in data.files:
                raise ValueError(
                    "OOF fold cache has no cache_key metadata: {}".format(path)
                )
            loaded_key = str(np.asarray(data["cache_key"]).tolist())
            if loaded_key != str(cache_key):
                raise ValueError(
                    "OOF fold cache parameters do not match current run: {}".format(
                        path
                    )
                )
        row_index = np.asarray(data["row_index"])
        prob_matrix = np.asarray(data["prob_matrix"], dtype=np.float64)
        true_idx = np.asarray(data["true_idx"])
    _validate_oof_arrays(prob_matrix, true_idx, class_names, row_index)
    return row_index, prob_matrix, true_idx

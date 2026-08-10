from functools import partial

import numpy as np

from cdskit.localize_models import (
    is_pretrained_localize_model_alias,
    resolve_localize_model_path,
)
from cdskit.localize_model import (
    BROAD_FEATURE_NAMES,
    FEATURE_NAMES,
    LOCALIZATION_CLASSES,
    apply_organism_group_constraints,
    extract_broad_localize_features,
    extract_localize_features,
    load_localize_model,
    normalize_localization_probability_matrix,
    postprocess_localization_probabilities,
    predict_perox_batch,
    predict_localization_and_peroxisome,
    predict_multilabel_localization,
    to_canonical_aa_sequence,
    translate_inframe_cds_to_aa,
    write_rows_json,
    write_rows_tsv,
    apply_targetp_feature_ltp_specialist_postprocess_batch,
    apply_targetp_specialist_postprocess_batch,
    compose_two_stage_ctp_ltp_probabilities,
)
from cdskit.util import (
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
    stop_if_invalid_codontable,
    stop_if_not_dna,
    stop_if_not_multiple_of_three,
    stop_if_not_protein,
)


MULTILABEL_MODEL_TYPES = {"multilabel_centroid_v1", "multilabel_cnn_v1"}
SINGLE_LABEL_BATCH_MODEL_TYPES = {
    "bilstm_attention_v1",
    "esm_head_v1",
    "nearest_centroid_v1",
    "targetp_blend_v1",
    "targetp_feature_ensemble_v1",
    "targetp_torch_v1",
}
DEFAULT_LOCALIZE_BATCH_SIZE = 512
DEFAULT_ESM_BATCH_SIZE = 128


def _is_true_arg(value):
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _float_or_default(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _configure_ml_threads(model, threads):
    def configure_estimators(value):
        if isinstance(value, dict):
            for nested in value.values():
                configure_estimators(nested)
            return
        if isinstance(value, (list, tuple)):
            for nested in value:
                configure_estimators(nested)
            return
        if hasattr(value, "n_jobs"):
            try:
                value.n_jobs = max(1, int(threads))
            except (AttributeError, TypeError, ValueError):
                pass

    configure_estimators(model)
    model_type = str(model.get("model_type", ""))
    if model_type == "targetp_blend_v1":
        for base_model in model.get("localization_model", {}).get("base_models", []):
            if isinstance(base_model, dict):
                _configure_ml_threads(model=base_model, threads=threads)
        return
    if model_type not in {"bilstm_attention_v1", "esm_head_v1", "targetp_torch_v1"}:
        return
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(max(1, int(threads)))
    try:
        torch.set_num_interop_threads(max(1, min(int(threads), 4)))
    except RuntimeError:
        # PyTorch permits configuring inter-op threads only before parallel work.
        pass


def _record_to_aa_sequence(record, codontable, seqtype):
    seqtype = str(seqtype or "dna").strip().lower()
    if seqtype == "protein":
        return to_canonical_aa_sequence(aa_seq=str(record.seq))
    if seqtype == "dna":
        return translate_inframe_cds_to_aa(
            cds_seq=str(record.seq),
            codontable=codontable,
            seq_id=record.id,
        )
    raise ValueError("--seq_type should be dna or protein.")


def _predict_single_record(
    record, codontable, seqtype, model, include_features, organism_group=""
):
    aa_seq = _record_to_aa_sequence(
        record=record,
        codontable=codontable,
        seqtype=seqtype,
    )
    if str(model.get("model_type", "")) in MULTILABEL_MODEL_TYPES:
        pred = predict_multilabel_localization(
            aa_seq=aa_seq,
            model=model,
            kingdom=organism_group,
        )
        class_order = list(model["localization_model"]["class_order"])
        row = {
            "seq_id": record.id,
            "predicted_labels": ";".join(pred["predicted_labels"]),
        }
        for class_name in class_order:
            row["p_{}".format(class_name)] = float(
                pred["class_probabilities"].get(class_name, 0.0)
            )
        if "peroxisome" in class_order:
            row["perox_signal_type"] = pred["perox_signal_type"]
        if include_features:
            for name, value in zip(
                BROAD_FEATURE_NAMES, pred["feature_values"], strict=False
            ):
                row[name] = float(value)
        return row

    pred = predict_localization_and_peroxisome(
        aa_seq=aa_seq,
        model=model,
        organism_group=organism_group,
    )
    row = {
        "seq_id": record.id,
        "predicted_class": pred["predicted_class"],
        "p_noTP": float(pred["class_probabilities"].get("noTP", 0.0)),
        "p_SP": float(pred["class_probabilities"].get("SP", 0.0)),
        "p_mTP": float(pred["class_probabilities"].get("mTP", 0.0)),
        "p_cTP": float(pred["class_probabilities"].get("cTP", 0.0)),
        "p_lTP": float(pred["class_probabilities"].get("lTP", 0.0)),
        "p_peroxisome": float(pred["perox_probability_yes"]),
        "perox_signal_type": pred["perox_signal_type"],
    }
    if include_features:
        for name, value in zip(FEATURE_NAMES, pred["feature_values"], strict=False):
            row[name] = float(value)
    return row


def _row_from_single_label_prediction(
    record_id,
    pred_class,
    class_probs,
    perox_probs,
    perox_signals,
    feature_vec,
    include_features,
):
    row = {
        "seq_id": record_id,
        "predicted_class": pred_class,
        "p_noTP": float(class_probs.get("noTP", 0.0)),
        "p_SP": float(class_probs.get("SP", 0.0)),
        "p_mTP": float(class_probs.get("mTP", 0.0)),
        "p_cTP": float(class_probs.get("cTP", 0.0)),
        "p_lTP": float(class_probs.get("lTP", 0.0)),
        "p_peroxisome": float(perox_probs.get("yes", 0.0)),
        "perox_signal_type": perox_signals["signal_type"],
    }
    if include_features:
        for name, value in zip(FEATURE_NAMES, feature_vec, strict=False):
            row[name] = float(value)
    return row


def _predict_single_label_records_batched(
    records,
    aa_sequences,
    model,
    include_features,
    organism_group="",
):
    model_type = str(model.get("model_type", ""))
    localization_model = model["localization_model"]
    feature_rows = [extract_localize_features(aa_seq=aa_seq) for aa_seq in aa_sequences]
    feature_matrix = np.asarray(
        [feature_vec for feature_vec, _ in feature_rows],
        dtype=np.float32,
    )
    prob_matrix, class_order, batch_details = _predict_model_probability_matrix(
        aa_sequences=aa_sequences,
        feature_matrix=feature_matrix,
        model_type=model_type,
        localization_model=localization_model,
        organism_group=organism_group,
    )

    perox_yes = predict_perox_batch(
        feature_matrix=feature_matrix,
        perox_model=model["perox_model"],
        aa_sequences=aa_sequences,
        organism_group=organism_group,
    )

    processed_probs = []
    pred_classes = []
    for i in range(len(records)):
        class_probs = {
            class_order[class_i]: float(prob_matrix[i, class_i])
            for class_i in range(len(class_order))
        }
        if model_type == "targetp_blend_v1":
            pred_class, class_probs = postprocess_localization_probabilities(
                class_probs=class_probs,
                localization_model=localization_model,
            )
        else:
            class_probs = apply_organism_group_constraints(
                class_probs=class_probs,
                organism_group=organism_group,
            )
            pred_class, class_probs = postprocess_localization_probabilities(
                class_probs=class_probs,
                localization_model=localization_model,
            )
        processed_probs.append(class_probs)
        pred_classes.append(pred_class)

    processed_matrix = np.asarray(
        [
            [row[class_name] for class_name in LOCALIZATION_CLASSES]
            for row in processed_probs
        ],
        dtype=np.float64,
    )
    if model_type == "targetp_blend_v1":
        base_matrices = batch_details["base_probabilities"]
        pred_classes = apply_targetp_specialist_postprocess_batch(
            aa_sequences=aa_sequences,
            base_prob_matrix=processed_matrix,
            prob_a_matrix=base_matrices[0],
            prob_b_matrix=base_matrices[1],
            localization_model=localization_model,
            organism_group=organism_group,
        )
    elif model_type == "targetp_feature_ensemble_v1":
        pred_classes = apply_targetp_feature_ltp_specialist_postprocess_batch(
            aa_sequences=aa_sequences,
            base_prob_matrix=processed_matrix,
            pred_classes=pred_classes,
            localization_model=localization_model,
            organism_group=organism_group,
        )

    rows = list()
    for i, record in enumerate(records):
        feature_vec, perox_signals = feature_rows[i]
        p_yes = float(perox_yes[i])
        rows.append(
            _row_from_single_label_prediction(
                record_id=record.id,
                pred_class=pred_classes[i],
                class_probs=processed_probs[i],
                perox_probs={"yes": p_yes, "no": 1.0 - p_yes},
                perox_signals=perox_signals,
                feature_vec=feature_vec,
                include_features=include_features,
            )
        )
    return rows


def _predict_model_probability_matrix(
    aa_sequences,
    feature_matrix,
    model_type,
    localization_model,
    organism_group,
):
    """Batch base predictors and recursively batch TargetP blend models."""
    num_rows = len(aa_sequences)
    strategy = str(localization_model.get("strategy", "single_stage")).strip().lower()
    if strategy in {"two_stage", "two_stage_ctp_ltp"}:
        stage1, stage1_order, _ = _predict_model_probability_matrix(
            aa_sequences=aa_sequences,
            feature_matrix=feature_matrix,
            model_type=model_type,
            localization_model=localization_model.get("stage1_model", {}),
            organism_group=organism_group,
        )
        stage2, stage2_order, _ = _predict_model_probability_matrix(
            aa_sequences=aa_sequences,
            feature_matrix=feature_matrix,
            model_type=model_type,
            localization_model=localization_model.get("stage2_model", {}),
            organism_group=organism_group,
        )
        base_matrix = np.zeros((num_rows, len(LOCALIZATION_CLASSES)), dtype=np.float64)
        no_tp_col = stage1_order.index("noTP")
        tp_col = stage1_order.index("TP") if "TP" in stage1_order else None
        base_matrix[:, 0] = stage1[:, no_tp_col]
        tp_probs = (
            stage1[:, tp_col]
            if tp_col is not None
            else np.maximum(0.0, 1.0 - base_matrix[:, 0])
        )
        for class_i, class_name in enumerate(LOCALIZATION_CLASSES[1:], start=1):
            if class_name in stage2_order:
                base_matrix[:, class_i] = (
                    tp_probs * stage2[:, stage2_order.index(class_name)]
                )
        totals = np.sum(base_matrix, axis=1, keepdims=True)
        totals[totals <= 0.0] = 1.0
        base_matrix /= totals
        if strategy == "two_stage":
            return base_matrix, list(LOCALIZATION_CLASSES), {}
        stage3_model = localization_model.get("stage3_model", None)
        if not isinstance(stage3_model, dict) or len(stage3_model) == 0:
            return base_matrix, list(LOCALIZATION_CLASSES), {}
        stage3, stage3_order, _ = _predict_model_probability_matrix(
            aa_sequences=aa_sequences,
            feature_matrix=feature_matrix,
            model_type=model_type,
            localization_model=stage3_model,
            organism_group=organism_group,
        )
        out_matrix = np.zeros_like(base_matrix)
        for row_i in range(num_rows):
            base_probs = {
                class_name: float(base_matrix[row_i, class_i])
                for class_i, class_name in enumerate(LOCALIZATION_CLASSES)
            }
            stage3_probs = {
                class_name: float(stage3[row_i, class_i])
                for class_i, class_name in enumerate(stage3_order)
            }
            out_probs, _ = compose_two_stage_ctp_ltp_probabilities(
                base_class_probs=base_probs,
                stage3_ctp_ltp_probs=stage3_probs,
                stage3_gate_threshold=localization_model.get(
                    "stage3_gate_threshold", 0.0
                ),
                stage3_blend_beta=localization_model.get("stage3_blend_beta", 1.0),
                stage3_ltp_threshold=localization_model.get(
                    "stage3_ltp_threshold", 0.5
                ),
            )
            out_matrix[row_i] = [out_probs[name] for name in LOCALIZATION_CLASSES]
        return out_matrix, list(LOCALIZATION_CLASSES), {}
    if str(localization_model.get("mode", "")).strip().lower() == "constant":
        class_order = list(localization_model.get("class_order", LOCALIZATION_CLASSES))
        class_label = str(localization_model.get("class_label", "")).strip()
        if class_label == "" and len(class_order) == 1:
            class_label = class_order[0]
        matrix = np.zeros((num_rows, len(class_order)), dtype=np.float64)
        matrix[:, class_order.index(class_label)] = 1.0
        return matrix, class_order, {}
    if model_type == "bilstm_attention_v1":
        from cdskit.localize_bilstm import predict_bilstm_attention_batch

        prob_matrix = predict_bilstm_attention_batch(
            aa_sequences=aa_sequences,
            localization_model=localization_model,
            device="cpu",
            batch_size=DEFAULT_LOCALIZE_BATCH_SIZE,
            feature_matrix=feature_matrix,
        )
        return np.asarray(prob_matrix), list(localization_model["class_order"]), {}
    elif model_type == "esm_head_v1":
        from cdskit.localize_esm_head import predict_esm_head_batch

        prob_matrix = predict_esm_head_batch(
            aa_sequences=aa_sequences,
            localization_model=localization_model,
            device="cpu",
            batch_size=DEFAULT_ESM_BATCH_SIZE,
        )
        return np.asarray(prob_matrix), list(localization_model["class_order"]), {}
    elif model_type == "targetp_torch_v1":
        from cdskit.targetp_torch import predict_targetp2_torch_batch

        prob_matrix = predict_targetp2_torch_batch(
            aa_sequences=aa_sequences,
            organism_groups=[organism_group] * len(aa_sequences),
            localization_model=localization_model,
            device="cpu",
            batch_size=DEFAULT_LOCALIZE_BATCH_SIZE,
        )
        return np.asarray(prob_matrix), list(LOCALIZATION_CLASSES), {}
    elif model_type == "targetp_feature_ensemble_v1":
        from cdskit.localize_model import predict_targetp_feature_ensemble_batch

        prob_matrix = predict_targetp_feature_ensemble_batch(
            aa_sequences=aa_sequences,
            localization_model=localization_model,
            organism_group=organism_group,
        )
        return prob_matrix, list(LOCALIZATION_CLASSES), {}
    elif model_type == "nearest_centroid_v1":
        mean = np.asarray(localization_model["mean"], dtype=np.float64)
        std = np.asarray(localization_model["std"], dtype=np.float64)
        centroids = np.asarray(localization_model["centroids"], dtype=np.float64)
        log_priors = np.asarray(localization_model["log_priors"], dtype=np.float64)
        z = (np.asarray(feature_matrix, dtype=np.float64) - mean) / std
        logits = (
            -0.5
            * np.sum(
                (z[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
                axis=2,
            )
            + log_priors[np.newaxis, :]
        )
        logits -= np.max(logits, axis=1, keepdims=True)
        prob_matrix = np.exp(logits)
        prob_matrix /= np.sum(prob_matrix, axis=1, keepdims=True)
        return prob_matrix, list(localization_model["class_order"]), {}
    elif model_type == "targetp_blend_v1":
        base_models = localization_model.get("base_models", [])
        if not isinstance(base_models, list) or len(base_models) != 2:
            raise ValueError("targetp_blend_v1 requires exactly two base_models.")
        base_matrices = []
        for base_model in base_models:
            base_matrix, base_order, _ = _predict_model_probability_matrix(
                aa_sequences=aa_sequences,
                feature_matrix=feature_matrix,
                model_type=str(base_model.get("model_type", "")).strip(),
                localization_model=base_model.get("localization_model", {}),
                organism_group=organism_group,
            )
            reordered = np.zeros(
                (num_rows, len(LOCALIZATION_CLASSES)),
                dtype=np.float64,
            )
            for class_i, class_name in enumerate(LOCALIZATION_CLASSES):
                if class_name in base_order:
                    reordered[:, class_i] = base_matrix[:, base_order.index(class_name)]
            reordered = normalize_localization_probability_matrix(
                probability_matrix=reordered,
                organism_group=organism_group,
            )
            base_matrices.append(reordered)
        alpha_by_class = localization_model.get("alpha_by_class", 1.0)
        alpha = np.ones((len(LOCALIZATION_CLASSES),), dtype=np.float64)
        if isinstance(alpha_by_class, dict):
            alpha[:] = [
                _float_or_default(alpha_by_class.get(class_name, 1.0), 1.0)
                for class_name in LOCALIZATION_CLASSES
            ]
        else:
            try:
                alpha[:] = float(alpha_by_class)
            except (TypeError, ValueError):
                alpha[:] = 1.0
        alpha = np.clip(alpha, 0.0, 1.0)
        prob_matrix = (
            alpha[np.newaxis, :] * base_matrices[0]
            + (1.0 - alpha[np.newaxis, :]) * base_matrices[1]
        )
        totals = np.sum(prob_matrix, axis=1, keepdims=True)
        totals[totals <= 0.0] = 1.0
        prob_matrix /= totals
        return (
            prob_matrix,
            list(LOCALIZATION_CLASSES),
            {
                "base_probabilities": base_matrices,
            },
        )
    raise ValueError("Unsupported batched localize model_type: {}".format(model_type))


def _predict_multilabel_cnn_records_batched(
    records,
    aa_sequences,
    model,
    include_features,
    organism_group="",
):
    from cdskit.localize_multilabel_cnn import predict_multilabel_cnn_batch

    localization_model = model["localization_model"]
    feature_rows = [
        extract_broad_localize_features(
            aa_seq=aa_seq,
            kingdom=organism_group,
        )
        for aa_seq in aa_sequences
    ]
    feature_matrix = None
    if int(localization_model.get("feature_dim", 0)) > 0:
        feature_matrix = np.asarray(
            [feature_vec for feature_vec, _ in feature_rows],
            dtype=np.float32,
        )
    pred = predict_multilabel_cnn_batch(
        aa_sequences=aa_sequences,
        localization_model=localization_model,
        device="cpu",
        batch_size=DEFAULT_LOCALIZE_BATCH_SIZE,
        feature_matrix=feature_matrix,
        apply_thresholds=True,
    )
    class_order = list(localization_model["class_order"])
    prob_matrix = pred["prob_matrix"]
    pred_matrix = pred["prediction_matrix"]
    rows = list()
    for i, record in enumerate(records):
        feature_vec, perox_signals = feature_rows[i]
        labels = [
            class_order[class_i]
            for class_i in range(len(class_order))
            if int(pred_matrix[i, class_i]) == 1
        ]
        row = {
            "seq_id": record.id,
            "predicted_labels": ";".join(labels),
        }
        for class_i, class_name in enumerate(class_order):
            row["p_{}".format(class_name)] = float(prob_matrix[i, class_i])
        if "peroxisome" in class_order:
            row["perox_signal_type"] = perox_signals["signal_type"]
        if include_features:
            for name, value in zip(BROAD_FEATURE_NAMES, feature_vec, strict=False):
                row[name] = float(value)
        rows.append(row)
    return rows


def _predict_records_batched_if_supported(
    records,
    codontable,
    seqtype,
    model,
    include_features,
    organism_group="",
):
    model_type = str(model.get("model_type", ""))
    if (
        model_type not in SINGLE_LABEL_BATCH_MODEL_TYPES
        and model_type != "multilabel_cnn_v1"
    ):
        return None
    if len(records) == 0:
        return []
    aa_sequences = [
        _record_to_aa_sequence(
            record=record,
            codontable=codontable,
            seqtype=seqtype,
        )
        for record in records
    ]
    if model_type == "multilabel_cnn_v1":
        return _predict_multilabel_cnn_records_batched(
            records=records,
            aa_sequences=aa_sequences,
            model=model,
            include_features=include_features,
            organism_group=organism_group,
        )
    return _predict_single_label_records_batched(
        records=records,
        aa_sequences=aa_sequences,
        model=model,
        include_features=include_features,
        organism_group=organism_group,
    )


def _resolve_output_fields(include_features, model=None):
    if (
        isinstance(model, dict)
        and str(model.get("model_type", "")) in MULTILABEL_MODEL_TYPES
    ):
        class_order = list(model["localization_model"]["class_order"])
        fields = ["seq_id", "predicted_labels"]
        fields.extend(["p_{}".format(class_name) for class_name in class_order])
        if "peroxisome" in class_order:
            fields.append("perox_signal_type")
        if include_features:
            fields.extend(BROAD_FEATURE_NAMES)
        return fields

    fields = [
        "seq_id",
        "predicted_class",
        "p_noTP",
        "p_SP",
        "p_mTP",
        "p_cTP",
        "p_lTP",
        "p_peroxisome",
        "perox_signal_type",
    ]
    if include_features:
        fields.extend(FEATURE_NAMES)
    return fields


def localize_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    seqtype = str(getattr(args, "seqtype", "dna") or "dna").strip().lower()
    if seqtype == "protein":
        stop_if_not_protein(records=records, label="--seq_file")
    elif seqtype == "dna":
        stop_if_not_dna(records=records, label="--seq_file")
        stop_if_not_multiple_of_three(records=records)
        stop_if_invalid_codontable(codontable=args.codontable, label="--codon_table")
    else:
        raise ValueError("--seq_type should be dna or protein.")

    if hasattr(args, "model_download"):
        allow_model_download = _is_true_arg(args.model_download)
    else:
        allow_model_download = not _is_true_arg(
            getattr(args, "no_model_download", False)
        )
    model_path = resolve_localize_model_path(
        model=args.model,
        allow_download=allow_model_download,
    )
    allow_unsafe_model = _is_true_arg(
        getattr(args, "allow_unsafe_model", False)
    ) or is_pretrained_localize_model_alias(args.model)
    if allow_unsafe_model:
        model = load_localize_model(path=model_path, allow_unsafe=True)
    else:
        model = load_localize_model(path=model_path)
    if isinstance(model.get("localization_model"), dict):
        model["localization_model"]["_runtime_offline"] = not allow_model_download
    if str(model.get("model_type", "")) not in MULTILABEL_MODEL_TYPES:
        model_classes = tuple(model["localization_model"]["class_order"])
        if model_classes != LOCALIZATION_CLASSES:
            txt = "Model class order mismatch: expected {}, got {}. Exiting."
            raise ValueError(
                txt.format(",".join(LOCALIZATION_CLASSES), ",".join(model_classes))
            )

    threads = resolve_threads(getattr(args, "threads", 1))
    _configure_ml_threads(model=model, threads=threads)
    rows = _predict_records_batched_if_supported(
        records=records,
        codontable=args.codontable,
        seqtype=seqtype,
        model=model,
        include_features=args.include_features,
        organism_group=getattr(args, "organism_group", ""),
    )
    if rows is None:
        worker = partial(
            _predict_single_record,
            codontable=args.codontable,
            seqtype=seqtype,
            model=model,
            include_features=args.include_features,
            organism_group=getattr(args, "organism_group", ""),
        )
        rows = parallel_map_ordered(items=records, worker=worker, threads=threads)

    report_path = args.report
    if report_path == "":
        report_path = "-"
    if report_path.lower().endswith(".json"):
        write_rows_json(rows=rows, output_path=report_path)
    else:
        write_rows_tsv(
            rows=rows,
            output_path=report_path,
            fieldnames=_resolve_output_fields(
                include_features=args.include_features,
                model=model,
            ),
        )

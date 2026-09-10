"""Scientific boundary cases and artifact compatibility, without model training."""

import json
import csv
import hashlib
import os

import numpy as np
import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.localize import (
    _predict_single_record,
    _predict_records_batched_if_supported,
)
from cdskit.localize_batch import predict_localization_batch
from cdskit.localize_decision import sequence_quality, threshold_decisions
from cdskit.localize_model import (
    BROAD_FEATURE_NAMES,
    FEATURE_NAMES,
    LOCALIZATION_CLASSES,
    PEROX_FEATURE_NAMES,
    detect_perox_signals,
    extract_broad_localize_features,
    extract_localize_features,
    extract_perox_features,
    load_localize_model,
    predict_localization_and_peroxisome,
    predict_multilabel_localization,
    save_localize_model,
)
from cdskit.localize_runtime import PredictionRuntime, prediction_runtime
from cdskit.localize_schema import (
    CURRENT_FEATURE_SCHEMA as CURRENT,
    LEGACY_FEATURE_SCHEMA as LEGACY,
    extraction_schema,
    feature_schema_scope,
    version_model,
)


NINE = "MRLQVVLGHLAAAA"
EIGHT = "MRLQVVVHLAAAA"


def constant_model(multilabel=False):
    if multilabel:
        return {
            "model_type": "multilabel_centroid_v1",
            "feature_names": list(BROAD_FEATURE_NAMES),
            "perox_model": {"mode": "embedded_multilabel"},
            "localization_model": {
                "mean": [0] * len(BROAD_FEATURE_NAMES),
                "std": [1] * len(BROAD_FEATURE_NAMES),
                "class_order": ["nucleus", "chloroplast", "peroxisome"],
                "label_models": [
                    {"mode": "constant", "probability": p} for p in (0.1, 0.9, 0.0)
                ],
                "class_thresholds": {
                    name: 0.5 for name in ("nucleus", "chloroplast", "peroxisome")
                },
                "ensure_one_label": True,
            },
        }
    return {
        "model_type": "nearest_centroid_v1",
        "feature_names": list(FEATURE_NAMES),
        "localization_model": {
            "mode": "constant",
            "class_order": list(LOCALIZATION_CLASSES),
            "class_label": "noTP",
        },
        "perox_model": {"mode": "constant", "yes_probability": 0.0},
    }


@pytest.mark.parametrize(
    "seq,current,legacy", [(NINE, True, False), (EIGHT, False, True)]
)
def test_pts2_nonapeptide_and_length_control(seq, current, legacy):
    assert detect_perox_signals(seq)["pts2_match"] is current
    assert detect_perox_signals(seq, LEGACY)["pts2_match"] is legacy
    for extract, names in (
        (extract_localize_features, FEATURE_NAMES),
        (extract_broad_localize_features, BROAD_FEATURE_NAMES),
        (extract_perox_features, PEROX_FEATURE_NAMES),
    ):
        before, _ = extract(seq, feature_schema=LEGACY)
        after, _ = extract(seq, feature_schema=CURRENT)
        expected = [
            i
            for i, name in enumerate(names)
            if name in ("pts2_match", "pts2_nterm_match")
        ]
        assert np.flatnonzero(before != after).tolist() == expected


def test_pts2_window_and_pts1_precedence():
    assert detect_perox_signals("A" * 31 + "RLQVVLGHL")["pts2_match"]
    assert not detect_perox_signals("A" * 32 + "RLQVVLGHL")["pts2_match"]
    both = detect_perox_signals(NINE + "SKL")
    assert both["pts1_match"] and both["pts2_match"]
    assert both["signal_type"] == "PTS1"
    assert extract_localize_features(NINE.lower() + "*")[1]["pts2_match"]
    # Central wildcard remains permissive in this limited length correction.
    assert detect_perox_signals("RLXXXXXHL")["pts2_match"]


def test_schema_scope_restores_after_failure():
    with pytest.raises(RuntimeError), feature_schema_scope(LEGACY):
        assert not detect_perox_signals(NINE)["pts2_match"]
        raise RuntimeError("test")
    assert extraction_schema() == CURRENT
    for invalid in ("", "future", 3):
        with pytest.raises(ValueError, match="schema"):
            extract_localize_features(NINE, feature_schema=invalid)


def test_unversioned_artifact_roundtrip_keeps_legacy(tmp_path):
    model = constant_model()
    path = tmp_path / "old.json"
    path.write_text(json.dumps(model))
    old = load_localize_model(str(path))
    assert old["feature_schema"] == LEGACY
    assert len(old["_artifact_sha256"]) == 64
    for seq in (NINE, EIGHT):
        pred = predict_localization_and_peroxisome(seq, old)
        assert pred["pts2_match"] == detect_perox_signals(seq, LEGACY)["pts2_match"]
    save_localize_model(old, str(tmp_path / "roundtrip.json"))
    saved = json.loads((tmp_path / "roundtrip.json").read_text())
    assert "_artifact_sha256" not in saved
    assert saved["feature_schema"] == LEGACY
    model["feature_schema"] = CURRENT
    save_localize_model(model, str(tmp_path / "new.json"))
    assert predict_localization_and_peroxisome(
        NINE, load_localize_model(str(tmp_path / "new.json"))
    )["pts2_match"]
    # Raw unversioned dictionaries are also legacy, never silently relabeled by saving.
    model.pop("feature_schema")
    save_localize_model(model, str(tmp_path / "raw.json"))
    assert load_localize_model(str(tmp_path / "raw.json"))["feature_schema"] == LEGACY


def test_invalid_artifact_schema_and_policy_rejected():
    model = constant_model()
    model["feature_schema"] = "future"
    with pytest.raises(ValueError, match="schema"):
        version_model(model)
    model["feature_schema"] = LEGACY
    model["localization_model"]["feature_schema"] = CURRENT
    with pytest.raises(ValueError, match="differ"):
        version_model(model)
    model = constant_model()
    model["localization_model"]["decision_policy"] = "future"
    with pytest.raises(ValueError, match="policy"):
        version_model(model)


@pytest.mark.parametrize("multilabel", [False, True])
def test_safe_abstention_wins_over_high_constant_and_forced_label(multilabel):
    model = constant_model(multilabel)
    model["localization_model"]["decision_policy"] = "safe-v1"
    predict = (
        predict_multilabel_localization
        if multilabel
        else predict_localization_and_peroxisome
    )
    for seq, reason in (
        ("", "empty_sequence"),
        ("X" * 100, "all_unknown"),
        ("M", "single_residue"),
    ):
        out = predict(seq, model)
        assert not out["score_available"]
        assert not out["forced_label"]
        assert not out["predicted_labels" if multilabel else "predicted_class"]
        assert out["quality_reason"] == reason
    assert predict("MMMM", model)["score_available"]


def test_taxonomy_mask_preserves_scores_and_cannot_be_forced_back():
    labels = ["nucleus", "chloroplast"]
    p = np.array([[0.1, 0.9]])
    with prediction_runtime(PredictionRuntime(taxonomy_id="9606")):
        pred = threshold_decisions(p, {}, labels, False)
        assert pred["decision_status"] == ["taxonomy_excluded"]
        assert not pred["prediction_matrix"].any()
        forced = threshold_decisions(p, {}, labels, True)
        assert forced["prediction_matrix"].tolist() == [[1, 0]]
        assert forced["forced_label"].tolist() == [True]
        none = threshold_decisions(p[:, 1:], {}, ["chloroplast"], True)
        assert not none["prediction_matrix"].any()
    np.testing.assert_array_equal(p, [[0.1, 0.9]])
    with prediction_runtime(PredictionRuntime(taxonomy_id="3702")):
        assert threshold_decisions(p, {}, labels)["prediction_matrix"].tolist() == [
            [0, 1]
        ]


def test_single_and_batch_detailed_output_match_and_report_annotation_separately():
    model = constant_model()
    records = [
        SeqRecord(Seq(s), id=str(i)) for i, s in enumerate([NINE, "M", "X" * 100])
    ]
    with prediction_runtime(PredictionRuntime(decision_policy="safe-v1")):
        batch = _predict_records_batched_if_supported(
            records, 1, "protein", model, True
        )
        single = [_predict_single_record(r, 1, "protein", model, True) for r in records]
        assert batch == single
        assert batch[0]["feature_schema"] == LEGACY
        assert batch[0]["signal_schema"] == CURRENT
        assert batch[0]["pts2_annotation_match"] is True
        assert (
            batch[0]["pts2_match"] == 0.0
        )  # Actual legacy feature, not the new annotation.
        assert batch[0]["perox_head_status"] == "constant"
        assert batch[0]["p_peroxisome"] == 0.0
        assert batch[1]["p_peroxisome"] is None
        assert batch[1]["predicted_class"] == ""
        json.dumps(batch, allow_nan=False)
        direct = predict_localization_batch([str(r.seq) for r in records], model)
        assert [r["score_available"] for r in direct] == [True, False, False]


def test_legacy_default_output_remains_legacy_and_explicit_override_works():
    model = constant_model()
    row = _predict_single_record(
        SeqRecord(Seq("M"), id="a"), 1, "protein", model, False
    )
    assert "decision_status" not in row
    assert row["predicted_class"] == "noTP"
    model["localization_model"]["decision_policy"] = "safe-v1"
    with prediction_runtime(PredictionRuntime(decision_policy="legacy")):
        assert predict_localization_and_peroxisome("M", model)["score_available"]
    with prediction_runtime(PredictionRuntime(report_schema="legacy")):
        with pytest.raises(ValueError, match="require"):
            _predict_single_record(
                SeqRecord(Seq("M"), id="a"), 1, "protein", model, False
            )


def test_quality_canonicalization():
    assert sequence_quality(" *") == "empty_sequence"
    assert sequence_quality("??") == "all_unknown"
    assert sequence_quality("m*") == "single_residue"
    assert sequence_quality("MXX") == ""
    with pytest.raises(ValueError, match="Internal stop"):
        sequence_quality("M*A")


def test_separate_perox_schema_does_not_change_localization_features():
    model = constant_model()
    dim = len(FEATURE_NAMES)
    i = FEATURE_NAMES.index("pts2_match")
    negative, positive = np.zeros(dim), np.zeros(dim)
    positive[i] = 1
    model["perox_model"] = {
        "mode": "centroid",
        "feature_schema": CURRENT,
        "mean": [0] * dim,
        "std": [1] * dim,
        "centroids": [negative.tolist(), positive.tolist()],
        "log_priors": [0, 0],
        "class_order": ["no", "yes"],
    }
    pred = predict_localization_and_peroxisome(NINE, model)
    assert pred["pts2_match"] is False
    assert pred["perox_probability_yes"] > 0.5
    batch = predict_localization_batch([NINE], model)[0]
    assert batch["perox_probability_yes"] == pytest.approx(
        pred["perox_probability_yes"]
    )
    assert batch["pts2_match"] is False


@pytest.mark.parametrize("extension", ["json", "tsv"])
def test_cli_v2_serialization_and_thread_policy(tmp_path, extension):
    from cdskit.cli import psr
    from cdskit.localize import localize_main

    # The centroid CLI uses threaded single-record inference for this input size.
    model = constant_model(multilabel=True)
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model))
    fasta = tmp_path / "input.faa"
    fasta.write_text("".join(f">s{i}\n{'M' if i % 2 else NINE}\n" for i in range(10)))
    out = tmp_path / ("result." + extension)
    args = psr.parse_args(
        [
            "localize",
            "--seq_file",
            str(fasta),
            "--seq_type",
            "protein",
            "--model",
            str(path),
            "--decision_policy",
            "safe-v1",
            "--taxonomy_id",
            "9606",
            "--threads",
            "2",
            "--report",
            str(out),
        ]
    )
    localize_main(args)
    if extension == "json":
        rows = json.loads(out.read_text())
        assert rows[1]["p_nucleus"] is None
    else:
        with out.open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        assert rows[1]["p_nucleus"] == ""
    assert [r["seq_id"] for r in rows] == [f"s{i}" for i in range(10)]
    for i, row in enumerate(rows):
        assert row["report_schema"] == "v2"
        assert len(row["source_model_sha256"]) == 64
        assert "chloroplast" not in row["predicted_labels"]
        assert row["decision_status"] == ("abstained" if i % 2 else "forced_label")
    assert rows[0]["perox_signal_type"] == "PTS2"


def test_nested_blend_preserves_each_base_schema():
    dim = len(FEATURE_NAMES)
    index = FEATURE_NAMES.index("pts2_match")
    centres = np.zeros((len(LOCALIZATION_CLASSES), dim))
    centres[1, index] = 1
    head = {
        "mean": [0] * dim,
        "std": [1] * dim,
        "centroids": centres.tolist(),
        "log_priors": [0] * len(LOCALIZATION_CLASSES),
        "class_order": list(LOCALIZATION_CLASSES),
    }
    child = {
        "model_type": "nearest_centroid_v1",
        "feature_schema": CURRENT,
        "localization_model": head,
    }
    parent = constant_model()
    parent["feature_schema"] = LEGACY
    parent["model_type"] = "targetp_blend_v1"
    parent["localization_model"] = {
        "class_order": list(LOCALIZATION_CLASSES),
        "base_models": [child, child],
        "alpha_by_class": {label: 0.5 for label in LOCALIZATION_CLASSES},
    }
    single = predict_localization_and_peroxisome(NINE, parent)
    batch = predict_localization_batch([NINE], parent)[0]
    assert single["predicted_class"] == batch["predicted_class"] == "SP"
    assert not single["pts2_match"]
    assert single["class_probabilities"]["SP"] == pytest.approx(
        batch["class_probabilities"]["SP"]
    )
    saved = version_model(parent)
    assert (
        saved["localization_model"]["base_models"][0]["localization_model"][
            "feature_schema"
        ]
        == CURRENT
    )


def test_pipeline_excludes_unavailable_scores_from_calibration_and_probability_metrics(
    tmp_path, monkeypatch
):
    from cdskit import localize_pipeline_stages as stages

    prediction = {
        "prob_matrix": np.array([[0.0], [1.0]]),
        "prediction_matrix": np.array([[0], [1]]),
        "score_available": np.array([False, True]),
        "decision_status": ["abstained", "predicted"],
    }
    with pytest.raises(ValueError, match="Unscored"):
        stages.require_scored_rows(prediction)
    monkeypatch.setattr(stages, "load_localize_model", lambda path: {})
    monkeypatch.setattr(stages, "predict", lambda *args, **kwargs: prediction)
    (tmp_path / "student.pt").write_bytes(b"test checkpoint identity")
    rows = [
        dict(accession=str(i), sequence=seq, localization_labels="nucleus")
        for i, seq in enumerate(["M", "MAA"])
    ]
    stages.evaluate_students(
        {"labels": ["nucleus"], "student": {"train_control": False, "batch_size": 2}},
        {"test": rows},
        tmp_path,
        tmp_path,
    )
    metrics = json.loads((tmp_path / "metrics.json").read_text())["models"]["student"]
    assert metrics["score_coverage"] == 0.5
    assert metrics["brier_score"] == 0.0  # Excludes the unavailable zero placeholder.
    assert metrics["micro_f1"] == pytest.approx(2 / 3)  # Still accounts for both rows.
    assert metrics["accepted_only"]["micro_f1"] == 1.0
    with np.load(tmp_path / "student.npz", allow_pickle=False) as result:
        assert result["score_available"].tolist() == [False, True]


def test_nested_schema_views_keep_runtime_cache_and_reject_conflicts():
    from cdskit.localize_schema import scoped_localization_head

    original = {}
    model = {"feature_schema": LEGACY, "localization_model": original}
    first = scoped_localization_head(model)
    first["_runtime_model_cache"]["cpu"] = object()
    second = scoped_localization_head(model)
    assert first["_runtime_model_cache"] is second["_runtime_model_cache"]
    assert first["_runtime_model_cache"] is original["_runtime_model_cache"]
    assert "feature_schema" not in original
    original["feature_schema"] = LEGACY
    assert scoped_localization_head(model) is original
    original["feature_schema"] = CURRENT
    with pytest.raises(ValueError, match="differ"):
        scoped_localization_head(model)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_scores_are_not_forced_to_a_label(value):
    with pytest.raises(ValueError, match="finite"):
        threshold_decisions([[value]], {}, ["nucleus"], True)
    from cdskit.localize_model import predict_multilabel_centroid_matrix

    model = constant_model(multilabel=True)["localization_model"]
    model["label_models"][0]["probability"] = value
    with pytest.raises(ValueError, match="finite"):
        predict_multilabel_centroid_matrix(
            np.zeros((1, len(BROAD_FEATURE_NAMES))), model, apply_thresholds=False
        )


@pytest.mark.parametrize("snapshot", [False, True])
def test_load_digest_describes_open_artifact_after_atomic_replacement(
    tmp_path, monkeypatch, snapshot
):
    import cdskit.localize_model as lm

    if not snapshot and os.name == "nt":
        pytest.skip(
            "Windows blocks replacement of an open file; use the snapshot case."
        )
    path = tmp_path / "model.json"
    original = json.dumps(constant_model()).encode()
    path.write_bytes(original)
    if snapshot:
        # Simulate an already-open descriptor to the original inode. Only the
        # first open uses the snapshot: reopening the path must see its new data.
        descriptor_path = tmp_path / "open-snapshot.json"
        descriptor_path.write_bytes(original)
        original_open = open
        opened = False

        def open_snapshot(filename, *args, **kwargs):
            nonlocal opened
            if filename == path and not opened:
                opened = True
                return original_open(descriptor_path, *args, **kwargs)
            return original_open(filename, *args, **kwargs)

        monkeypatch.setattr(lm, "open", open_snapshot, raising=False)
    replacement = tmp_path / "replacement.json"
    changed = constant_model()
    changed["localization_model"]["class_label"] = "SP"
    replacement.write_text(json.dumps(changed))
    json_load = json.load

    def replace_after_parse(stream):
        result = json_load(stream)
        replacement.replace(path)
        return result

    monkeypatch.setattr(lm.json, "load", replace_after_parse)
    model = lm.load_localize_model(path)
    assert model["localization_model"]["class_label"] == "noTP"
    assert model["_artifact_sha256"] == hashlib.sha256(original).hexdigest()
    assert model["_artifact_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()


def test_in_place_artifact_change_is_rejected(tmp_path, monkeypatch):
    import cdskit.localize_model as lm

    path = tmp_path / "model.json"
    path.write_text(json.dumps(constant_model()))
    json_load = json.load

    def change_after_parse(stream):
        result = json_load(stream)
        with path.open("ab") as output:
            output.write(b" ")
        return result

    monkeypatch.setattr(lm.json, "load", change_after_parse)
    with pytest.raises(ValueError, match="changed in place"):
        lm.load_localize_model(path)


def test_legacy_feature_fitting_records_the_actual_schema():
    from cdskit.localize_model import (
        fit_nearest_centroid_classifier,
        fit_perox_binary_classifier,
    )

    features = np.stack(
        [
            extract_localize_features(seq, feature_schema=LEGACY)[0]
            for seq in (NINE, EIGHT)
        ]
    )
    head = fit_nearest_centroid_classifier(
        features, ["no", "yes"], ["no", "yes"], feature_schema=LEGACY
    )
    assert head["feature_schema"] == LEGACY
    with feature_schema_scope(LEGACY):
        assert (
            fit_perox_binary_classifier(features, ["no", "no"])["feature_schema"]
            == LEGACY
        )
    with pytest.raises(ValueError, match="schema"):
        fit_perox_binary_classifier(features, ["no", "yes"], feature_schema="future")


def test_strata_exclude_unscored_placeholders():
    from cdskit.localize_evaluation import stratified_metrics
    from cdskit.deeploc_benchmark import compute_multilabel_metrics

    result = stratified_metrics(
        [{"sequence": "M"}, {"sequence": "MAAA"}],
        np.array([[1.0], [1.0]]),
        np.array([[0], [1]]),
        np.array([[0.0], [0.8]]),
        ["nucleus"],
        compute_multilabel_metrics,
        score_available=[False, True],
    )["length:<=512"]
    assert result["brier_score"] == pytest.approx(0.04)
    assert result["micro_f1"] == pytest.approx(2 / 3)
    assert result["score_coverage"] == 0.5


def test_calibration_rejects_mismatched_availability():
    from cdskit.localize_pipeline_stages import require_scored_rows

    with pytest.raises(ValueError, match="Unscored"):
        require_scored_rows({"prob_matrix": np.zeros((2, 1)), "score_available": []})

import json

import numpy as np
import pytest

from cdskit.deeploc_benchmark import build_label_matrix, compute_multilabel_metrics
from cdskit.localize_evaluation import probability_metrics
from cdskit.localize_labels import observed_targets, validate_label_evidence
from cdskit.localize_model import (
    infer_labels_from_uniprot_cc,
    fit_perox_binary_classifier,
)
from cdskit.localize_splits import sequence_folds
from cdskit.targetp_labeling import (
    strict_uniprot_targetp_label,
    targeting_label_from_evidence,
)


def evidence(label="SP", **changes):
    return dict(
        label=label,
        state="positive",
        evidence_type="experimental",
        source="curated",
        source_version="v1",
        reference="PMID:example",
        feature_type="SIGNAL",
        **changes,
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Cytoplasm. Note=Does not localize to mitochondria.",
        "Secreted. Note=Secreted by an unconventional pathway.",
    ],
)
def test_location_proxies_do_not_invent_targeting_or_negative(text):
    assert infer_labels_from_uniprot_cc(text) == (None, "unknown", False)


def test_legacy_missing_conversion_is_explicit():
    assert infer_labels_from_uniprot_cc("", legacy=True) == ("noTP", "no", False)
    assert (
        strict_uniprot_targetp_label(
            "Cytoplasm. Note=Does not localize to mitochondria.", "non_plant"
        )[0]
        != "mTP"
    )
    assert (
        strict_uniprot_targetp_label(
            "Secreted. Note=Secreted by an unconventional pathway.", "non_plant"
        )[0]
        is None
    )


def test_evidence_accepts_unknown_coordinates_but_not_proxy():
    record = evidence(start=None, end=None)
    assert targeting_label_from_evidence(json.dumps([record])) == "SP"
    for kind in ("localization_proxy", "sequence_analysis", "similarity", "dataset"):
        with pytest.raises(ValueError, match="Experimental policy"):
            targeting_label_from_evidence(
                json.dumps([dict(record, evidence_type=kind)])
            )
    with pytest.raises(ValueError, match="coordinates"):
        targeting_label_from_evidence(json.dumps([dict(record, start=9, end=1)]))
    with pytest.raises(ValueError, match="requires SIGNAL"):
        targeting_label_from_evidence(
            json.dumps([dict(record, feature_type="TRANSIT")])
        )


def test_evidence_requires_coverage_and_reference():
    with pytest.raises(ValueError, match="Every observed"):
        validate_label_evidence("[]", ["nucleus"], [])
    with pytest.raises(ValueError, match=r"requires label|Evidence requires"):
        validate_label_evidence(
            json.dumps([dict(evidence("nucleus"), reference="")]), ["nucleus"], []
        )


def test_partial_matrix_and_masked_metrics_ignore_unknown_predictions():
    labels = ["nucleus", "cytoplasm"]
    rows = [
        dict(localization_labels="nucleus", negative_labels=""),
        dict(localization_labels="", negative_labels="nucleus"),
    ]
    y = build_label_matrix(rows, labels, "localization_labels")
    np.testing.assert_equal(y, [[1, np.nan], [0, np.nan]])
    a = np.array([[1, 1], [0, 0]])
    b = np.array([[1, 0], [0, 1]])
    assert compute_multilabel_metrics(y, a, labels) == compute_multilabel_metrics(
        y, b, labels
    )
    scores = compute_multilabel_metrics(y, a, labels)
    assert scores["micro_f1"] == 1
    assert scores["subset_accuracy"] is None
    assert scores["by_label"]["cytoplasm"]["f1"] is None
    assert scores["observed_count"] == 2
    assert probability_metrics(y, a.astype(float), labels)["brier_score"] == 0


def test_all_unknown_metrics_are_not_zero_accuracy():
    y = np.full((2, 2), np.nan)
    score = compute_multilabel_metrics(y, np.zeros((2, 2)), ["a", "b"])
    assert score["micro_f1"] is None
    assert score["macro_f1"] is None
    assert probability_metrics(y, np.zeros((2, 2)), ["a", "b"])["brier_score"] is None


def test_mask_can_hide_arbitrary_placeholder_without_imputing_truth():
    mask = np.array([[True, False]])
    a, actual = observed_targets([[1, 999]], mask)
    np.testing.assert_array_equal(a, [[1, 0]])
    np.testing.assert_array_equal(actual, mask)
    with pytest.raises(ValueError, match="binary"):
        observed_targets([[0.5]])


def test_perox_training_ignores_unknown_and_rejects_no_observations():
    model = fit_perox_binary_classifier(
        np.array([[0], [1], [999]]), ["yes", "no", "unknown"]
    )
    np.testing.assert_array_equal(model["mean"], [0.5])
    with pytest.raises(ValueError, match="No observed"):
        fit_perox_binary_classifier([[1]], ["unknown"])


def test_exact_duplicates_stay_together_and_provided_leak_rejected():
    sequences = ["MAAA", "MAAA", "MCCC", "MCCC", "MDDD", "MDDD", "MEEE", "MEEE"]
    labels = ["SP"] * 4 + ["noTP"] * 4
    folds, _, report = sequence_folds(sequences, labels, 2, 1)
    assert not set(sequences[i] for i in folds[0]) & set(sequences[i] for i in folds[1])
    assert report["exact_group_overlap"] == "none"
    with pytest.raises(ValueError, match="overlap"):
        sequence_folds(sequences, labels, 2, 1, fold_ids=["a", "b"] * 4)


def test_supplied_group_prevents_distinct_sequences_crossing_folds():
    with pytest.raises(ValueError, match="overlap"):
        sequence_folds(
            ["MAAA", "MCCC"],
            ["SP", "SP"],
            2,
            1,
            fold_ids=["a", "b"],
            group_ids=["g", "g"],
        )


def test_nested_postprocessing_does_not_see_outer_targets():
    from cdskit.localize_learn import evaluate_cross_validation

    n = 20
    x = np.random.default_rng(5).normal(size=(n, 3))
    sequences = ["M" + "A" * i + "G" for i in range(n)]
    labels = ["SP", "mTP", "cTP", "lTP", "noTP"] * 4
    folds = ["a"] * 10 + ["b"] * 10
    options = dict(
        temperature=True, thresholds=True, objective="macro", two_stage=False
    )

    def run(truth):
        return evaluate_cross_validation(
            x,
            sequences,
            truth,
            ["no"] * n,
            2,
            1,
            "nearest_centroid",
            {},
            "cpu",
            fold_ids=folds,
            postprocess=options,
        )["nested_postprocess"]

    first = run(labels)
    changed = ["noTP"] * 10 + labels[10:]
    second = run(changed)
    assert first["folds"][0] == second["folds"][0]
    for a, b in zip(first["oof_rows"][:10], second["oof_rows"][:10], strict=True):
        assert a["class_probabilities"] == b["class_probabilities"]
        assert a["predicted_class"] == b["predicted_class"]


def test_homology_audit_rejects_residual_hits_and_missing_tool(monkeypatch):
    from cdskit import perox_benchmark
    from cdskit.localize_splits import audit_homology_partitions

    partitions = {"train": [{"sequence": "MAAA"}], "test": [{"sequence": "MCCC"}]}
    monkeypatch.setattr(
        perox_benchmark,
        "mmseqs_homology_report",
        lambda *a, **k: {"status": "ok", "hit_query_count": 1},
    )
    with pytest.raises(ValueError, match="Cross-partition homology hits"):
        audit_homology_partitions(partitions)
    monkeypatch.setattr(
        perox_benchmark,
        "mmseqs_homology_report",
        lambda *a, **k: {"status": "unavailable"},
    )
    with pytest.raises(ValueError, match="unavailable"):
        audit_homology_partitions(partitions)


def test_conflicting_canonical_groups_rejected_even_within_one_fold():
    with pytest.raises(ValueError, match="conflicting group"):
        sequence_folds(
            ["MAAA", "M A A A", "MCCC"],
            ["SP"] * 3,
            2,
            1,
            fold_ids=["a", "a", "b"],
            group_ids=["x", "y", "z"],
        )


@pytest.mark.parametrize("bad", [None, float("nan"), " "])
def test_missing_fold_values_are_not_independent_groups(bad):
    with pytest.raises(ValueError, match="Complete fold"):
        sequence_folds(["MAAA", "MCCC"], ["SP"] * 2, 2, 1, fold_ids=["a", bad])


def test_evidence_cannot_cover_opposing_observations():
    with pytest.raises(ValueError, match="conflict"):
        validate_label_evidence("[]", ["nucleus"], ["nucleus"])

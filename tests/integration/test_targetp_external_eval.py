import csv
import json
import math
from types import SimpleNamespace

import pytest

from cdskit.targetp_external_eval import (
    _parse_threshold_grid,
    build_deeploc_hpa_broad_rows,
    build_deeploc_sorting_rows,
    build_uniprot_holdout_rows,
    compute_single_label_metrics,
    evaluate_prediction_threshold_calibration,
    filter_rows_by_mmseqs_similarity,
    load_fixed_uniprot_holdout_rows,
    load_targetp_exclusion_keys,
    run_external_evaluation,
    stratified_sample_rows,
)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_threshold_grid_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="must be finite"):
        _parse_threshold_grid(value)
    with pytest.raises(ValueError, match="must be finite"):
        evaluate_prediction_threshold_calibration([], threshold_grid=[float(value)])


def test_external_evaluation_rejects_nonfinite_grid_before_outputs(tmp_path):
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="must be finite"):
        run_external_evaluation(
            model_path="model.pt",
            targetp_tsv="targetp.tsv",
            deeploc_dir="deeploc",
            uniprot_tsv="uniprot.tsv",
            out_dir=str(output),
            threshold_grid=[math.nan],
        )
    assert not output.exists()


def test_external_evaluation_rolls_back_prediction_on_later_failure(
    tmp_path, monkeypatch
):
    import cdskit.targetp_external_eval as external_eval

    output = tmp_path / "out"
    output.mkdir()
    first_prediction = output / "deeploc_sorting_predictions.tsv"
    first_prediction.write_text("previous result\n", encoding="utf-8")
    monkeypatch.setattr(
        external_eval,
        "load_targetp_exclusion_keys",
        lambda targetp_tsv: {"rows": [], "accessions": set(), "sequences": set()},
    )
    monkeypatch.setattr(
        external_eval,
        "build_deeploc_sorting_rows",
        lambda **kwargs: ([], {}),
    )
    monkeypatch.setattr(external_eval, "predict_rows", lambda **kwargs: [])

    def fail_later(**kwargs):
        raise OSError("simulated later dataset failure")

    monkeypatch.setattr(external_eval, "build_deeploc_hpa_broad_rows", fail_later)
    with pytest.raises(OSError, match="simulated later dataset failure"):
        run_external_evaluation(
            model_path="model.pt",
            targetp_tsv="targetp.tsv",
            deeploc_dir="deeploc",
            uniprot_tsv="uniprot.tsv",
            out_dir=str(output),
        )
    assert first_prediction.read_text(encoding="utf-8") == "previous result\n"
    assert sorted(path.name for path in output.iterdir()) == [first_prediction.name]


def test_external_evaluation_commits_complete_result_set(tmp_path, monkeypatch):
    import cdskit.targetp_external_eval as external_eval

    monkeypatch.setattr(
        external_eval,
        "load_targetp_exclusion_keys",
        lambda targetp_tsv: {"rows": [], "accessions": set(), "sequences": set()},
    )
    monkeypatch.setattr(
        external_eval, "build_deeploc_sorting_rows", lambda **kwargs: ([], {})
    )
    monkeypatch.setattr(
        external_eval, "build_deeploc_hpa_broad_rows", lambda **kwargs: ([], {})
    )
    monkeypatch.setattr(
        external_eval, "build_uniprot_holdout_rows", lambda **kwargs: ([], {})
    )
    monkeypatch.setattr(external_eval, "predict_rows", lambda **kwargs: [])
    monkeypatch.setattr(
        external_eval,
        "filter_rows_by_mmseqs_similarity",
        lambda **kwargs: ([], {"status": "empty"}),
    )
    output = tmp_path / "out"
    result = run_external_evaluation(
        model_path="model.pt",
        targetp_tsv="targetp.tsv",
        deeploc_dir="deeploc",
        uniprot_tsv="uniprot.tsv",
        out_dir=str(output),
        threshold_calibration=False,
    )
    assert len(list(output.iterdir())) == 6
    saved = json.loads((output / "targetp_external_eval.json").read_text())
    assert saved == result
    assert saved["deeploc_sorting"]["predictions_tsv"] == str(
        output / "deeploc_sorting_predictions.tsv"
    )


def _write_tsv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_deeploc_sorting_rows_remove_targetp_exact_overlaps(temp_dir):
    targetp = temp_dir / "targetp.tsv"
    sorting = temp_dir / "sorting.tsv"
    _write_tsv(
        targetp,
        ["accession", "sequence"],
        [{"accession": "P1", "sequence": "MATS"}],
    )
    _write_tsv(
        sorting,
        ["source", "accession", "kingdom", "sequence", "sorting_signal_labels"],
        [
            {
                "source": "deeploc",
                "accession": "P1.1",
                "kingdom": "Metazoa",
                "sequence": "MATS",
                "sorting_signal_labels": "SP",
            },
            {
                "source": "deeploc",
                "accession": "P2",
                "kingdom": "Viridiplantae",
                "sequence": "MAAA",
                "sorting_signal_labels": "TH",
            },
            {
                "source": "deeploc",
                "accession": "P3",
                "kingdom": "Metazoa",
                "sequence": "MBBB",
                "sorting_signal_labels": "GPI",
            },
        ],
    )

    rows, skipped = build_deeploc_sorting_rows(
        path=str(sorting),
        targetp_keys=load_targetp_exclusion_keys(str(targetp)),
    )

    assert [row["true_class"] for row in rows] == ["lTP"]
    assert rows[0]["organism_group"] == "plant"
    assert skipped["targetp_exact_overlap"] == 1
    assert skipped["no_targetp_equivalent_label"] == 1


def test_deeploc_hpa_broad_maps_mature_locations_to_targetp_proxy(temp_dir):
    targetp = temp_dir / "targetp.tsv"
    hpa = temp_dir / "hpa.tsv"
    _write_tsv(targetp, ["accession", "sequence"], [])
    _write_tsv(
        hpa,
        ["source", "accession", "kingdom", "sequence", "localization_labels"],
        [
            {
                "source": "hpa",
                "accession": "H1",
                "kingdom": "Metazoa",
                "sequence": "MAAA",
                "localization_labels": "mitochondrion;nucleus",
            },
            {
                "source": "hpa",
                "accession": "H2",
                "kingdom": "Metazoa",
                "sequence": "MBBB",
                "localization_labels": "nucleus;cytoplasm",
            },
        ],
    )

    rows, skipped = build_deeploc_hpa_broad_rows(
        path=str(hpa),
        targetp_keys=load_targetp_exclusion_keys(str(targetp)),
    )

    assert [row["true_class"] for row in rows] == ["mTP", "noTP"]
    assert skipped == {}


def test_uniprot_holdout_uses_cdskit_cc_label_rules_and_skips_ambiguous(temp_dir):
    targetp = temp_dir / "targetp.tsv"
    uniprot = temp_dir / "uniprot.tsv"
    _write_tsv(
        targetp,
        ["accession", "sequence"],
        [{"accession": "P1", "sequence": "MATS"}],
    )
    _write_tsv(
        uniprot,
        ["accession", "sequence", "cc_subcellular_location", "lineage_ids"],
        [
            {
                "accession": "P1",
                "sequence": "MATS",
                "cc_subcellular_location": "SUBCELLULAR LOCATION: Secreted.",
                "lineage_ids": "2759, 33208",
            },
            {
                "accession": "P2",
                "sequence": "MAAA",
                "cc_subcellular_location": "SUBCELLULAR LOCATION: Chloroplast.",
                "lineage_ids": "2759, 33090",
            },
            {
                "accession": "P3",
                "sequence": "MBBB",
                "cc_subcellular_location": "SUBCELLULAR LOCATION: Secreted. Mitochondrion.",
                "lineage_ids": "2759",
            },
        ],
    )

    rows, skipped = build_uniprot_holdout_rows(
        path=str(uniprot),
        targetp_keys=load_targetp_exclusion_keys(str(targetp)),
    )

    assert [row["true_class"] for row in rows] == ["cTP"]
    assert rows[0]["organism_group"] == "plant"
    assert skipped["targetp_exact_overlap"] == 1
    assert skipped["ambiguous_uniprot_cc"] == 1


def test_uniprot_holdout_can_skip_nonplant_plastid_labels(temp_dir):
    targetp = temp_dir / "targetp.tsv"
    uniprot = temp_dir / "uniprot.tsv"
    _write_tsv(targetp, ["accession", "sequence"], [])
    _write_tsv(
        uniprot,
        ["accession", "sequence", "cc_subcellular_location", "lineage_ids"],
        [
            {
                "accession": "P1",
                "sequence": "MAAA",
                "cc_subcellular_location": "SUBCELLULAR LOCATION: Chloroplast.",
                "lineage_ids": "2759, 33090",
            },
            {
                "accession": "P2",
                "sequence": "MBBB",
                "cc_subcellular_location": "SUBCELLULAR LOCATION: Chloroplast.",
                "lineage_ids": "2759, 33208",
            },
        ],
    )

    rows, skipped = build_uniprot_holdout_rows(
        path=str(uniprot),
        targetp_keys=load_targetp_exclusion_keys(str(targetp)),
        strict_targetp_organism_labels=True,
    )

    assert [row["accession"] for row in rows] == ["P1"]
    assert skipped["nonplant_plastid"] == 1


def test_strict_uniprot_holdout_uses_targetp_compatible_ltp_labels(temp_dir):
    targetp = temp_dir / "targetp.tsv"
    uniprot = temp_dir / "uniprot.tsv"
    _write_tsv(targetp, ["accession", "sequence"], [])
    _write_tsv(
        uniprot,
        ["accession", "sequence", "cc_subcellular_location", "lineage_ids"],
        [
            {
                "accession": "ER1",
                "sequence": "MAAA",
                "cc_subcellular_location": "SUBCELLULAR LOCATION: Endoplasmic reticulum lumen.",
                "lineage_ids": "2759, 33208",
            },
            {
                "accession": "THM1",
                "sequence": "MBBB",
                "cc_subcellular_location": "SUBCELLULAR LOCATION: Plastid, chloroplast thylakoid membrane.",
                "lineage_ids": "2759, 33090",
            },
            {
                "accession": "THL1",
                "sequence": "MCCC",
                "cc_subcellular_location": "SUBCELLULAR LOCATION: Plastid, chloroplast thylakoid lumen.",
                "lineage_ids": "2759, 33090",
            },
            {
                "accession": "STR1",
                "sequence": "MDDD",
                "cc_subcellular_location": "SUBCELLULAR LOCATION: Chloroplast stroma.",
                "lineage_ids": "2759, 33090",
            },
        ],
    )

    rows, skipped = build_uniprot_holdout_rows(
        path=str(uniprot),
        targetp_keys=load_targetp_exclusion_keys(str(targetp)),
        strict_targetp_organism_labels=True,
    )

    assert [(row["accession"], row["true_class"]) for row in rows] == [
        ("ER1", "noTP"),
        ("THL1", "lTP"),
        ("STR1", "cTP"),
    ]
    assert skipped["thylakoid_not_lumen"] == 1


def test_fixed_uniprot_holdout_rows_are_normalized_and_can_be_strict(temp_dir):
    holdout = temp_dir / "holdout.tsv"
    _write_tsv(
        holdout,
        [
            "source",
            "accession",
            "sequence",
            "organism_group",
            "true_class",
            "external_labels",
        ],
        [
            {
                "source": "fixed",
                "accession": "H1",
                "sequence": "MAAA",
                "organism_group": "plant",
                "true_class": "cTP",
                "external_labels": "cTP",
            },
            {
                "source": "fixed",
                "accession": "H2",
                "sequence": "MBBB",
                "organism_group": "non_plant",
                "true_class": "lTP",
                "external_labels": "lTP",
            },
            {
                "source": "fixed",
                "accession": "H3",
                "sequence": "MCCC",
                "organism_group": "non_plant",
                "true_class": "other",
                "external_labels": "other",
            },
        ],
    )

    rows, skipped = load_fixed_uniprot_holdout_rows(
        path=str(holdout),
        strict_targetp_organism_labels=True,
    )

    assert [row["accession"] for row in rows] == ["H1"]
    assert skipped["inconsistent_targetp_organism_label"] == 1
    assert skipped["unknown_true_class"] == 1


def test_external_eval_metrics_and_stratified_sampling_are_deterministic():
    rows = [
        {"true_class": "SP", "predicted_class": "SP"},
        {"true_class": "SP", "predicted_class": "noTP"},
        {"true_class": "noTP", "predicted_class": "noTP"},
    ]

    metrics = compute_single_label_metrics(rows)

    assert metrics["n_rows"] == 3
    assert metrics["accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["by_class"]["SP"]["recall"] == pytest.approx(0.5)
    assert metrics["by_class"]["noTP"]["precision"] == pytest.approx(0.5)

    sample = stratified_sample_rows(
        rows=[
            {"true_class": "SP", "accession": "S1"},
            {"true_class": "SP", "accession": "S2"},
            {"true_class": "noTP", "accession": "N1"},
            {"true_class": "noTP", "accession": "N2"},
        ],
        max_per_class=1,
        seed=7,
    )

    assert [row["true_class"] for row in sample] == ["noTP", "SP"]
    assert len(sample) == 2


def test_prediction_threshold_calibration_reports_oracle_and_foldwise_metrics():
    rows = [
        {
            "true_class": "noTP",
            "p_noTP": 0.45,
            "p_SP": 0.55,
            "p_mTP": 0.0,
            "p_cTP": 0.0,
            "p_lTP": 0.0,
        },
        {
            "true_class": "noTP",
            "p_noTP": 0.46,
            "p_SP": 0.54,
            "p_mTP": 0.0,
            "p_cTP": 0.0,
            "p_lTP": 0.0,
        },
        {
            "true_class": "SP",
            "p_noTP": 0.10,
            "p_SP": 0.90,
            "p_mTP": 0.0,
            "p_cTP": 0.0,
            "p_lTP": 0.0,
        },
        {
            "true_class": "SP",
            "p_noTP": 0.11,
            "p_SP": 0.89,
            "p_mTP": 0.0,
            "p_cTP": 0.0,
            "p_lTP": 0.0,
        },
    ]

    result = evaluate_prediction_threshold_calibration(
        rows=rows,
        threshold_grid=[1.0, 2.0],
        cv_folds=2,
        seed=3,
    )

    assert result["argmax_metrics"]["macro_f1"] < result["oracle_metrics"]["macro_f1"]
    assert result["oracle_thresholds"]["SP"] == pytest.approx(2.0)
    assert result["cv_folds"] == 2
    assert result["cv_metrics"]["macro_f1"] == pytest.approx(
        result["oracle_metrics"]["macro_f1"]
    )
    assert all(
        fold["thresholds"]["SP"] == pytest.approx(2.0) for fold in result["folds"]
    )


def test_mmseqs_similarity_filter_removes_hit_queries(monkeypatch):
    def fake_run(command, **kwargs):
        del kwargs
        with open(command[4], "w", encoding="utf-8") as output:
            output.write("q0\tt0\t100\t4\t4\t4\t0\t50\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "cdskit.targetp_external_eval.shutil.which",
        lambda executable: executable,
    )
    monkeypatch.setattr(
        "cdskit.targetp_external_eval.subprocess.run",
        fake_run,
    )

    kept, report = filter_rows_by_mmseqs_similarity(
        rows=[
            {"accession": "Q0", "sequence": "MATS"},
            {"accession": "Q1", "sequence": "MAAA"},
        ],
        targetp_rows=[
            {"accession": "T0", "sequence": "MATS"},
        ],
        min_seq_id=0.30,
        min_coverage=0.80,
        threads=2,
        enabled=True,
    )

    assert [row["accession"] for row in kept] == ["Q1"]
    assert report["available"] is True
    assert report["removed"] == 1
    assert report["kept"] == 1
    assert report["status"] == "ok"

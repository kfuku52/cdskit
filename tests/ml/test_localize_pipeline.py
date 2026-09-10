"""Offline end-to-end exercise of the public teacher/student stages."""

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from cdskit.localize_pipeline import run_pipeline
from cdskit.localize_pipeline_config import load_config, load_partitions
from cdskit.localize_pipeline_stages import read_teacher_predictions


@pytest.fixture
def pipeline_config(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    encoder = tmp_path / "encoder"
    encoder.mkdir()
    vocab = ["<cls>", "<pad>", "<eos>", "<unk>", *"LAGVSERTIDPKQNFYMHWCXBUZO", "<mask>"]
    (encoder / "vocab.txt").write_text("\n".join(vocab))
    transformers.EsmTokenizer(str(encoder / "vocab.txt")).save_pretrained(encoder)
    transformers.EsmModel(
        transformers.EsmConfig(
            vocab_size=len(vocab),
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=64,
            pad_token_id=1,
            mask_token_id=len(vocab) - 1,
        )
    ).save_pretrained(encoder, safe_serialization=True)
    data = tmp_path / "data.tsv"
    data.write_text(
        "accession\tsequence\tlocalization_labels\tsplit\tcluster_id\n"
        + "\n".join(
            "{}\t{}\t{}\t{}\tc{}".format(i, seq, label, split, i)
            for i, (seq, label, split) in enumerate(
                [
                    ("MAAA", "nucleus", "train"),
                    ("MCCC", "cytoplasm", "train"),
                    ("MDDD", "nucleus;cytoplasm", "train"),
                    ("MEEE", "nucleus", "train"),
                    ("MFFF", "nucleus", "validation"),
                    ("MGGG", "cytoplasm", "validation"),
                    ("MHHH", "nucleus", "test"),
                    ("MIII", "cytoplasm", "test"),
                ]
            )
        )
        + "\n"
    )
    config = {
        "schema_version": 1,
        "labels": ["nucleus", "cytoplasm"],
        "data": {"path": "data.tsv"},
        "teacher": {
            "model_name": "encoder",
            "window": 16,
            "overlap": 4,
            "epochs": 1,
            "batch_size": 2,
            "device": "cpu",
            "cache_dir": "cache",
        },
        "student": {
            "epochs": 1,
            "batch_size": 2,
            "device": "cpu",
            "seq_len": 16,
            "embed_dim": 4,
            "num_filters": 4,
            "kernel_sizes": [3],
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_pipeline_roundtrip_resume_and_cpu_export(
    pipeline_config, tmp_path, monkeypatch
):
    root = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cdskit.cli",
            "localize-learn",
            "--stage",
            "all",
            "--config",
            str(pipeline_config),
            "--run_dir",
            str(root),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((root / "evaluate/metrics.json").read_text())
    assert set(report["models"]) == {"student", "control"}
    for name in (
        "teacher/training.json",
        "distill/student-training.json",
        "distill/control-training.json",
    ):
        history = json.loads((root / name).read_text())
        assert history["selected_epoch"] == 1
        assert history["history"][0]["validation_bce"] >= 0
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    run_pipeline(pipeline_config, root)
    assert before == {p: p.stat().st_mtime_ns for p in before}
    with pytest.raises(ValueError, match="exists"):
        run_pipeline(pipeline_config, root, resume=False)
    fasta = tmp_path / "input.fa"
    fasta.write_text(">query\nMAAA\n")
    output = tmp_path / "localize.tsv"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['transformers'] = None; from cdskit.cli import main; main()",
            "localize",
            "--seq_file",
            str(fasta),
            "--seq_type",
            "protein",
            "--model",
            str(root / "distill/student.pt"),
            "--report",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "query" in output.read_text()
    # A separate CPU run consumes only the saved probabilities and safe teacher hash.
    shutil.rmtree(tmp_path / "encoder")
    import cdskit.localize_multilabel_plm as plm

    def forbidden(*args, **kwargs):
        raise AssertionError("Distillation must not instantiate the encoder")

    monkeypatch.setattr(plm.ResidueEncoder, "__init__", forbidden)
    student_config = json.loads(pipeline_config.read_text())
    student_config["teacher"] = {}
    pipeline_config.write_text(json.dumps(student_config))
    other = run_pipeline(pipeline_config, tmp_path / "cpu", teacher_run=root)
    assert (other / "evaluate/metrics.json").exists()
    cfg = load_config(pipeline_config)
    with np.load(root / "predict/probabilities.npz", allow_pickle=False) as saved:
        payload = {k: saved[k] for k in saved.files}
    payload["accessions"] = payload["accessions"][::-1]
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **payload)
    with pytest.raises(ValueError, match="alignment"):
        read_teacher_predictions(
            cfg, load_partitions(cfg), root / "teacher/model.pt", bad
        )
    (root / "predict/probabilities.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="output changed"):
        run_pipeline(
            pipeline_config, tmp_path / "corrupt-run", stage="distill", teacher_run=root
        )


def test_pipeline_interruption_and_input_changes(
    pipeline_config, tmp_path, monkeypatch
):
    from cdskit import localize_pipeline_stages as stages

    original = stages.fit_teacher

    def interrupted(config, partitions, output):
        (output / "partial.pt").write_bytes(b"partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(stages, "fit_teacher", interrupted)
    root = tmp_path / "run"
    with pytest.raises(RuntimeError, match="interrupted"):
        run_pipeline(pipeline_config, root, "teacher")
    assert list(root.iterdir()) == [root / "run.json"]
    monkeypatch.setattr(stages, "fit_teacher", original)
    run_pipeline(pipeline_config, root, "teacher")
    lock = root / ".pipeline.lock"
    lock.write_text("owner")
    with pytest.raises(ValueError, match="locked"):
        run_pipeline(pipeline_config, root, "teacher")
    assert lock.read_text() == "owner"
    lock.unlink()
    data = tmp_path / "data.tsv"
    data.write_text(data.read_text().replace("MIII", "MKKK"))
    with pytest.raises(ValueError, match="changed"):
        run_pipeline(pipeline_config, root, "teacher")


@pytest.mark.parametrize("layout", ["legacy", "windows", "separate_termini"])
@pytest.mark.parametrize("pooling", ["mean", "light_attention", "label_attention"])
def test_student_layouts_and_no_test(pipeline_config, tmp_path, layout, pooling):
    config = json.loads(pipeline_config.read_text())
    config["student"].update(sequence_layout=layout, train_control=False)
    config["teacher"]["pooling"] = pooling
    data = tmp_path / "data.tsv"
    data.write_text(
        "\n".join(
            line for line in data.read_text().splitlines() if "\ttest\t" not in line
        )
        + "\n"
    )
    pipeline_config.write_text(json.dumps(config))
    root = run_pipeline(pipeline_config, tmp_path / "run")
    assert (root / "distill/student.pt").exists()
    assert not (root / "distill/control.pt").exists()
    assert not (root / "evaluate").exists()
    with pytest.raises(ValueError, match="test partition"):
        run_pipeline(pipeline_config, root, "evaluate")


@pytest.mark.parametrize("nested", ["cache", "run"])
def test_encoder_cannot_contain_mutable_pipeline_outputs(
    pipeline_config, tmp_path, nested
):
    config = json.loads(pipeline_config.read_text())
    root = tmp_path / "run"
    if nested == "cache":
        config["teacher"]["cache_dir"] = "encoder/cache"
    else:
        root = tmp_path / "encoder/run"
    pipeline_config.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="overlap"):
        run_pipeline(pipeline_config, root, "teacher")
    assert not root.exists()


@pytest.mark.parametrize("changed", ["data", "config", "encoder", "code"])
def test_changed_input_during_training_is_not_published(
    pipeline_config, tmp_path, monkeypatch, changed
):
    from cdskit import localize_pipeline_stages as stages

    original = stages.fit_teacher

    def change_input(config, partitions, output):
        original(config, partitions, output)
        if changed == "data":
            data = tmp_path / "data.tsv"
            data.write_text(data.read_text().replace("MIII", "MLLL"))
        elif changed == "config":
            raw = json.loads(pipeline_config.read_text())
            raw["student"]["epochs"] = 2
            pipeline_config.write_text(json.dumps(raw))
        elif changed == "encoder":
            (tmp_path / "encoder" / "config.json").write_text("{}")
        else:
            from cdskit import localize_pipeline

            monkeypatch.setattr(localize_pipeline, "code_identity", lambda: "changed")

    monkeypatch.setattr(stages, "fit_teacher", change_input)
    root = tmp_path / "run"
    with pytest.raises(ValueError, match="changed"):
        run_pipeline(pipeline_config, root, "teacher")
    assert not (root / "teacher").exists()


@pytest.mark.parametrize("ensure_one_label", [False, True])
def test_single_label_model_can_predict_negative(
    pipeline_config, tmp_path, ensure_one_label
):
    from cdskit.localize_model import load_localize_model
    from cdskit.localize_pipeline_stages import predict

    config = json.loads(pipeline_config.read_text())
    config["labels"] = ["nucleus"]
    config["ensure_one_label"] = ensure_one_label
    config["student"]["train_control"] = False
    data = tmp_path / "data.tsv"
    data.write_text(
        data.read_text()
        .replace("nucleus;cytoplasm", "nucleus")
        .replace("cytoplasm", "")
    )
    pipeline_config.write_text(json.dumps(config))
    root = run_pipeline(pipeline_config, tmp_path / "run")
    for path in ("distill/student.pt", "teacher/model.pt"):
        model = load_localize_model(str(root / path))
        model["localization_model"]["class_thresholds"] = {"nucleus": 1.0}
        result = predict(model, [{"sequence": "MHHH"}])
        assert bool(result["prediction_matrix"].any()) == ensure_one_label


def test_all_positive_class_has_nonzero_training_weight():
    from cdskit.localize_multilabel_cnn import _class_pos_weight

    weights = _class_pos_weight(np.array([[1, 0, 1], [1, 0, 0], [1, 0, 0]]))
    assert weights[0] == 1.0
    assert weights[1] == 1.0
    assert weights[2] == 2.0


def test_test_labels_do_not_affect_training(pipeline_config, tmp_path):
    from cdskit.localize_model import load_localize_model

    first = run_pipeline(pipeline_config, tmp_path / "first")
    data = tmp_path / "data.tsv"
    data.write_text(
        data.read_text()
        .replace("MHHH\tnucleus", "MHHH\tcytoplasm")
        .replace("MIII\tcytoplasm", "MIII\tnucleus")
    )
    second = run_pipeline(pipeline_config, tmp_path / "second")
    for artifact in ("teacher/model.pt", "distill/student.pt", "distill/control.pt"):
        a = load_localize_model(str(first / artifact))["localization_model"]
        b = load_localize_model(str(second / artifact))["localization_model"]
        assert a["class_thresholds"] == b["class_thresholds"]
        assert a["training_history"] == b["training_history"]
        for key in a["state_dict"]:
            np.testing.assert_array_equal(a["state_dict"][key], b["state_dict"][key])


def test_observed_schema_pipeline_preserves_unknown(pipeline_config, tmp_path):
    import csv

    config = json.loads(pipeline_config.read_text())
    config["schema_version"] = 2
    pipeline_config.write_text(json.dumps(config))
    data = tmp_path / "data.tsv"
    with data.open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows, columns = list(reader), reader.fieldnames
    for i, row in enumerate(rows):
        positives = row["localization_labels"].split(";")
        negative = (
            [label for label in config["labels"] if label not in positives]
            if i % 2
            else []
        )
        row["negative_labels"] = ";".join(negative)
        row["label_evidence"] = json.dumps(
            [
                dict(
                    label=label,
                    state=state,
                    evidence_type="experimental",
                    source="test",
                    source_version="1",
                    reference="fixture",
                )
                for state, names in [("positive", positives), ("negative", negative)]
                for label in names
            ]
        )
    with data.open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[*columns, "negative_labels", "label_evidence"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    partitions = load_partitions(load_config(pipeline_config))
    assert "negative_labels" in partitions["test"][0]
    output = run_pipeline(pipeline_config, tmp_path / "observed")
    with np.load(output / "evaluate" / "student.npz") as saved:
        assert np.isnan(saved["target"]).any()
        np.testing.assert_array_equal(
            saved["observation_mask"], np.isfinite(saved["target"])
        )
    report = json.loads((output / "evaluate" / "metrics.json").read_text())
    assert report["models"]["student"]["unknown_count"] == 1
    assert report["student_minus_control"]["cluster_count"] == 2

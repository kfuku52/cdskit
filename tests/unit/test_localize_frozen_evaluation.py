import csv
import json

import numpy as np
import pytest

from cdskit import localize_frozen_evaluation as frozen


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    data = tmp_path / "data.tsv"
    rows = []
    for i, (split, sequence, positive) in enumerate(
        [
            ("train", "MAAA", True),
            ("validation", "MCCC", False),
            ("test", "MDDD", True),
            ("test", "MEEE", False),
        ]
    ):
        rows.append(
            dict(
                accession=str(i),
                sequence=sequence,
                split=split,
                cluster_id=str(i),
                localization_labels="nucleus" if positive else "",
                negative_labels="" if positive else "nucleus",
                label_evidence=json.dumps(
                    [
                        dict(
                            label="nucleus",
                            state="positive" if positive else "negative",
                            evidence_type="experimental",
                            source="fixture",
                            source_version="1",
                            reference="fixture",
                        )
                    ]
                ),
            )
        )
    with data.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            dict(schema_version=2, data=dict(path=str(data)), labels=["nucleus"])
        )
    )
    protocol = tmp_path / "protocol-input.json"
    protocol.write_text(
        json.dumps(
            dict(
                population="synthetic test",
                annotation_protocol="fixture",
                selection_history="fixture",
                primary_metric="macro_f1",
                test_used_for_selection=False,
                comparison=dict(reference="control", candidate="student"),
                development_tsvs=[],
            )
        )
    )
    models = {}
    for name in ("control", "student"):
        model = tmp_path / (name + ".pt")
        model.write_bytes(b"fixture")
        models[name] = model
    monkeypatch.setattr(
        frozen,
        "audit_homology_partitions",
        lambda *args: [{"status": "ok", "fixture_only": True}],
    )
    monkeypatch.setattr(
        frozen,
        "load_localize_model",
        lambda path: {"localization_model": {"class_order": ["nucleus"]}},
    )
    monkeypatch.setattr(
        frozen,
        "_predict_model_on_rows",
        lambda model, rows: {
            "prob_matrix": np.array([[0.8], [0.2]]),
            "prediction_matrix": np.array([[1], [0]]),
        },
    )
    return config, models, protocol


def test_freeze_score_and_refuse_rescore(inputs, tmp_path):
    path = frozen.freeze_evaluation(*inputs, tmp_path / "result")
    result = frozen.evaluate_frozen(path)
    report = json.loads(result.read_text())
    assert report["models"]["student"]["macro_f1"] == 1
    assert report["paired_difference"]["percentile_95"]["macro_f1"] == [0, 0]
    with pytest.raises(ValueError, match="already scored"):
        frozen.evaluate_frozen(path)
    with pytest.raises(FileExistsError):
        frozen.freeze_evaluation(*inputs, tmp_path / "result")


def test_changed_model_and_reused_test_are_rejected(inputs, tmp_path):
    path = frozen.freeze_evaluation(*inputs, tmp_path / "result")
    inputs[1]["student"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen model"):
        frozen.evaluate_frozen(path)
    protocol = json.loads(inputs[2].read_text())
    protocol["test_used_for_selection"] = True
    inputs[2].write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="Previously inspected"):
        frozen.freeze_evaluation(*inputs, tmp_path / "other")


def test_legacy_contract_cannot_claim_experimental_evaluation(inputs, tmp_path):
    config = json.loads(inputs[0].read_text())
    config["schema_version"] = 1
    inputs[0].write_text(json.dumps(config))
    with pytest.raises(ValueError, match="schema 2"):
        frozen.freeze_evaluation(*inputs, tmp_path / "result")


def test_mutation_during_audit_cannot_be_frozen(inputs, tmp_path, monkeypatch):
    def mutate(*args):
        inputs[2].write_text(inputs[2].read_text() + "\n")
        return []

    monkeypatch.setattr(frozen, "audit_homology_partitions", mutate)
    with pytest.raises(ValueError, match="Frozen model"):
        frozen.freeze_evaluation(*inputs, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_test_rows_are_rechecked_after_loading(inputs, tmp_path, monkeypatch):
    path = frozen.freeze_evaluation(*inputs, tmp_path / "result")
    original = frozen.load_partitions

    def altered(config):
        partitions = original(config)
        partitions["test"][0]["sequence"] = "MFFFF"
        return partitions

    monkeypatch.setattr(frozen, "load_partitions", altered)
    with pytest.raises(ValueError, match="test rows changed"):
        frozen.evaluate_frozen(path)


def test_encoder_content_addition_invalidates_freeze(inputs, tmp_path, monkeypatch):
    from cdskit import localize_multilabel_plm

    encoder_dir = tmp_path / "encoder"
    encoder_dir.mkdir()
    (encoder_dir / "weights").write_text("original")

    class Encoder:
        def __init__(self, config):
            self.identity = sorted(p.name for p in encoder_dir.iterdir())

    monkeypatch.setattr(localize_multilabel_plm, "ResidueEncoder", Encoder)
    monkeypatch.setattr(
        frozen,
        "load_localize_model",
        lambda path: {
            "localization_model": {
                "class_order": ["nucleus"],
                "encoder": {"model_name": str(encoder_dir)},
                "encoder_identity": ["weights"],
            }
        },
    )
    path = frozen.freeze_evaluation(*inputs, tmp_path / "result")
    (encoder_dir / "tokenizer").write_text("changed")
    with pytest.raises(ValueError, match="encoder changed"):
        frozen.evaluate_frozen(path)


def test_positive_only_panel_cannot_establish_f1(inputs, tmp_path, monkeypatch):
    original = frozen.load_partitions

    def positive_only(config):
        partitions = original(config)
        partitions["test"] = partitions["test"][:1]
        return partitions

    monkeypatch.setattr(frozen, "load_partitions", positive_only)
    with pytest.raises(ValueError, match="positive-only"):
        frozen.freeze_evaluation(*inputs, tmp_path / "result")

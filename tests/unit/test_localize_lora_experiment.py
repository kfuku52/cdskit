import json
from pathlib import Path

import pytest

from cdskit.localize_evaluation import assert_disjoint, dataset_digest
from scripts import localize_homology_components as components
from scripts.localize_lora_experiment import main


def test_experiment_rejects_duplicate_seeds_before_data_or_output_writes(tmp_path):
    output = tmp_path / "run"
    with pytest.raises(SystemExit) as error:
        main(["--output", str(output), "--seeds", "1", "1"])
    assert error.value.code == 2
    assert not output.exists()


def test_experiment_preserves_completed_directory_before_reading_inputs(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    marker = output / "protocol.json"
    marker.write_text("completed")
    with pytest.raises(SystemExit) as error:
        main(["--output", str(output)])
    assert error.value.code == 2
    assert marker.read_text() == "completed"


def test_components_join_indirect_homology_and_existing_groups():
    seed = [0, "b", "c", "unrelated", 0]
    groups, count = components.component_groups(seed, [(0, 1), (2, 1)])
    assert groups == ["0", "0", "0", "3", "0"]
    assert count == 2
    assert seed == [0, "b", "c", "unrelated", 0]


def test_component_payload_identifies_unmodified_source(tmp_path, monkeypatch):
    rows = [
        dict(accession=str(i), sequence="M" + aa * 5, fold_id=str(i % 2))
        for i, aa in enumerate("ACDE")
    ]
    source_digest = dataset_digest(rows)
    source = tmp_path / "source.tsv"
    source.write_text("source snapshot")
    clusters = tmp_path / "seed.json"
    clusters.write_text(
        json.dumps({"dataset_sha256": source_digest, "groups": ["a", "b", "c", "d"]})
    )
    monkeypatch.setattr(components, "_read_prepared_tsv", lambda path: rows)
    monkeypatch.setattr(components.shutil, "which", lambda name: "/test/mmseqs")
    monkeypatch.setattr(
        components.subprocess, "check_output", lambda *a, **k: "fixture-version"
    )

    def search(command, **kwargs):
        Path(command[4]).write_text("r0\tr1\t100\t1\t1\n")

    def audit(partitions, **kwargs):
        assert_disjoint(partitions["0"], partitions["1"])
        return []

    monkeypatch.setattr(components.subprocess, "run", search)
    monkeypatch.setattr(components, "audit_homology_partitions", audit)
    output = tmp_path / "components"
    report = components.prepare(source, clusters, output)
    payload = json.loads((output / "clusters.json").read_text())
    assert payload["dataset_sha256"] == source_digest == dataset_digest(rows)
    assert all("cluster_id" not in row for row in rows)
    assert report["kept_rows"] == 2
    assert report["removed_cross_fold_component_rows"] == 2

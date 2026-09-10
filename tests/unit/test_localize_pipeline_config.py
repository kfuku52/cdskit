import json

import pytest

from cdskit.cli import psr
from cdskit.command_paths import command_paths
from cdskit.localize_pipeline_config import load_config, load_partitions


@pytest.fixture
def config_file(tmp_path):
    (tmp_path / "data.tsv").write_text(
        "accession\tsequence\tlocalization_labels\tsplit\tcluster_id\n"
        "a\tMAAA\tnucleus\ttrain\tc1\n"
        "b\tMCCC\tcytoplasm\tvalidation\tc2\n"
        "c\tMDDD\tnucleus;cytoplasm\ttest\tc3\n"
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"schema_version": 1, "data": {"path": "data.tsv"}}))
    return path


def test_config_defaults_paths_and_cli(config_file):
    config = load_config(config_file)
    assert config["decision_policy"] == "safe-v1"
    assert config["feature_schema"] == "localize-pts2-9-v2"
    assert config["ensure_one_label"] is False
    assert config["data"]["path"] == str(config_file.parent / "data.tsv")
    assert len(load_partitions(config)["test"]) == 1
    args = psr.parse_args(
        [
            "localize-learn",
            "--stage",
            "distill",
            "--config",
            str(config_file),
            "--run_dir",
            "run",
        ]
    )
    assert command_paths(args) == ([str(config_file)], [])
    assert psr.parse_args(["localize-learn"]).stage == "train"


@pytest.mark.parametrize(
    "change",
    [
        {"teacher": {"loss": "unknown"}},
        {"teacher": {"selection_metric": "test_f1"}},
        {"extra": 1},
        {"schema_version": True},
        {"threads": 0},
        {"ensure_one_label": "no"},
        {"decision_policy": "future"},
        {"student": {"seed": 2**32}},
        {"teacher": {"learning_rate": float("nan")}},
        {"labels": ["nucleus", "nucleus"]},
        {"labels": ["invalid"]},
        {"student": {"kernel_sizes": [2]}},
        {"teacher": {"overlap": 1000}},
        {"student": {"batch_size": True}},
        {"student": {"dropout": 1}},
        {"data": {"path": ""}},
    ],
)
def test_invalid_config(config_file, change):
    config = json.loads(config_file.read_text())
    config.update(change)
    config_file.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(config_file)


@pytest.mark.parametrize(
    "old,new",
    [
        ("MCCC", "maaa*"),
        ("c2", "c1"),
        ("validation", "other"),
        ("cytoplasm", "invalid"),
        ("b\t", "a\t"),
        ("c2", ""),
        ("MCCC", "XXXX"),
        ("MCCC", "M"),
    ],
)
def test_partition_errors(config_file, old, new):
    data = config_file.parent / "data.tsv"
    data.write_text(data.read_text().replace(old, new))
    with pytest.raises(ValueError):
        load_partitions(load_config(config_file))


@pytest.mark.parametrize(
    "extension,text",
    [
        ("json", '{"schema_version":1,"schema_version":1}'),
        ("yaml", "schema_version: 1\nschema_version: 1\n"),
    ],
)
def test_duplicate_keys(tmp_path, extension, text):
    if extension == "yaml":
        pytest.importorskip("yaml")
    path = tmp_path / ("config." + extension)
    path.write_text(text)
    with pytest.raises(ValueError, match="Duplicate"):
        load_config(path)


def test_stage_and_legacy_flags_rejected(config_file):
    from cdskit.localize_learn import localize_learn_main

    for argv in (
        ["--config", str(config_file)],
        ["--stage", "distill", "--model_arch", "esm_head"],
        ["--stage", "all"],
    ):
        with pytest.raises(ValueError):
            localize_learn_main(psr.parse_args(["localize-learn", *argv]))


def test_yaml_matches_json(config_file):
    yaml = pytest.importorskip("yaml")
    path = config_file.with_suffix(".yaml")
    path.write_text(yaml.safe_dump(json.loads(config_file.read_text())))
    assert load_config(path) == load_config(config_file)


def test_loaded_defaults_are_independent(config_file):
    first = load_config(config_file)
    second = load_config(config_file)
    first["student"]["kernel_sizes"].append(17)
    assert second["student"]["kernel_sizes"] == [3, 5, 9, 15]


@pytest.mark.parametrize("sequence", ["", "X", "XXXX", "M"])
def test_safe_test_rows_retain_abstention_coverage(config_file, sequence):
    data = config_file.parent / "data.tsv"
    data.write_text(data.read_text().replace("MDDD", sequence))
    config = load_config(config_file)
    assert load_partitions(config)["test"][0]["sequence"] == sequence
    if not sequence or set(sequence) == {"X"}:
        config["decision_policy"] = "legacy"
        with pytest.raises(ValueError, match="sequence"):
            load_partitions(config)

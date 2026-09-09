"""Strict, portable configuration and partition input for staged localization learning."""

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from cdskit.deeploc_benchmark import DEEPLOC_LOCALIZATION_LABELS
from cdskit.localize_evaluation import assert_disjoint
from cdskit.localize_model import to_canonical_aa_sequence
from cdskit.tsvio import read_tsv


TEACHER_DEFAULTS = {
    "model_name": "facebook/esm2_t33_650M_UR50D",
    "revision": "",
    "cache_dir": "",
    "local_files_only": False,
    "pooling": "mean",
    "window": 1000,
    "overlap": 128,
    "epochs": 12,
    "batch_size": 8,
    "learning_rate": 0.001,
    "patience": 3,
    "seed": 1,
    "device": "auto",
}
STUDENT_DEFAULTS = {
    "seq_len": 512,
    "embed_dim": 32,
    "num_filters": 32,
    "kernel_sizes": [3, 5, 9, 15],
    "dropout": 0.25,
    "epochs": 12,
    "batch_size": 256,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "seed": 1,
    "device": "auto",
    "patience": 3,
    "use_class_weight": True,
    "sequence_layout": "separate_termini",
    "mask_padding": True,
    "distillation_weight": 0.5,
    "train_control": True,
}
DATA_DEFAULTS = {
    "path": "",
    "id_col": "accession",
    "sequence_col": "sequence",
    "label_col": "localization_labels",
    "split_col": "split",
    "cluster_col": "cluster_id",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _unique_mapping(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate configuration key: {}".format(key))
        result[key] = value
    return result


def _read_config(path):
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".json":
        return json.loads(text, object_pairs_hook=_unique_mapping)
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "YAML configuration requires PyYAML; install cdskit[ml]."
        ) from exc

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        return _unique_mapping(
            (loader.construct_object(k), loader.construct_object(v, deep=True))
            for k, v in node.value
        )

    UniqueLoader.add_constructor("tag:yaml.org,2002:map", mapping)
    return yaml.load(text, Loader=UniqueLoader)


def _section(value, defaults, name):
    if not isinstance(value, dict) or set(value) - set(defaults):
        raise ValueError("Unknown keys or invalid mapping in {}.".format(name))
    result = {**deepcopy(defaults), **value}
    for key, default in defaults.items():
        item = result[key]
        valid = type(item) is type(default)
        if isinstance(default, float):
            valid = type(item) in (int, float) and math.isfinite(item)
        if not valid:
            raise ValueError("Invalid type/value for {}.{}.".format(name, key))
    return result


def _validate_training(config):
    for section in ("teacher", "student"):
        settings = config[section]
        for key in ("epochs", "batch_size", "patience"):
            if settings[key] < 1:
                raise ValueError("{}.{} must be positive.".format(section, key))
        if not 0 <= settings["seed"] < 2**32 or settings["learning_rate"] <= 0:
            raise ValueError("Invalid seed or learning_rate in {}.".format(section))
        if settings["device"] not in ("cpu", "cuda", "mps", "auto"):
            raise ValueError("Invalid device in {}.".format(section))
    teacher, student = config["teacher"], config["student"]
    if teacher["pooling"] not in ("mean", "light_attention", "label_attention"):
        raise ValueError("Invalid teacher.pooling.")
    if not 0 <= teacher["overlap"] < teacher["window"] or teacher["window"] < 4:
        raise ValueError("Invalid teacher window/overlap.")
    if student["sequence_layout"] not in ("legacy", "separate_termini", "windows"):
        raise ValueError("Invalid student.sequence_layout.")
    for key in ("seq_len", "embed_dim", "num_filters"):
        if student[key] < 1:
            raise ValueError("student.{} must be positive.".format(key))
    if student["seq_len"] < 4:
        raise ValueError("student.seq_len must be at least 4.")
    if not student["kernel_sizes"] or any(
        type(k) is not int or k < 1 or k % 2 == 0 for k in student["kernel_sizes"]
    ):
        raise ValueError("student.kernel_sizes must contain positive odd integers.")
    if (
        not 0 <= student["dropout"] < 1
        or not 0 <= student["distillation_weight"] <= 1
        or student["weight_decay"] < 0
    ):
        raise ValueError("Invalid dropout, distillation_weight or weight_decay.")


def load_config(path):
    path = Path(path).expanduser().resolve()
    raw = _read_config(path)
    allowed = {
        "schema_version",
        "data",
        "labels",
        "threads",
        "teacher",
        "student",
        "ensure_one_label",
    }
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("Unknown keys or invalid pipeline configuration.")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("Pipeline configuration requires schema_version: 1.")
    config: dict[str, Any] = {"schema_version": 1, "threads": raw.get("threads", 1)}
    if type(config["threads"]) is not int or config["threads"] < 1:
        raise ValueError("threads must be a positive integer.")
    for name, defaults in (
        ("data", DATA_DEFAULTS),
        ("teacher", TEACHER_DEFAULTS),
        ("student", STUDENT_DEFAULTS),
    ):
        config[name] = _section(raw.get(name, {}), defaults, name)
    labels = raw.get("labels", list(DEEPLOC_LOCALIZATION_LABELS))
    if (
        not isinstance(labels, list)
        or not labels
        or any(not isinstance(x, str) for x in labels)
        or len(labels) != len(set(labels))
        or set(labels) - set(DEEPLOC_LOCALIZATION_LABELS)
    ):
        raise ValueError("labels must be unique supported localization names.")
    config["labels"] = labels
    config["ensure_one_label"] = raw.get("ensure_one_label", False)
    if type(config["ensure_one_label"]) is not bool:
        raise ValueError("ensure_one_label must be a boolean.")
    if not config["data"]["path"]:
        raise ValueError("data.path is required.")
    for section, key in (("data", "path"), ("teacher", "cache_dir")):
        if config[section][key]:
            config[section][key] = str(
                (path.parent / Path(config[section][key]).expanduser()).resolve()
            )
    source = Path(config["teacher"]["model_name"]).expanduser()
    if (path.parent / source).is_dir():
        config["teacher"]["model_name"] = str((path.parent / source).resolve())
    _validate_training(config)
    return config


def load_partitions(config):
    settings = config["data"]
    columns = [
        settings[key] for key in ("id_col", "sequence_col", "label_col", "split_col")
    ]
    if (
        len(set(columns)) != len(columns)
        or not all(columns)
        or settings["cluster_col"] in columns
    ):
        raise ValueError("Input column names must be nonempty and distinct.")
    rows = read_tsv(settings["path"], required_columns=columns)
    partitions: dict[str, list[dict[str, str]]] = {
        name: [] for name in ("train", "validation", "test")
    }
    seen = set()
    for row in rows:
        accession = row[settings["id_col"]].strip()
        sequence = to_canonical_aa_sequence(row[settings["sequence_col"]])
        split = row[settings["split_col"]].strip()
        labels = [x.strip() for x in row[settings["label_col"]].split(";") if x.strip()]
        if not accession or accession in seen or not sequence or set(sequence) == {"X"}:
            raise ValueError(
                "Empty/duplicate ID or empty/unknown sequence: {}".format(accession)
            )
        if split not in partitions or set(labels) - set(config["labels"]):
            raise ValueError(
                "Invalid split or localization labels: {}".format(accession)
            )
        seen.add(accession)
        partitions[split].append(
            {
                "accession": accession,
                "sequence": sequence,
                "localization_labels": ";".join(
                    x for x in config["labels"] if x in labels
                ),
                "cluster_id": row.get(settings["cluster_col"], "").strip(),
            }
        )
    if not partitions["train"] or not partitions["validation"]:
        raise ValueError("Nonempty train and validation partitions are required.")
    all_rows = [row for rows in partitions.values() for row in rows]
    if any(row["cluster_id"] for row in all_rows) and not all(
        row["cluster_id"] for row in all_rows
    ):
        raise ValueError("Supply cluster IDs for every row or omit them for every row.")
    assert_disjoint(partitions["train"], partitions["validation"])
    assert_disjoint(partitions["train"] + partitions["validation"], partitions["test"])
    return partitions

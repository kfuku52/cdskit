"""Restartable stages for public localization teacher/student experiments."""

import json
import os
import platform
import shutil
import socket
import tempfile
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

from cdskit.localize_pipeline_config import (
    digest,
    file_digest,
    load_config,
    load_partitions,
)
from cdskit import localize_pipeline_stages as stages
from cdskit.util import atomic_write_json

STAGES = ("teacher", "predict", "distill", "evaluate")


def code_identity():
    return digest(
        {p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    )


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "absent"


def environment():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("cdskit", "numpy", "torch", "transformers", "PyYAML"):
        result[name] = package_version(name)
    return result


@contextmanager
def run_lock(root):
    lock = root / ".pipeline.lock"
    try:
        handle = lock.open("x")
    except FileExistsError as exc:
        raise ValueError(
            "Run is locked: {}. Check its owner before removing a stale lock.".format(
                lock
            )
        ) from exc
    try:
        with handle:
            json.dump({"pid": os.getpid(), "host": socket.gethostname()}, handle)
        yield
    finally:
        lock.unlink()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest(root):
    manifest = read_json(root / "run.json")
    if manifest.get("fingerprint") != digest(manifest["identity"]):
        raise ValueError("Run manifest identity is corrupt: {}".format(root))
    return manifest


def verify_stage(root, name):
    manifest = read_manifest(root)
    record = read_json(root / name / "stage.json")
    if (
        record.get("run_fingerprint") != manifest["fingerprint"]
        or record.get("stage") != name
    ):
        raise ValueError("Stage provenance differs: {}".format(name))
    outputs = record.get("outputs", {})
    if not outputs or any(Path(path).name != path for path in outputs):
        raise ValueError("Invalid stage output inventory.")
    for path, checksum in outputs.items():
        if file_digest(root / name / path) != checksum:
            raise ValueError("Stage output changed: {}/{}".format(name, path))
    return record


def input_identity(config, partitions, data_sha256):
    model_dir = Path(config["teacher"]["model_name"])
    encoder_files = {}
    if model_dir.is_dir():
        encoder_files = {
            str(p.relative_to(model_dir)): file_digest(p)
            for p in sorted(model_dir.rglob("*"))
            if p.is_file()
        }
    return {
        "schema_version": 1,
        "config": config,
        "code_sha256": code_identity(),
        "data_sha256": data_sha256,
        "partitions": stages.partition_identity(partitions),
        "encoder_files": encoder_files,
    }


def load_inputs(config_path):
    config_sha256 = file_digest(config_path)
    config = load_config(config_path)
    data_sha256 = file_digest(config["data"]["path"])
    partitions = load_partitions(config)
    identity = input_identity(config, partitions, data_sha256)
    if config_sha256 != file_digest(config_path) or data_sha256 != file_digest(
        config["data"]["path"]
    ):
        raise ValueError("Pipeline inputs changed while reading them.")
    return config, partitions, identity


def prepare_manifest(root, identity):
    manifest = {
        "identity": identity,
        "fingerprint": digest(identity),
        "environment": environment(),
    }
    if (root / "run.json").exists():
        if read_manifest(root)["fingerprint"] != manifest["fingerprint"]:
            raise ValueError(
                "Run inputs/configuration/code changed; use a new run directory."
            )
    else:
        if any(p.name != ".pipeline.lock" for p in root.iterdir()):
            raise ValueError("New run directory must be empty.")
        atomic_write_json(str(root / "run.json"), manifest)
    return manifest


def check_source(source, config, partitions):
    if (source / ".pipeline.lock").exists():
        raise ValueError("Teacher source run is currently locked.")
    identity = read_manifest(source)["identity"]
    if identity["config"]["labels"] != config["labels"] or any(
        identity["partitions"][name] != stages.partition_identity(partitions)[name]
        for name in ("train", "validation")
    ):
        raise ValueError(
            "Teacher source labels or training/validation partitions differ."
        )


def stage_dependencies(root, source, name):
    required = {
        "teacher": [],
        "predict": [(source, "teacher")],
        "distill": [(source, "teacher"), (source, "predict")],
        "evaluate": [(root, "distill")],
    }[name]
    result = {}
    for directory, stage in required:
        result[stage] = digest(verify_stage(directory, stage))
    if name == "distill":
        prediction = verify_stage(source, "predict")
        if prediction["dependencies"].get("teacher") != result["teacher"]:
            raise ValueError("Prediction stage references a different teacher.")
    return result


def execute_stage(
    root, source, name, config, partitions, manifest, resume, config_path
):
    dependencies = stage_dependencies(root, source, name)
    destination = root / name
    if destination.exists():
        record = verify_stage(root, name)
        if not resume or record["dependencies"] != dependencies:
            raise ValueError(
                "Stage exists or its dependencies changed: {}".format(name)
            )
        return
    temporary = Path(tempfile.mkdtemp(prefix=".{}-".format(name), dir=root))
    try:
        teacher = source / "teacher" / "model.pt"
        predictions = source / "predict" / "probabilities.npz"
        if name == "teacher":
            stages.fit_teacher(config, partitions, temporary)
        elif name == "predict":
            stages.make_teacher_predictions(config, partitions, teacher, temporary)
        elif name == "distill":
            stages.fit_students(config, partitions, teacher, predictions, temporary)
        else:
            stages.evaluate_students(config, partitions, root / "distill", temporary)
        _, _, current_identity = load_inputs(config_path)
        if (
            digest(current_identity) != manifest["fingerprint"]
            or read_manifest(root)["fingerprint"] != manifest["fingerprint"]
        ):
            raise ValueError(
                "Pipeline inputs/configuration/code changed during execution."
            )
        if dependencies != stage_dependencies(root, source, name):
            raise ValueError("Stage dependencies changed during execution.")
        record = {
            "stage": name,
            "run_fingerprint": manifest["fingerprint"],
            "dependencies": dependencies,
            "environment": environment(),
            "outputs": {
                p.name: file_digest(p) for p in temporary.iterdir() if p.is_file()
            },
        }
        atomic_write_json(str(temporary / "stage.json"), record)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def validate_pipeline_paths(config_path, config, root, source):
    for path in (
        config_path,
        config["data"]["path"],
        config["teacher"]["cache_dir"],
        config["teacher"]["model_name"],
    ):
        if path and Path(path).expanduser().resolve().is_relative_to(root):
            raise ValueError(
                "Configuration, data, encoder and cache must be outside the run directory."
            )
    encoder = Path(config["teacher"]["model_name"]).expanduser().resolve()
    cache = config["teacher"]["cache_dir"]
    if encoder.is_dir():
        for path in (root, Path(cache).resolve() if cache else None):
            if path is not None and (
                path.is_relative_to(encoder) or encoder.is_relative_to(path)
            ):
                raise ValueError("Encoder, cache and run directories must not overlap.")
    if cache:
        cache_path = Path(cache).resolve()
        for directory in (root, source):
            if cache_path.is_relative_to(directory) or directory.is_relative_to(
                cache_path
            ):
                raise ValueError("Cache and run directories must not overlap.")
        if cache_path.exists() and not cache_path.is_dir():
            raise ValueError("teacher.cache_dir must be a directory.")


def run_pipeline(config_path, run_dir, stage="all", teacher_run=None, resume=True):
    if stage not in (*STAGES, "all"):
        raise ValueError("Invalid pipeline stage.")
    config_path = Path(config_path).expanduser().resolve()
    config, partitions, identity = load_inputs(config_path)
    root = Path(run_dir).expanduser().resolve()
    source = Path(teacher_run).expanduser().resolve() if teacher_run else root
    validate_pipeline_paths(config_path, config, root, source)
    if source != root:
        if source.is_relative_to(root) or root.is_relative_to(source):
            raise ValueError(
                "Teacher source and student run directories must not overlap."
            )
        check_source(source, config, partitions)
        if stage in ("teacher", "predict"):
            raise ValueError(
                "teacher_run is only supported for distill, evaluate or all."
            )
    selected = list(STAGES) if stage == "all" else [stage]
    if stage == "all":
        if source != root:
            selected = ["distill", "evaluate"]
        if not partitions["test"]:
            selected.remove("evaluate")
    if "evaluate" in selected and not partitions["test"]:
        raise ValueError("The evaluate stage requires a test partition.")
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root):
        new_run = not (root / "run.json").exists()
        audit = {"status": "provided_groups_only; homology not independently checked"}
        if new_run and config["data"]["homology_audit"] == "mmseqs":
            from cdskit.localize_splits import audit_homology_partitions

            audit = {
                "status": "ok",
                "pairs": audit_homology_partitions(partitions, config["threads"]),
            }
        _, _, current_identity = load_inputs(config_path)
        if current_identity != identity:
            raise ValueError("Pipeline inputs changed during partition audit.")
        manifest = prepare_manifest(root, identity)
        if new_run:
            manifest["partition_audit"] = audit
            atomic_write_json(str(root / "run.json"), manifest)
        import torch

        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(config["threads"])
            for name in selected:
                execute_stage(
                    root,
                    source,
                    name,
                    config,
                    partitions,
                    manifest,
                    resume,
                    config_path,
                )
        finally:
            torch.set_num_threads(previous_threads)
    return root


def pipeline_main(args):
    from cdskit.cli import p_localize_learn

    allowed = {
        "stage",
        "config",
        "run_dir",
        "teacher_run",
        "resume",
        "handler",
        "command",
        "debug",
    }
    defaults = vars(p_localize_learn.parse_args([]))
    for action in p_localize_learn._actions:
        if action.dest not in allowed and getattr(
            args, action.dest, defaults.get(action.dest)
        ) != defaults.get(action.dest):
            raise ValueError(
                "Legacy option --{} cannot be mixed with staged configuration.".format(
                    action.dest
                )
            )
    if not args.config or not args.run_dir:
        raise ValueError("Staged learning requires --config and --run_dir.")
    result = run_pipeline(
        args.config, args.run_dir, args.stage, args.teacher_run, args.resume
    )
    print("Localization pipeline artifacts: {}".format(result))

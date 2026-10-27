#!/usr/bin/env python3
"""Matched frozen/LoRA development comparison; never opens the historical HPA set."""

import argparse
from collections import defaultdict
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from cdskit.deeploc_benchmark import (
    DEEPLOC_LOCALIZATION_LABELS as LABELS,
    _read_prepared_tsv,
    build_label_matrix,
    compute_multilabel_metrics,
)
from cdskit.localize_decision import sequence_quality
from cdskit.localize_evaluation import (
    assert_disjoint,
    dataset_digest,
    normalize_partition_id,
    paired_cluster_bootstrap,
    probability_metrics,
)
from cdskit.localize_model import (
    _tune_binary_threshold,
    load_localize_model,
    save_localize_model,
)
from cdskit.localize_multilabel_plm import fit_multilabel_plm, predict_multilabel_plm
from cdskit.localize_pipeline import code_identity
from cdskit.localize_pipeline_config import file_digest
from cdskit.localize_splits import audit_homology_partitions
from cdskit.util import atomic_write_json


def prepare_partitions(args):
    rows = _read_prepared_tsv(args.training_tsv)
    clusters = json.loads(Path(args.clusters).read_text())
    if (
        clusters.get("dataset_sha256") != dataset_digest(rows)
        or len(clusters.get("groups", [])) != len(rows)
        or clusters.get("report", {}).get("status") != "ok"
        or clusters["report"].get("tool") != "mmseqs"
        or clusters["report"].get("min_seq_id") != 0.3
        or clusters["report"].get("coverage") != 0.8
        or clusters["report"].get("cov_mode") != 0
    ):
        raise ValueError(
            "Cached MMseqs clusters do not match this source dataset/protocol."
        )
    memberships = defaultdict(set)
    for row, group in zip(rows, clusters["groups"], strict=True):
        group = normalize_partition_id(group)
        if group is None:
            raise ValueError("Missing MMseqs cluster identifier.")
        row["cluster_id"] = str(group)
        memberships[str(group)].add(row["fold_id"])
    clean = [row for row in rows if len(memberships[row["cluster_id"]]) == 1]
    eligible = [
        row
        for row in clean
        if not args.max_length or len(row["sequence"]) <= args.max_length
    ]
    partitions = {
        "test": [row for row in eligible if row["fold_id"] == args.test_fold],
        "validation": [
            row for row in eligible if row["fold_id"] == args.validation_fold
        ],
        "train": [
            row
            for row in eligible
            if row["fold_id"] not in (args.test_fold, args.validation_fold)
        ],
    }
    for name, values in partitions.items():
        if name != "test":
            values = [row for row in values if not sequence_quality(row["sequence"])]
        values = sorted(
            values,
            key=lambda row: hashlib.sha256(
                (str(args.subset_seed) + ":" + row["accession"]).encode()
            ).hexdigest(),
        )
        limit = getattr(args, name + "_rows")
        partitions[name] = values[:limit] if limit else values
        if not partitions[name]:
            raise ValueError("Empty experiment partition: " + name)
    assert_disjoint(partitions["train"], partitions["validation"])
    assert_disjoint(partitions["train"] + partitions["validation"], partitions["test"])
    homology = audit_homology_partitions(partitions, threads=4)
    audit = {
        "source_rows": len(rows),
        "cross_fold_cluster_rows_removed": len(rows) - len(clean),
        "length_eligible_rows": len(eligible),
        "audit_scope": "Source-digest-verified MMseqs clusters and a fresh cross-partition search.",
        "cross_partition_homology": homology,
        "partitions": {},
    }
    for name, values in partitions.items():
        y = build_label_matrix(values, LABELS, "localization_labels")
        audit["partitions"][name] = {
            "rows": len(values),
            "sha256": dataset_digest(values),
            "positive_counts": dict(zip(LABELS, (y == 1).sum(0).tolist(), strict=True)),
        }
    return partitions, audit


def _release(model, torch):
    model["localization_model"].pop("_runtime_model_cache", None)
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_model(args, partitions, output, kind, seed, torch):
    stem = output / (kind + "_seed" + str(seed))
    config = {
        "model_name": args.model_name,
        "revision": args.revision,
        "local_files_only": True,
        "window": args.window,
        "overlap": args.overlap,
        "pooling": "light_attention",
        "memory_cache_bytes": 512 * 1024**2,
        "cache_dir": str(output / "frozen_embeddings") if kind == "frozen" else "",
    }
    if kind == "lora":
        config.update(
            lora_rank=args.rank,
            lora_alpha=args.alpha,
            lora_learning_rate=args.adapter_lr,
        )
    sequences = {
        name: [row["sequence"] for row in values] for name, values in partitions.items()
    }
    targets = {
        name: build_label_matrix(values, LABELS, "localization_labels")
        for name, values in partitions.items()
    }
    started = time.perf_counter()
    print("TRAIN", kind, "seed", seed, flush=True)
    head = fit_multilabel_plm(
        sequences["train"],
        targets["train"],
        list(LABELS),
        config,
        validation_sequences=sequences["validation"],
        validation_y=targets["validation"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=seed,
        learning_rate=0.001,
        patience=3,
        device=args.device,
        progress_callback=lambda item: print(kind, seed, json.dumps(item), flush=True),
    )
    head.update(ensure_one_label=False, decision_policy="safe-v1")
    validation = predict_multilabel_plm(
        sequences["validation"], head, device=args.device, batch_size=args.batch_size
    )
    for j, label in enumerate(LABELS):
        observed = np.isfinite(targets["validation"][:, j])
        if len(np.unique(targets["validation"][observed, j])) == 2:
            head["class_thresholds"][label] = _tune_binary_threshold(
                validation["prob_matrix"][observed, j],
                targets["validation"][observed, j],
                objective="f1",
            )
    model = {
        "model_type": "multilabel_plm_lora_v1"
        if kind == "lora"
        else "multilabel_plm_v1",
        "feature_names": [],
        "localization_model": head,
        "perox_model": {"mode": "embedded_multilabel"},
        "metadata": {
            "experiment": "matched_lora_development",
            "seed": seed,
            "partitions": {
                name: dataset_digest(values) for name, values in partitions.items()
            },
        },
    }
    # Persist and reload the actual artifact used to score development-test rows.
    _release(model, torch)
    save_localize_model(model, str(stem.with_suffix(".pt")))
    model = load_localize_model(str(stem.with_suffix(".pt")))
    pred = predict_multilabel_plm(
        sequences["test"],
        model["localization_model"],
        device=args.device,
        batch_size=args.batch_size,
    )
    scored = np.asarray(pred["score_available"], dtype=bool)
    metrics = compute_multilabel_metrics(
        targets["test"], pred["prediction_matrix"], list(LABELS)
    )
    metrics.update(
        probability_metrics(
            targets["test"][scored], pred["prob_matrix"][scored], list(LABELS)
        )
    )
    metrics.update(
        selected_epoch=head["selected_epoch"],
        training_history=head["training_history"],
        elapsed_seconds=time.perf_counter() - started,
        score_coverage=float(scored.mean()),
        class_thresholds=head["class_thresholds"],
        model_sha256=file_digest(stem.with_suffix(".pt")),
        adapter_parameters=sum(
            value.numel() for value in head.get("lora_state_dict", {}).values()
        ),
    )
    np.savez_compressed(
        stem.with_suffix(".npz"),
        target=targets["test"],
        probability=pred["prob_matrix"],
        prediction=pred["prediction_matrix"],
        score_available=scored,
        observed_mask=np.isfinite(targets["test"]),
        decision_status=np.asarray(pred["decision_status"]),
        quality_reason=np.asarray(pred["quality_reason"]),
        accession=np.array([row["accession"] for row in partitions["test"]]),
        cluster_id=np.array([row["cluster_id"] for row in partitions["test"]]),
        labels=np.array(LABELS),
    )
    atomic_write_json(str(stem.with_suffix(".json")), metrics)
    print(
        "RESULT",
        kind,
        seed,
        {
            key: metrics[key]
            for key in (
                "macro_f1",
                "micro_f1",
                "macro_average_precision_observed",
                "elapsed_seconds",
            )
        },
        flush=True,
    )
    _release(model, torch)
    return metrics, pred["prediction_matrix"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training_tsv",
        default="data/localize_bench/deeploc21/deeploc21_localization_train_validation.tsv",
    )
    parser.add_argument(
        "--clusters",
        default="data/localize_bench/full_localization_20260908/clusters.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument(
        "--revision", default="08e4846e537177426273712802403f7ba8261b6c"
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--test_fold", default="0")
    parser.add_argument("--validation_fold", default="4")
    parser.add_argument("--subset_seed", type=int, default=20261027)
    parser.add_argument("--train_rows", type=int, default=0)
    parser.add_argument("--validation_rows", type=int, default=0)
    parser.add_argument("--test_rows", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8)
    parser.add_argument("--adapter_lr", type=float, default=1e-4)
    args = parser.parse_args(argv)
    if args.test_fold == args.validation_fold or any(
        getattr(args, name) < 0
        for name in ("train_rows", "validation_rows", "test_rows", "max_length")
    ):
        parser.error(
            "Use distinct test/validation folds and nonnegative subset limits."
        )
    if len(set(args.seeds)) != len(args.seeds) or any(
        not 0 <= seed < 2**32 for seed in args.seeds
    ):
        parser.error("seeds must be unique integers in [0, 2**32).")
    if args.epochs < 1 or args.batch_size < 1 or args.rank < 1:
        parser.error("epochs, batch_size and rank must be positive.")
    if args.window < 4 or not 0 <= args.overlap < args.window:
        parser.error("Require window >= 4 and 0 <= overlap < window.")
    if any(
        not np.isfinite(value) or value <= 0 for value in (args.alpha, args.adapter_lr)
    ):
        parser.error("alpha and adapter_lr must be finite and positive.")
    output = Path(args.output)
    if output.exists():
        parser.error("Output exists; choose a fresh experiment directory.")
    import torch

    torch.set_num_threads(4)
    partitions, audit = prepare_partitions(args)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "scope": "Development comparison on previously examined DeepLoc data; not independent biological validation.",
        "labels": list(LABELS),
        "label_policy": "Original DeepLoc closed-world dataset convention; zeros are not experimentally verified absence.",
        "configuration": vars(args),
        "partition_audit": audit,
        "source_sha256": file_digest(args.training_tsv),
        "clusters_sha256": file_digest(args.clusters),
        "code_sha256": code_identity(),
        "script_sha256": file_digest(__file__),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "numpy")
            },
            "mmseqs": subprocess.run(
                ["mmseqs", "version"], check=True, text=True, capture_output=True
            ).stdout.strip(),
        },
        "selection": "Validation BCE for epochs; validation-only independent F1 thresholds; no forced label.",
        "adoption_criteria": {
            "macro_f1_delta": 0.02,
            "macro_ap_delta_min": 0,
            "micro_f1_delta_min": -0.005,
            "requires": "5 folds x 3 seeds, positive paired cluster-bootstrap interval, and a new independent external test.",
        },
    }
    atomic_write_json(str(output / "protocol.json"), protocol)
    atomic_write_json(str(output / "partitions.json"), partitions)
    source = output / "source"
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "cdskit",
        source / "cdskit",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (source / "scripts").mkdir()
    shutil.copy2(__file__, source / "scripts" / Path(__file__).name)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(Path(__file__).resolve().parents[1] / name, source / name)
    results = []
    for seed in args.seeds:
        frozen, prediction_a = run_model(
            args, partitions, output, "frozen", seed, torch
        )
        lora, prediction_b = run_model(args, partitions, output, "lora", seed, torch)
        comparison = {
            "seed": seed,
            "frozen": frozen,
            "lora": lora,
            "delta": {
                key: lora[key] - frozen[key]
                for key in ("macro_f1", "micro_f1", "macro_average_precision_observed")
            },
            "paired_cluster_bootstrap": paired_cluster_bootstrap(
                build_label_matrix(partitions["test"], LABELS, "localization_labels"),
                prediction_a,
                prediction_b,
                [row["cluster_id"] for row in partitions["test"]],
                list(LABELS),
                compute_multilabel_metrics,
            ),
        }
        results.append(comparison)
        atomic_write_json(str(output / "comparison.json"), results)
    return results


if __name__ == "__main__":
    main()

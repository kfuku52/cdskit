#!/usr/bin/env python3
"""Summarize full-data OOF, clustered paired uncertainty, and fixed external results."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cdskit.deeploc_benchmark import (
    DEEPLOC_LOCALIZATION_LABELS as LABELS,
    compute_multilabel_metrics,
    _read_prepared_tsv,
)
from cdskit.util import atomic_write_json


def paired_cluster_intervals(target, baseline, candidate, groups, iterations=1000):
    _, membership = np.unique(np.asarray(groups), return_inverse=True)
    n_group = int(membership.max()) + 1
    all_counts = []
    for prediction in (baseline, candidate):
        counts = np.zeros((n_group, 3, target.shape[1]))
        for j, values in enumerate(
            (target * prediction, (1 - target) * prediction, target * (1 - prediction))
        ):
            for k in range(target.shape[1]):
                counts[:, j, k] = np.bincount(
                    membership, weights=values[:, k], minlength=n_group
                )
        all_counts.append(counts)
    rng = np.random.default_rng(20260908)
    deltas = []
    for _ in range(iterations):
        weights = np.bincount(rng.integers(n_group, size=n_group), minlength=n_group)
        scores = []
        for counts in all_counts:
            tp, fp, fn = (counts * weights[:, None, None]).sum(0)
            scores.append(
                [
                    np.mean(2 * tp / np.maximum(1, 2 * tp + fp + fn)),
                    2 * tp.sum() / max(1, 2 * tp.sum() + fp.sum() + fn.sum()),
                ]
            )
        deltas.append(np.subtract(scores[1], scores[0]))
    return dict(
        clusters=n_group,
        iterations=iterations,
        seed=20260908,
        delta_macro_f1_95=np.quantile(
            np.asarray(deltas)[:, 0], [0.025, 0.975]
        ).tolist(),
        delta_micro_f1_95=np.quantile(
            np.asarray(deltas)[:, 1], [0.025, 0.975]
        ).tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument(
        "--training_tsv",
        default="data/localize_bench/deeploc21/deeploc21_localization_train_validation.tsv",
    )
    args = parser.parse_args()
    root = Path(args.experiment_dir)
    records, predictions = {}, {}
    for recipe in ["cnn_legacy", "cnn_termini"]:
        for seed in [1, 2, 3]:
            directory = root / "{}_seed{}".format(recipe, seed)
            if (directory / "metrics.json").exists():
                name = "{} seed{}".format(recipe, seed)
                records[name] = json.loads((directory / "metrics.json").read_text())
                predictions[name] = np.load(directory / "oof.npz")
    for seed in [1, 2, 3]:
        directory = root / "integration_seed{}".format(seed)
        for recipe in ["integrated", "student_control", "distilled"]:
            if (directory / (recipe + "_metrics.json")).exists():
                name = "{} seed{}".format(recipe, seed)
                records[name] = json.loads(
                    (directory / (recipe + "_metrics.json")).read_text()
                )
                predictions[name] = np.load(directory / (recipe + "_oof.npz"))
    if not (root / "clusters.json").exists():
        from cdskit.perox_benchmark import mmseqs_cluster_assignments
        from cdskit.localize_evaluation import dataset_digest

        source_rows = _read_prepared_tsv(args.training_tsv)
        groups, report = mmseqs_cluster_assignments(source_rows, threads=2)
        atomic_write_json(
            str(root / "clusters.json"),
            dict(
                groups=groups, report=report, dataset_sha256=dataset_digest(source_rows)
            ),
        )
    clusters = json.loads((root / "clusters.json").read_text())
    if (
        clusters.get("dataset_sha256")
        != records["cnn_legacy seed1"]["config"]["dataset_sha256"]
    ):
        raise ValueError("Cluster input digest differs from model evaluation input.")
    from collections import defaultdict

    cluster_folds = defaultdict(set)
    for cluster, fold in zip(
        clusters["groups"], predictions["cnn_legacy seed1"]["folds"], strict=True
    ):
        cluster_folds[cluster].add(str(fold))
    crossing = {cluster for cluster, folds in cluster_folds.items() if len(folds) > 1}
    clean_mask = np.array([cluster not in crossing for cluster in clusters["groups"]])
    clean_results = {
        name: compute_multilabel_metrics(
            pred["target"][clean_mask], pred["prediction"][clean_mask], LABELS
        )
        for name, pred in predictions.items()
    }

    if clusters["report"]["status"] != "ok":
        raise ValueError("Cluster confidence intervals require successful clustering.")
    intervals = {}
    for name, pred in predictions.items():
        seed = name.rsplit("seed", 1)[1]
        baseline = predictions["cnn_legacy seed" + seed]
        np.testing.assert_array_equal(baseline["target"], pred["target"])
        np.testing.assert_array_equal(baseline["folds"], pred["folds"])
        if not name.startswith("cnn_legacy"):
            intervals[name] = paired_cluster_intervals(
                pred["target"],
                baseline["prediction"],
                pred["prediction"],
                clusters["groups"],
            )
    external_path = root / "final/hpa_metrics.json"
    external = json.loads(external_path.read_text()) if external_path.exists() else {}
    homology_path = root / "external_homology.json"
    external_homology = (
        json.loads(homology_path.read_text()) if homology_path.exists() else {}
    )
    runtime = {
        p.stem: json.loads(p.read_text()) for p in (root / "runtime").glob("*.json")
    }
    summary = dict(
        cross_fold_homology=dict(clusters=len(crossing), rows=int((~clean_mask).sum())),
        clean_subset=clean_results,
        development={
            name: {
                k: val[k]
                for k in ["macro_f1", "micro_f1", "macro_average_precision_observed"]
            }
            for name, val in records.items()
        },
        intervals=intervals,
        external=external,
        external_homology=external_homology,
        runtime=runtime,
    )
    means = {}
    for recipe in ["cnn_legacy", "integrated", "student_control", "distilled"]:
        values = [
            [value["macro_f1"], value["micro_f1"]]
            for name, value in records.items()
            if name.startswith(recipe + " seed")
        ]
        if values:
            means[recipe] = dict(
                seeds=len(values),
                mean=np.mean(values, axis=0).tolist(),
                sd=np.std(values, axis=0, ddof=1).tolist() if len(values) > 1 else None,
            )
    summary["development_seed_summary"] = means
    if "integrated" in runtime and "distilled" in runtime:
        assert (
            runtime["integrated"]["dataset_sha256"]
            == runtime["distilled"]["dataset_sha256"]
        )
        teacher = np.load(root / "runtime/integrated.json.npz")
        student = np.load(root / "runtime/distilled.json.npz")
        summary["runtime_output_agreement"] = dict(
            exact_label_set=float(
                np.mean(np.all(teacher["prediction"] == student["prediction"], axis=1))
            ),
            per_label=float(np.mean(teacher["prediction"] == student["prediction"])),
            probability_rmse=float(
                np.sqrt(np.mean((teacher["probability"] - student["probability"]) ** 2))
            ),
        )
    atomic_write_json(str(root / "summary.json"), summary)
    lines = [
        "# Full-data localization experiment",
        "",
        "28,303 Swiss-Prot rows, official five outer partitions. Inner validation only for stopping, thresholds and blend weights.",
        "",
        "| Model | Macro F1 | Micro F1 | Macro AP |",
        "| --- | ---: | ---: | ---: |",
    ]
    eligible = {
        name: value
        for name, value in means.items()
        if value["mean"][1] >= means["cnn_legacy"]["mean"][1] - 0.01
    }
    recommended = max(eligible, key=lambda name: eligible[name]["mean"][0])
    retention = (
        all(
            means["integrated"]["mean"][i] - means["distilled"]["mean"][i] <= 0.02
            for i in (0, 1)
        )
        if "integrated" in means and "distilled" in means
        else None
    )
    overview = [
        "# Result and recommendation",
        "",
        "Development-selected accuracy candidate: {}. Compact/distilled models produce different outputs; assess them against the same-sized hard-label control as well as the teacher.".format(
            recommended
        ),
        "",
        "| Model | Seeds | Mean macro F1 | Mean micro F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in means.items():
        overview.append(
            "| {} | {} | {:.4f} | {:.4f} |".format(
                name, values["seeds"], *values["mean"]
            )
        )
    if external and "integrated" in external:
        base, integrated = external["cnn_legacy"], external["integrated"]
        perox = integrated["by_label"]["peroxisome"]
        overview += [
            "",
            "HPA integrated-minus-baseline differences: macro F1 {:+.4f}, micro F1 {:+.4f}. Integrated peroxisome F1 is {:.4f} on {} positives. Inspect the per-label table; aggregate gains do not establish improvement in every compartment.".format(
                integrated["macro_f1"] - base["macro_f1"],
                integrated["micro_f1"] - base["micro_f1"],
                perox["f1"],
                perox["support"],
            ),
        ]
    if retention is not None:
        overview += [
            "",
            "Teacher retention (both mean macro and micro within 0.02 of the teacher): {}. No HPA thresholds were tuned.".format(
                "PASS" if retention else "FAIL"
            ),
        ]
    overview += [""]
    lines = overview + lines
    for name, metrics in records.items():
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} |".format(
                name,
                metrics["macro_f1"],
                metrics["micro_f1"],
                metrics["macro_average_precision_observed"],
            )
        )
    lines += [
        "",
        "## Additional homology stress subset",
        "",
        "{} MMseqs clusters / {} rows cross official folds. Excluding those rows leaves {} evaluated proteins; models retain their original training folds.".format(
            len(crossing), int((~clean_mask).sum()), int(clean_mask.sum())
        ),
        "",
        "| Model | Macro F1 | Micro F1 |",
        "| --- | ---: | ---: |",
    ]
    for name, metrics in clean_results.items():
        lines.append(
            "| {} | {:.4f} | {:.4f} |".format(
                name, metrics["macro_f1"], metrics["micro_f1"]
            )
        )
    lines += [
        "",
        "## Paired cluster-bootstrap differences versus same-seed baseline",
        "",
        "95% percentile intervals; 1,000 resamples of MMseqs 30% identity / 80% coverage clusters. These describe test-sample uncertainty, not hyperparameter-selection uncertainty.",
        "",
    ]
    for name, interval in intervals.items():
        lines.append(
            "- {}: macro {}, micro {}".format(
                name, interval["delta_macro_f1_95"], interval["delta_micro_f1_95"]
            )
        )
    if external:
        lines += [
            "",
            "## Fixed HPA stress test",
            "",
            "1,717 human proteins. Historically inspected dataset; not a newly collected final holdout. No HPA threshold tuning.",
            "",
            "| Model | Macro F1 | Micro F1 | Macro AP |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, metrics in external.items():
            lines.append(
                "| {} | {:.4f} | {:.4f} | {:.4f} |".format(
                    name,
                    metrics["macro_f1"],
                    metrics["micro_f1"],
                    metrics["macro_average_precision_observed"],
                )
            )
    if external:
        lines += [
            "",
            "## Per-label HPA F1",
            "",
            "| Label | Positives | Baseline | Integrated | Control | Distilled |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for label in LABELS:
            vals = [
                external[name]["by_label"][label]
                for name in ["cnn_legacy", "integrated", "student_control", "distilled"]
            ]
            lines.append(
                "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                    label, vals[0]["support"], *[v["f1"] for v in vals]
                )
            )
    if external_homology:
        lines += [
            "",
            "## HPA without detected training-source homology",
            "",
            "| Model | Rows | Macro F1 | Micro F1 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, metrics in external_homology["subsets"].items():
            if name.endswith("_nohit"):
                lines.append(
                    "| {} | {} | {:.4f} | {:.4f} |".format(
                        name,
                        metrics["n_rows"],
                        metrics["macro_f1"],
                        metrics["micro_f1"],
                    )
                )
    if runtime:
        lines += [
            "",
            "## CPU measurements",
            "",
            "256 length-stratified HPA proteins, one CPU thread, warmup and three repeats in separate processes. Different models have different predictions; this is an accuracy/runtime tradeoff, not equivalent-output optimization.",
            "",
            "| Model | Median seconds | Peak process MiB | File MiB |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, measurements in runtime.items():
            lines.append(
                "| {} | {:.3f} | {:.1f} | {:.2f} |".format(
                    name,
                    measurements["median_seconds"],
                    measurements["peak_process_rss_bytes"] / 2**20,
                    measurements["model_bytes"] / 2**20,
                )
            )
    lines += [
        "",
        "## Per-label F1, development seed1",
        "",
        "| Label | Support | Baseline | Integrated | Control | Distilled |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if all(
        name + " seed1" in records
        for name in ["cnn_legacy", "integrated", "student_control", "distilled"]
    ):
        for label in LABELS:
            vals = [
                records[name + " seed1"]["by_label"][label]
                for name in ["cnn_legacy", "integrated", "student_control", "distilled"]
            ]
            lines.append(
                "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                    label, vals[0]["support"], *[v["f1"] for v in vals]
                )
            )
    (root / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

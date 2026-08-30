#!/usr/bin/env python3
"""Compare with the last retained successful benchmark and write a CI summary."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    repository = os.environ["GITHUB_REPOSITORY"]
    endpoint = f"repos/{repository}/actions/workflows/benchmark.yml/runs?status=success&per_page=10"
    runs = json.loads(subprocess.check_output(["gh", "api", endpoint], text=True))[
        "workflow_runs"
    ]
    with tempfile.TemporaryDirectory(prefix="cdskit-baseline-") as temporary:
        baseline = None
        # Old runs may have expired artifacts. Find the newest retained one.
        for run in runs:
            if run["head_branch"] != os.environ["GITHUB_REF_NAME"]:
                continue
            artifacts = json.loads(
                subprocess.check_output(
                    [
                        "gh",
                        "api",
                        f"repos/{repository}/actions/runs/{run['id']}/artifacts",
                    ],
                    text=True,
                )
            )["artifacts"]
            retained = [
                artifact
                for artifact in artifacts
                if not artifact["expired"]
                and artifact["name"].startswith("cpu-hotpaths-")
            ]
            if not retained:
                continue
            subprocess.run(
                [
                    "gh",
                    "run",
                    "download",
                    str(run["id"]),
                    "--repo",
                    repository,
                    "--name",
                    retained[0]["name"],
                    "--dir",
                    temporary,
                ],
                check=True,
            )
            baseline = Path(temporary) / "benchmark.json"
            if not baseline.is_file():
                raise RuntimeError("Benchmark artifact is missing benchmark.json.")
            break
        command = [
            sys.executable,
            "-m",
            "cdskit.benchmark_hotpaths",
            "--full",
            "--repeats",
            "5",
        ]
        if baseline is not None:
            command.extend(["--baseline", str(baseline)])
        with open("benchmark.json", "w", encoding="utf-8") as output:
            subprocess.run(command, stdout=output, check=True)
    report = json.loads(Path("benchmark.json").read_text())
    comparison = report.get("comparison", {"status": "no_baseline", "workloads": {}})
    if comparison["status"] == "review":
        print(
            "::warning::Benchmark output changed or median runtime increased by more than 30%; inspect the benchmark summary."
        )
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
        summary.write("Benchmark comparison: **" + comparison["status"] + "**\n\n")
        if comparison.get("reason"):
            summary.write(comparison["reason"] + "\n\n")
        summary.write(
            "| Workload | Median seconds | Current / previous | Status |\n|---|---:|---:|---|\n"
        )
        for name, result in report["benchmarks"].items():
            delta = comparison["workloads"].get(name, {})
            ratio = delta.get("time_ratio")
            ratio_text = "—" if ratio is None else f"{ratio:.2f}"
            summary.write(
                f"| {name} | {result['median_seconds']:.6f} | {ratio_text} | {delta.get('status', 'not compared')} |\n"
            )
        summary.write(
            f"\nProcess peak RSS: {report['process_peak_rss_bytes']} bytes. Full environment and output fingerprints are in the artifact.\n"
        )


if __name__ == "__main__":
    main()

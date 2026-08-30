"""Metadata, output fingerprints and comparisons for repeatable benchmarks."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import platform
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import sys
from typing import Any

import numpy as np

from cdskit import __version__


def output_fingerprint(value: Any) -> str:
    """Hash complete numerical/structured outputs without serializing huge arrays."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, np.ndarray):
            digest.update(b"array")
            digest.update(str((item.dtype.descr, item.shape)).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(f"dict:{len(item)}:".encode())
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"sequence:{len(item)}:".encode())
            for element in item:
                visit(element)
        else:
            if isinstance(item, np.generic):
                item = item.item()
            encoded = json.dumps(item, sort_keys=True, allow_nan=False).encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)

    visit(value)
    return digest.hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def environment_metadata() -> dict[str, Any]:
    dependencies = {
        package: _package_version(package)
        for package in ("numpy", "biopython", "matplotlib")
    }
    processor = platform.processor()
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as source:
                for line in source:
                    if line.startswith("model name"):
                        processor = line.partition(":")[2].strip()
                        break
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            processor = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    source_root = Path(__file__).resolve().parents[1]
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=source_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if Path(git_root).resolve() != source_root:
            raise OSError("The installed package is not the repository checkout.")
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=source_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cdskit_version": __version__,
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": processor or platform.machine(),
        "cpu_count": os.cpu_count(),
        "dependencies": dependencies,
        "runner_image": os.environ.get("ImageVersion"),
    }


def peak_rss_bytes() -> int | None:
    """Process high-water RSS (not a per-workload allocation measurement)."""
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def compare_reports(
    current: dict[str, Any], baseline: dict[str, Any], max_slowdown: float = 0.3
) -> dict[str, Any]:
    if current.get("schema_version") != 2 or baseline.get("schema_version") != 2:
        return {
            "status": "incomparable",
            "reason": "Benchmark schema differs; record a new baseline.",
            "workloads": {},
        }
    keys = (
        "python",
        "implementation",
        "system",
        "machine",
        "processor",
        "cpu_count",
        "dependencies",
    )
    differences = [
        key
        for key in keys
        if current["environment"].get(key) != baseline["environment"].get(key)
    ]
    if differences:
        return {
            "status": "incomparable",
            "reason": "Environment differs: " + ", ".join(differences),
            "workloads": {},
        }
    comparisons: dict[str, dict[str, Any]] = {}
    for name, result in current["benchmarks"].items():
        previous = baseline["benchmarks"].get(name)
        if previous is None or previous.get("workload") != result.get("workload"):
            comparisons[name] = {
                "status": "incomparable",
                "reason": "Workload differs or has no baseline.",
            }
            continue
        if previous.get("output_sha256") != result.get("output_sha256"):
            comparisons[name] = {
                "status": "changed_output",
                "reason": "Output changed; review correctness before comparing speed.",
            }
            continue
        if not result.get("output_sha256"):
            comparisons[name] = {
                "status": "incomparable",
                "reason": "No output fingerprint.",
            }
            continue
        old_seconds = float(previous["median_seconds"])
        ratio = (
            float(result["median_seconds"]) / old_seconds if old_seconds > 0 else None
        )
        comparisons[name] = {
            "status": "slowdown"
            if ratio is not None and ratio > 1.0 + max_slowdown
            else "ok",
            "time_ratio": ratio,
        }
    review = any(
        item["status"] in {"slowdown", "changed_output"}
        for item in comparisons.values()
    )
    comparable = any(item["status"] == "ok" for item in comparisons.values())
    return {
        "status": "review" if review else "ok" if comparable else "incomparable",
        "workloads": comparisons,
        "max_slowdown": max_slowdown,
    }

#!/usr/bin/env python3
"""Choose platform coverage from changed paths; scheduled/manual runs cover all."""

from __future__ import annotations

import json
import os
import re
import subprocess


def platform_coverage_required(paths: list[str]) -> bool:
    for path in paths:
        if path.startswith("tests/ml/"):
            continue
        if path in {"pyproject.toml", "uv.lock", "MANIFEST.in"} or path.startswith(
            (".github/", "scripts/", "tests/")
        ):
            return True
        if path.startswith("cdskit/") and not path.startswith(
            (
                "cdskit/localize",
                "cdskit/targetp_",
                "cdskit/perox_",
                "cdskit/deeploc_",
                "cdskit/esm_embeddings",
            )
        ):
            return True
    return False


def core_matrix(event: str, paths: list[str]) -> list[dict[str, str]]:
    exhaustive = event in {"schedule", "workflow_dispatch"}
    platforms = exhaustive or platform_coverage_required(paths)
    versions = ["3.10", "3.14"]
    if exhaustive or (platforms and event == "push"):
        versions = ["3.10", "3.11", "3.12", "3.13", "3.14"]
    matrix = [
        {"os": "ubuntu-latest", "python-version": version} for version in versions
    ]
    if platforms:
        matrix.extend(
            {"os": runner, "python-version": version}
            for runner in ("macos-latest", "windows-latest")
            for version in (["3.14"] if event == "pull_request" else ["3.12", "3.14"])
        )
    return matrix


def main() -> None:
    event = os.environ["GITHUB_EVENT_NAME"]
    paths: list[str] = []
    if event not in {"schedule", "workflow_dispatch"}:
        base = os.environ.get("BASE_SHA", "")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", base) or base == "0" * 40:
            paths = ["pyproject.toml"]
        else:
            available = subprocess.run(
                ["git", "cat-file", "-e", base], capture_output=True
            )
            if available.returncode:
                subprocess.run(
                    ["git", "fetch", "--no-tags", "--depth=1", "origin", base],
                    check=True,
                )
            changed = subprocess.check_output(
                ["git", "diff", "--name-only", "-z", base, "HEAD"]
            )
            paths = changed.decode().rstrip("\0").split("\0")
    matrix = core_matrix(event, paths)
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write("core_matrix=" + json.dumps(matrix) + "\n")
    print(
        json.dumps(
            {"platform_coverage": platform_coverage_required(paths), "matrix": matrix},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

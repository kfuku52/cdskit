#!/usr/bin/env python3
"""Enforce combined branch/statement coverage floors for critical modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


MODULE_FLOORS = {
    "cdskit/atomicio.py": 90.0,
    "cdskit/command_runtime.py": 85.0,
    "cdskit/localize_models.py": 85.0,
    "cdskit/targetp_training.py": 90.0,
}


def validate_coverage(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    files = payload.get("files", {})
    for filename, floor in MODULE_FLOORS.items():
        summary = files.get(filename, {}).get("summary")
        if not isinstance(summary, dict):
            failures.append(f"{filename}: missing from coverage report")
            continue
        observed = float(summary.get("percent_covered", 0.0))
        if observed < floor:
            failures.append(
                f"{filename}: {observed:.1f}% is below the {floor:.1f}% floor"
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    report_path = Path(args[0] if args else "coverage.json")
    with report_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    failures = validate_coverage(payload)
    if failures:
        print("Critical coverage checks failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Critical module coverage floors passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

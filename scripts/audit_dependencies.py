#!/usr/bin/env python3
"""Audit installed dependencies, including the upstream release of CPU PyTorch."""

from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import Distribution, distributions
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def audit_requirements(installed: Iterable[Distribution]) -> list[str]:
    requirements = []
    for distribution in installed:
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        if direct_url.get("dir_info", {}).get("editable", False):
            print(f"Skipping editable distribution: {name}", flush=True)
            continue
        version = distribution.version
        if name == "torch" and version.endswith("+cpu"):
            # Official CPU wheels share the upstream release's advisories, but
            # PyPI has no JSON record for their CPU-index-only local version.
            upstream = version.removesuffix("+cpu")
            print(
                f"Auditing CPU torch {version} against upstream {upstream} advisories.",
                flush=True,
            )
            version = upstream
        requirements.append(f"{name}=={version}")
    return sorted(set(requirements))


def audit(requirements: list[str]) -> int:
    if not requirements:
        raise ValueError("No installed dependencies to audit.")
    with tempfile.TemporaryDirectory(prefix="cdskit-audit-") as temporary:
        root = Path(temporary)
        source = root / "requirements.txt"
        output = root / "audit.json"
        source.write_text("\n".join(requirements) + "\n", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--requirement",
                str(source),
                "--disable-pip",
                "--no-deps",
                "--format",
                "json",
                "--output",
                str(output),
            ],
            check=False,
        )
        if not output.is_file():
            return result.returncode or 1
        report = json.loads(output.read_text(encoding="utf-8"))
        dependencies = report["dependencies"]
        unresolved = [item for item in dependencies if item.get("skip_reason")]
        vulnerable = [item for item in dependencies if item.get("vulns")]
        if result.returncode or unresolved or vulnerable:
            print(json.dumps(report, indent=2), file=sys.stderr)
            return result.returncode or 1
        if len(dependencies) != len(requirements):
            raise ValueError("The audit did not return every requested dependency.")
    print(f"Audited {len(requirements)} installed dependencies; none were skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(audit(audit_requirements(distributions())))

#!/usr/bin/env python3
"""Test an actual built wheel from outside the source tree in a fresh environment."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main() -> None:
    wheels = list(Path(sys.argv[1]).resolve().glob("cdskit-*.whl"))
    if len(wheels) != 1:
        raise ValueError(
            f"Expected exactly one fresh cdskit wheel, found {len(wheels)}."
        )
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for the clean wheel smoke check.")
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/example_plot.fasta"
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    with tempfile.TemporaryDirectory(prefix="cdskit-wheel-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        bindir = environment / ("Scripts" if os.name == "nt" else "bin")
        python = str(bindir / ("python.exe" if os.name == "nt" else "python"))
        cli = str(bindir / ("cdskit.exe" if os.name == "nt" else "cdskit"))

        def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.run(
                command, cwd=root, env=env, check=True, text=True, **kwargs
            )

        run([uv, "venv", str(environment), "--python", sys.executable])
        run([uv, "pip", "install", "--python", python, str(wheels[0])])
        run(
            [
                python,
                "-I",
                "-c",
                "import cdskit, pathlib, sys; assert pathlib.Path(cdskit.__file__).is_relative_to(pathlib.Path(sys.prefix))",
            ]
        )
        run([cli, "--version"])
        padded = run([cli, "pad"], input=">wheel-smoke\nATGA\n", capture_output=True)
        if padded.stdout.splitlines() != [">wheel-smoke", "ATGANN"]:
            raise RuntimeError("Installed CLI returned unexpected padding output.")
        plot = root / "smoke.svg"
        run(
            [
                cli,
                "plot",
                "--seq_file",
                str(fixture),
                "--out_file",
                str(plot),
                "--format",
                "svg",
            ]
        )
        if "<svg" not in plot.read_text(encoding="utf-8"):
            raise RuntimeError("Installed CLI did not create an SVG plot.")
        run([uv, "pip", "check", "--python", python])
    print("Fresh installed-wheel import, CLI, padding and SVG checks passed.")


if __name__ == "__main__":
    main()

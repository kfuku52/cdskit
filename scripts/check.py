#!/usr/bin/env python3
"""Run reproducible local/CI checks without replacing another task's environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "quick": "core",
    "core": "core",
    "ml": "full",
    "coverage": "full",
    "quality": "full",
    "build": "build",
    "all": "full",
}


def run(command: list[str], env: dict[str, str]) -> None:
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=PROFILES)
    parser.add_argument(
        "--python", default="3.12", choices=["3.10", "3.11", "3.12", "3.13", "3.14"]
    )
    parser.add_argument(
        "--ml-backend",
        default="cpu",
        choices=["cpu", "default"],
        help="Default uses the GPU-capable ml extra; CPU uses ml-cpu.",
    )
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    if extra and args.check not in {"quick", "core", "ml", "coverage"}:
        parser.error("Extra pytest arguments are only supported for test checks.")
    uv = shutil.which("uv")
    if uv is None:
        parser.error(
            "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
        )
    profile = PROFILES[args.check]
    suffix = f"-{args.ml_backend}" if profile == "full" else ""
    environment = ROOT / ".venvs" / f"{profile}-{args.python}{suffix}"
    env = dict(os.environ)
    env["UV_PROJECT_ENVIRONMENT"] = str(environment)
    sync = [uv, "sync", "--locked", "--python", args.python]
    if profile == "core":
        sync += ["--no-dev", "--group", "test"]
    elif profile == "full":
        sync += [
            "--group",
            "dev",
            "--extra",
            "ml-cpu" if args.ml_backend == "cpu" else "ml",
        ]
    else:
        sync += ["--no-dev", "--group", "build"]
    run(sync, env)
    bindir = environment / ("Scripts" if os.name == "nt" else "bin")
    python = str(bindir / ("python.exe" if os.name == "nt" else "python"))
    # Direct executable wrappers use /usr/bin/env python. Their children must
    # use this profile too, regardless of the caller's active environment.
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(environment)
    if profile == "full":
        # Never let importorskip silently turn the full check into core-only.
        run(
            [
                python,
                "-c",
                "import numpy, Bio, matplotlib, torch, sklearn, transformers",
            ],
            env,
        )
    if args.check in {"quality", "all"}:
        for command in (
            ["ruff", "check", "cdskit", "scripts", "tests"],
            ["ruff", "format", "--check", "cdskit", "scripts", "tests"],
            ["mypy", "cdskit"],
            ["ruff", "check", "cdskit", "--select", "C901"],
            ["bandit", "-q", "-r", "cdskit", "-lll"],
        ):
            run([python, "-m", *command], env)
    if args.check in {"quick", "core", "ml", "coverage", "all"}:
        test = [python, "-m", "pytest", "-q"]
        if args.check in {"quick", "core"}:
            run([python, "-m", "compileall", "-q", "cdskit", "scripts"], env)
            test += [
                "tests/unit",
                "tests/integration",
                "-m",
                "not ml and not subprocess" if args.check == "quick" else "not ml",
            ]
        elif args.check == "ml":
            test += ["-m", "ml"]
        else:
            test += [
                "-n",
                "2",
                "--dist=worksteal",
                "--cov=cdskit",
                "--cov-report=term-missing",
                "--cov-report=json:coverage.json",
            ]
        run([*test, "--durations=20", "--durations-min=0.05", *extra], env)
        if args.check in {"coverage", "all"}:
            run([python, "scripts/check_coverage.py", "coverage.json"], env)
    if args.check == "all":
        run([python, "scripts/audit_dependencies.py"], env)
    if args.check in {"build", "all"}:
        # A fresh directory prevents an old wheel from passing the smoke check.
        with tempfile.TemporaryDirectory(prefix="cdskit-dist-") as dist:
            run([python, "-m", "build", "--outdir", dist], env)
            run([python, "scripts/check_wheel.py", dist], env)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None

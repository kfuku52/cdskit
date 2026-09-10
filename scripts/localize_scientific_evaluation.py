#!/usr/bin/env python3
"""Freeze a new external localization evaluation, then score unchanged models."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdskit.localize_frozen_evaluation import evaluate_frozen, freeze_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--protocol", required=True)
    freeze.add_argument(
        "--model",
        action="append",
        required=True,
        help="NAME=PATH; repeat for each fixed model",
    )
    freeze.add_argument("--output", required=True)
    score = sub.add_parser("evaluate")
    score.add_argument("manifest")
    args = parser.parse_args()
    if args.command == "evaluate":
        result = evaluate_frozen(args.manifest)
    else:
        models = {}
        for value in args.model:
            name, separator, path = value.partition("=")
            if not separator or name in models:
                parser.error("Models require distinct NAME=PATH entries.")
            models[name] = path
        result = freeze_evaluation(args.config, models, args.protocol, args.output)
    print(result)


if __name__ == "__main__":
    main()

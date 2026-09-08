#!/usr/bin/env python3
"""Compare localization recipes on identical folds, without opening external tests."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdskit.deeploc_benchmark import run_deeploc21_benchmark
from cdskit.util import atomic_write_json


RECIPES = {
    "cnn_legacy": ("cnn", {"sequence_layout": "legacy", "mask_padding": "no"}),
    "cnn_masked": ("cnn", {"sequence_layout": "legacy", "mask_padding": "yes"}),
    "cnn_termini": (
        "cnn",
        {"sequence_layout": "separate_termini", "mask_padding": "yes"},
    ),
    "cnn_windows": ("cnn", {"sequence_layout": "windows", "mask_padding": "yes"}),
    "plm_mean": ("plm", {"plm_pooling": "mean"}),
    "plm_light": ("plm", {"plm_pooling": "light_attention"}),
    "plm_label": ("plm", {"plm_pooling": "label_attention"}),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared_dir", default="data/localize_bench/deeploc21")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--recipes", default="cnn_legacy,cnn_masked,cnn_termini,cnn_windows"
    )
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--epochs", default=6, type=int)
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"]
    )
    parser.add_argument("--plm_model_name", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--plm_revision", default="")
    parser.add_argument("--plm_cache_dir", default="data/localize_bench/embeddings")
    args = parser.parse_args(argv)
    recipes, seeds = args.recipes.split(","), [int(x) for x in args.seeds.split(",")]
    if any(recipe not in RECIPES for recipe in recipes):
        parser.error("Unknown recipe. Choose from: " + ",".join(RECIPES))
    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for recipe in recipes:
        arch, changes = RECIPES[recipe]
        for seed in seeds:
            params = dict(
                epochs=args.epochs,
                seed=seed,
                device=args.device,
                plm_model_name=args.plm_model_name,
                plm_revision=args.plm_revision,
                plm_cache_dir=args.plm_cache_dir,
                **changes,
            )
            stem = os.path.join(args.out_dir, "{}_seed{}".format(recipe, seed))
            if os.path.exists(stem + ".json"):
                raise FileExistsError(
                    "Use a fresh output directory; preserving existing experiment: "
                    + stem
                )
            print("Running {} seed {}".format(recipe, seed), flush=True)
            result = run_deeploc21_benchmark(
                args.prepared_dir,
                model_arch=arch,
                dl_params=params,
                comparison_json=stem + ".json",
                comparison_md=stem + ".md",
                evaluate_external=False,
            )
            metrics = result["cross_validation"]
            results.append(
                dict(
                    recipe=recipe,
                    seed=seed,
                    dataset_sha256=metrics["dataset_sha256"],
                    macro_f1=metrics["macro_f1"],
                    micro_f1=metrics["micro_f1"],
                    macro_ap=metrics["macro_average_precision_observed"],
                )
            )
            atomic_write_json(os.path.join(args.out_dir, "summary.json"), results)
    return results


if __name__ == "__main__":
    main()

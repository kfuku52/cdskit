"""Interleave two implementations with identical baseline dependencies.

Complements benchmark_followup_changes.py's fresh-process RSS/whole-snapshot
comparison. Only the named implementation module changes between paired calls.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

from benchmark_followup_changes import CASES, build_case


MODULES = {
    "trimcodon_runs": "trimcodon",
    "trimcodon_fragmented": "trimcodon",
    "split_many": "split",
    "split_long": "split",
    "split_short": "split",
    "plot_msa": "plot",
    "gap_coordinates": "gapjust",
    "validate_ambiguity": "validate",
    "validate_gap_only": "validate",
    "backalign": "backalign",
    "backalign_lower_wildcard": "backalign",
    "stats_counts": "stats",
    "stats_long": "stats",
    "stats_empty": "stats",
    "reliability_bins": "localize_evaluation",
    "reliability_dense": "localize_evaluation",
    "group_folds": "localize_splits",
    "group_folds_many_labels": "localize_splits",
    "group_folds_small": "localize_splits",
    "alphabet_dna": "util",
    "alphabet_protein": "util",
}


def load_module(label, source, module_name):
    path = source / "cdskit" / f"{module_name}.py"
    name = f"cdskit._paired_{label}_{module_name}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, hashlib.sha256(path.read_bytes()).hexdigest()


def measure(run, loops):
    start, cpu_start = time.perf_counter(), time.process_time()
    for _ in range(loops):
        result = run()
    cpu = (time.process_time() - cpu_start) / loops
    wall = (time.perf_counter() - start) / loops
    return result, {"cpu_seconds": cpu, "wall_seconds": wall}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--minimum-batch-cpu", type=float, default=0.02)
    args = parser.parse_args()
    if args.rounds < 1 or args.minimum_batch_cpu <= 0:
        parser.error("rounds and minimum batch CPU must be positive")
    sources = {"before": args.before.resolve(), "after": args.after.resolve()}
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = "1"
    os.environ["CDSKIT_OFFLINE"] = "1"
    sys.path.insert(0, str(sources["before"]))
    import cdskit
    from cdskit.benchmarking import environment_metadata, output_fingerprint

    if Path(cdskit.__file__).resolve().parent.parent != sources["before"]:
        raise RuntimeError("Wrong baseline dependencies imported.")
    report = {
        "schema_version": 1,
        "sources": {name: str(path) for name, path in sources.items()},
        "command": sys.argv,
        "environment": environment_metadata(),
        "method": "alternating ABBA/BAAB; baseline dependencies; one warmup per implementation",
        "rounds": args.rounds,
        "results": {},
    }
    for case in args.cases:
        module_name = MODULES[case]
        canonical_name = f"cdskit.{module_name}"
        runs, adapters, hashes, samples, expected = {}, {}, {}, {}, set()
        with tempfile.TemporaryDirectory(prefix="cdskit-paired-") as directory:
            for label, source in sources.items():
                module, hashes[label] = load_module(label, source, module_name)
                previous = sys.modules.get(canonical_name)
                sys.modules[canonical_name] = module
                root = Path(directory) / label
                root.mkdir()
                try:
                    runs[label], adapters[label], workload = build_case(
                        case, root, None
                    )
                finally:
                    if previous is None:
                        del sys.modules[canonical_name]
                    else:
                        sys.modules[canonical_name] = previous
                result, warm = measure(runs[label], 1)
                expected.add(output_fingerprint(adapters[label](result)))
                samples[label] = []
                if label == "before":
                    loops = max(
                        1,
                        min(
                            1000,
                            int(
                                args.minimum_batch_cpu / max(warm["cpu_seconds"], 1e-9)
                            ),
                        ),
                    )
            if len(expected) != 1:
                raise RuntimeError(f"{case}: baseline and candidate outputs differ.")
            for round_number in range(args.rounds):
                order = (
                    ("before", "after", "after", "before")
                    if round_number % 2 == 0
                    else ("after", "before", "before", "after")
                )
                for label in order:
                    result, sample = measure(runs[label], loops)
                    if output_fingerprint(adapters[label](result)) not in expected:
                        raise RuntimeError(
                            f"{case}: output changed during measurement."
                        )
                    samples[label].append(sample)
            summary = {
                label: {
                    f"median_{metric}": statistics.median(
                        value[metric] for value in values
                    )
                    for metric in ("cpu_seconds", "wall_seconds")
                }
                for label, values in samples.items()
            }
            report["results"][case] = {
                "workload": workload,
                "loops_per_sample": loops,
                "samples": samples,
                "summary": summary,
                "module_sha256": hashes,
                "output_sha256": expected.pop(),
            }
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            ratio = (
                summary["before"]["median_cpu_seconds"]
                / summary["after"]["median_cpu_seconds"]
            )
            print(f"{case}: CPU ratio {ratio:.3f}x; outputs identical", flush=True)


if __name__ == "__main__":
    main()

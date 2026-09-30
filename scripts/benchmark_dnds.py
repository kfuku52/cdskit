#!/usr/bin/env python3
"""Compare pairwise YN00 implementations on identical aligned CDS inputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cdskit.atomicio import (
    atomic_write_json,
    validate_distinct_paths,
    validate_output_paths,
)
from cdskit.benchmarking import environment_metadata, output_fingerprint, peak_rss_bytes
from cdskit.codonutil import definite_stop_patterns, get_forward_table, get_stop_codons
from cdskit.tsvio import validate_fieldnames

PAML_CODES = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 9: 6, 10: 7, 12: 8, 13: 9, 15: 10}


def _joint_sense_sequences(seqs, forward, stop_patterns):
    codons = [
        [seq[start : start + 3] for start in range(0, len(seq), 3)] for seq in seqs
    ]
    for side in codons:
        last = next(
            (
                i
                for i in range(len(side) - 1, -1, -1)
                if not any(base in "-?." for base in side[i])
            ),
            None,
        )
        if any(codon in stop_patterns and i != last for i, codon in enumerate(side)):
            raise ValueError("Benchmark input has a definite internal stop")
    retained = [
        (a, b) for a, b in zip(*codons, strict=True) if a in forward and b in forward
    ]
    if not retained:
        raise ValueError("Benchmark pair has no concrete aligned sense codons")
    return tuple("".join(pair[side] for pair in retained) for side in (0, 1))


def read_pairs(path, codon_table=1):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    forward = get_forward_table(codon_table)
    stop_patterns = frozenset(
        definite_stop_patterns(get_stop_codons(codon_table).difference(forward))
    )
    required = ("pair_id", "sequence_1", "sequence_2")
    alphabet = frozenset("ACGTRYSWKMBDHVNX-?.")
    result = []
    seen = set()
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", strict=True)
        header = validate_fieldnames(next(reader, None), path, required)
        positions = [header.index(name) for name in required]
        for line, values in enumerate(reader, start=2):
            if not values:
                continue
            if len(values) != len(header):
                raise ValueError(f"TSV row width mismatch at line {line}: {path}")
            identifier, first, second = (values[i] for i in positions)
            if not identifier.strip() or identifier in seen:
                raise ValueError("Benchmark pair_id values must be nonempty and unique")
            seen.add(identifier)
            if not first.isascii() or not second.isascii():
                raise ValueError("Invalid DNA alphabet in dnds benchmark input")
            seqs = [seq.upper().replace("U", "T") for seq in (first, second)]
            if len(seqs[0]) != len(seqs[1]) or len(seqs[0]) % 3:
                raise ValueError(
                    "Benchmark inputs must be equally aligned, in-frame CDS"
                )
            if any(set(seq) - alphabet for seq in seqs):
                raise ValueError("Invalid DNA alphabet in dnds benchmark input")
            result.append(
                (identifier, *_joint_sense_sequences(seqs, forward, stop_patterns))
            )
    if not result:
        raise ValueError("Benchmark input has no pairs")
    return result


def parse_paml(text):
    values = []
    for section in text.split("(B)")[1:]:
        for line in section.split("(C)", 1)[0].splitlines():
            fields = line.split()
            if (
                len(fields) == 13
                and fields[:2] == ["2", "1"]
                and fields[8] == fields[11] == "+-"
            ):
                row = {
                    "dN": float(fields[7]),
                    "dS": float(fields[10]),
                    "kappa": float(fields[5]),
                }
                # PAML can print NaN for unidentifiable sites (e.g. ATG/ATG).
                # Keep these missing, never turn them into zero or invalid JSON.
                values.append(
                    {
                        key: value if math.isfinite(value) else None
                        for key, value in row.items()
                    }
                )
    return values


def paml_chunk(rows, code):
    with tempfile.TemporaryDirectory(prefix="cdskit-yn00-bench-") as temp:
        work = Path(temp)
        (work / "pairs.phy").write_text(
            "".join(f"2 {len(a)}\ntarget  {a}\nquery  {b}\n" for _, a, b in rows),
            encoding="ascii",
        )
        (work / "yn00.ctl").write_text(
            "seqfile = pairs.phy\noutfile = yn.out\nverbose = 0\nnoisy = 0\n"
            f"icode = {PAML_CODES[code]}\nweighting = 0\ncommonkappa = 0\ncommonf3x4 = 0\nndata = {len(rows)}\n",
            encoding="ascii",
        )
        subprocess.run(
            ["yn00", "yn00.ctl"], cwd=work, capture_output=True, text=True, check=True
        )
        values = parse_paml((work / "yn.out").read_text(encoding="utf-8"))
        if len(values) != len(rows):
            raise ValueError(
                f"PAML produced {len(values)} results for {len(rows)} pairs"
            )
        return values


def run_paml(rows, code, workers, individual=False):
    size = 1 if individual else math.ceil(len(rows) / workers)
    chunks = [rows[i : i + size] for i in range(0, len(rows), size)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [
            value
            for chunk in pool.map(lambda chunk: paml_chunk(chunk, code), chunks)
            for value in chunk
        ]


def measure(worker, repeats, fingerprint=output_fingerprint):
    values = worker()
    expected = fingerprint(values)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        current = worker()
        samples.append(time.perf_counter() - start)
        if fingerprint(current) != expected:
            raise ValueError("Repeated benchmark changed its output")
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "warmup_runs": 1,
        "output_sha256": expected,
    }, values


def benchmark_cli(rows, code, threads, repeats):
    """Time fresh CLI processes, including import and TSV input/output."""
    with tempfile.TemporaryDirectory(prefix="cdskit-dnds-cli-bench-") as temp:
        pairs, output = Path(temp) / "pairs.tsv", Path(temp) / "ds.tsv"
        with pairs.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("pair_id", "sequence_1", "sequence_2"))
            writer.writerows(rows)
        command = [
            sys.executable,
            "-c",
            "from cdskit.cli import main; raise SystemExit(main())",
            "dnds",
            "--pairs_file",
            str(pairs),
            "--out_file",
            str(output),
            "--codon_table",
            str(code),
            "--threads",
            str(threads),
        ]

        def worker():
            subprocess.run(command, capture_output=True, check=True)
            return output

        result, _ = measure(
            worker, repeats, lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        )
    # Run this before any other child process in main: the child high-water
    # mark then describes only these fresh CLI launches, not PAML or metadata.
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        result["peak_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
    except ImportError:
        result["peak_rss_bytes"] = None
    result["input_preparation_in_timing"] = False
    result["import_and_tsv_io_in_timing"] = True
    result["memory_scope"] = "maximum child-process RSS across fresh CLI launches"
    return result


def paml_agreement(values, paml):
    errors, coverage, missing = {}, {}, {}
    for name in ("dN", "dS", "kappa"):
        native_name = name + "_diagnostic" if name in ("dN", "dS") else name
        differences = [
            abs(row[native_name] - other[name])
            for row, other in zip(values, paml, strict=True)
            if row[native_name] is not None and other[name] is not None
        ]
        errors[name] = max(differences, default=None)
        coverage[name] = len(differences)
        missing[name] = {
            "both_missing": sum(
                row[native_name] is None and other[name] is None
                for row, other in zip(values, paml, strict=True)
            ),
            "native_missing_only": sum(
                row[native_name] is None and other[name] is not None
                for row, other in zip(values, paml, strict=True)
            ),
            "reference_missing_only": sum(
                row[native_name] is not None and other[name] is None
                for row, other in zip(values, paml, strict=True)
            ),
        }
    if any(counts["reference_missing_only"] for counts in missing.values()):
        raise ValueError(
            f"Native finite diagnostics have no finite PAML reference: {missing}"
        )
    if any(error is not None and error > 0.000051 for error in errors.values()):
        raise ValueError(
            f"YN00 outputs disagree beyond PAML's printed precision: {errors}"
        )
    return {
        "agreement_max_absolute_error": errors,
        "agreement_compared_pairs": coverage,
        "agreement_includes_saturated_diagnostics": True,
        "agreement_all_pairs_compared": all(
            count == len(values) for count in coverage.values()
        ),
        "agreement_missing_pairs": missing,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--codon-table", type=int, default=1)
    parser.add_argument("--paml-only", action="store_true")
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument(
        "--cli", action="store_true", help="Also benchmark fresh CLI processes"
    )
    parser.add_argument("--individual", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("threads and repeats must be positive")
    if args.native_only and args.paml_only:
        parser.error("native-only and paml-only are mutually exclusive")
    if args.cli and args.paml_only:
        parser.error("cli requires native estimation")
    if args.native_only and args.individual:
        parser.error("individual requires PAML benchmarking")
    if not args.native_only and args.codon_table not in PAML_CODES:
        parser.error(
            "PAML does not support this codon-table mapping; use --native-only"
        )
    if str(args.pairs) == "-" or str(args.output) == "-":
        parser.error("pairs and output require file paths, not '-'")
    validate_distinct_paths(inputs=[args.pairs], outputs=[args.output])
    validate_output_paths([args.output])
    rows = read_pairs(args.pairs, args.codon_table)
    cli = (
        benchmark_cli(rows, args.codon_table, args.threads, args.repeats)
        if args.cli
        else None
    )
    report = {
        "schema_version": 2,
        "environment": environment_metadata(),
        "command": [sys.executable, *sys.argv],
        "runtime_image": os.environ.get("CDSKIT_BENCH_RUNTIME_IMAGE"),
        "host_cpu": os.environ.get("CDSKIT_BENCH_HOST_CPU"),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[1] / "cdskit/dnds.py",
                Path(__file__).resolve().parents[1] / "cdskit/codonutil.py",
                Path(__file__).resolve().parents[1] / "cdskit/tsvio.py",
                Path(__file__).resolve().parents[1] / "cdskit/cli.py",
            )
        },
        "pair_count": len(rows),
        "input_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        "threads": args.threads,
        "codon_table": args.codon_table,
        "method": "YN00 weighting=0 commonkappa=0 commonf3x4=0",
        "alignment_in_timing": False,
        "native_import_in_timing": False,
        "memory_scope": "whole benchmark process high-water mark, not per engine",
        "benchmarks": {"cdskit_cli": cli} if cli else {},
    }
    if not args.native_only:
        benchmark, paml = measure(
            lambda: run_paml(rows, args.codon_table, args.threads, args.individual),
            args.repeats,
        )
        executable = Path(shutil.which("yn00") or "")
        report["paml_executable_sha256"] = hashlib.sha256(
            executable.read_bytes()
        ).hexdigest()
        report["benchmarks"]["paml_individual" if args.individual else "paml_batch"] = (
            benchmark
        )
        report["paml_values"] = paml
    if not args.paml_only:
        from cdskit.dnds import estimate_pairs

        native, values = measure(
            lambda: estimate_pairs(
                [(a, b) for _, a, b in rows], args.codon_table, args.threads
            ),
            args.repeats,
        )
        report["benchmarks"]["cdskit_batch"] = native
        report["native_status_counts"] = {
            status: sum(row["status"] == status for row in values)
            for status in sorted({row["status"] for row in values})
        }
        if not args.native_only:
            report.update(paml_agreement(values, paml))
            report["speedup_vs_paml"] = (
                benchmark["median_seconds"] / native["median_seconds"]
            )
    report["process_peak_rss_bytes"] = peak_rss_bytes()
    validate_distinct_paths(inputs=[args.pairs], outputs=[args.output])
    atomic_write_json(args.output, report, indent=2)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("environment", "paml_values")
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

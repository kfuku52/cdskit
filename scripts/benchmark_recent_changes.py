"""Compare time, peak RSS and outputs in fresh processes across source snapshots.

Example:
  python scripts/benchmark_recent_changes.py --source old=/tmp/old \
    --source current=. --output /tmp/recent-changes.json
Use the same interpreter/dependencies for every snapshot. CPU threads are fixed
at one. Model workloads use an existing checkpoint; nothing is downloaded.
"""

import argparse
import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace


CASES = (
    "translate",
    "filter",
    "filter_long",
    "pad",
    "longestorf",
    "longestorf_ambiguous",
    "backtrim",
    "gapjust_gff",
    "targetp_features",
    "targetp_features_legacy",
    "cnn_predict",
    "cnn_train",
)


def build_case(name, root, model_path):
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    rng = random.Random(20260910)
    # Independent, reproducible coding sequences with only a terminal stop.
    sense = [
        a + b + c
        for a in "ACGT"
        for b in "ACGT"
        for c in "ACGT"
        if a + b + c not in {"TAA", "TAG", "TGA"}
    ]
    sequences = [
        "ATG" + "".join(rng.choices(sense, k=498)) + "TAA" for _ in range(1000)
    ]

    def common(value):
        return value

    if name == "translate":
        from cdskit.translate import translate_sequence_string

        sequence = "".join(sequences) * 2

        def run():
            return translate_sequence_string(sequence, 1, False)

        workload = {"nt": len(sequence)}
    elif name in ("filter", "filter_long"):
        from cdskit.filter import analyze_record

        records = (
            [SeqRecord(Seq("ATG" * 10_000_000), id="long")]
            if name == "filter_long"
            else [SeqRecord(Seq(s), id=str(i)) for i, s in enumerate(sequences)]
        )

        def run():
            return [analyze_record(r, 1, True) for r in records]

        new_fields = {
            "possible_stop_codons",
            "context_dependent_codons",
            "internal_possible_stop_codons",
            "codon_semantics_version",
        }

        def common(values):
            return [
                {k: v for k, v in row.items() if k not in new_fields} for row in values
            ]

        workload = {"records": len(records), "nt_per_record": len(records[0])}
    elif name == "pad":
        from cdskit.pad import process_record_padding

        kwargs = (
            {"include_report": False}
            if "include_report" in inspect.signature(process_record_padding).parameters
            else {}
        )

        def run():
            return [
                process_record_padding(str(i), s[1:], 1, "N", **kwargs)
                for i, s in enumerate(sequences)
            ]

        def common(values):
            return [
                {k: row[k] for k in ("new_seq", "is_no_stop", "was_padded")}
                for row in values
            ]

        workload = {"records": len(sequences), "nt_per_record": 1499, "report": False}
    elif name.startswith("longestorf"):
        from cdskit.longestcds import choose_best_candidate

        alphabet = "ACGT" if name == "longestorf" else "ACGTN"
        sequences = ["".join(rng.choices(alphabet, k=3000)) for _ in range(300)]

        def run():
            return [vars(choose_best_candidate(s, 1)) for s in sequences]

        workload = {
            "records": len(sequences),
            "nt_per_record": 3000,
            "alphabet": alphabet,
        }
    elif name == "backtrim":
        from cdskit.backtrim import backtrim_main
        from cdskit.translate import translate_sequence_string

        sequences = ["".join(rng.choices(sense, k=3000)) for _ in range(200)]
        source, target, output = [
            root / p for p in ("source.fa", "target.fa", "output.fa")
        ]
        source.write_text("".join(f">{i}\n{s}\n" for i, s in enumerate(sequences)))
        target.write_text(
            "".join(
                f">{i}\n{translate_sequence_string(s, 1, False)[::2]}\n"
                for i, s in enumerate(sequences)
            )
        )
        args = SimpleNamespace(
            seqfile=str(source),
            trimmed_aa_aln=str(target),
            outfile=str(output),
            inseqformat="fasta",
            outseqformat="fasta",
            codontable=1,
            threads=1,
        )

        def run():
            backtrim_main(args)
            return output.read_text()

        workload = {"records": 200, "source_codons": 3000, "kept_codons": 1500}
    elif name == "gapjust_gff":
        from cdskit.gapjust import gapjust_main

        sequence = ("ACGT" * 225 + "N" * 100) * 1000
        source, gff, output, outgff = [
            root / p for p in ("genome.fa", "input.gff", "out.fa", "out.gff")
        ]
        source.write_text("".join(f">s{i}\n{sequence}\n" for i in range(10)))
        gff.write_text(
            "##gff-version 3\n"
            + "".join(
                f"s{i}\tsynthetic\tgene\t{j * 1000 + 1}\t{j * 1000 + 800}\t.\t+\t.\tID=g{i}_{j}\n"
                for i in range(10)
                for j in range(1000)
            )
        )
        args = SimpleNamespace(
            seqfile=str(source),
            ingff=str(gff),
            outfile=str(output),
            outgff=str(outgff),
            inseqformat="fasta",
            outseqformat="fasta",
            gap_len=10,
            threads=1,
        )

        def run():
            gapjust_main(args)
            return [output.read_text(), outgff.read_text()]

        workload = {
            "nt": 10_000_000,
            "records": 10,
            "gaps": 10_000,
            "gff_features": 10_000,
        }
    elif name in ("targetp_features", "targetp_features_legacy"):
        import cdskit
        from cdskit.localize_model import extract_targetp_feature_ensemble_features

        sequences = [
            "M" + "".join(rng.choices("ACDEFGHIKLMNPQRSTVWY", k=499))
            for _ in range(1000)
        ]

        def run():
            scope = contextlib.nullcontext()
            if (
                name.endswith("_legacy")
                and (Path(cdskit.__file__).parent / "localize_schema.py").is_file()
            ):
                from cdskit.localize_schema import (
                    feature_schema_scope,
                    LEGACY_FEATURE_SCHEMA,
                )

                scope = feature_schema_scope(LEGACY_FEATURE_SCHEMA)
            with scope:
                return [
                    extract_targetp_feature_ensemble_features(s, "plant")
                    for s in sequences
                ]

        workload = {
            "records": 1000,
            "aa_per_record": 500,
            "feature_schema": "legacy" if name.endswith("_legacy") else "default",
        }
    else:
        import numpy as np
        import torch
        from cdskit.localize_multilabel_cnn import (
            fit_multilabel_cnn_classifier,
            predict_multilabel_cnn_batch,
        )

        torch.set_num_threads(1)
        sequences = [
            "M" + "".join(rng.choices("ACDEFGHIKLMNPQRSTVWY", k=length - 1))
            for length in rng.choices([100, 300, 500, 1000, 2000], k=512)
        ]
        if name == "cnn_predict":
            from cdskit.localize_model import load_localize_model

            model = load_localize_model(str(model_path))["localization_model"]
            if model.get("feature_dim", 0):
                raise ValueError("Use a sequence-only legacy CNN checkpoint.")

            def run():
                return predict_multilabel_cnn_batch(
                    sequences, model, device="cpu", batch_size=64
                )

            def common(result):
                return {
                    key: result[key] for key in ("prob_matrix", "prediction_matrix")
                }

            workload = {
                "records": 512,
                "aa_lengths": [100, 300, 500, 1000, 2000],
                "batch_size": 64,
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            }
        else:
            targets = np.random.default_rng(1).integers(0, 2, (512, 10))
            options = dict(
                seq_len=512,
                embed_dim=16,
                num_filters=16,
                kernel_sizes=(3, 5, 9),
                epochs=2,
                batch_size=64,
                seed=1,
                device="cpu",
                sequence_layout="legacy",
                mask_padding=False,
                tune_thresholds=False,
            )
            supported = inspect.signature(fit_multilabel_cnn_classifier).parameters
            options = {key: value for key, value in options.items() if key in supported}

            def run():
                return fit_multilabel_cnn_classifier(
                    sequences, targets, [str(i) for i in range(10)], **options
                )

            # Predict with each newly fitted head, outside the measured interval.
            def common(head):
                return predict_multilabel_cnn_batch(
                    sequences[:32], head, apply_thresholds=False
                )["prob_matrix"]

            workload = {
                "records": 512,
                "epochs": 2,
                "layout": "legacy",
                "mask_padding": False,
                "batch_size": 64,
                "thresholds_compared": False,
            }
    return run, common, workload


def worker(args):
    sys.path.insert(0, str(args.root.resolve()))
    import cdskit
    from cdskit.benchmarking import (
        environment_metadata,
        output_fingerprint,
        peak_rss_bytes,
    )

    if Path(cdskit.__file__).resolve().parent.parent != args.root.resolve():
        raise RuntimeError("Wrong source snapshot imported.")
    with tempfile.TemporaryDirectory(prefix="cdskit-bench-case-") as temporary:
        run, common, workload = build_case(args.case, Path(temporary), args.model)
        with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet):
            warmed = run()
            expected = output_fingerprint(common(warmed))
            del warmed
            samples = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                result = run()
                samples.append(time.perf_counter() - started)
                if output_fingerprint(common(result)) != expected:
                    raise RuntimeError("Output changed across repetitions.")
                del result
        print(
            json.dumps(
                dict(
                    workload=workload,
                    samples_seconds=samples,
                    median_seconds=statistics.median(samples),
                    warmup_runs=1,
                    peak_rss_bytes=peak_rss_bytes(),
                    output_sha256=expected,
                    environment=environment_metadata(),
                ),
                sort_keys=True,
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", action="append", default=[], help="NAME=SOURCE_ROOT"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "data/localize_bench/full_localization_20260908/final/cnn_legacy.pt"
        ),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.model = args.model.resolve()
    if args.case:
        worker(args)
        return
    if not args.source or not args.output:
        parser.error("source and output are required")
    sources = {
        name: str(Path(root).resolve())
        for name, root in (value.split("=", 1) for value in args.source)
    }
    report = {
        "schema_version": 1,
        "sources": sources,
        "repeats": args.repeats,
        "results": {name: {} for name in sources},
    }
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        CDSKIT_OFFLINE="1",
    )
    # Alternate source order per case to reduce systematic thermal/order bias.
    for i, case in enumerate(args.cases):
        order = list(sources.items())
        if i % 2:
            order.reverse()
        for name, root in order:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--case",
                case,
                "--root",
                root,
                "--model",
                str(args.model),
                "--repeats",
                str(args.repeats),
            ]
            process = subprocess.run(
                command, text=True, capture_output=True, env=environment, check=True
            )
            value = json.loads(process.stdout)
            report["results"][name][case] = value
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
            print(
                f"{name} {case}: {value['median_seconds']:.4f}s, {value['peak_rss_bytes'] / 1024**2:.1f} MiB",
                flush=True,
            )


if __name__ == "__main__":
    main()

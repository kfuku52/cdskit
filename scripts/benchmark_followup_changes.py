"""Compare follow-up CPU/memory candidates with complete output fingerprints.

Uses deterministic synthetic inputs, fresh source-snapshot workers and the
existing benchmark runner. No model downloads or biological training are used.
"""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
from types import SimpleNamespace

import benchmark_recent_changes as runner


CASES = (
    "trimcodon_runs",
    "trimcodon_fragmented",
    "split_many",
    "split_long",
    "split_short",
    "plot_msa",
    "gap_coordinates",
    "validate_ambiguity",
    "validate_gap_only",
    "backalign",
    "backalign_lower_wildcard",
    "stats_counts",
    "stats_long",
    "stats_empty",
    "reliability_bins",
    "reliability_dense",
    "group_folds",
    "group_folds_many_labels",
    "group_folds_small",
    "alphabet_dna",
    "alphabet_protein",
)


def build_case(name, root, model_path):
    del model_path
    import numpy as np
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    rng = random.Random(20261023)

    def common(value):
        return value

    if name.startswith("trimcodon"):
        from cdskit.trimcodon import trimcodon_main

        source, output = root / "input.fa", root / "output.fa"
        pattern = (
            ["ATG"] * 900 + ["---"] * 100
            if name.endswith("runs")
            else ["ATG", "---"] * 500
        )
        sequence = "".join(pattern) * 10
        source.write_text("".join(f">s{i}\n{sequence}\n" for i in range(100)))
        args = SimpleNamespace(
            seqfile=str(source),
            outfile=str(output),
            inseqformat="fasta",
            outseqformat="fasta",
            codontable=1,
            min_clean_fraction=0.5,
            report="",
        )

        def run():
            with contextlib.redirect_stderr(io.StringIO()) as messages:
                trimcodon_main(args)
            return messages.getvalue()

        def common(value):
            return [value, hashlib.sha256(output.read_bytes()).hexdigest()]

        workload = {"records": 100, "codons": 10000, "pattern": name}
    elif name.startswith("split_"):
        from cdskit.split import build_split_output_paths, split_main

        source = root / "input.fa"
        count, length = {
            "split_many": (100000, 300),
            "split_long": (100, 300000),
            "split_short": (10, 90),
        }[name]
        sequence = ("ATGGCN---acg" * ((length + 11) // 12))[:length]
        with source.open("w") as stream:
            for i in range(count):
                stream.write(f">s{i}\n{sequence}\n")
        args = SimpleNamespace(
            seqfile=str(source),
            prefix=str(root / "split"),
            inseqformat="fasta",
            outseqformat="fasta",
        )
        outputs = build_split_output_paths(args.prefix, args.outseqformat)

        def run():
            with contextlib.redirect_stderr(io.StringIO()) as messages:
                split_main(args)
            return messages.getvalue()

        def common(value):
            from pathlib import Path

            return [
                value,
                *[
                    hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    for path in outputs
                ],
            ]

        workload = {"records": count, "nt_per_record": len(sequence), "outputs": 3}
    elif name == "plot_msa":
        from cdskit.plot import _build_msa_summary

        triplets = ["ATG", "GCT", "TAG", "TAR", "TAN", "GCN", "---", "...", "NNN"]
        records = [
            SeqRecord(Seq("".join(rng.choices(triplets, k=1000))), id=str(i))
            for i in range(100)
        ]

        def run():
            return _build_msa_summary(records, 28)

        workload = {"records": 100, "codons": 1000, "code": 28}
    elif name == "gap_coordinates":
        from cdskit.gapjust import vectorized_coordinate_update

        edits = [
            {
                "original_gap_start": i * 1000 + 900,
                "original_gap_length": 100,
                "target_gap_length": 10,
            }
            for i in range(10000)
        ]
        starts = np.arange(10000, dtype=np.int64) * 1000 + 1
        ends = starts + 799

        def run():
            return vectorized_coordinate_update(starts, ends, edits)

        workload = {"intervals": 10000, "edits": 10000, "range_aware": True}
    elif name == "validate_ambiguity":
        from cdskit.validate import sequence_ambiguous_codon_counts

        sequence = "ATGGCNTAR---...?NNNacg" * 150000

        def run():
            return sequence_ambiguous_codon_counts(sequence)

        workload = {"nt": len(sequence), "legacy_helper": True}
    elif name == "validate_gap_only":
        from cdskit.validate import is_gap_only_sequence

        sequence = "-?.NXnx" * 500000

        def run():
            return is_gap_only_sequence(sequence)

        workload = {"nt": len(sequence), "gap_only": True}
    elif name.startswith("backalign"):
        from cdskit.backalign import backalign_sequence_strings

        sequence = "ATGGCTAAA" * 10000
        protein = "M-AK." * 10000
        if name.endswith("lower_wildcard"):
            sequence = sequence.lower()
            protein = "X-X?." * 10000

        def run():
            return backalign_sequence_strings(sequence, protein, 1, "s", False)

        workload = {"nt": len(sequence), "aligned_aa": len(protein), "code": 1}
    elif name.startswith("stats_"):
        from cdskit.stats import record_stats

        sequence = "ACGTacgtNNry-?." * 10000
        count = 100
        if name == "stats_long":
            sequence = (sequence * 215)[:30000000]
            count = 1
        elif name == "stats_empty":
            sequence = ""
        records = [SeqRecord(Seq(sequence), id=str(i)) for i in range(count)]

        def run():
            return [record_stats(record) for record in records]

        workload = {"records": len(records), "nt_per_record": len(sequence)}
    elif name.startswith("reliability_"):
        from cdskit.localize_evaluation import reliability_bins

        generator = np.random.default_rng(20261023)
        probability = generator.random((100000, 10))
        if name.endswith("dense"):
            probability.fill(1)
        target = generator.integers(0, 2, probability.shape)

        def run():
            return reliability_bins(target, probability)

        workload = {"shape": list(target.shape), "bins": 10}
    elif name.startswith("group_folds"):
        from cdskit.localize_splits import stratified_group_ids

        count = 20 if name.endswith("small") else 100000
        classes = 20 if name.endswith("many_labels") else 5
        names = (
            ["noTP", "SP", "mTP", "cTP", "lTP"]
            if classes == 5
            else [f"label{i}" for i in range(classes)]
        )
        labels = rng.choices(names, k=count)
        group_size = 2 if name.endswith("small") else 50
        groups = [str(i // group_size) for i in range(len(labels))]

        def run():
            return stratified_group_ids(labels, groups, 5, 1).tolist()

        workload = {
            "rows": len(labels),
            "groups": len(set(groups)),
            "labels": classes,
            "folds": 5,
            "seed": 1,
        }
    else:
        from cdskit.util import stop_if_not_dna, stop_if_not_protein

        dna = name == "alphabet_dna"
        alphabet = (
            "ACGTRYSWKMBDHVNXacgtryswkmbdhvnx-?."
            if dna
            else "ABCDEFGHIKLMNPQRSTVWXYZJUO*abcdefghiklmnpqrstvwxyzjuo-?."
        )
        sequence = "".join(rng.choices(alphabet, k=100000))
        records = [SeqRecord(Seq(sequence), id=str(i)) for i in range(100)]
        validate = stop_if_not_dna if dna else stop_if_not_protein

        def run():
            return validate(records)

        workload = {
            "records": len(records),
            "residues": len(sequence),
            "alphabet": name,
        }
    return run, common, workload


def main():
    # The shared runner checks each worker internally. Also reject different
    # outputs across snapshots before reporting a successful comparison.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case")
    options, _ = parser.parse_known_args()
    runner.main(
        cases=CASES, build_case_fn=build_case, script_path=__file__, description=__doc__
    )
    if options.case is None and options.output is not None:
        report = json.loads(options.output.read_text())
        cases = next(iter(report["results"].values()))
        for case in cases:
            outputs = {
                rows[case]["output_sha256"] for rows in report["results"].values()
            }
            if len(outputs) != 1:
                raise RuntimeError(
                    f"{case}: source snapshots produced different outputs."
                )


if __name__ == "__main__":
    main()

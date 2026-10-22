"""Compare sequence-kernel changes using fresh, verified source snapshots.

Uses the existing snapshot runner; models are generated locally, never downloaded.
Example: python scripts/benchmark_sequence_kernels.py --source before=/tmp/before \
    --source after=. --repeats 7 --output /tmp/sequence-kernels.json
"""

import hashlib
from itertools import product
import random
from types import SimpleNamespace

import benchmark_recent_changes as runner

CASES = (
    "hammer",
    "hammer_no_gap_guard",
    "translate_short",
    "translate_long",
    "translate_mixed",
    "codonstats_short",
    "codonstats_long",
    "codonstats_mixed",
    "maxalign16",
    "maxalign20",
    "orf_short",
    "orf_long",
    "orf_code28",
    "dnds",
    "bilstm",
)


def build_case(name, root, model_path):
    del model_path
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    rng = random.Random(20261022)

    def common(value):
        return value

    if name.startswith("hammer"):
        from cdskit.hammer import hammer_main

        source, destination = root / "input.fa", root / "output.fa"
        codons = ["ATG", "GCT", "AAA", "GCN", "TTY", "TAA", "---", "NNN"]
        with source.open("w") as stream:
            for i in range(500):
                stream.write(f">seq{i}\n{''.join(rng.choices(codons, k=5000))}\n")
        args = SimpleNamespace(
            seqfile=str(source),
            outfile=str(destination),
            inseqformat="fasta",
            outseqformat="fasta",
            codontable=1,
            nail="100",
            prevent_gap_only=name == "hammer",
            threads=1,
        )

        def run():
            hammer_main(args)

        def common(value):
            return hashlib.sha256(destination.read_bytes()).hexdigest()

        workload = {
            "records": 500,
            "codons": 5000,
            "prevent_gap_only": args.prevent_gap_only,
        }
    elif name.startswith("translate"):
        from cdskit.translate import translate_sequence_string

        sequences = {
            "translate_short": ["ATGGCTAAA" * 10] * 1000,
            "translate_long": ["ATGGCTAAA" * 33333],
            "translate_mixed": ["ATGGCNTARTAN---NNN" * 5000],
        }[name]

        def run():
            return [
                translate_sequence_string(sequence, 1, False) for sequence in sequences
            ]

        workload = {"records": len(sequences), "nt_per_record": len(sequences[0])}
    elif name.startswith("codonstats"):
        from cdskit.codonstats import summarize_record

        if name == "codonstats_short":
            sequences = ["ATGGCCTAA" * 10] * 300
        elif name == "codonstats_long":
            sequences = [
                "".join(
                    rng.choices(
                        ["".join(c) for c in product("ACGT", repeat=3)], k=100000
                    )
                )
            ]
        else:
            sequences = ["ATGGCNTARTAN---NNN" * 167] * 30
        records = [
            SeqRecord(Seq(sequence), id=str(i)) for i, sequence in enumerate(sequences)
        ]

        def run():
            return [summarize_record(record, 1) for record in records]

        workload = {"records": len(records), "nt_per_record": len(records[0])}
    elif name.startswith("maxalign"):
        from cdskit.maxalign import solve_exact

        count = int(name.removeprefix("maxalign"))
        matrix = [[rng.random() > 0.1 for _ in range(80)] for _ in range(count)]

        def run():
            return solve_exact(matrix, threads=1)

        workload = {"records": count, "codon_sites": 80, "threads": 1}
    elif name.startswith("orf"):
        from cdskit.longestcds import choose_best_candidate

        length = 150 if name == "orf_short" else 300000
        sequences = [
            "".join(rng.choices("ACGTRYSWKMBDHVNX-?.", k=length))
            for _ in range(32 if length == 150 else 1)
        ]
        code = 28 if name == "orf_code28" else 11

        def run():
            return [
                vars(choose_best_candidate(sequence, code)) for sequence in sequences
            ]

        workload = {"records": len(sequences), "nt_per_record": length, "code": code}
    elif name == "dnds":
        from cdskit.dnds import estimate_pairs

        codons = [
            "".join(c)
            for c in product("ACGT", repeat=3)
            if "".join(c) not in {"TAA", "TAG", "TGA"}
        ]
        pairs = []
        for _ in range(256):
            first = rng.choices(codons, k=300)
            second = [
                rng.choice(codons) if rng.random() < 0.15 else codon for codon in first
            ]
            pairs.append(("".join(first), "".join(second)))

        def run():
            return estimate_pairs(pairs)

        workload = {"pairs": 256, "codons": 300, "code": 1, "threads": 1}
    else:
        import torch
        from cdskit.localize_bilstm import (
            DEFAULT_AA_TO_IDX,
            _build_bilstm_attention_module,
            predict_bilstm_attention_batch,
        )

        torch.set_num_threads(1)
        torch.manual_seed(20261022)
        module = _build_bilstm_attention_module(
            torch, torch.nn, len(DEFAULT_AA_TO_IDX), 32, 128, 2, 0.0, 5
        )
        head = dict(
            aa_to_idx=dict(DEFAULT_AA_TO_IDX),
            class_order=["noTP", "SP", "mTP", "cTP", "lTP"],
            seq_len=64,
            embed_dim=32,
            hidden_dim=128,
            num_layers=2,
            dropout=0.0,
            state_dict=module.state_dict(),
        )
        sequences = ["".join(rng.choices("ACDEFGHIKLMNPQRSTVWY", k=64))]

        def run():
            return predict_bilstm_attention_batch(sequences, head, device="cpu")

        workload = {
            "records": 1,
            "aa_length": 64,
            "hidden_dim": 128,
            "layers": 2,
            "threads": 1,
            "random_local_head": True,
        }
    return run, common, workload


if __name__ == "__main__":
    runner.main(
        cases=CASES, build_case_fn=build_case, script_path=__file__, description=__doc__
    )

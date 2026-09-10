"""Print review probes; these observations are not assertions of desired behavior.

Run from the repository with .venvs/core-3.12/bin/python; --model optionally
uses a trusted, safely loadable CNN checkpoint and requires the ML environment.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.backtrim import find_kept_aa_sites, trim_codon_records
from cdskit.codonutil import summarize_codons
from cdskit.gapjust import (
    apply_gap_justifications_to_gff,
    normalize_record_gap_lengths,
)
from cdskit.localize_learn import build_stratified_folds, evaluate_cross_validation
from cdskit.localize_model import (
    LOCALIZATION_CLASSES,
    detect_perox_signals,
    extract_localize_features,
    infer_labels_from_uniprot_cc,
)
from cdskit.longestcds import choose_best_candidate
from cdskit.pad import process_record_padding
from cdskit.translate import translate_sequence_string
from cdskit.util import GFF_DTYPE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="")
    args = parser.parse_args()
    result = {}
    result["pts2"] = {
        seq: detect_perox_signals(seq) for seq in ("MRLQVVLGHLAAAA", "MRLQVVVHLAAAA")
    }
    result["stop_semantics"] = [
        {
            "code": code,
            "sequence": seq,
            "translation": translate_sequence_string(seq, code, False),
            "qc": summarize_codons(seq, code),
        }
        for code, seq in (
            (27, "ATGTGAAAA"),
            (28, "ATGTAATAGTGAAAA"),
            (31, "ATGTAATAGAAA"),
            (1, "ATGTARAAA"),
        )
    ]
    result["padding"] = {
        seq: process_record_padding("example", seq, 1, "N")
        for seq in ("ATGTAAGGG", "ATGTARAAA")
    }
    result["weak_labels"] = {
        text: infer_labels_from_uniprot_cc(text)
        for text in (
            "",
            "SUBCELLULAR LOCATION: Mitochondrion outer membrane.",
            "SUBCELLULAR LOCATION: Secreted. Note=Secreted by an unconventional pathway.",
            "SUBCELLULAR LOCATION: Cytoplasm. Note=Does not localize to mitochondria.",
        )
    }
    labels = [label for label in LOCALIZATION_CLASSES for _ in range(2)]
    sequences = ["M" + aa * 20 for aa in "ARDLS" for _ in range(2)]
    features = np.array([extract_localize_features(seq)[0] for seq in sequences])
    folds = build_stratified_folds(labels, 2, 1)
    cv = evaluate_cross_validation(
        features,
        sequences,
        labels,
        ["no"] * 10,
        2,
        1,
        "nearest_centroid",
        {},
        "cpu",
    )
    result["duplicate_cv"] = {
        "folds": [fold.tolist() for fold in folds],
        "shared_sequences": len(
            {sequences[i] for i in folds[0]} & {sequences[i] for i in folds[1]}
        ),
        "accuracy": cv["class_accuracy_mean"],
    }
    record = SeqRecord(Seq("ATGNNNAAACCCGGGTTTAAA"), id="s")
    gff = {
        "data": np.array(
            [
                ("s", ".", "CDS", 1, 9, ".", "+", "0", "ID=c1;Parent=t"),
                ("s", ".", "CDS", 13, 21, ".", "+", "0", "ID=c2;Parent=t"),
            ],
            dtype=GFF_DTYPE,
        )
    }
    edits, *_ = normalize_record_gap_lengths(record, 4)
    try:
        apply_gap_justifications_to_gff(gff, {"s": edits})
    except ValueError as exc:
        result["gapjust_rejection"] = str(exc)
    result["gapjust"] = {
        "sequence": str(record.seq),
        "start_end_phase": gff["data"][["start", "end", "phase"]].tolist(),
        "consistent_downstream_phase": 2,
    }
    records = [
        SeqRecord(Seq("GCTGCC"), id="s1"),
        SeqRecord(Seq("GCTGCT"), id="s2"),
    ]
    kept, _ = find_kept_aa_sites(["AA", "AA"], ["A", "A"])
    result["backtrim"] = {
        "kept_zero_based": kept,
        "sequences": [str(r.seq) for r in trim_codon_records(records, kept)],
    }
    candidate = choose_best_candidate("ATGTAA" + "C" * 100, 1)
    result["longestorf"] = {
        "input_length": 106,
        "output": candidate.output_seq,
        "category": candidate.category,
    }
    if args.model:
        import torch
        from cdskit.localize_model import load_localize_model
        from cdskit.localize_multilabel_cnn import predict_multilabel_cnn_batch

        torch.set_num_threads(1)
        head = load_localize_model(args.model)["localization_model"]
        probes = ["", "X" * 100, "M"]
        prediction = predict_multilabel_cnn_batch(probes, head, device="cpu")
        result["model_sha256"] = hashlib.sha256(
            Path(args.model).read_bytes()
        ).hexdigest()
        result["low_information"] = [
            {
                "sequence": seq,
                "labels": [
                    name for name, active in zip(head["class_order"], row) if active
                ],
                "max_score": float(prob.max()),
            }
            for seq, row, prob in zip(
                probes, prediction["prediction_matrix"], prediction["prob_matrix"]
            )
        ]
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline, fixed-protocol checks against external CDS annotations and controls.

Never tune padding/ORF rules on this report. This small panel is not an estimate
of population accuracy, and a stop-free simulated frameshift is not a repair.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from Bio import __version__ as biopython_version
from Bio.Seq import Seq

from cdskit.codonutil import CODON_SEMANTICS_VERSION
from cdskit.longestcds import choose_best_candidate
from cdskit.pad import process_record_padding
from cdskit.translate import translate_sequence_string

ROOT = Path(__file__).resolve().parents[1]


def padding_case(seq_id, sequence, code, category, head=0, tail=0):
    result = process_record_padding(seq_id, sequence, code, "N")
    choice = result["candidates"][result["selected_candidate_0based"]]
    evaluable = category in ("intact", "end_deletion")
    return {
        "id": seq_id,
        "category": category,
        "input": sequence,
        "head_deleted": head,
        "tail_deleted": tail,
        "selected_head": choice["head_padding"],
        "selected_tail": choice["tail_padding"],
        "stop_free_after_padding": result["is_no_stop"],
        "frame_changed": choice["head_padding"] != 0,
        "tied": len(result["tied_candidates_0based"]) > 1,
        "correct_frame": ((choice["head_padding"] - head) % 3 == 0)
        if evaluable
        else None,
        "correct_placement": (
            choice["head_padding"] == head and choice["tail_padding"] == tail
        )
        if evaluable
        else None,
    }


def evaluate_padding(record):
    cds = record["cds"]
    seq_id = record["accession"]
    code = record["codon_table"]
    cases = [padding_case(seq_id, cds, code, "intact")]
    for head in range(3):
        for tail in range(3):
            if head or tail:
                fragment = cds[head : len(cds) - tail if tail else len(cds)]
                cases.append(
                    padding_case(seq_id, fragment, code, "end_deletion", head, tail)
                )
    for fraction in (0.25, 0.5, 0.75):
        pos = (int(len(cds) * fraction) // 3) * 3
        cases.append(
            padding_case(
                seq_id, cds[:pos] + cds[pos + 1 :], code, "internal_frameshift"
            )
        )
        cases.append(
            padding_case(
                seq_id, cds[:pos] + "A" + cds[pos:], code, "internal_frameshift"
            )
        )
        cases.append(
            padding_case(
                seq_id, cds[:pos] + "TAA" + cds[pos + 3 :], code, "internal_stop"
            )
        )
    return cases


def synthetic_records():
    # Fixed controls cover short/long and GC-poor/rich coding sequences.
    rng = random.Random(20260910)
    for target_gc in (0.2, 0.5, 0.8):
        weights = [
            (1 - target_gc) / 2,
            target_gc / 2,
            target_gc / 2,
            (1 - target_gc) / 2,
        ]
        for length in (10, 100):
            for replicate in range(10):
                codons = []
                while len(codons) < length:
                    codon = "".join(rng.choices("ACGT", weights=weights, k=3))
                    if codon not in ("TAA", "TAG", "TGA"):
                        codons.append(codon)
                yield {
                    "accession": f"synthetic_gc{target_gc}_len{length}_{replicate}",
                    "cds": "ATG" + "".join(codons) + "TAA",
                    "codon_table": 1,
                }


def evaluate_orf(record):
    transcript = record["transcript"]
    rows = []
    for reverse in (False, True):
        seq = str(Seq(transcript).reverse_complement()) if reverse else transcript
        start, end = record["cds_start_1based"], record["cds_end_1based"]
        strand = "+"
        if reverse:
            start, end = len(seq) - end + 1, len(seq) - start + 1
            strand = "-"
        for selection in ("complete-first", "longest"):
            candidate = choose_best_candidate(seq, record["codon_table"], selection)
            assert candidate is not None
            rows.append(
                {
                    "id": record["accession"],
                    "reverse_input": reverse,
                    "selection": selection,
                    "expected": [strand, start, end],
                    "observed": [
                        candidate.strand,
                        candidate.start_1based,
                        candidate.end_1based,
                    ],
                    "exact_boundary_match": (
                        candidate.strand,
                        candidate.start_1based,
                        candidate.end_1based,
                    )
                    == (strand, start, end),
                    "strand_frame_match": candidate.strand == strand
                    and (
                        (candidate.end_1based - end) % 3 == 0
                        if strand == "-"
                        else (candidate.start_1based - start) % 3 == 0
                    ),
                }
            )
    return rows


def summarize(cases):
    summary = defaultdict(lambda: defaultdict(int))
    for row in cases:
        group = summary[row["category"]]
        group["n"] += 1
        for metric in (
            "stop_free_after_padding",
            "frame_changed",
            "tied",
            "correct_frame",
            "correct_placement",
        ):
            if row[metric] is not None:
                group[metric] += int(row[metric])
    return dict(summary)


def evaluate(dataset, include_cases=False):
    records = dataset["records"]
    translations = [
        {
            "id": record["accession"],
            "matches_external_translation": translate_sequence_string(
                record["cds"], record["codon_table"], False, complete_cds=True
            )
            == record["annotated_translation"],
        }
        for record in records
    ]
    external = [case for record in records for case in evaluate_padding(record)]
    synthetic = [
        case for record in synthetic_records() for case in evaluate_padding(record)
    ]
    report = {
        "protocol_version": "1",
        "codon_semantics_version": CODON_SEMANTICS_VERSION,
        "biopython_version": biopython_version,
        "source": dataset["source"],
        "limitations": dataset["limitations"],
        "translations": translations,
        "external_padding_summary": summarize(external),
        "synthetic_padding_summary": summarize(synthetic),
        "orf": [row for record in records for row in evaluate_orf(record)],
        "synthetic_strata": {
            group: summarize(
                [
                    case
                    for case in synthetic
                    if "_".join(case["id"].split("_")[1:3]) == group
                ]
            )
            for group in sorted(
                {"_".join(case["id"].split("_")[1:3]) for case in synthetic}
            )
        },
    }
    if include_cases:
        report.update(
            external_padding_cases=external, synthetic_padding_cases=synthetic
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "tests/fixtures/codon_evaluation/refseq.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-cases",
        action="store_true",
        help="Include every perturbed sequence and decision for auditing.",
    )
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    report = evaluate(json.loads(raw), include_cases=args.include_cases)
    report["dataset_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

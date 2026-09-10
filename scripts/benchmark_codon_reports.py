#!/usr/bin/env python3
"""Compare compact ORF reports to the pre-review report path in separate runs.

Run each implementation in a fresh process so the peak RSS is comparable:
  python scripts/benchmark_codon_reports.py --implementation reference
  python scripts/benchmark_codon_reports.py --implementation current
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.codonutil import summarize_codons
from cdskit.longestcds import candidate_report, iter_candidates, selection_key


def reference_report(record, code, selection, order):
    """Pre-review path: materialize and scan each nested candidate separately."""
    candidates = sorted(
        iter_candidates(str(record.seq), code),
        key=lambda c: selection_key(c, selection),
        reverse=True,
    )
    rows = []
    primary = selection_key(candidates[0], selection)[:2] if candidates else None
    for index, candidate in enumerate(candidates):
        summary = summarize_codons(candidate.output_seq, code, "physical")
        rows.append(
            {
                **asdict(candidate),
                "rank": index + 1,
                "selected": index == 0,
                "tied_before_deterministic_order": selection_key(candidate, selection)[
                    :2
                ]
                == primary,
                "stop_evidence": "definite" if candidate.has_stop else "unconfirmed",
                "possible_stop_codons": summary["possible_stop"],
                "context_dependent_codons": summary["context_dependent"],
            }
        )
    return {
        "seq_id": record.id,
        "input_order": order,
        "selection": selection,
        "candidates": rows,
        "selected_candidate_0based": 0 if rows else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation", choices=["reference", "current"], required=True
    )
    parser.add_argument(
        "--workload", choices=["nested-starts", "refseq-tp53"], default="nested-starts"
    )
    args = parser.parse_args()
    sequence = "ATG" * 2000
    if args.workload == "refseq-tp53":
        dataset = (
            Path(__file__).resolve().parents[1]
            / "tests/fixtures/codon_evaluation/refseq.json"
        )
        sequence = next(
            r["transcript"]
            for r in json.loads(dataset.read_text())["records"]
            if r["accession"] == "NM_000546.6"
        )
    record = SeqRecord(Seq(sequence), id=args.workload)
    function = (
        reference_report if args.implementation == "reference" else candidate_report
    )
    function(record, 1, "complete-first", 1)  # Warm up code/codon caches.
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        report = function(record, 1, "complete-first", 1)
        samples.append(time.perf_counter() - start)
    # Coordinates reconstruct every omitted unselected sequence. Compare the
    # common fields separately from intentionally more compact serialization.
    common = [
        {
            key: value
            for key, value in row.items()
            if key not in ("output_seq", "missing_codons", "ambiguous_codons")
        }
        for row in report["candidates"]
    ]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "implementation": args.implementation,
        "workload": args.workload,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "input_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "nt": len(sequence),
        "candidates": len(common),
        "seconds": samples,
        "median_seconds": statistics.median(samples),
        "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "serialized_bytes": len(json.dumps(report)),
        "common_fields_sha256": hashlib.sha256(
            json.dumps(common, sort_keys=True).encode()
        ).hexdigest(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

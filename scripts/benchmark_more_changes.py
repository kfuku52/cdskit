"""Compare additional sequence kernels with verified source snapshots.

Use --paired --before OLD --after NEW for interleaved module comparisons, or
--source before=OLD --source after=NEW for fresh-process timing and RSS.
"""

import argparse
import json
from pathlib import Path
import random
import sys

import benchmark_followup_paired as paired_runner
import benchmark_recent_changes as fresh_runner


CASES = (
    "summary_short",
    "summary_long",
    "summary_mixed",
    "summary_code28",
    "summary_missing",
    "summary_partial",
    "mask_short",
    "mask_mixed",
    "mask_gap",
    "mask_clean",
    "label_collision",
    "label_unique",
    "pad_random",
    "pad_rich",
    "pad_rare",
    "pad_dual",
)
MODULES = {
    case: {"summary": "codonutil", "mask": "mask", "label": "label", "pad": "pad"}[
        case.split("_")[0]
    ]
    for case in CASES
}


def build_case(name, root, model_path):
    del root, model_path
    rng = random.Random(20261023)

    def common(value):
        return value

    if name.startswith("summary"):
        from cdskit.codonutil import summarize_codons

        sequence = "ATGGCTAAA" * 33333
        code = 28 if name.endswith("code28") else 1
        if name == "summary_short":
            sequence = "ATGGCNTAA" * 10
        elif name in {"summary_mixed", "summary_code28"}:
            sequence = "ATGGCNTARTAN---NNN" * 16667
        elif name == "summary_missing":
            sequence = "ATGTAR" + "---" * 100000
        elif name == "summary_partial":
            sequence += "A"

        def run():
            return summarize_codons(sequence, code)

        workload = {
            "nt": len(sequence),
            "code": code,
            "terminal_policy": "last_evaluable",
        }
    elif name.startswith("mask"):
        from cdskit.mask import mask_sequence_string

        sequence = "ATGGCNTARTAN---A--" * 50000
        flags = (True, True)
        if name == "mask_short":
            sequence = "ATGGCN---A--" * 3
        elif name == "mask_gap":
            sequence = "ATG---A--G.T..." * 60000
            flags = (False, False)
        elif name == "mask_clean":
            sequence = "ATGGCTAAA" * 100000

        def run():
            return mask_sequence_string(sequence, 1, "NNN", *flags)

        workload = {
            "nt": len(sequence),
            "code": 1,
            "mask_ambiguous": flags[0],
            "mask_stop": flags[1],
        }
    elif name.startswith("label"):
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        from cdskit.label import uniquify_label_ids

        names = [f"AA{i:06}" for i in range(1200) for _ in range(2)]
        if name == "label_unique":
            names = [f"AA{i:06}" for i in range(2400)]
        records = [SeqRecord(Seq("ATG"), id=value) for value in names]

        def run():
            for record, value in zip(records, names, strict=True):
                record.id = value
                record.description = "description"
            return uniquify_label_ids(records, clip_len=8)

        def common(value):
            return value, [(record.id, record.description) for record in records]

        workload = {
            "records": len(records),
            "clip_len": 8,
            "scenario": name,
            "includes_input_reset": True,
        }
    else:
        from cdskit.pad import count_internal_stop_codons

        sequence = "".join(rng.choices("ACGT", k=300000))
        code = 27 if name == "pad_dual" else 1
        if name == "pad_rich":
            sequence = "TAATARTGA" * 33334
        elif name == "pad_rare":
            sequence = "ATGGCTAAA" * 33333

        def run():
            return count_internal_stop_codons(sequence, code)

        workload = {"nt": len(sequence), "code": code, "scenario": name}
    return run, common, workload


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case")
    options, _ = parser.parse_known_args()
    if options.paired:
        sys.argv.remove("--paired")
        paired_runner.CASES = CASES
        paired_runner.MODULES = MODULES
        paired_runner.build_case = build_case
        paired_runner.main()
        return
    fresh_runner.main(
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
                raise RuntimeError(f"{case}: snapshots produced different outputs.")


if __name__ == "__main__":
    main()

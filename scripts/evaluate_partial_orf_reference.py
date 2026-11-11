"""Compare clipped ORF selection against GENCODE partial CDS coordinates."""

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
from pathlib import Path
import re

from cdskit.atomicio import atomic_write_json, validate_distinct_paths
from cdskit.codonutil import get_forward_table, get_stop_codons
from cdskit.longestcds import choose_best_candidate
from cdskit.provenance import file_digest
from evaluate_orf_reference import fasta_records

MODES = {
    (True, False): "partial-5prime",
    (False, True): "partial-3prime",
    (True, True): "partial-both",
}
SENSE = get_forward_table(1)
STOPS = get_stop_codons(1)


def read_annotation(path):
    transcripts, phases = {}, {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            chrom, _, feature, start, end, _, strand, phase, attrs = line.split("\t")
            if chrom == "chrM" or feature not in {"transcript", "CDS"}:
                continue
            if 'transcript_type "protein_coding"' not in attrs:
                continue
            missing = ('tag "cds_start_NF"' in attrs, 'tag "cds_end_NF"' in attrs)
            match = re.search(r'transcript_id "([^"]+)"', attrs)
            if missing not in MODES or match is None:
                continue
            identifier = match[1]
            if feature == "transcript":
                transcripts[identifier] = missing
            elif phase in {"0", "1", "2"}:
                first = int(start) if strand == "+" else -int(end)
                if identifier not in phases or first < phases[identifier][0]:
                    phases[identifier] = (first, int(phase))
    return transcripts, phases


def reference_panel(fasta, annotation):
    transcripts, phases = read_annotation(annotation)
    counts = defaultdict(Counter)
    selected = {}
    for title, sequence in fasta_records(fasta):
        identifier, gene, *_ = title.split("|")
        match = re.search(r"\|CDS:(\d+)-(\d+)\|", title)
        if identifier not in transcripts or identifier not in phases or match is None:
            continue
        left, right = map(int, match.groups())
        left -= 1
        missing_start, missing_stop = transcripts[identifier]
        mode = MODES[missing_start, missing_stop]
        counts[mode]["joined"] += 1
        if not missing_stop and sequence[right - 3 : right] not in STOPS:
            if sequence[right : right + 3] in STOPS:
                right += 3
        if missing_start:
            left += phases[identifier][1]
        if missing_stop:
            right -= (right - left) % 3
        if (missing_start and left >= 3) or (
            missing_stop and len(sequence) - right >= 3
        ):
            counts[mode]["boundary_away_from_input_end"] += 1
            continue
        coding = sequence[left:right]
        sense = coding if missing_stop else coding[:-3]
        if (
            not (0 <= left < right <= len(sequence))
            or set(sequence) - set("ACGT")
            or len(sense) < 3
            or len(coding) % 3
            or (not missing_stop and coding[-3:] not in STOPS)
            or (not missing_start and coding[:3] != "ATG")
            or any(sense[i : i + 3] not in SENSE for i in range(0, len(sense), 3))
        ):
            counts[mode]["invalid_sequence"] += 1
            continue
        rank = sha256(identifier.encode("utf-8")).digest()
        key = (gene, mode)
        if key not in selected or rank < selected[key][0]:
            selected[key] = rank, (mode, sequence, left + 1, right)
    return [row for _, row in selected.values()], counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [args.transcripts, args.annotation]
    validate_distinct_paths(inputs=inputs, outputs=[args.output])
    panel, counts = reference_panel(*inputs)
    if not panel:
        raise ValueError("No eligible partial reference transcripts could be scored.")
    for mode, sequence, start, end in panel:
        counts[mode]["selected"] += 1
        for selection in ("complete-first", "longest", mode):
            found = choose_best_candidate(
                sequence, 1, selection, strand="forward", start_policy="atg-only"
            )
            counts[mode][selection] += found is not None and (
                found.start_1based,
                found.end_1based,
            ) == (start, end)
    atomic_write_json(
        args.output,
        {
            "inputs_sha256": {path.name: file_digest(path) for path in inputs},
            "groups": counts,
        },
        indent=2,
    )


if __name__ == "__main__":
    main()

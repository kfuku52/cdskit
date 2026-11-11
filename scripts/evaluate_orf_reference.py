"""Compare ORF search policies against released transcript annotations."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
from hashlib import sha256
from pathlib import Path
import re

from Bio.SeqIO.FastaIO import SimpleFastaParser

from cdskit.atomicio import atomic_write_json, validate_distinct_paths
from cdskit.codonutil import get_forward_table, get_stop_codons
from cdskit.longestcds import choose_best_candidate
from cdskit.provenance import file_digest

CODE = 1
POLICIES = {
    "default": {},
    "longest": {"selection": "longest"},
    "forward": {"strand": "forward"},
    "atg_only": {"start_policy": "atg-only"},
    "forward_atg_only": {"strand": "forward", "start_policy": "atg-only"},
}
SAMPLE_SEED = b"cdskit-orf-reference-2026-11-11-v1"


def fasta_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        yield from SimpleFastaParser(handle)


def human_transcripts(transcripts, annotation):
    features = defaultdict(set)
    required = {"transcript", "start_codon", "stop_codon"}
    with gzip.open(annotation, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            chromosome, _, feature, *_, attributes = line.split("\t")
            if chromosome == "chrM" or feature not in required:
                continue
            if feature == "transcript" and (
                'transcript_type "protein_coding"' not in attributes
                or 'tag "cds_start_NF"' in attributes
                or 'tag "cds_end_NF"' in attributes
            ):
                continue
            match = re.search(r'transcript_id "([^"]+)"', attributes)
            if match:
                features[match[1]].add(feature)
    complete = {identifier for identifier, seen in features.items() if seen == required}
    for header, sequence in fasta_records(transcripts):
        identifier, gene, *_ = header.split("|")
        match = re.search(r"\|CDS:(\d+)-(\d+)\|", header)
        if match and identifier in complete:
            start, end = map(int, match.groups())
            yield gene, identifier, sequence, start - 1, end


def plant_transcripts(transcripts, coding_sequences):
    cds = {
        header.split()[0]: sequence
        for header, sequence in fasta_records(coding_sequences)
    }
    for header, sequence in fasta_records(transcripts):
        identifier = header.split()[0]
        match = re.search(r" gene:(AT[1-5]G\d+)(?:\s|$)", header)
        if identifier not in cds or not match:
            continue
        coding = cds[identifier]
        start = sequence.find(coding)
        if start >= 0 and sequence.find(coding, start + 1) < 0:
            yield match[1], identifier, sequence, start, start + len(coding)


def select_transcripts(records):
    selected = {}
    excluded = 0
    sense = get_forward_table(CODE)
    stops = get_stop_codons(CODE)
    for gene, identifier, sequence, start, end in records:
        sequence = sequence.upper()
        # Release FASTA coordinates may omit the terminal stop.
        if sequence[end - 3 : end] not in stops and sequence[end : end + 3] in stops:
            end += 3
        coding = sequence[start:end]
        if (
            not (0 <= start <= end - 6 and end <= len(sequence))
            or set(sequence) - set("ACGT")
            or len(coding) % 3
            or coding[-3:] not in stops
            or any(coding[i : i + 3] not in sense for i in range(0, len(coding) - 3, 3))
        ):
            excluded += 1
            continue
        rank = sha256(SAMPLE_SEED + identifier.encode("utf-8")).digest()
        if gene not in selected or rank < selected[gene][0]:
            selected[gene] = rank, (sequence, start, end)
    return [row for _, row in selected.values()], excluded


def score_transcript(row):
    sequence, start, end = row
    scores = {}
    for name, options in POLICIES.items():
        candidate = choose_best_candidate(sequence, CODE, **options)
        forward = candidate.strand == "+"
        matched_start = forward and candidate.start_1based == start + 1
        matched_stop = forward and candidate.end_1based == end
        matched_frame = forward and (candidate.start_1based - 1 - start) % 3 == 0
        scores[name] = forward, matched_frame, matched_start, matched_stop
    return sequence[start : start + 3] == "ATG", end - start <= 300, scores


def summarize(rows):
    groups = defaultdict(lambda: defaultdict(Counter))
    for has_atg, short, scores in rows:
        base_exact = all(scores["default"])
        strata = (
            "all",
            "annotated_atg" if has_atg else "annotated_non_atg",
            "cds_le_300_nt" if short else "cds_gt_300_nt",
        )
        for stratum in strata:
            for policy, score in scores.items():
                counts = groups[stratum][policy]
                exact = all(score)
                counts.update(
                    {
                        "rows": 1,
                        **dict(
                            zip(
                                ("strand", "frame", "start", "stop"), score, strict=True
                            )
                        ),
                        "exact": exact,
                        "gained": exact and not base_exact,
                        "lost": base_exact and not exact,
                    }
                )
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("species", choices=("human", "arabidopsis"))
    parser.add_argument("--transcripts", type=Path, required=True, help="Gzip FASTA.")
    parser.add_argument(
        "--annotation",
        type=Path,
        required=True,
        help="Gzip GENCODE GTF or plant CDS FASTA.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    inputs = [args.transcripts, args.annotation]
    validate_distinct_paths(inputs=inputs, outputs=[args.output])
    load = human_transcripts if args.species == "human" else plant_transcripts
    panel, excluded = select_transcripts(load(*inputs))
    if not panel:
        raise ValueError("No complete reference transcripts could be scored.")
    if args.workers == 1:
        groups = summarize(map(score_transcript, panel))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            groups = summarize(executor.map(score_transcript, panel, chunksize=64))
    atomic_write_json(
        args.output,
        {
            "species": args.species,
            "genetic_code": CODE,
            "inputs_sha256": {path.name: file_digest(path) for path in inputs},
            "excluded_after_coordinate_selection": excluded,
            "groups": groups,
        },
        indent=2,
    )


if __name__ == "__main__":
    main()

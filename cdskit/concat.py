"""Assemble an ordered codon supermatrix through a disk-backed transpose."""

import csv
import json
import re
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryFile

from Bio.SeqIO.FastaIO import SimpleFastaParser

from cdskit.output_bundle import new_directory
from cdskit.sequence_io import open_text
from cdskit.tsvio import read_tsv

_BAD_DNA = re.compile(rb"[^ACGTRYSWKMBDHVN?-]")
_GAP = re.compile(rb"-+")
_FILL_GAPS = bytes.maketrans(b"\0", b"-")


def _identifiers(rows, key, pattern):
    names = [row[key] for row in rows]
    if not names or any(not re.fullmatch(pattern, name) for name in names):
        raise ValueError(f"Missing or invalid {key} in manifest")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate {key} in manifest")
    return names


def _source_sites(row, parent, length):
    if not row.get("backtrim_json"):
        return range(1, length + 1)
    data = json.loads((parent / row["backtrim_json"]).read_text(encoding="utf-8"))
    sites = data["selected_sites"]
    source_length = data["source_column_count"]
    if not (
        data.get("status") == "success"
        and data.get("output_complete") is True
        and data.get("index_base") == 0
        and data.get("target_column_count") == length
        and type(source_length) is int
        and isinstance(sites, list)
        and len(sites) == length
        and all(type(site) is int and 0 <= site < source_length for site in sites)
        and all(a < b for a, b in pairwise(sites))
    ):
        raise ValueError(f"Invalid or incomplete backtrim mapping for {row['gene']}")
    return [site + 1 for site in sites]


def _store_gene(path, labels, spool, base, coverage, gene):
    width = 0
    present: dict[str, int] = {}
    cursor = spool.tell()
    with open_text(path) as handle:
        # SimpleFastaParser otherwise silently skips a preamble.
        first = handle.read(1)
        if first != ">":
            raise ValueError(f"Expected FASTA header at start of {path}")
        handle.seek(0)
        for header, sequence in SimpleFastaParser(handle):
            label = header.split(maxsplit=1)[0] if header.strip() else ""
            if label not in labels or label in present:
                raise ValueError(f"{path}: unknown or duplicate taxon {label!r}")
            seq = sequence.upper().encode("ascii")
            if not seq or len(seq) % 3 or _BAD_DNA.search(seq):
                raise ValueError(f"{path}: invalid codon alignment for {label}")
            if width and len(seq) != width:
                raise ValueError(f"{path}: unequal aligned lengths")
            if any(m.start() % 3 or m.end() % 3 for m in _GAP.finditer(seq)):
                raise ValueError(f"{path}: partial codon gap for {label}")
            width = len(seq)
            target = base + labels[label] * width
            if target != cursor:
                spool.seek(target)
            spool.write(seq)
            cursor = target + width
            present[label] = width - seq.count(b"-") - seq.count(b"?") - seq.count(b"N")
    if not present:
        raise ValueError(f"{path}: empty alignment")
    for label in labels:
        coverage.writerow(
            (gene, label, int(label in present), width // 3, present.get(label, 0))
        )
    spool.truncate(base + len(labels) * width)
    return width


def _transpose(spool, blocks, labels, handle):
    # Read a batch of taxa contiguously from each gene. Bound scratch by row width.
    step = max(1, (4 << 20) // sum(width for _, width in blocks))
    for first in range(0, len(labels), step):
        size = min(step, len(labels) - first)
        parts = []
        for base, width in blocks:
            spool.seek(base + first * width)
            parts.append(memoryview(spool.read(size * width).translate(_FILL_GAPS)))
        for i in range(size):
            handle.write(b">" + labels[first + i].encode("utf-8") + b"\n")
            handle.write(
                b"".join(
                    p[i * w : (i + 1) * w]
                    for p, (_, w) in zip(parts, blocks, strict=True)
                )
            )
            handle.write(b"\n")


def concat_main(args):
    labels = _identifiers(
        read_tsv(args.taxa_tsv, ["taxon"]), "taxon", r"[^\s>\x00-\x1f\x7f]+"
    )
    by_label = {label: i for i, label in enumerate(labels)}
    rows = read_tsv(args.genes_tsv, ["gene", "alignment"])
    _identifiers(rows, "gene", r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
    parent = Path(args.genes_tsv).parent
    blocks = []
    base = end = 0
    with (
        new_directory(args.out_dir) as out,
        TemporaryFile(dir=out) as spool,
        (out / "coverage.tsv").open("w", encoding="utf-8", newline="") as cov,
        (out / "sites.tsv").open("w", encoding="utf-8") as sites,
        (out / "partitions.txt").open("w", encoding="utf-8") as partitions,
        (out / "partitions.tsv").open("w", encoding="utf-8") as ranges,
    ):
        coverage = csv.writer(cov, delimiter="\t", lineterminator="\n")
        coverage.writerow(
            ("gene", "taxon", "present", "aligned_codons", "nonmissing_nt")
        )
        ranges.write("gene\tnt_start\tnt_end\tcodon_start\tcodon_end\n")
        sites.write("matrix_codon\tgene\talignment_codon\tsource_codon\n")
        for row in rows:
            gene = row["gene"]
            width = _store_gene(
                parent / row["alignment"], by_label, spool, base, coverage, gene
            )
            blocks.append((base, width))
            codon = end // 3
            for i, source in enumerate(_source_sites(row, parent, width // 3), 1):
                sites.write(f"{codon + i}\t{gene}\t{i}\t{source}\n")
            partitions.write(f"DNA, {gene} = {end + 1}-{end + width}\n")
            ranges.write(
                f"{gene}\t{end + 1}\t{end + width}\t{codon + 1}\t{(end + width) // 3}\n"
            )
            base += len(labels) * width
            end += width
        with (out / "alignment.fa").open("wb") as handle:
            _transpose(spool, blocks, labels, handle)

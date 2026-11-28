"""Run local conserved-domain searches and read compatible precomputed TSVs."""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from cdskit.alignment_io import (
    alignment_sequences,
    file_digest,
    read_fasta,
    sequence_digest,
    staged_outputs,
    write_fasta,
    write_table,
)
from cdskit.cliutil import resolve_threads
from cdskit.domain_database import resolve_database
from cdskit.sequence_io import open_text

RPS_FIELDS = (
    "qacc",
    "sacc",
    "pident",
    "length",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "sstart",
    "send",
    "evalue",
    "bitscore",
    "qlen",
    "slen",
    "stitle",
)
DOMAIN_FIELDS = {"qacc", "sacc", "qstart", "qend", "sstart", "send", "bitscore"}


@dataclass(frozen=True, slots=True)
class Domain:
    model: str
    start: int
    end: int
    first: int
    last: int
    score: float


def domain_fields(seq_type):
    return (*RPS_FIELDS, "qframe") if seq_type == "dna" else RPS_FIELDS


def tool_path(program):
    path = shutil.which(program)
    if path is None:
        raise FileNotFoundError(f"Required executable is unavailable: {program}")
    return path


def run_search(sequences, args, seq_type, directory, expected_database=None):
    program = "rpstblastn" if seq_type == "dna" else "rpsblast"
    executable = tool_path(program)
    database = resolve_database(args)
    if (
        expected_database is not None
        and database.identity["files"] != expected_database["files"]
    ):
        raise ValueError(
            "The database for missing queries differs from the precomputed domain manifest."
        )
    labels = {f"cdskit{i}": gene for i, gene in enumerate(sequences, 1)}
    input_file = directory / "domain-query.fa"
    write_fasta(
        input_file, {key: sequences[gene] for key, gene in labels.items()}, labels
    )
    result = directory / "domain-hits.tsv"
    fields = domain_fields(seq_type)
    command = [
        executable,
        "-query",
        str(input_file),
        "-db",
        str(database.prefix),
        "-out",
        str(result),
        "-evalue",
        str(args.evalue),
        "-outfmt",
        "6 " + " ".join(fields),
        "-num_threads",
        str(resolve_threads(args.threads)),
    ]
    if args.rps_mt_mode == "queries":
        command.extend(("-mt_mode", "1"))
    if seq_type == "dna":
        command.extend(("-query_gencode", str(args.codontable)))
    subprocess.run(command, check=True)
    return _search_rows(result, fields, labels, sequences, program), database


def _search_rows(result, fields, labels, sequences, program):
    seen = set()
    with result.open(encoding="utf-8") as handle:
        for line in handle:
            row = line.rstrip("\r\n").split("\t")
            if len(row) != len(fields) or row[0] not in labels:
                raise ValueError(f"Unexpected {program} output row: {line[:160]!r}")
            row[0] = labels[row[0]]
            if row[12] != str(len(sequences[row[0]])):
                raise ValueError(f"Unexpected {program} query length: {row[0]}")
            seen.add(row[0])
            yield row
    for gene in sequences:
        if gene not in seen:
            row = [""] * len(fields)
            row[0], row[12] = gene, len(sequences[gene])
            yield row


def _validate_rows(rows, seq_type, lengths):
    by_gene: dict[str, list[Domain] | None] = {}
    frame_hits: dict[str, dict[int, list[Domain]]] = {}
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Non-rectangular domain table.")
        gene = row["qacc"]
        if not gene:
            raise ValueError("Empty domain-table query ID.")
        if gene not in lengths:
            continue
        length = lengths[gene]
        if row.get("qlen") and int(row["qlen"]) != length:
            raise ValueError(f"Domain-table query length does not match FASTA: {gene}")
        if not row["sacc"]:
            if gene in by_gene:
                raise ValueError(f"Duplicate or conflicting no-hit row: {gene}")
            by_gene[gene] = None
            continue
        hits = by_gene.setdefault(gene, [])
        if hits is None:
            raise ValueError(f"Conflicting hit and no-hit rows: {gene}")
        a, b = int(row["qstart"]), int(row["qend"])
        first, last = sorted((int(row["sstart"]), int(row["send"])))
        score = float(row["bitscore"])
        if min(a, b, first) < 1 or max(a, b) > length or not math.isfinite(score):
            raise ValueError(f"Invalid domain coordinates or score for {gene}.")
        if seq_type == "dna":
            frame = int(row["qframe"])
            if frame not in (-3, -2, -1, 1, 2, 3) or (a > b) != (frame < 0):
                raise ValueError(f"Invalid translated search frame for {gene}.")
            if frame < 0:
                a, b = -a, -b
            frame_hits.setdefault(gene, {}).setdefault(frame, []).append(
                Domain(row["sacc"], a, b, first, last, score)
            )
        else:
            hits.append(Domain(row["sacc"], min(a, b), max(a, b), first, last, score))
    for gene, frames in frame_hits.items():
        by_gene[gene] = max(
            frames.values(), key=lambda group: max(hit.score for hit in group)
        )
    return {gene: hits or [] for gene, hits in by_gene.items()}


def read_domains(path, seq_type, sequences):
    with open_text(path) as handle:
        rows = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        required = DOMAIN_FIELDS | ({"qframe"} if seq_type == "dna" else set())
        header = rows.fieldnames or []
        if not required <= set(header) or len(set(header)) != len(header):
            raise ValueError(f"Missing or duplicate domain-search columns in {path}.")
        return _validate_rows(
            rows, seq_type, {key: len(seq) for key, seq in sequences.items()}
        )


def validate_domain_manifest(path, table, sequence_type, sequences, args):
    with open_text(path) as handle:
        manifest = json.load(handle)
    if (
        manifest["schema_version"] != 1
        or manifest["coordinate_unit"] != ("nt" if sequence_type == "dna" else "aa")
        or manifest["sequence_sha256"] != sequence_digest(sequences)
        or manifest["table_sha256"] != file_digest(table)
        or manifest["evalue"] != args.evalue
        or (sequence_type == "dna" and manifest["codon_table"] != args.codontable)
    ):
        raise ValueError(
            "The domain-search manifest does not match the current input and parameters."
        )
    return manifest["database"]


def annotate(sequences, args, seq_type, scratch, existing=None, expected_database=None):
    domains = dict(existing or {})
    missing = {
        gene: sequence for gene, sequence in sequences.items() if gene not in domains
    }
    if missing:
        if existing is not None and not args.search_missing_domains:
            raise ValueError(
                f"Domain table lacks {len(missing)} sequence(s); first missing ID: {next(iter(missing))}. "
                "Supply complete results or use --search_missing_domains."
            )
        rows, _ = run_search(missing, args, seq_type, scratch, expected_database)
        fields = domain_fields(seq_type)
        domains.update(
            _validate_rows(
                (dict(zip(fields, row, strict=True)) for row in rows),
                seq_type,
                {gene: len(sequence) for gene, sequence in missing.items()},
            )
        )
    return domains


def rpsblast_main(args):
    if args.manifest == "-":
        raise ValueError("--manifest must name a file.")
    if args.inseqformat != "fasta":
        raise ValueError("Domain search requires FASTA input.")
    if not math.isfinite(args.evalue) or args.evalue <= 0:
        raise ValueError("--evalue must be positive and finite.")
    sequence_type = "dna" if args.seq_type == "dna" else "aa"
    raw = read_fasta(args.seqfile)
    sequences = alignment_sequences(raw, args.seq_type, args.codontable, args.aa_file)
    inputs = (args.seqfile, args.aa_file, args.db_prefix)
    with staged_outputs(
        inputs, {"table": args.outfile, "manifest": args.manifest}
    ) as outputs:
        with tempfile.TemporaryDirectory(prefix="cdskit-rpsblast-") as temporary:
            rows, database = run_search(sequences, args, sequence_type, Path(temporary))
            table = (
                Path(temporary) / "domains.tsv"
                if outputs["table"] == "-"
                else outputs["table"]
            )
            write_table(
                table, domain_fields(sequence_type), rows, quoting=csv.QUOTE_NONE
            )
            if args.manifest:
                manifest = dict(
                    schema_version=1,
                    input_type=args.seq_type,
                    coordinate_unit="nt" if sequence_type == "dna" else "aa",
                    sequence_sha256=sequence_digest(sequences),
                    table_sha256=file_digest(table),
                    evalue=args.evalue,
                    codon_table=args.codontable,
                    database=database.identity,
                    program="rpstblastn" if sequence_type == "dna" else "rpsblast",
                )
                Path(outputs["manifest"]).write_text(
                    json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
                )
            if outputs["table"] == "-":
                with open_text(table) as handle:
                    shutil.copyfileobj(handle, sys.stdout)

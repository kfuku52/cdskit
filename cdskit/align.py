"""Align DNA, codons, or proteins with MAFFT, optionally using a query backbone."""

from __future__ import annotations

import csv
import gzip
import io
import math
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from itertools import compress
from pathlib import Path

from cdskit.alignment_io import (
    GAPS,
    alignment_sequences,
    read_fasta,
    staged_outputs,
    write_fasta,
    write_table,
)
from cdskit.backalign import backalign_sequence_strings
from cdskit.cliutil import resolve_threads
from cdskit.domain_search import (
    Domain,
    annotate,
    read_domains,
    tool_path,
    validate_domain_manifest,
)
from cdskit.sequence_io import open_text

RESIDUE_RUN = re.compile(r"[^.-]+")


def architecture(domains):
    """Collapse overlapping Pfam alternatives while retaining distinct domain copies."""
    chosen: list[Domain] = []
    for hit in sorted(
        domains, key=lambda h: (-h.score, h.start, h.end, h.model, h.first, h.last)
    ):
        if not any(
            2 * (min(hit.end, prev.end) - max(hit.start, prev.start) + 1)
            >= min(hit.end - hit.start + 1, prev.end - prev.start + 1)
            for prev in chosen
        ):
            chosen.append(hit)
    return sorted(chosen, key=lambda h: (h.start, h.end, h.model))


def architecture_match(query, candidate):
    """Count shared profile positions in the best ordered one-to-one domain match."""
    previous = [0] * (len(candidate) + 1)
    for q in query:
        current = [0]
        for i, hit in enumerate(candidate):
            overlap = 0
            if hit.model == q.model:
                overlap = max(0, min(hit.last, q.last) - max(hit.first, q.first) + 1)
            current.append(max(previous[i + 1], current[-1], previous[i] + overlap))
        previous = current
    return previous[-1]


def fasta_offsets(handle):
    """Index MAFFT's wrapped FASTA by record using byte offsets."""
    starts = []
    offset = 0
    previous = b"\n"
    while block := handle.read(4 * 1024 * 1024):
        if previous == b"\n" and block.startswith(b">"):
            starts.append(offset)
        cursor = block.find(b"\n>")
        while cursor != -1:
            starts.append(offset + cursor + 1)
            cursor = block.find(b"\n>", cursor + 2)
        offset += len(block)
        previous = block[-1:]

    records = {}
    for start, end in zip(starts, [*starts[1:], offset] if starts else [], strict=True):
        handle.seek(start)
        fields = handle.readline()[1:].split(None, 1)
        gene = fields[0].decode() if fields else ""
        if not gene or gene in records:
            raise ValueError(f"Expected unique, nonempty MAFFT IDs: {gene!r}")
        records[gene] = (start, end - start)
    return records


def occupied_columns(rows):
    """Mark positions with a residue in at least one backbone row."""
    width = len(next(iter(rows.values())))
    occupied = bytearray(width)
    for row in rows.values():
        if len(row) != width:
            raise ValueError("MAFFT produced an unaligned backbone row")
        for match in RESIDUE_RUN.finditer(row):
            start, end = match.span()
            occupied[start:end] = b"\1" * (end - start)
    return occupied


def write_column_map(original, final, column_map):
    old_columns = occupied_columns(original)
    new_columns = occupied_columns(final)
    if old_columns.count(1) != new_columns.count(1):
        raise ValueError("MAFFT changed the number of occupied backbone columns")
    for gene, old_row in original.items():
        if "".join(compress(old_row, old_columns)) != "".join(
            compress(final[gene], new_columns)
        ):
            raise ValueError(f"MAFFT changed the backbone alignment for {gene}")

    occupied = compress(range(1, len(old_columns) + 1), old_columns)
    with column_map.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("alignment_column", "backbone_column"))
        writer.writerows(
            (i, next(occupied) if present else "")
            for i, present in enumerate(new_columns, 1)
        )


def _annotations(path, field, convert):
    if path is None:
        return {}
    with open_text(path) as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        fields = rows.fieldnames or []
        if not {"gene_id", field} <= set(fields) or len(fields) != len(set(fields)):
            raise ValueError(f"Invalid annotation columns in {path}.")
        result = {}
        for row in rows:
            gene = row["gene_id"]
            if not gene or gene in result:
                raise ValueError(
                    f"Duplicate or empty annotation ID in {path}: {gene!r}"
                )
            result[gene] = convert(row[field])
    return result


def select_backbone(
    candidates, shared, query_domains, domains, species, priorities, size
):
    if len(shared) > size:
        raise ValueError(
            "--backbone_size is smaller than the number of matching query sequences."
        )
    profiles = {
        g: architecture(hits) for g, hits in sorted(query_domains.items()) if hits
    }
    profile_lengths = {
        g: sum(hit.last - hit.first + 1 for hit in hits) for g, hits in profiles.items()
    }
    models = {hit.model for hits in profiles.values() for hit in hits}
    evidence = {}
    for gene in candidates:
        hits = architecture(hit for hit in domains.get(gene, ()) if hit.model in models)
        score, match = (0.0, "")
        if hits:
            score, match = max(
                (architecture_match(profile, hits) / profile_lengths[ref], ref)
                for ref, profile in profiles.items()
            )
        evidence[gene] = (
            species.get(gene, ""),
            priorities.get(gene, 1),
            match if score else "",
            score,
        )
    ranked = sorted(
        candidates.keys() - shared,
        key=lambda g: (evidence[g][1], -evidence[g][3], not evidence[g][0], g),
    )
    selected = dict.fromkeys(sorted(shared), "query")
    represented = {evidence[g][0] for g in selected if evidence[g][0]}
    for g in ranked:
        if len(selected) >= size:
            break
        label = evidence[g][0]
        if label and label not in represented:
            selected[g] = "species_representative"
            represented.add(label)
    for g in ranked:
        if len(selected) >= size:
            break
        selected.setdefault(g, "additional_representative")
    return selected, evidence, len(profiles)


def _query_domains(args, candidates, query, seq_type, scratch, shared):
    # If a reference uses a candidate ID with different residues, search the
    # reference separately instead of lending it another sequence's hits.
    distinct = {g: seq for g, seq in query.items() if g not in shared}
    union = candidates | {g: seq for g, seq in distinct.items() if g not in candidates}
    existing = None
    expected_database = None
    if args.rpsblast_tsv:
        if args.domain_manifest:
            expected_database = validate_domain_manifest(
                args.domain_manifest, args.rpsblast_tsv, seq_type, candidates, args
            )
        existing = read_domains(args.rpsblast_tsv, seq_type, union)
    domains = annotate(union, args, seq_type, scratch, existing, expected_database)
    separate = {g: seq for g, seq in distinct.items() if g in candidates}
    if separate:
        alternate = annotate(
            separate,
            args,
            seq_type,
            scratch,
            {} if existing is not None else None,
            expected_database,
        )
    else:
        alternate = {}
    return domains, {g: alternate[g] if g in alternate else domains[g] for g in query}


def _run_mafft(executable, source, output, seq_type, threads, options):
    command = [
        executable,
        "--nuc" if seq_type == "dna" else "--amino",
        "--anysymbol",
        "--quiet",
        "--thread",
        str(threads),
        "--threadit",
        "0",
        *options,
        str(source),
    ]
    with output.open("wb") as handle:
        subprocess.run(command, stdout=handle, check=True)


@contextmanager
def _fasta_output(path):
    with path.open("wb") as raw:
        if path.suffix.lower() == ".gz":
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0
            ) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as text:
                    yield text
        else:
            with io.TextIOWrapper(raw, encoding="utf-8", newline="\n") as text:
                yield text


def _write_output(
    alignment, candidates, selected, destination, raw_codons, codon_table
):
    final_core = {}
    width = None
    with alignment.open("rb") as aligned, _fasta_output(destination) as handle:
        offsets = fasta_offsets(aligned)
        if offsets.keys() != candidates.keys():
            raise ValueError("MAFFT changed the sequence IDs.")
        for gene, original in candidates.items():
            start, length = offsets[gene]
            aligned.seek(start)
            data = aligned.read(length).partition(b"\n")[2]
            sequence = data.translate(None, b" \t\r\n").upper().decode("ascii")
            if width is None:
                width = len(sequence)
            if len(sequence) != width or sequence.translate(GAPS) != original:
                raise ValueError(
                    f"MAFFT changed residues or produced an unaligned row: {gene}"
                )
            if selected is not None and gene in selected:
                final_core[gene] = sequence
            if raw_codons is not None:
                sequence = backalign_sequence_strings(
                    raw_codons[gene], sequence, codon_table, gene, False
                )
                if (
                    len(sequence) != 3 * width
                    or sequence.translate(GAPS) != raw_codons[gene]
                ):
                    raise ValueError(
                        f"Codon alignment did not preserve all input codons: {gene}"
                    )
            handle.write(f">{gene}\n{sequence}\n")
    return final_core


def _backbone(args, candidates, target_type, scratch):
    if args.backbone_file:
        core = read_fasta(args.backbone_file, aligned=True)
        if any(candidates.get(g) != row.translate(GAPS) for g, row in core.items()):
            raise ValueError(
                "Precomputed backbone sequences must match candidate IDs and residues."
            )
        occupied_columns(core)
        return dict.fromkeys(core, "provided_backbone"), {}, 0, core
    query = alignment_sequences(
        read_fasta(args.query_file),
        args.query_type or args.seq_type,
        args.codontable,
        args.query_aa_file,
    )
    if args.seq_type == "dna" and not (args.search_translated or args.rpsblast_tsv):
        raise ValueError(
            "DNA domain guidance requires --search_translated or a precomputed domain table."
        )
    shared = {g for g, seq in query.items() if candidates.get(g) == seq}
    domains, query_domains = _query_domains(
        args, candidates, query, target_type, scratch, shared
    )
    species = _annotations(args.species_tsv, "species", str)
    priorities = _annotations(args.priority_tsv, "priority", int)
    selected, evidence, profiles = select_backbone(
        candidates,
        shared,
        query_domains,
        domains,
        species,
        priorities,
        args.backbone_size,
    )
    return selected, evidence, profiles, None


def _align_backbone(args, candidates, target_type, scratch):
    executable = tool_path("mafft")
    threads = resolve_threads(args.threads)
    selected, evidence, profiles, provided = _backbone(
        args, candidates, target_type, scratch
    )
    source, core, result = (
        scratch / name for name in ("backbone.fa", "backbone.aln.fa", "family.fa")
    )
    if provided is not None:
        write_fasta(core, provided, provided)
    else:
        write_fasta(source, candidates, selected)
        if len(selected) == 1:
            core = source
        else:
            options = (
                ["--localpair", "--maxiterate", "1000"]
                if len(selected) < len(candidates)
                else ["--auto"]
            )
            _run_mafft(executable, source, core, target_type, threads, options)
    if len(selected) < len(candidates):
        fragments = scratch / "fragments.fa"
        write_fasta(
            fragments, candidates, (gene for gene in candidates if gene not in selected)
        )
        _run_mafft(
            executable,
            core,
            result,
            target_type,
            threads,
            ["--auto", "--addfragments", str(fragments)],
        )
    else:
        result = core
    original = read_fasta(core, aligned=True)
    if original.keys() != selected.keys():
        raise ValueError("MAFFT changed the backbone IDs.")
    if any(candidates[g] != seq.translate(GAPS) for g, seq in original.items()):
        raise ValueError("MAFFT changed backbone residues.")
    return result, selected, evidence, profiles, original


def _write_report(path, candidates, selected, evidence, profiles):
    rank = {gene: i for i, gene in enumerate(selected, 1)}
    values = (evidence.get(g, ("", 1, "", 0.0)) for g in candidates)
    write_table(
        path,
        (
            "gene_id",
            "backbone_rank",
            "backbone_reason",
            "species",
            "priority",
            "best_domain_query",
            "ordered_domain_coverage",
            "query_domain_profiles",
        ),
        (
            (
                g,
                rank.get(g, ""),
                selected.get(g, "added"),
                species,
                priority,
                match,
                f"{score:.6f}",
                profiles,
            )
            for g, (species, priority, match, score) in zip(
                candidates, values, strict=True
            )
        ),
    )


def _validate_options(args):
    if args.inseqformat != "fasta" or args.outseqformat != "fasta":
        raise ValueError("Alignment requires FASTA input and output.")
    if not math.isfinite(args.evalue) or args.evalue <= 0:
        raise ValueError("--evalue must be positive and finite.")
    if args.query_type and (
        not args.query_file
        or (
            args.query_type != args.seq_type
            and (args.seq_type, args.query_type) != ("codon", "aa")
        )
    ):
        raise ValueError(
            "--query_type must match --seq_type, or be aa for a codon alignment."
        )
    if args.query_aa_file and not args.query_file:
        raise ValueError("--query_aa_file requires --query_file.")
    if args.backbone_file and any(
        (args.query_file, args.rpsblast_tsv, args.species_tsv, args.priority_tsv)
    ):
        raise ValueError(
            "--backbone_file supplies the selected alignment; omit reference and ranking inputs."
        )
    if args.domain_manifest and not args.rpsblast_tsv:
        raise ValueError("--domain_manifest requires --rpsblast_tsv.")
    if args.strategy == "query_backbone":
        if not args.query_file and not args.backbone_file:
            raise ValueError(
                "Query backbone alignment requires --query_file or --backbone_file."
            )
        if args.backbone_size < 2:
            raise ValueError("--backbone_size must be at least two.")
    elif any(
        (
            args.query_file,
            args.backbone_file,
            args.rpsblast_tsv,
            args.backbone_report,
            args.column_map,
            args.species_tsv,
            args.priority_tsv,
            args.search_translated,
            args.search_missing_domains,
        )
    ):
        raise ValueError("Backbone options require --strategy query_backbone.")


def align_main(args):
    _validate_options(args)
    raw = read_fasta(args.seqfile)
    target_type = "dna" if args.seq_type == "dna" else "aa"
    candidates = alignment_sequences(raw, args.seq_type, args.codontable, args.aa_file)
    raw_codons = raw if args.seq_type == "codon" else None
    inputs = (
        args.seqfile,
        args.aa_file,
        args.query_file,
        args.query_aa_file,
        args.rpsblast_tsv,
        args.domain_manifest,
        args.backbone_file,
        args.species_tsv,
        args.priority_tsv,
    )
    targets = dict(
        alignment=args.outfile, report=args.backbone_report, columns=args.column_map
    )
    with (
        staged_outputs(inputs, targets) as staged,
        tempfile.TemporaryDirectory(prefix="cdskit-align-") as path,
    ):
        scratch = Path(path)
        files = {
            key: scratch / f"{key}.out" if value == "-" else Path(value)
            for key, value in staged.items()
        }
        if args.strategy == "query_backbone":
            msa, selected, evidence, profiles, original = _align_backbone(
                args, candidates, target_type, scratch
            )
        else:
            selected = None
            source, msa = scratch / "input.fa", scratch / "family.fa"
            if len(candidates) == 1:
                write_fasta(msa, candidates, candidates)
            else:
                write_fasta(source, candidates, candidates)
                _run_mafft(
                    tool_path("mafft"),
                    source,
                    msa,
                    target_type,
                    resolve_threads(args.threads),
                    ["--auto"],
                )
        final = _write_output(
            msa, candidates, selected, files["alignment"], raw_codons, args.codontable
        )
        if selected is not None:
            write_column_map(
                original, final, files.get("columns", scratch / "column-map.tsv")
            )
            if "report" in files:
                _write_report(files["report"], candidates, selected, evidence, profiles)
            print(
                f"MAFFT backbone: {len(selected)} representatives, {len(candidates) - len(selected)} added, "
                f"{profiles} queries with annotated domains",
                file=sys.stderr,
            )
        for key, value in staged.items():
            if value == "-":
                with files[key].open(encoding="utf-8") as handle:
                    shutil.copyfileobj(handle, sys.stdout)

"""Stream annotated GenBank CDS copies and one unambiguous sequence per locus."""

import csv
import hashlib
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

from Bio import SeqIO
from Bio.Data import CodonTable
from Bio.Seq import Seq, UndefinedSequenceError
from Bio.SeqFeature import ExactPosition, UnknownPosition

from cdskit.output_bundle import new_directory
from cdskit.sequence_io import open_text

FIELDS = (
    "feature_id",
    "accession_version",
    "organism",
    "taxid",
    "source_status",
    "specimen_voucher",
    "isolate",
    "population",
    "ecotype",
    "strain",
    "gene",
    "locus_tag",
    "protein_id",
    "strand",
    "location",
    "codon_start",
    "transl_table",
    "translation_check",
    "copy_status",
    "decision",
    "reason",
    "gene_file",
)


def values(qualifiers, key):
    return "|".join(qualifiers.get(key, ()))


def _source_for(feature, sources):
    loc = feature.location
    if not sources or loc is None:
        return {}, "missing"
    if any(
        isinstance(boundary, UnknownPosition)
        for p in loc.parts
        for boundary in (p.start, p.end)
    ):
        return {}, "unknown_coordinates"
    hits = [
        s
        for s in sources
        if s.location is not None
        and all(
            any(
                part.ref == region.ref
                and part.ref_db == region.ref_db
                and not isinstance(region.start, UnknownPosition)
                and not isinstance(region.end, UnknownPosition)
                and region.start <= part.start
                and part.end <= region.end
                for region in s.location.parts
            )
            for part in loc.parts
        )
    ]
    if len(hits) == 1:
        return hits[0].qualifiers, "matched"
    return {}, "ambiguous" if hits else "missing"


def inspect_feature(record, feature, number, sources):
    source, source_status = _source_for(feature, sources)
    q = feature.qualifiers
    gene = values(q, "gene")
    tag = values(q, "locus_tag")
    loc = feature.location
    row = dict(
        feature_id=f"{record.id}.cds{number}",
        accession_version=record.id,
        organism=values(source, "organism") or record.annotations.get("organism", ""),
        source_status=source_status,
        taxid="|".join(
            x[6:] for x in source.get("db_xref", ()) if x.startswith("taxon:")
        ),
        specimen_voucher=values(source, "specimen_voucher"),
        isolate=values(source, "isolate"),
        population=values(source, "pop_variant"),
        ecotype=values(source, "ecotype"),
        strain=values(source, "strain"),
        gene=gene,
        locus_tag=tag,
        protein_id=values(q, "protein_id"),
        strand=loc.strand if loc else "",
        location=str(loc) if loc is not None else "",
        codon_start=values(q, "codon_start") or "1",
        transl_table=values(q, "transl_table") or "1",
        translation_check="unavailable",
        gene_file="",
        copy_status="single",
    )
    reasons = []
    if "pseudo" in q or "pseudogene" in q:
        reasons.append("pseudogene")
    if not gene and not tag:
        reasons.append("missing_gene_identifier")
    raw = ""
    if loc is None or any(p.ref or p.ref_db for p in loc.parts):
        reasons.append("unavailable_location")
    elif any(
        isinstance(p.start, UnknownPosition)
        or isinstance(p.end, UnknownPosition)
        or p.start < 0
        or p.end > len(record.seq)
        for p in loc.parts
    ):
        reasons.append("unavailable_location")
    else:
        try:
            raw = str(feature.extract(record.seq)).upper()
        except (TypeError, ValueError, UndefinedSequenceError):
            reasons.append("unavailable_sequence")
    seq = ""
    try:
        offset = int(row["codon_start"]) - 1
        if offset not in (0, 1, 2):
            raise ValueError
        table = CodonTable.unambiguous_dna_by_id[int(row["transl_table"])]
    except (ValueError, KeyError):
        reasons.append("invalid_frame_or_table")
    else:
        seq = raw[offset:]
        if not seq or len(seq) % 3:
            reasons.append("incomplete_codon")
        elif re.search(r"[^ACGTRYSWKMBDHVN]", seq) or set(seq) == {"N"}:
            reasons.append("invalid_or_missing_dna")
        elif "transl_except" in q or "exception" in q or "ribosomal_slippage" in q:
            reasons.append("translation_exception")
        else:
            try:
                aa = str(Seq(seq).translate(table=table.id))
            except (ValueError, CodonTable.TranslationError):
                reasons.append("invalid_codon")
            else:
                # Alternative starts are methionine only for complete 5' ends.
                first = loc.parts[0]
                boundary = first.end if first.strand == -1 else first.start
                if (
                    offset == 0
                    and isinstance(boundary, ExactPosition)
                    and seq[:3] in table.start_codons
                ):
                    aa = "M" + aa[1:]
                protein = aa.removesuffix("*")
                if "*" in protein:
                    reasons.append("internal_stop")
                if "translation" in q:
                    annotated = "".join(values(q, "translation").split()).removesuffix(
                        "*"
                    )
                    row["translation_check"] = (
                        "match" if protein == annotated else "mismatch"
                    )
                    if protein != annotated:
                        reasons.append("translation_mismatch")
    return row, gene or tag, raw, seq, reasons


def export_cds(args):
    if (
        args.outfile != "-"
        or args.outseqformat != "fasta"
        or not args.extract_cds
        or args.list_seqname_keys
        or args.seqnamefmt != "organism_accessions"
    ):
        raise ValueError(
            "--all_cds_dir uses its own FASTA names, format and feature table"
        )
    source = nullcontext(sys.stdin) if args.seqfile == "-" else open_text(args.seqfile)
    with source as stream, new_directory(args.all_cds_dir) as out:
        (out / "copies").mkdir()
        (out / "genes").mkdir()
        paths = {}
        buffers: dict[Path, list[str]] = defaultdict(list)
        pending = 0

        def flush():
            for path, entries in buffers.items():
                with path.open("a", encoding="utf-8") as handle:
                    handle.writelines(entries)
            buffers.clear()

        def append(directory, filename, label, seq):
            nonlocal pending
            entry = f">{label}\n{seq}\n"
            buffers[out / directory / filename].append(entry)
            pending += len(entry)
            if pending >= 2 << 20:
                flush()
                pending = 0

        seen = set()
        with (out / "features.tsv").open("w", encoding="utf-8", newline="") as audit:
            writer = csv.DictWriter(audit, FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for record in SeqIO.parse(stream, "genbank"):
                if (
                    record.id in seen
                    or not record.id
                    or any(c.isspace() for c in record.id)
                ):
                    raise ValueError(f"Duplicate or invalid record ID: {record.id}")
                seen.add(record.id)
                if "protein" in record.annotations.get("molecule_type", "").lower():
                    raise ValueError(f"Expected nucleotide record: {record.id}")
                sources = [f for f in record.features if f.type == "source"]
                groups = defaultdict(list)
                for number, f in enumerate(record.features, 1):
                    if f.type == "CDS":
                        item = inspect_feature(record, f, number, sources)
                        groups[item[1]].append(item)
                for gene, items in groups.items():
                    if gene not in paths:
                        stem = re.sub(r"[^A-Za-z0-9_-]", "_", gene)[:80] or "unnamed"
                        paths[gene] = (
                            f"{stem}.{hashlib.sha256(gene.encode()).hexdigest()[:16]}.fa"
                        )
                    filename = paths[gene]
                    eligible = {
                        (seq, row["transl_table"])
                        for row, _, _, seq, why in items
                        if not why
                    }
                    copies = {raw for _, _, raw, _, _ in items}
                    emitted = False
                    for row, _, raw, seq, reasons in items:
                        row["gene_file"] = filename
                        if raw:
                            append("copies", filename, row["feature_id"], raw)
                        if len(items) > 1:
                            row["copy_status"] = (
                                "identical"
                                if len(copies) == 1 and "" not in copies
                                else "discordant_or_unknown"
                            )
                        if not reasons and len(eligible) > 1:
                            reasons.append("discordant_copies")
                        if not reasons and emitted:
                            reasons.append("identical_duplicate")
                        row["decision"] = "excluded" if reasons else "included"
                        row["reason"] = "|".join(reasons)
                        writer.writerow(row)
                        if not reasons:
                            append("genes", filename, record.id, seq)
                            emitted = True
            if not seen:
                raise ValueError("No GenBank records found")
        flush()

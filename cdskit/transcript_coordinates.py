"""Project transcript ends onto selected CDSs and choose the longest UTR per side.

Coordinates are zero-based half-open. Source exon paths and outer transcript
ends stay fixed; a boundary change never translates a UTR along the genome.
"""

from bisect import bisect_right
from collections import defaultdict
from operator import itemgetter
from typing import Any, NamedTuple

from Bio.Seq import Seq
from cdskit.gff_attributes import parse_gff_attributes
from cdskit.sequence_io import open_text

UTR_FEATURE_TYPES = {"5": "five_prime_UTR", "3": "three_prime_UTR"}
_CHILD_KINDS = {
    kind: kind
    for kind in (
        "exon",
        "cds",
        "utr",
        "five_prime_utr",
        "three_prime_utr",
        "5utr",
        "3utr",
    )
}
_MODEL_CHILD_KEYS = frozenset(("ID", "Parent", "transcript_id", "gene_id", "coge_fid"))
_BLOCK_START = itemgetter(0)


class _TranscriptChild(NamedTuple):
    kind: str
    seqid: str
    strand: str
    start: str
    end: str
    phase: str
    gene: str
    coge: tuple[tuple[str, ...], tuple[str, ...]] | None


class TranscriptGroups:
    """Place generated children before the source GFF group's closing directive."""

    def __init__(self, transcripts):
        self.transcripts = transcripts
        self.current, self.closed, self.placements = set(), set(), {}

    def add(self, attrs):
        self.current.update(
            tid
            for key in ("ID", "Parent")
            for tid in attrs.get(key, ())
            if tid in self.transcripts
        )

    def close(self, position):
        if self.current & self.closed:
            raise ValueError("Transcript spans closed GFF groups")
        if position is not None:
            self.placements.update((tid, position) for tid in self.current)
        self.closed.update(self.current)
        self.current.clear()


def _utr_low_side(model, side):
    if side not in UTR_FEATURE_TYPES or model["strand"] not in {"+", "-"}:
        raise ValueError("UTR coordinates require side 5 or 3 and a resolved strand")
    return (side == "5") == (model["strand"] == "+")


def coding_boundary(model, side):
    """Return the CDS endpoint, including a verified separate terminal stop."""
    low = _utr_low_side(model, side)
    if side == "3" and "annotated_stop_boundary" in model:
        return model["annotated_stop_boundary"]
    return min(b[0] for b in model["cds"]) if low else max(b[1] for b in model["cds"])


def terminal_utr_blocks(model, side, *, boundary=None):
    """Clip donor exons outside a CDS endpoint, without extending or selecting ends.

    Native export uses the model's own CDS endpoint. Representative completion
    may supply its selected CDS endpoint after checking donor eligibility.
    """
    low = _utr_low_side(model, side)
    boundary = coding_boundary(model, side) if boundary is None else boundary
    return (
        [[a, min(b, boundary)] for a, b in model["exons"] if a < boundary]
        if low
        else [[max(a, boundary), b] for a, b in model["exons"] if b > boundary]
    )


def exons_within_reference(model, length):
    """Use the same genome-bound check for native export and final projection."""
    return (
        length is not None
        and bool(model["exons"])
        and all(0 <= a < b <= length for a, b in model["exons"])
    )


def merge_blocks(blocks):
    result: list[list[int]] = []
    for a, b in sorted(set(tuple(block[:2]) for block in blocks)):
        if a < 0 or b <= a:
            raise ValueError("Invalid transcript interval")
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def _record_separate_stop(model, rows):
    """Record a separate stop when it covers the next three exonic bases."""
    if not rows or model.get("problem"):
        return
    annotated = set()
    for seqid, strand, start_text, end_text in rows:
        if (seqid, strand) != (model["seqid"], model["strand"]):
            return
        try:
            start, end = int(start_text) - 1, int(end_text)
        except ValueError:
            return
        if not 0 <= start < end:
            return
        annotated.add((start, end))
    if sum(end - start for start, end in annotated) != 3:
        return

    low = _utr_low_side(model, "3")
    after_cds = terminal_utr_blocks(model, "3")
    expected = []
    remaining = 3
    for start, end in reversed(after_cds) if low else after_cds:
        length = min(remaining, end - start)
        expected.append([end - length, end] if low else [start, start + length])
        remaining -= length
        if remaining == 0:
            break
    if not remaining and sorted(expected) == merge_blocks(annotated):
        model["annotated_stop_boundary"] = expected[-1][0 if low else 1]


def _coge_terminal_segments(model, rows):
    """Recognize CoGe's separate coding and terminal noncoding segments.

    This is a source-specific interpretation, not repair of arbitrary incomplete
    exon paths. CoGe feature identity, unique Parent links and declared RNA bounds
    must independently support the union. Internal or overlapping segments fail.
    """
    fid = model["attrs"].get("coge_fid", ())
    if (
        model.get("problem")
        or model.get("source", "").lower() != "coge"
        or len(fid) != 1
        or not fid[0].isdecimal()
        or not model["exons"]
        or model["attrs"].get("ID") != (model["id"],)
        or model["attrs"].get("Parent") != (model["gene"],)
    ):
        return None
    low, high = model["bounds"]
    coding_low, coding_high = model["cds"][0][0], model["cds"][-1][1]
    for row in rows:
        if row.coge is None:
            return None
        parents, feature = row.coge
        if parents != (model["id"],) or len(feature) != 1 or not feature[0].isdecimal():
            return None
        if row.kind == "exon" and feature != fid:
            return None
    if any(not (b <= coding_low or a >= coding_high) for a, b in model["exons"]):
        return None
    blocks = [*model["exons"], *[b[:2] for b in model["cds"]], *model["utr"]]
    if any(not low <= a < b <= high for a, b in blocks):
        return None
    result = merge_blocks(blocks)
    return result if (result[0][0], result[-1][1]) == (low, high) else None


def read_models(text, *, strict=True):
    """Read explicit GFF3 or GTF transcript relationships, including shared rows."""
    return read_model_rows(text.splitlines(), strict=strict)


def _contained_blocks(blocks, exons):
    for block in blocks:
        index = bisect_right(exons, block[0], key=_BLOCK_START) - 1
        if index < 0 or block[1] > exons[index][1]:
            return False
    return True


def read_model_rows(lines, *, strict=True):
    """Read a line iterator without retaining the complete annotation text."""
    models: dict[str, Any] = {}
    children: dict[str, list[_TranscriptChild]] = defaultdict(list)
    genes: set[str] = set()
    shared_names: dict[str, str] = {}
    stop_rows = defaultdict(list)
    for line in lines:
        line = line.rstrip("\r\n")
        if line == "##FASTA":
            break
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) != 9:
            raise ValueError("Invalid transcript GFF/GTF row")
        kind = f[2].lower()
        is_transcript = kind.endswith(("rna", "transcript"))
        attrs = parse_gff_attributes(
            f[8], keys=None if is_transcript else _MODEL_CHILD_KEYS
        )
        if kind.endswith("gene"):
            genes.add(next(iter(attrs.get("ID", ())), ""))
        if is_transcript:
            tid = next(iter(attrs.get("ID", ()) or attrs.get("transcript_id", ())), "")
            parents = attrs.get("Parent", ()) or attrs.get("gene_id", ())
            # Standalone noncoding RNAs need no coding-locus owner. Validate
            # ownership only when this transcript actually has CDS children.
            if not tid:
                continue
            if tid in models:
                models[tid]["problem"] = "duplicate_transcript_identity"
                continue
            models[tid] = dict(
                id=tid,
                gene=parents[0] if len(parents) == 1 else "",
                seqid=f[0],
                strand=f[6],
                cds=[],
                exons=[],
                utr=[],
                attrs=attrs,
                source=f[1],
                bounds=(int(f[3]) - 1, int(f[4])),
            )
        elif kind == "stop_codon":
            owners = attrs.get("Parent", ()) or attrs.get("transcript_id", ())
            for tid in owners:
                stop_rows[tid].append((f[0], f[6], f[3], f[4]))
        elif kind in _CHILD_KINDS:
            parents = attrs.get("Parent", ())
            gene = next(iter(attrs.get("gene_id", ())), "")
            row = _TranscriptChild(
                _CHILD_KINDS[kind],
                shared_names.setdefault(f[0], f[0]),
                f[6],
                f[3],
                f[4],
                f[7],
                shared_names.setdefault(gene, gene),
                (parents, attrs.get("coge_fid", ()))
                if f[1].lower() == "coge"
                else None,
            )
            for tid in parents or attrs.get("transcript_id", ()):
                children[tid].append(row)
    # Release raw child fields as transcripts are finalized, instead of keeping
    # both every parsed feature and every completed model alive until return.
    for tid in list(children):
        rows = children.pop(tid)
        stops = stop_rows.pop(tid, ())
        if not any(row.kind == "cds" for row in rows):
            continue
        if tid not in models:
            gene = rows[0].gene or (tid if tid in genes else "")
            if not gene:
                raise ValueError("Unresolved transcript parent: " + tid)
            models[tid] = dict(
                id=tid,
                gene=gene,
                seqid=rows[0].seqid,
                strand=rows[0].strand,
                cds=[],
                exons=[],
                utr=[],
                attrs={},
            )
        model = models[tid]
        if not model["gene"]:
            model["problem"] = "unresolved_or_ambiguous_gene_owner"
        if strict and model.get("problem"):
            raise ValueError(
                "Transcript requires unique identity and one gene owner: " + tid
            )
        for row in rows:
            if (row.seqid, row.strand) != (
                model["seqid"],
                model["strand"],
            ) or row.strand not in {
                "+",
                "-",
            }:
                if strict:
                    raise ValueError("Transcript features disagree on contig or strand")
                model["problem"] = "multiple_contigs_or_strands"
                continue
            block = [int(row.start) - 1, int(row.end)]
            if block[0] < 0 or block[1] <= block[0]:
                raise ValueError("Invalid transcript coordinates")
            if row.kind == "cds":
                model["cds"].append(
                    [*block, int(row.phase) if row.phase in {"0", "1", "2"} else None]
                )
            else:
                model["exons" if row.kind == "exon" else "utr"].append(block)
        model["cds"] = sorted(
            set(tuple(b) for b in model["cds"]),
            key=lambda b: (b[0], b[1], -1 if b[2] is None else b[2]),
        )
        if any(
            a[1] > b[0] for a, b in zip(model["cds"], model["cds"][1:], strict=False)
        ):
            if strict:
                raise ValueError("Overlapping transcript CDS blocks")
            model["problem"] = "overlapping_cds"
        model["exons"] = merge_blocks(model["exons"] or [*model["cds"], *model["utr"]])
        cds_contained = _contained_blocks(model["cds"], model["exons"])
        if not cds_contained:
            interpreted = _coge_terminal_segments(model, rows)
            if interpreted is not None:
                model["declared_exons"] = model["exons"]
                model["exons"] = interpreted
                model["exon_path_basis"] = (
                    "coge_coding_plus_terminal_noncoding_segments"
                )
                cds_contained = _contained_blocks(model["cds"], model["exons"])
        if not cds_contained:
            if strict:
                raise ValueError("CDS is not contained in its transcript exons")
            model["problem"] = "cds_outside_exons"
        if not _contained_blocks(model["utr"], model["exons"]):
            if strict:
                raise ValueError("UTR is not contained in its transcript exons")
            model["problem"] = "utr_outside_exons"
        _record_separate_stop(model, stops)
    return {tid: m for tid, m in models.items() if m["cds"]}


def load_models(path):
    with open_text(path, "rt") as stream:
        return read_model_rows(stream)


def project(donor, selected, side, genome, *, reference_lengths=None):
    """Retain donor exons outside selected CDS; audit any new terminal splice."""
    if (donor["seqid"], donor["strand"]) != (selected["seqid"], selected["strand"]):
        raise ValueError("UTR projection requires the same contig and strand")
    if reference_lengths is None:
        length = (
            genome.get_reference_length(donor["seqid"])
            if donor["seqid"] in genome.references
            else None
        )
    else:
        length = reference_lengths.get(donor["seqid"])
    if not exons_within_reference(donor, length):
        return dict(
            blocks=[],
            length=0,
            reason="source_outside_genome",
            boundary_shift=0,
            new_junction=False,
        )
    low = _utr_low_side(selected, side)
    old_boundary = coding_boundary(donor, side)
    boundary = coding_boundary(selected, side)
    outer = donor["exons"][0][0] if low else donor["exons"][-1][1]
    if old_boundary <= outer if low else old_boundary >= outer:
        return dict(
            blocks=[],
            length=0,
            reason="no_source_utr",
            boundary_shift=0,
            new_junction=False,
        )
    blocks = terminal_utr_blocks(donor, side, boundary=boundary)
    reason, new_junction, motif, canonical = "projected", False, None, None
    if not blocks:
        reason = "cds_reaches_transcript_end"
    else:
        edge = blocks[-1][1] if low else blocks[0][0]
        if edge != boundary:
            # The CDS boundary lies in a donor intron or outside its exon path.
            # This is the requested coordinate projection, including novel
            # terminal connections. Record their motifs without imposing an
            # additional UTR-loss gate or claiming biological splice support.
            a, b = sorted((edge, boundary))
            intron = (
                genome.fetch(selected["seqid"], a, b)
                if b - a < 4
                else genome.fetch(selected["seqid"], a, a + 2)
                + genome.fetch(selected["seqid"], b - 2, b)
            ).upper()
            if selected["strand"] == "-":
                intron = str(Seq(intron).reverse_complement())
            new_junction, motif = True, intron[:2] + "-" + intron[-2:]
            canonical = b - a >= 4 and (intron[:2], intron[-2:]) in {
                ("GT", "AG"),
                ("GC", "AG"),
                ("AT", "AC"),
            }
    return dict(
        blocks=blocks,
        length=sum(b - a for a, b in blocks),
        reason=reason,
        boundary_shift=boundary - old_boundary,
        new_junction=new_junction,
        junction_motif=motif,
        junction_canonical=canonical,
    )

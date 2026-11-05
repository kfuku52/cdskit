"""Project transcript ends onto selected CDSs and choose the longest UTR per side.

Coordinates are zero-based half-open. Source exon paths and outer transcript
ends stay fixed; a boundary change never translates a UTR along the genome.
"""

from collections import defaultdict
from typing import Any

from Bio.Seq import Seq
from cdskit.gff_attributes import parse_gff_attributes
from cdskit.sequence_io import open_text

UTR_FEATURE_TYPES = {"5": "five_prime_UTR", "3": "three_prime_UTR"}


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
    """Return the unchanged zero-based CDS endpoint for a transcript side."""
    return (
        min(b[0] for b in model["cds"])
        if _utr_low_side(model, side)
        else max(b[1] for b in model["cds"])
    )


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
    for kind, fields, attrs in rows:
        if (
            fields[1].lower() != "coge"
            or attrs.get("Parent") != (model["id"],)
            or len(attrs.get("coge_fid", ())) != 1
            or not attrs["coge_fid"][0].isdecimal()
        ):
            return None
        if kind == "exon" and attrs["coge_fid"] != fid:
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


def read_model_rows(lines, *, strict=True):
    """Read a line iterator without retaining the complete annotation text."""
    models: dict[str, Any] = {}
    children, genes = defaultdict(list), set()
    for line in lines:
        line = line.rstrip("\r\n")
        if line == "##FASTA":
            break
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) != 9:
            raise ValueError("Invalid transcript GFF/GTF row")
        attrs = parse_gff_attributes(f[8])

        def scalar(key, attrs=attrs):
            return next(iter(attrs.get(key, ())), "")

        kind = f[2].lower()
        if kind.endswith("gene"):
            genes.add(scalar("ID"))
        if kind.endswith(("rna", "transcript")):
            tid = scalar("ID") or scalar("transcript_id")
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
        elif kind in {
            "exon",
            "cds",
            "utr",
            "five_prime_utr",
            "three_prime_utr",
            "5utr",
            "3utr",
        }:
            owners = attrs.get("Parent", ()) or attrs.get("transcript_id", ())
            # Child IDs and free-form attributes are not model geometry. Keep
            # only ownership and the CoGe interpreter's explicit source proof.
            # Original GFF rows are preserved separately by each writer.
            attrs = {
                key: attrs[key]
                for key in ("Parent", "gene_id", "coge_fid")
                if key in attrs
            }
            f[8] = ""
            for tid in owners:
                children[tid].append((kind, f, attrs))
    # Release raw child fields as transcripts are finalized, instead of keeping
    # both every parsed feature and every completed model alive until return.
    for tid in list(children):
        rows = children.pop(tid)
        if not any(kind == "cds" for kind, _, _ in rows):
            continue
        if tid not in models:
            gene = next(iter(rows[0][2].get("gene_id", ())), "") or (
                tid if tid in genes else ""
            )
            if not gene:
                raise ValueError("Unresolved transcript parent: " + tid)
            models[tid] = dict(
                id=tid,
                gene=gene,
                seqid=rows[0][1][0],
                strand=rows[0][1][6],
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
        for kind, f, _ in rows:
            if (f[0], f[6]) != (model["seqid"], model["strand"]) or f[6] not in {
                "+",
                "-",
            }:
                if strict:
                    raise ValueError("Transcript features disagree on contig or strand")
                model["problem"] = "multiple_contigs_or_strands"
                continue
            block = [int(f[3]) - 1, int(f[4])]
            if block[0] < 0 or block[1] <= block[0]:
                raise ValueError("Invalid transcript coordinates")
            if kind == "cds":
                model["cds"].append(
                    [*block, int(f[7]) if f[7] in {"0", "1", "2"} else None]
                )
            else:
                model["exons" if kind == "exon" else "utr"].append(block)
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
        if any(
            not any(x <= a < b <= y for x, y in model["exons"])
            for a, b, _ in model["cds"]
        ):
            interpreted = _coge_terminal_segments(model, rows)
            if interpreted is not None:
                model["declared_exons"] = model["exons"]
                model["exons"] = interpreted
                model["exon_path_basis"] = (
                    "coge_coding_plus_terminal_noncoding_segments"
                )
        for a, b, _ in model["cds"]:
            if not any(x <= a < b <= y for x, y in model["exons"]):
                if strict:
                    raise ValueError("CDS is not contained in its transcript exons")
                model["problem"] = "cds_outside_exons"
        if any(
            not any(x <= a < b <= y for x, y in model["exons"]) for a, b in model["utr"]
        ):
            if strict:
                raise ValueError("UTR is not contained in its transcript exons")
            model["problem"] = "utr_outside_exons"
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

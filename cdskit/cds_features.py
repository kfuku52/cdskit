"""CDS phase carry and strand-aware reconstruction over indexed references."""

import sys
from Bio.Seq import Seq


def phase_conventions(blocks):
    """Distinguish GFF3 skip-count phases from complementary frame fields."""
    if not blocks or any(row["phase"] not in {"0", "1", "2"} for row in blocks):
        return set()
    first, cumulative = int(blocks[0]["phase"]), 0
    conventions = {"gff3", "complementary"}
    for row in blocks:
        phase = int(row["phase"])
        if phase != (first - cumulative) % 3:
            conventions.discard("gff3")
        if phase != (first + cumulative) % 3:
            conventions.discard("complementary")
        cumulative += row["end"] - row["start"]
    return conventions


def reconstruct_sequence(blocks, seqid, strand, genome):
    """Fetch biological CDS without per-exon phase trimming."""
    if strand not in {"+", "-"}:
        raise ValueError("Unknown CDS contig or strand: " + seqid)
    # FastaFile already maintains a reference-name hash index. Preserve the
    # original reference-sequence protocol without importing a new dependency
    # for callers supplying a different genome implementation.
    native_fasta = getattr(sys.modules.get("pysam"), "FastaFile", None)
    references = (
        genome
        if type(genome) is native_fasta and type(seqid) is str
        else genome.references
    )
    if seqid not in references:
        raise ValueError("Unknown CDS contig or strand: " + seqid)
    pieces = []
    for start, end, _phase in blocks:
        if start < 0 or end <= start or end > genome.get_reference_length(seqid):
            raise ValueError("CDS outside genome bounds: " + seqid)
        piece = genome.fetch(seqid, start, end).upper()
        pieces.append(str(Seq(piece).reverse_complement()) if strand == "-" else piece)
    return "".join(pieces)

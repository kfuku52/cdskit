"""Protein alignment coverage, identity and phase-aware junction similarity."""

from dataclasses import dataclass
import math
from Bio.Align import PairwiseAligner


@dataclass(frozen=True)
class PairScore:
    score: float
    identity: float
    coverage_a: float
    coverage_b: float
    junction_similarity: float
    aligned_residues: int
    internal_loss_a: bool
    internal_loss_b: bool
    bounded: bool = False

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def protein_aligner(
    *, match_score=2.0, mismatch_score=-1.0, open_gap_score=-5.0, extend_gap_score=-0.5
) -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_score
    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score
    return aligner


def project_junctions(
    junctions: tuple, segments: list, protein_length: int, columns: int
) -> set:
    """Project only requested boundaries through monotone alignment segments."""
    positions, index = set(), 0
    for residue, phase in sorted(junctions):
        if residue == protein_length:
            positions.add((columns, phase))
        elif 0 <= residue < protein_length:
            while index < len(segments) and residue >= segments[index][1]:
                index += 1
            if index < len(segments):
                start, end, column = segments[index]
                if start <= residue < end:
                    positions.add((column + residue - start, phase))
    return positions


def _gap_scores_at_most(aligner, maximum_open):
    if hasattr(aligner, "open_internal_insertion_score"):
        names = (
            (f"{mode}_{end}_{kind}_score", mode)
            for mode in ("open", "extend")
            for end in ("left", "internal", "right")
            for kind in ("insertion", "deletion")
        )
    else:  # Biopython 1.80-1.85 use the original target/query names.
        names = (
            (f"{kind}_{end}_{mode}_gap_score", mode)
            for mode in ("open", "extend")
            for end in ("left", "internal", "right")
            for kind in ("target", "query")
        )
    return all(
        getattr(aligner, name) <= (maximum_open if mode == "open" else 0.0)
        for name, mode in names
    )


def _identical_alignment_is_optimal(aligner):
    """Only bypass a caller's aligner when its scoring proves the shortcut."""
    try:
        if (
            aligner.substitution_matrix is not None
            or aligner.wildcard is not None
            or not math.isfinite(aligner.match_score)
            or aligner.match_score <= 0
            or aligner.match_score < aligner.mismatch_score
        ):
            return False
        return _gap_scores_at_most(aligner, 0.0)
    except (AttributeError, ValueError):
        return False


def _proven_ungapped_match_count(protein_a, protein_b, aligner):
    if (
        len(protein_a) != len(protein_b)
        or not protein_a.isascii()
        or not protein_b.isascii()
    ):
        return None
    if protein_a == protein_b:
        if _identical_alignment_is_optimal(aligner):
            return len(protein_a) - sum(protein_a.count(residue) for residue in "XBZJ")
        return None
    if len(protein_a) < 128 or type(aligner) is not PairwiseAligner:
        return None
    # On equal-length inputs, a gapped global alignment must open a gap
    # in each sequence and lose at least one aligned pair. Its score is at
    # most 2*(length - 1) - 10, below 2*length - 9 for three mismatches.
    mismatches = matches = 0
    for first, second in zip(protein_a, protein_b, strict=True):
        if first != second:
            mismatches += 1
            if mismatches > 3:
                return None
        elif first not in "XBZJ":
            matches += 1
    try:
        if (
            aligner.mode != "global"
            or aligner.substitution_matrix is not None
            or aligner.wildcard is not None
            or aligner.match_score != 2.0
            or aligner.mismatch_score != -1.0
            or not 0 <= aligner.epsilon < 1
            or not _gap_scores_at_most(aligner, -5.0)
        ):
            return None
    except (AttributeError, ValueError):
        return None
    return matches


def score_protein_junctions(
    signature_a: tuple,
    signature_b: tuple,
    *,
    max_alignment_cells: int,
    aligner_factory,
    identity_weight=0.85,
    junction_weight=0.15,
    internal_gap_threshold=9,
) -> PairScore:
    protein_a, boundaries_a = signature_a
    protein_b, boundaries_b = signature_b
    if (
        not protein_a
        or not protein_b
        or len(protein_a) * len(protein_b) > max_alignment_cells
    ):
        return PairScore(0.0, 0.0, 0.0, 0.0, 0.0, 0, False, False, True)
    aligned = matches = column = 0
    internal_loss_a = internal_loss_b = False
    segments_a, segments_b = [], []
    aligner = aligner_factory()
    ungapped_matches = _proven_ungapped_match_count(protein_a, protein_b, aligner)
    if ungapped_matches is not None:
        aligned = column = len(protein_a)
        matches = ungapped_matches
        if boundaries_a:
            segments_a.append((0, aligned, 0))
        if boundaries_b:
            segments_b.append((0, aligned, 0))
    else:
        coordinates = aligner.align(protein_a, protein_b)[0].coordinates
        for index in range(coordinates.shape[1] - 1):
            start_a, end_a = int(coordinates[0, index]), int(coordinates[0, index + 1])
            start_b, end_b = int(coordinates[1, index]), int(coordinates[1, index + 1])
            width_a, width_b = end_a - start_a, end_b - start_b
            width = max(width_a, width_b)
            if width_a and width_b:
                aligned += width_a
                matches += sum(
                    a == b and a not in "XBZJ"
                    for a, b in zip(
                        protein_a[start_a:end_a], protein_b[start_b:end_b], strict=True
                    )
                )
            elif width_a >= internal_gap_threshold and 0 < start_b < len(protein_b):
                internal_loss_a = True
            elif width_b >= internal_gap_threshold and 0 < start_a < len(protein_a):
                internal_loss_b = True
            if boundaries_a and width_a:
                segments_a.append((start_a, end_a, column))
            if boundaries_b and width_b:
                segments_b.append((start_b, end_b, column))
            column += width
    junctions_a = project_junctions(boundaries_a, segments_a, len(protein_a), column)
    junctions_b = project_junctions(boundaries_b, segments_b, len(protein_b), column)
    union = junctions_a | junctions_b
    junction_similarity = len(junctions_a & junctions_b) / len(union) if union else 1.0
    identity = matches / max(1, aligned)
    coverage_a, coverage_b = aligned / len(protein_a), aligned / len(protein_b)
    # The identity contribution is also multiplied by reciprocal coverage:
    # a perfect short overlap cannot receive the same score as a complete match.
    reciprocal = min(coverage_a, coverage_b)
    score = reciprocal * (
        identity_weight * identity + junction_weight * junction_similarity
    )
    return PairScore(
        score,
        identity,
        coverage_a,
        coverage_b,
        junction_similarity,
        aligned,
        internal_loss_a,
        internal_loss_b,
    )

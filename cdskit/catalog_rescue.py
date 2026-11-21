"""Screen provisional protein representatives against the published catalogue."""

from collections import defaultdict
from collections.abc import Iterable, Mapping
import csv
from dataclasses import dataclass
import math
from pathlib import Path

from cdskit.homology import MAX_EVALUE


CATALOG_MIN_IDENTITY = 90.0
CATALOG_MIN_QUERY_COVERAGE = 0.90
CATALOG_DIAMOND_COLUMNS = (
    "qseqid",
    "sseqid",
    "pident",
    "qstart",
    "qend",
    "qlen",
    "sstart",
    "send",
    "slen",
    "bitscore",
    "evalue",
)
RESCUE_COLUMNS = (
    "parent_group_id",
    "candidate_cds_id",
    "selected_orf_id",
    "transcript_id",
    "protein_length",
    "decision",
    "catalog_target_id",
    "pident",
    "query_coverage",
    "target_coverage",
)


@dataclass(frozen=True)
class CatalogHit:
    target: str
    identity: float
    query_coverage: float
    target_coverage: float
    bitscore: float


@dataclass(frozen=True)
class RescueCandidate:
    name: str
    parent: str
    orf_id: str
    transcript: str
    peptide: str


def read_catalog_hits(
    path: Path, queries: Mapping[str, str], catalogue: Mapping[str, str]
) -> dict[str, list[CatalogHit]]:
    """Read the headerless DIAMOND outfmt 6 table in CATALOG_DIAMOND_COLUMNS order."""
    hits: dict[str, list[CatalogHit]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        for row in reader:
            try:
                query, target, pident, qs, qe, ql, ts, te, tl, score, raw_evalue = row
                identity, bitscore, evalue = map(float, (pident, score, raw_evalue))
                start, end, qlen, tstart, tend, tlen = map(
                    int, (qs, qe, ql, ts, te, tl)
                )
                if (
                    not all(map(math.isfinite, (identity, bitscore, evalue)))
                    or not 0 <= identity <= 100
                    or bitscore <= 0
                    or evalue < 0
                    or not 1 <= start <= end <= qlen == len(queries[query])
                    or not 1 <= tstart <= tend <= tlen == len(catalogue[target])
                ):
                    raise ValueError("Invalid catalogue alignment geometry or score")
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"{path}:{reader.line_num}: invalid catalogue hit"
                ) from error
            if evalue <= MAX_EVALUE:
                hits[query].append(
                    CatalogHit(
                        target,
                        identity,
                        (end - start + 1) / qlen,
                        (tend - tstart + 1) / tlen,
                        bitscore,
                    )
                )
    return hits


def best_hit(hits: Iterable[CatalogHit]) -> CatalogHit | None:
    return max(hits, key=lambda hit: (hit.bitscore, hit.target), default=None)


def compare_catalogue(
    candidate: RescueCandidate, catalogue: Mapping[str, str], hits: Iterable[CatalogHit]
) -> tuple[str, CatalogHit | None]:
    hits = tuple(hits)
    # Keep exact extensions for follow-up with RNA reads.
    extensions = [
        h
        for h in hits
        if len(catalogue[h.target]) < len(candidate.peptide)
        and catalogue[h.target] in candidate.peptide
    ]
    parent_extension = best_hit(h for h in extensions if h.target == candidate.parent)
    if parent_extension:
        return "possible_extension_review", parent_extension
    covered = best_hit(
        h
        for h in hits
        if h.identity >= CATALOG_MIN_IDENTITY
        and h.query_coverage >= CATALOG_MIN_QUERY_COVERAGE
    )
    if covered:
        return "covered_by_catalog", covered
    if extensions:
        return "possible_extension_review", best_hit(extensions)
    return "add", best_hit(hits)


def decide_catalog_rescues(
    candidates: Iterable[RescueCandidate],
    catalogue: Mapping[str, str],
    source_by_group: Mapping[str, str],
    hits: Mapping[str, list[CatalogHit]],
) -> list[dict[str, object]]:
    """Keep incumbent representatives and evaluate additional ORFs in stable order."""
    original_sources = set(source_by_group.values())
    used_sources = original_sources.copy()
    exact = {peptide: name for name, peptide in catalogue.items()}
    seen = set(source_by_group)
    decisions: list[dict[str, object]] = []
    for candidate in sorted(candidates, key=lambda c: (-len(c.peptide), c.name)):
        if candidate.name in seen:
            raise ValueError(f"Duplicate rescue candidate ID: {candidate.name}")
        seen.add(candidate.name)
        target = exact.get(candidate.peptide, "")
        hit = None
        if candidate.parent not in source_by_group:
            decision = "no_published_parent_review"
        elif not candidate.peptide or "*" in candidate.peptide:
            decision = "invalid_peptide_review"
        elif candidate.peptide == catalogue.get(candidate.parent):
            decision, target = "same_as_parent", candidate.parent
        elif target:
            decision = (
                "identical_in_catalog"
                if target in catalogue
                else "identical_to_another_addition"
            )
        elif candidate.transcript in used_sources:
            decision = (
                "alternate_of_published_transcript_review"
                if candidate.transcript in original_sources
                else "alternate_of_rescued_transcript_review"
            )
        else:
            decision, hit = compare_catalogue(
                candidate, catalogue, hits.get(candidate.name, ())
            )
            if decision == "add":
                exact[candidate.peptide] = candidate.name
                used_sources.add(candidate.transcript)
        decisions.append(
            dict(
                parent_group_id=candidate.parent,
                candidate_cds_id=candidate.name,
                selected_orf_id=candidate.orf_id,
                transcript_id=candidate.transcript,
                protein_length=len(candidate.peptide),
                decision=decision,
                catalog_target_id=target or (hit.target if hit else ""),
                pident=100.0 if target else hit.identity if hit else "",
                query_coverage=1.0 if target else hit.query_coverage if hit else "",
                target_coverage=1.0 if target else hit.target_coverage if hit else "",
            )
        )
    return decisions

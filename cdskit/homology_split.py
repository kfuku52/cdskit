"""Review distant protein hits and accept splits corroborated by genomic loci."""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
import csv
from itertools import chain
from pathlib import Path

from cdskit.homology import HIT_COLUMNS, Hit, comparable_hits, parse_hit, qualifies


SPLIT_MIN_COVERAGE = 0.80
SPLIT_COLUMNS = (
    "original_group_id",
    "transcripts",
    "decision",
    "near_best_targets",
    "transcript_target_families",
    "locus_ids",
    "applied_groups",
)
TargetSupport = dict[str, list[set[str]]]
SplitCandidate = tuple[str, list[str], TargetSupport]
FAMILY_SPLIT_DECISIONS = frozenset(
    {
        "review_distinct_targets",
        "review_same_locus",
        "review_locus_conflict",
        "split_supported",
    }
)


def read_split_targets(
    path: Path, source_name: Callable[[str], str], source_ids: set[str]
) -> TargetSupport:
    """Collect near-best broad hits separately for each candidate ORF."""
    candidates: dict[str, dict[str, Hit]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        if tuple(next(reader, ())) != HIT_COLUMNS:
            raise ValueError("Invalid protein hit header for split review")
        for row in reader:
            query, hit = parse_hit(row)
            if (
                source_name(query) in source_ids
                and qualifies(hit)
                and min(hit.query_coverage, hit.target_coverage) >= SPLIT_MIN_COVERAGE
            ):
                by_target = candidates[query]
                previous = by_target.get(hit.target)
                if previous is None or hit.rank > previous.rank:
                    by_target[hit.target] = hit
    support: TargetSupport = defaultdict(list)
    for query, hits in candidates.items():
        support[source_name(query)].append(set(comparable_hits(hits)))
    return support


def connected_targets(target_sets: Iterable[Iterable[str]]) -> dict[str, str]:
    """Map targets sharing evidence to stable connected-component labels."""
    parents: dict[str, str] = {}

    def root(item: str) -> str:
        parents.setdefault(item, item)
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    for target_set in target_sets:
        targets = iter(target_set)
        first = next(targets, None)
        if first is None:
            continue
        base = root(first)
        for target in targets:
            other = root(target)
            if base != other:
                keep, merge = sorted((base, other))
                parents[merge] = keep
                base = keep
    return {target: root(target) for target in parents}


def potential_splits(
    groups: Mapping[str, list[str]], target_support: TargetSupport
) -> Iterator[SplitCandidate]:
    """Skip groups whose supported ORFs agree on at least one accession."""
    for group, members in groups.items():
        supports = {
            member: target_support[member]
            for member in members
            if member in target_support
        }
        evidence = list(chain.from_iterable(supports.values()))
        if len(evidence) > 1 and not set.intersection(*evidence):
            yield group, members, supports


def review_group(
    members: list[str],
    supports: TargetSupport,
    families: Mapping[str, str],
    loci: Mapping[str, str],
) -> tuple[str, dict[str, set[str]]]:
    transcript_families = {}
    for transcript, per_orf in supports.items():
        labels = [
            {families.get(target, target) for target in targets} for targets in per_orf
        ]
        transcript_families[transcript] = set.intersection(*labels)
    if any(not targets for targets in transcript_families.values()):
        return "review_conflicting_orfs", transcript_families
    # A transcript supported by both target classes also connects the classes.
    if len(set(connected_targets(transcript_families.values()).values())) < 2:
        return "compatible_targets", transcript_families
    if any(member not in loci for member in members):
        return "review_distinct_targets", transcript_families
    by_locus: dict[str, set[str]] = {loci[member]: set() for member in members}
    for transcript, targets in transcript_families.items():
        by_locus[loci[transcript]].update(targets)
    if len(by_locus) < 2:
        return "review_same_locus", transcript_families
    union = set().union(*by_locus.values())
    if not all(by_locus.values()) or len(union) != sum(map(len, by_locus.values())):
        return "review_locus_conflict", transcript_families
    return "split_supported", transcript_families


def propose_family_partition(
    members: Iterable[str],
    transcript_families: Mapping[str, Iterable[str]],
    original_representative: str,
) -> tuple[dict[str, str], list[str]]:
    """Put ambiguous transcripts on the original representative's side."""
    targets = {member: tuple(labels) for member, labels in transcript_families.items()}
    families = connected_targets(targets.values())
    by_member = {member: families[labels[0]] for member, labels in targets.items()}
    counts = Counter(by_member.values())
    if len(counts) < 2:
        raise ValueError(
            "A family split requires at least two disconnected target families"
        )
    primary = by_member.get(original_representative) or min(
        counts, key=lambda family: (-counts[family], family)
    )
    ordered = [primary, *sorted(counts.keys() - {primary})]
    return {member: by_member.get(member, primary) for member in members}, ordered


def review_selected_families(
    expected: Mapping[str, str],
    observed: Mapping[str, str],
    loci: Mapping[str, str],
    *,
    similar_proteins: bool = False,
) -> str:
    """Check provisional output ORFs before accepting a family-based division."""
    if len(set(loci.values())) < len(loci):
        return "retain_same_locus"
    if len(loci) == len(expected):
        return "supported_distinct_loci"
    if similar_proteins or len(set(observed.values())) < len(observed):
        return "retain_related_representatives"
    if expected != observed:
        return "retain_unconfirmed_representatives"
    return "distinct_representatives"


def plan_splits(
    candidates: Iterable[SplitCandidate],
    related_targets: Iterable[Iterable[str]],
    loci: Mapping[str, str],
    *,
    apply: bool = False,
) -> tuple[dict[str, str], list[dict[str, object]]]:
    families = connected_targets(related_targets)
    assignments: dict[str, str] = {}
    audit: list[dict[str, object]] = []
    for group, members, supports in candidates:
        decision, labels = review_group(members, supports, families, loci)
        locus_ids = sorted({loci[member] for member in members if member in loci})
        children = {}
        if apply and decision == "split_supported":
            children = {
                locus: f"{group}.split{i}" for i, locus in enumerate(locus_ids, 1)
            }
            assignments.update((member, children[loci[member]]) for member in members)
            decision = "split_applied"
        audit.append(
            dict(
                original_group_id=group,
                transcripts=len(members),
                decision=decision,
                near_best_targets={
                    transcript: sorted(set().union(*per_orf))
                    for transcript, per_orf in supports.items()
                },
                transcript_target_families={
                    transcript: sorted(family_set)
                    for transcript, family_set in labels.items()
                },
                locus_ids=locus_ids,
                applied_groups=children,
            )
        )
    return assignments, audit

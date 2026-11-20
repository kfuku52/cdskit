"""Rank a length-based representative with qualified protein alignments."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

HIT_COLUMNS = (
    "query_id",
    "target_id",
    "evalue",
    "bitscore",
    "query_start",
    "query_end",
    "query_length",
    "target_start",
    "target_end",
    "target_length",
)
MIN_COVERAGE = 0.60
MAX_EVALUE = 1e-5
MIN_SCORE_GAIN = 1.10


@dataclass(frozen=True)
class Hit:
    target: str
    evalue: float
    bitscore: float
    query_coverage: float
    target_coverage: float

    @property
    def rank(self):
        return self.bitscore, self.target_coverage, self.query_coverage, -self.evalue


def parse_hit(row):
    (
        query,
        target,
        raw_evalue,
        raw_bits,
        q_start,
        q_end,
        q_len,
        t_start,
        t_end,
        t_len,
    ) = row
    evalue, bits = float(raw_evalue), float(raw_bits)
    q_start, q_end, q_len, t_start, t_end, t_len = map(
        int, (q_start, q_end, q_len, t_start, t_end, t_len)
    )
    if (
        not query
        or not target
        or not math.isfinite(evalue)
        or not math.isfinite(bits)
        or evalue < 0
        or bits <= 0
    ):
        raise ValueError("Invalid protein hit identity or score")
    if not (1 <= q_start <= q_end <= q_len and 1 <= t_start <= t_end <= t_len):
        raise ValueError("Invalid protein hit coordinates")
    return query, Hit(
        target,
        evalue,
        bits,
        (q_end - q_start + 1) / q_len,
        (t_end - t_start + 1) / t_len,
    )


def qualifies(hit):
    return (
        hit.evalue <= MAX_EVALUE
        and min(hit.query_coverage, hit.target_coverage) >= MIN_COVERAGE
    )


def read_hits(path, query_ids):
    """Keep the best qualifying HSP per query and target."""
    hits = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        if tuple(next(reader, ())) != HIT_COLUMNS:
            raise ValueError(f"Expected protein hit header: {' '.join(HIT_COLUMNS)}")
        for row in reader:
            try:
                query, hit = parse_hit(row)
                if query not in query_ids:
                    raise ValueError(f"Unknown hit query: {query}")
            except (ValueError, OverflowError) as error:
                raise ValueError(f"{path}:{reader.line_num}: {error}") from error
            if not qualifies(hit):
                continue
            targets = hits.setdefault(query, {})
            previous = targets.get(hit.target)
            if previous is None or hit.rank > previous.rank:
                targets[hit.target] = hit
    return hits


def comparable_hits(hits):
    best = max(hit.bitscore for hit in hits.values())
    return {
        target: hit
        for target, hit in hits.items()
        if hit.bitscore * MIN_SCORE_GAIN >= best
    }


def best_hit(hits):
    # Stable accession order resolves equal evidence independently of row order.
    return max((hits[target] for target in sorted(hits)), key=lambda hit: hit.rank)


def select_representative(records, hits):
    baseline = max(records, key=lambda record: len(record.seq))
    if len(records) == 1:
        evidence = hits.get(baseline.id)
        hit = best_hit(evidence) if evidence else None
        return baseline, baseline, "single_candidate", hit, hit
    supported = [
        (record, comparable_hits(hits[record.id]))
        for record in records
        if record.id in hits
    ]
    if not supported:
        return baseline, baseline, "no_qualified_hit", None, None
    if baseline.id not in hits:
        common = set(supported[0][1]).intersection(
            *(targets for _, targets in supported[1:])
        )
        if not common:
            return baseline, baseline, "conflicting_targets", None, None
        # Choose a single shared target before comparing candidates on that target.
        anchor = sorted(common)[0]
        winner, targets = max(supported, key=lambda result: result[1][anchor].rank)
        return baseline, winner, "supported_alternative", None, targets[anchor]
    baseline_hit = best_hit(hits[baseline.id])
    comparable = [
        (record, targets[baseline_hit.target])
        for record, targets in supported
        if baseline_hit.target in targets and record.id != baseline.id
    ]
    stronger = [
        (record, hit)
        for record, hit in comparable
        if hit.target_coverage >= baseline_hit.target_coverage
        and hit.bitscore >= baseline_hit.bitscore * MIN_SCORE_GAIN
    ]
    if stronger:
        winner, hit = max(stronger, key=lambda result: result[1].rank)
        return baseline, winner, "stronger_shared_target", baseline_hit, hit
    return (
        baseline,
        baseline,
        "retain_length_representative",
        baseline_hit,
        baseline_hit,
    )


def hit_values(hit):
    if hit is None:
        return ("",) * 5
    return (
        hit.target,
        hit.evalue,
        hit.bitscore,
        hit.query_coverage,
        hit.target_coverage,
    )


def write_selection_report(path, selections):
    fields = ("target_id", "evalue", "bitscore", "query_coverage", "target_coverage")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "group_id",
                "baseline_id",
                "selected_id",
                "reason",
                "selected_description",
                *(f"baseline_{field}" for field in fields),
                *(f"selected_{field}" for field in fields),
            )
        )
        for group, baseline, selected, reason, before, after in selections:
            writer.writerow(
                (
                    group,
                    baseline.id,
                    selected.id,
                    reason,
                    selected.description,
                    *hit_values(before),
                    *hit_values(after),
                )
            )

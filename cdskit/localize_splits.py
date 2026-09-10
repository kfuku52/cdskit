"""Shared sequence-group splitting and explicit audit provenance."""

import numpy as np

from cdskit.localize_evaluation import (
    assert_disjoint,
    dataset_digest,
    normalize_partition_id,
)
from cdskit.localize_model import to_canonical_aa_sequence


def sequence_folds(
    sequences, labels, n_folds, seed, fold_ids=None, group_ids=None, method="exact"
):
    if len(sequences) != len(labels):
        raise ValueError("Sequence and label counts must match.")
    if n_folds < 2:
        raise ValueError(
            "At least two independent groups are required for inner evaluation."
        )
    if fold_ids is None and group_ids is None:
        from cdskit.localize_learn import _validate_cross_validation_labels

        _validate_cross_validation_labels(labels)
    rows = [
        {"accession": str(i), "sequence": to_canonical_aa_sequence(seq)}
        for i, seq in enumerate(sequences)
    ]
    if not all(row["sequence"] for row in rows):
        raise ValueError("Empty canonical sequence in CV input.")
    if group_ids is not None and (
        len(group_ids) != len(rows) or any(not valid_partition_id(v) for v in group_ids)
    ):
        raise ValueError("Complete group IDs are required for every CV row.")
    if method not in ("exact", "mmseqs", "random"):
        raise ValueError("Unknown CV split method.")
    groups = (
        [str(value) for value in group_ids]
        if group_ids is not None
        else [row["sequence"] for row in rows]
    )
    report = {
        "method": "provided" if fold_ids is not None else method,
        "homology": "provided groups; not independently rechecked"
        if group_ids is not None
        else "not checked",
    }
    if group_ids is not None and method == "mmseqs":
        raise ValueError(
            "Choose provided groups or MMseqs groups; neither may silently replace the other."
        )
    if method == "mmseqs":
        from cdskit.perox_benchmark import mmseqs_cluster_assignments

        groups, homology = mmseqs_cluster_assignments(
            rows, min_seq_id=0.3, coverage=0.8, threads=1
        )
        if homology["status"] != "ok":
            raise ValueError("Homology partitioning failed: {}".format(homology))
        report.update(homology=homology)
    seen: dict[str, str] = {}
    for row, group in zip(rows, groups, strict=True):
        previous = seen.setdefault(row["sequence"], group)
        if previous != group:
            raise ValueError(
                "Identical canonical sequences have conflicting group IDs."
            )
    for row, group in zip(rows, groups, strict=True):
        row["cluster_id"] = str(group)
    if fold_ids is None and method == "random":
        from cdskit.localize_learn import build_stratified_folds

        folds = build_stratified_folds(labels, n_folds, seed)
    else:
        values = np.asarray(
            [str(v) if valid_partition_id(v) else "" for v in fold_ids]
            if fold_ids is not None
            else stratified_group_ids(labels, groups, n_folds, seed)
        )
        if len(values) != len(rows) or any(not valid_partition_id(v) for v in values):
            raise ValueError("Complete fold IDs are required.")
        folds = [
            np.flatnonzero(values == value) for value in sorted(set(values.tolist()))
        ]
    if len(folds) < 2:
        raise ValueError("At least two independent folds are required.")
    violations = []
    for fold in folds:
        held = set(fold.tolist())
        try:
            assert_disjoint(
                [r for i, r in enumerate(rows) if i not in held],
                [rows[i] for i in fold],
            )
        except ValueError as exc:
            if method != "random" or fold_ids is not None:
                raise
            violations.append(str(exc))
    if method == "mmseqs":
        report["cross_partition_homology"] = audit_homology_partitions(
            {str(i): [rows[j] for j in fold] for i, fold in enumerate(folds)}
        )
    report.update(
        exact_group_overlap="detected" if violations else "none",
        violations=violations,
        data_sha256=dataset_digest(rows),
        n_groups=len(set(groups)),
        n_folds=len(folds),
    )
    return folds, groups, report


def stratified_group_ids(labels, groups, n_folds, seed):
    """Greedy class balance without splitting a group (including mixed labels)."""
    if (
        len(labels) != len(groups)
        or n_folds < 2
        or any(not valid_partition_id(g) for g in groups)
    ):
        raise ValueError(
            "Matching labels and complete groups, and at least two folds, are required."
        )
    classes = sorted(set(labels))
    members: dict[str, list[int]] = {}
    for i, group in enumerate(groups):
        members.setdefault(str(group), []).append(i)
    if len(members) < 2:
        raise ValueError("At least two independent groups are required for evaluation.")
    n_folds = min(n_folds, len(members))
    counts = np.zeros((n_folds, len(classes)))
    sizes = np.zeros(n_folds)
    rng = np.random.default_rng(seed)
    keys = list(members)
    rng.shuffle(keys)
    keys.sort(key=lambda key: len(members[key]), reverse=True)
    result = np.empty(len(labels), dtype=object)
    for key in keys:
        ids = members[key]
        vector = np.asarray([sum(labels[i] == name for i in ids) for name in classes])
        costs = (counts * vector).sum(axis=1)
        fold = min(range(n_folds), key=lambda i: (costs[i], sizes[i]))
        result[ids] = str(fold)
        counts[fold] += vector
        sizes[fold] += len(ids)
    return result


def audit_homology_partitions(partitions, threads=1):
    """Recheck cross-partition hits, rather than trusting clustering representatives."""
    from cdskit.perox_benchmark import mmseqs_homology_report

    result = []
    names = list(partitions)
    for i, name in enumerate(names):
        for other in names[i + 1 :]:
            left, right = partitions[name], partitions[other]
            if not left or not right:
                continue

            # This is a sequence audit, not a peroxisome-label evaluation.
            def clean(rows):
                return [
                    {
                        "accession": str(row.get("accession", j)),
                        "sequence": row["sequence"],
                    }
                    for j, row in enumerate(rows)
                ]

            report = mmseqs_homology_report(
                clean(left), clean(right), min_seq_id=0.3, coverage=0.8, threads=threads
            )
            if report["status"] != "ok":
                raise ValueError(
                    "Cross-partition homology audit unavailable: {}".format(report)
                )
            if report["hit_query_count"]:
                raise ValueError(
                    "Cross-partition homology hits between {} and {}; rebuild groups before evaluation.".format(
                        name, other
                    )
                )
            result.append(
                {
                    "partitions": [name, other],
                    **{
                        key: report[key]
                        for key in (
                            "status",
                            "tool",
                            "min_seq_id",
                            "coverage",
                            "cov_mode",
                            "hit_query_count",
                        )
                    },
                }
            )
    return result


def valid_partition_id(value):
    return normalize_partition_id(value) is not None

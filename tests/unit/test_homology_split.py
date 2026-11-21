from cdskit.homology import HIT_COLUMNS
from cdskit.homology_split import (
    plan_splits,
    potential_splits,
    propose_family_partition,
    read_split_targets,
    review_selected_families,
)


def test_family_partition_retains_unassigned_transcripts_on_primary_side():
    members = ["left", "right", "extra", "unknown"]
    evidence = {"left": {"A"}, "right": {"B"}, "extra": {"B", "B2"}}
    groups, ordered = propose_family_partition(members, evidence, "left")
    assert ordered == ["A", "B"]
    assert groups == {"left": "A", "right": "B", "extra": "B", "unknown": "A"}
    groups, ordered = propose_family_partition(members, evidence, "unknown")
    assert ordered == ["B", "A"] and groups["unknown"] == "B"


def test_review_uses_selected_orfs_and_independent_locus_support():
    expected = {"left": "A", "right": "B"}
    assert (
        review_selected_families(expected, expected, {}) == "distinct_representatives"
    )
    assert (
        review_selected_families(expected, {"left": "A"}, {})
        == "retain_unconfirmed_representatives"
    )
    assert (
        review_selected_families(expected, {"left": "A", "right": "A"}, {})
        == "retain_related_representatives"
    )
    assert (
        review_selected_families(expected, expected, {}, similar_proteins=True)
        == "retain_related_representatives"
    )
    assert (
        review_selected_families(expected, expected, {"left": "l1", "right": "l1"})
        == "retain_same_locus"
    )
    assert (
        review_selected_families(
            expected, {}, {"left": "l1", "right": "l2"}, similar_proteins=True
        )
        == "supported_distinct_loci"
    )


def review(supports, loci, related=(), *, apply=True):
    groups = {"g1": ["t1", "t2"]}
    assignments, (row,) = plan_splits(
        potential_splits(groups, supports), related, loci, apply=apply
    )
    return assignments, row["decision"]


def test_homology_split_requires_distinct_targets_and_corroborating_loci():
    supports = {"t1": [{"A", "Aliased"}], "t2": [{"B"}]}
    loci = {"t1": "loc2", "t2": "loc1"}
    assert review(supports, loci) == (
        {"t1": "g1.split2", "t2": "g1.split1"},
        "split_applied",
    )
    assert review(supports, loci, apply=False) == ({}, "split_supported")
    assert review(supports, {}) == ({}, "review_distinct_targets")
    assert review(supports, {"t1": "loc1", "t2": "loc1"}) == ({}, "review_same_locus")
    assert review(
        supports, loci, [("Aliased", "intermediate"), ("intermediate", "B")]
    ) == ({}, "compatible_targets")


def test_homology_split_reviews_conflicting_orfs_and_unsupported_loci():
    loci = {"t1": "loc1", "t2": "loc2"}
    assert review({"t1": [{"A"}, {"B"}]}, loci) == ({}, "review_conflicting_orfs")
    assert not list(potential_splits({"g1": ["t1", "t2"]}, {"t1": [{"A", "B"}, {"A"}]}))
    groups = {"g1": ["t1", "t2", "unmapped"]}
    targets = {"t1": [{"A"}], "t2": [{"B"}]}
    assignments, (row,) = plan_splits(
        potential_splits(groups, targets), (), loci, apply=True
    )
    assert assignments == {}
    assert row["decision"] == "review_distinct_targets"
    loci["unmapped"] = "loc3"
    assignments, (row,) = plan_splits(
        potential_splits(groups, targets), (), loci, apply=True
    )
    assert assignments == {}
    assert row["decision"] == "review_locus_conflict"


def test_homology_split_requires_broad_near_best_hits_per_orf(tmp_path):
    path = tmp_path / "hits.tsv"
    rows = (
        ("t1", "A", "1e-50", 150, 1, 95, 100, 3, 96, 100),
        ("t1", "B", "1e-49", 140, 1, 95, 100, 3, 96, 100),
        ("t1", "weak", "1e-20", 50, 1, 100, 100, 1, 100, 100),
        ("t1", "domain", "1e-50", 150, 1, 79, 100, 1, 90, 100),
        ("t1", "short", "1e-50", 160, 1, 85, 100, 1, 79, 100),
        ("t1.1", "C", "1e-51", 155, 1, 100, 100, 1, 100, 100),
        ("not_in_group", "C", "1e-51", 155, 1, 100, 100, 1, 100, 100),
    )
    path.write_text("\n".join("\t".join(map(str, row)) for row in (HIT_COLUMNS, *rows)))
    support = read_split_targets(path, lambda query: query.split(".")[0], {"t1"})
    assert support == {"t1": [{"A", "B"}, {"C"}]}

import pytest

from cdskit.catalog_rescue import (
    CatalogHit,
    RescueCandidate,
    decide_catalog_rescues,
    read_catalog_hits,
)


def run_candidates(candidates, hits):
    return {
        r["candidate_cds_id"]: (r["decision"], r["catalog_target_id"])
        for r in decide_catalog_rescues(
            candidates,
            {"g1": "PA", "g2": "PEPT", "g3": "OTHER"},
            {"g1": "g1-i1", "g2": "g2-i1", "g3": "g3-i1"},
            hits,
        )
    }


def candidate(number, sequence, transcript=None):
    source = transcript or f"g1-i{number + 1}"
    return RescueCandidate(f"g1.split{number}", "g1", source, source, sequence)


def test_catalog_rescue_screens_all_groups_and_requires_candidate_coverage():
    candidates = [
        candidate(1, "PA"),
        candidate(2, "PEPT"),
        candidate(3, "FULLQ"),
        candidate(4, "OTHERQ"),
        candidate(5, "DIFFQ"),
    ]
    hits = {
        "g1.split3": [CatalogHit("g2", 99.0, 0.89, 1.0, 100.0)],
        "g1.split4": [CatalogHit("g3", 91.0, 0.90, 1.0, 80.0)],
        "g1.split5": [
            CatalogHit("g2", 89.0, 0.98, 1.0, 100.0),
            CatalogHit("g3", 90.0, 0.90, 1.0, 60.0),
        ],
    }
    assert run_candidates(candidates, hits) == {
        "g1.split1": ("same_as_parent", "g1"),
        "g1.split2": ("identical_in_catalog", "g2"),
        "g1.split3": ("add", "g2"),
        "g1.split4": ("covered_by_catalog", "g3"),
        "g1.split5": ("covered_by_catalog", "g3"),
    }


def test_catalog_rescue_keeps_incumbents_and_deduplicates_additions():
    candidates = [
        candidate(1, "PROT", "g1-i1"),
        candidate(2, "QQPAK"),
        candidate(3, "QNEW"),
        candidate(4, "QNEW"),
        candidate(5, "OTHERFORM", "g1-i4"),
    ]
    hits = {"g1.split2": [CatalogHit("g1", 100, 0.4, 1.0, 40)]}
    assert run_candidates(candidates[:4], hits) == {
        "g1.split1": ("alternate_of_published_transcript_review", ""),
        "g1.split2": ("possible_extension_review", "g1"),
        "g1.split3": ("add", ""),
        "g1.split4": ("identical_to_another_addition", "g1.split3"),
    }
    decisions = run_candidates(candidates, hits)
    assert decisions["g1.split5"] == ("add", "")
    assert decisions["g1.split3"] == ("alternate_of_rescued_transcript_review", "")


def test_read_catalog_hits_checks_ids_and_alignment_lengths(tmp_path):
    path = tmp_path / "diamond.tsv"
    path.write_text("candidate\ttarget\t90\t1\t9\t10\t2\t10\t12\t70\t1e-20\n")
    assert read_catalog_hits(path, {"candidate": "A" * 10}, {"target": "A" * 12}) == {
        "candidate": [CatalogHit("target", 90.0, 0.9, 0.75, 70.0)]
    }
    with pytest.raises(ValueError, match=r"diamond.tsv:1: invalid catalogue hit"):
        read_catalog_hits(path, {"candidate": "A" * 11}, {"target": "A" * 12})

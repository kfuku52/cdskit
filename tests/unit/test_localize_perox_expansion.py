"""Evidence must apply to the predicted organelle and the supplied isoform."""

import copy

import numpy as np
import pytest

from scripts.localize_perox_expansion import experimental_perox_row, compare_recall


@pytest.fixture
def entry():
    return {
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
        "primaryAccession": "QTEST",
        "organism": {
            "lineage": ["Eukaryota", "Metazoa"],
            "taxonId": 9606,
            "scientificName": "Homo sapiens",
        },
        "sequence": {"value": "MAAASKL"},
        "comments": [
            {
                "commentType": "SUBCELLULAR LOCATION",
                "subcellularLocations": [
                    {
                        "location": {
                            "id": "SL-0204",
                            "value": "Peroxisome",
                            "evidences": [
                                {
                                    "evidenceCode": "ECO:0000269",
                                    "source": "PubMed",
                                    "id": "123",
                                }
                            ],
                        }
                    }
                ],
            }
        ],
    }


def test_perox_evidence_is_retained_with_source(entry):
    row, reason = experimental_perox_row(entry)
    assert reason == ""
    assert row["peroxisome"] == 1
    assert row["evidence"] == ["PubMed:123"]
    assert row["taxon_id"] == 9606


def test_experiment_on_other_location_does_not_validate_perox(entry):
    other = copy.deepcopy(entry["comments"][0]["subcellularLocations"][0])
    other["location"]["id"] = "SL-0173"
    entry["comments"][0]["subcellularLocations"][0]["location"]["evidences"][0][
        "evidenceCode"
    ] = "ECO:0000250"
    entry["comments"][0]["subcellularLocations"].append(other)
    assert experimental_perox_row(entry)[0] is None


def test_experimental_note_does_not_validate_inferred_location(entry):
    entry["comments"][0]["note"] = {
        "texts": [{"evidences": [{"evidenceCode": "ECO:0000269"}]}]
    }
    entry["comments"][0]["subcellularLocations"][0]["location"]["evidences"][0][
        "evidenceCode"
    ] = "ECO:0000250"
    assert experimental_perox_row(entry)[0] is None


@pytest.mark.parametrize(
    "change", ["isoform", "fragment", "unreviewed", "non_eukaryote"]
)
def test_ineligible_entries_rejected(entry, change):
    if change == "isoform":
        entry["comments"][0]["molecule"] = "Isoform 2"
    elif change == "fragment":
        entry["proteinDescription"] = {"flag": "Fragments"}
    elif change == "unreviewed":
        entry["entryType"] = "UniProtKB unreviewed (TrEMBL)"
    else:
        entry["organism"]["lineage"] = ["Bacteria"]
    assert experimental_perox_row(entry)[0] is None


def test_paired_recall_counts():
    base = np.asarray([True, True, False, False])
    new = np.asarray([True, False, True, True])
    result = compare_recall(base, new, np.asarray(["a", "b", "c", "d"]))
    assert result["baseline_tp"] == 2
    assert result["integrated_tp"] == 3
    assert result["baseline_only"] == 1
    assert result["integrated_only"] == 2
    assert result["both_detected"] == 1
    assert result["neither"] == 0
    assert result["integrated_recall"] == 0.75
    assert compare_recall(base[:0], new[:0], np.asarray([])) == {"n": 0}


def test_ensembl_protein_ids_include_isoforms_and_strip_version():
    from scripts.localize_perox_expansion import ensembl_protein_ids

    entry = {
        "uniProtKBCrossReferences": [
            {
                "database": "Ensembl",
                "properties": [
                    {"key": "ProteinId", "value": "ENSP00000001.4"},
                    {"key": "GeneId", "value": "ENSG00000001"},
                ],
            },
            {
                "database": "Ensembl",
                "properties": [{"key": "ProteinId", "value": "ENSP00000002"}],
            },
        ]
    }
    assert ensembl_protein_ids(entry) == {"ENSP00000001", "ENSP00000002"}

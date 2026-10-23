"""Check trimming against the established per-codon output and record contract."""

import copy
import random

from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
import numpy as np
import pytest

from cdskit.trimcodon import (
    _trim_records_to_codon_sites,
    trim_record_to_codon_sites,
)
from cdskit.util import replace_record_sequence


def _record(sequence):
    record = SeqRecord(
        Seq(sequence), id="record", name="name", description="description"
    )
    record.annotations = {"molecule_type": "DNA", "nested": {"value": 1}}
    record.dbxrefs = ["synthetic:1"]
    record.features = [SeqFeature(SimpleLocation(0, len(record)), type="test")]
    record.letter_annotations["quality"] = list(range(len(record)))
    return record


def _per_codon_oracle(record, sites):
    result = copy.copy(record)
    sequence = str(record.seq)
    codons = [sequence[site * 3 : site * 3 + 3] for site in sites]
    return replace_record_sequence(result, "".join(codons))


def _assert_same_record(actual, expected, original, oracle_original):
    assert str(actual.seq) == str(expected.seq)
    for name in (
        "id",
        "name",
        "description",
        "annotations",
        "dbxrefs",
        "features",
        "letter_annotations",
    ):
        assert getattr(actual, name) == getattr(expected, name)
    assert actual is not original
    assert actual.annotations is original.annotations
    assert original.letter_annotations == oracle_original.letter_annotations
    assert original.features == oracle_original.features


@pytest.mark.parametrize("length", [0, 1, 2, 3, 5, 180, 30000])
def test_command_ranges_match_per_codon_oracle(length):
    sequence = ("GCAACGtgcnnN-?." * ((length + 14) // 15))[:length]
    sites = list(range(length // 3))
    rng = random.Random(19)
    for kept in (
        [],
        sites,
        sites[::2],
        sites[2:-2],
        [site for site in sites if site % 100 < 50],
        [site for site in sites if rng.random() < 0.7],
    ):
        records = [_record(sequence), _record(sequence[::-1])]
        oracle_records = copy.deepcopy(records)
        results = _trim_records_to_codon_sites(records, kept)
        assert len(results) == len(records)
        for actual, original, oracle_original in zip(
            results, records, oracle_records, strict=True
        ):
            expected = _per_codon_oracle(oracle_original, kept)
            _assert_same_record(actual, expected, original, oracle_original)


@pytest.mark.parametrize(
    "sites",
    [
        [],
        [0, 1, 2, 3],
        [3, 1, 2, 0],
        [0, 0, 2, 2],
        [-3, -2, -1, 0, 1],
        [-100, 0, 100],
        [np.int64(0), np.int64(2)],
        [True, False, True],
    ],
)
def test_direct_api_retains_order_duplicates_and_slice_semantics(sites):
    record = _record("GCAACGtgcnnN")
    oracle_record = copy.deepcopy(record)
    actual = trim_record_to_codon_sites(record, iter(sites))
    expected = _per_codon_oracle(oracle_record, sites)
    _assert_same_record(actual, expected, record, oracle_record)


@pytest.mark.parametrize("site", [0.0, None, "0"])
def test_direct_api_rejects_non_index_sites(site):
    with pytest.raises(TypeError):
        trim_record_to_codon_sites(_record("GCA"), [site])

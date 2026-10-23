"""Public alphabet diagnostics compared with an independent set-based oracle."""

from __future__ import annotations

from itertools import product
import random
from types import SimpleNamespace

import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit import util


_DNA = frozenset("ACGTRYSWKMBDHVNXacgtryswkmbdhvnx-?.")
_PROTEIN = frozenset("ABCDEFGHIKLMNPQRSTVWXYZJUO*abcdefghiklmnpqrstvwxyzjuo-?.")


def _oracle(records, seqtype, label):
    allowed = {"dna": _DNA, "protein": _PROTEIN, "auto": _DNA | _PROTEIN}[seqtype]
    invalid_ids = []
    invalid_chars = set()
    for record in records:
        bad = set(str(record.seq)) - allowed
        if bad:
            invalid_ids.append(str(record.id))
            invalid_chars.update(bad)
    if not invalid_ids:
        return None
    shown = ",".join(invalid_ids[:10]) + (",..." if len(invalid_ids) > 10 else "")
    chars = "".join(sorted(invalid_chars))
    if seqtype == "dna":
        return (
            f"Invalid non-DNA character(s) were detected in {label} ({shown}) [chars: {chars}]. "
            "DNA-only input is required (use T instead of U). Exiting.\n"
        )
    if seqtype == "protein":
        return (
            f"Invalid non-protein character(s) were detected in {label} ({shown}) [chars: {chars}]. "
            "Protein-only input is required. Exiting.\n"
        )
    return (
        f"Invalid sequence character(s) were detected in {label} ({shown}) [chars: {chars}]. "
        "DNA or protein input is required when --seq_type=auto. Exiting.\n"
    )


def _assert_diagnostics(records, seqtype, label="--custom_input"):
    validators = [
        lambda: util.stop_if_not_seqtype(records, seqtype=seqtype, label=label),
        lambda: util.stop_if_not_seqtype(records, seqtype=seqtype.upper(), label=label),
    ]
    if seqtype == "dna":
        validators.append(lambda: util.stop_if_not_dna(records, label=label))
    elif seqtype == "protein":
        validators.append(lambda: util.stop_if_not_protein(records, label=label))
    expected = _oracle(records, seqtype, label)
    for validate in validators:
        if expected is None:
            assert validate() is None
        else:
            with pytest.raises(ValueError) as error:
                validate()
            assert type(error.value) is ValueError
            assert str(error.value) == expected


@pytest.mark.parametrize("seqtype", ["dna", "protein", "auto"])
def test_full_ascii_alphabet_matches_set_oracle(seqtype):
    # Single characters distinguish upper/lower case and DNA/protein membership.
    records = [SeqRecord(Seq(chr(value)), id=str(value)) for value in range(128)]
    for record in records:
        _assert_diagnostics([record], seqtype)
    _assert_diagnostics(records, seqtype)


@pytest.mark.parametrize("seqtype", ["dna", "protein", "auto"])
def test_empty_and_allowed_records(seqtype):
    _assert_diagnostics([], seqtype)
    alphabet = _DNA if seqtype == "dna" else _PROTEIN
    _assert_diagnostics(
        [
            SeqRecord(Seq(""), id="empty"),
            SeqRecord(Seq("".join(sorted(alphabet))), id="all"),
        ],
        seqtype,
    )


@pytest.mark.parametrize("seqtype", ["dna", "protein", "auto"])
def test_invalid_ids_keep_order_duplicates_and_ten_record_limit(seqtype):
    records = [SeqRecord(Seq("A!"), id=f"seq{index // 2}") for index in range(14)]
    records.insert(3, SeqRecord(Seq("ACG"), id="valid"))
    _assert_diagnostics(records, seqtype)
    # A character found only after the ten displayed records is still reported.
    records[-1].seq = Seq("1")
    _assert_diagnostics(records, seqtype, label="input with spaces")


@pytest.mark.parametrize("seqtype", ["dna", "protein", "auto"])
def test_unicode_controls_and_surrogates_keep_sorted_diagnostics(seqtype):
    # SeqRecord stores ASCII; record-like objects cover existing str conversion.
    characters = "éΩ😀\ud800\udfff\x00\x01\n\r\t!Uu"
    records = [
        SimpleNamespace(seq="ACG" + characters, id="unicode"),
        SimpleNamespace(seq="!", id=17),
    ]
    _assert_diagnostics(records, seqtype)


@pytest.mark.parametrize("seqtype", ["dna", "protein", "auto"])
@pytest.mark.parametrize(
    "invalid_position", [0, 1, 65535, 65536, 65537, 131072, 131073]
)
def test_invalid_positions_and_chunk_boundaries(seqtype, invalid_position):
    sequence = "A" * invalid_position + "!" + "C" * (131073 - invalid_position) + "1"
    _assert_diagnostics([SeqRecord(Seq(sequence), id="long")], seqtype)


@pytest.mark.parametrize("seqtype", ["dna", "protein", "auto"])
def test_long_invalid_records_keep_all_unique_characters(seqtype):
    sequence = "!1 " * 50000 + "\tZ"
    _assert_diagnostics([SeqRecord(Seq(sequence), id="long-invalid")], seqtype)


@pytest.mark.parametrize("seqtype", ["dna", "protein", "auto"])
def test_seeded_mixed_records_match_set_oracle(seqtype):
    rng = random.Random(20261023)
    alphabet = "ACGTacgtNXnx-?.Uu*EJOPZ!123éΩ\x00\n\ud800"
    records = [
        SimpleNamespace(
            seq="".join(rng.choices(alphabet, k=rng.randrange(80))), id=f"r{i % 7}"
        )
        for i in range(40)
    ]
    _assert_diagnostics(records, seqtype)
    for sequence in map("".join, product("AaUu*!", repeat=3)):
        _assert_diagnostics([SimpleNamespace(seq=sequence, id="short")], seqtype)


@pytest.mark.parametrize("seqtype", ["rna", "DNA ", None, 1, ""])
def test_invalid_seqtype_preserves_error(seqtype):
    with pytest.raises(ValueError) as error:
        util.stop_if_not_seqtype([], seqtype=seqtype)
    assert str(error.value) == (
        f"Invalid --seq_type: {seqtype}. Choose from dna, protein, auto. Exiting.\n"
    )


class _DifferentLength(str):
    def __len__(self):
        return 1


class _DifferentIterator(str):
    def __iter__(self):
        return iter("A!")


class _DifferentSlices(str):
    def __getitem__(self, key):
        if isinstance(key, slice):
            return "A"
        return super().__getitem__(key)


class _SequenceString:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


@pytest.mark.parametrize("seqtype", ["dna", "protein", "auto"])
@pytest.mark.parametrize(
    "sequence",
    [_DifferentLength("A!"), _DifferentIterator("A"), _DifferentSlices("A!")],
    ids=["custom-length", "custom-iterator", "custom-slices"],
)
def test_str_subclass_operations_keep_original_set_behavior(seqtype, sequence):
    records = [SimpleNamespace(seq=_SequenceString(sequence), id="custom-string")]
    assert str(records[0].seq) is sequence
    _assert_diagnostics(records, seqtype)

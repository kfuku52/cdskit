"""MSA preparation retains its translation, consensus and drawing contracts."""

from collections import Counter
from functools import lru_cache
from itertools import product
from types import SimpleNamespace

import matplotlib
import pytest
from Bio.Data import CodonTable, IUPACData
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit import plot


def _records(sequences):
    return [
        SeqRecord(Seq(sequence), id=f"row{index}")
        for index, sequence in enumerate(sequences)
    ]


def _reference_summary(records, codontable):
    """Use DNA expansions and simple column counts, independent of plot helpers."""
    table = CodonTable.unambiguous_dna_by_id[int(codontable)]

    @lru_cache(maxsize=None)
    def translate(codon):
        if len(codon) != 3:
            return "?"
        if set(codon) & set("-?."):
            return "-"
        if all(base in IUPACData.ambiguous_dna_values for base in codon):
            expansions = product(
                *(IUPACData.ambiguous_dna_values[base] for base in codon)
            )
            if all(
                "".join(expansion) in table.stop_codons
                and "".join(expansion) not in table.forward_table
                for expansion in expansions
            ):
                return "*"
        if set(codon) - set("ACGT"):
            return "?"
        return table.forward_table.get(codon, "*")

    strings = [str(record.seq).upper() for record in records]
    sites = len(strings[0]) // 3 if strings else 0
    rows = [
        [sequence[3 * site : 3 * site + 3] for site in range(sites)]
        for sequence in strings
    ]
    consensus_codons = []
    consensus_aas = []
    frequencies = []
    for site in range(sites):
        column = [row[site] for row in rows]
        concrete = [
            codon
            for codon in column
            if len(codon) == 3 and not set(codon) - set("ACGT")
        ]
        if concrete:
            counts = Counter(concrete)
            consensus = min(counts, key=lambda codon: (-counts[codon], codon))
            aa_counts = Counter(translate(codon) for codon in concrete)
            frequencies.append(
                [
                    (aa, count / len(concrete))
                    for aa, count in sorted(
                        aa_counts.items(), key=lambda item: (item[1], item[0])
                    )
                ]
            )
        else:
            consensus_bases = []
            for offset in range(3):
                counts = Counter(
                    codon[offset]
                    for codon in column
                    if len(codon) == 3 and codon[offset] not in "-?."
                )
                consensus_bases.append(
                    min(counts, key=lambda base: (-counts[base], base))
                    if counts
                    else "-"
                )
            consensus = "".join(consensus_bases)
            frequencies.append([])
        consensus_codons.append(consensus)
        consensus_aas.append(translate(consensus))
    return {
        "seq_strings": strings,
        "seq_codons": rows,
        "seq_aas": [[translate(codon) for codon in row] for row in rows],
        "consensus_codons": consensus_codons,
        "consensus_aas": consensus_aas,
        "aa_logo_frequencies": frequencies,
    }


@pytest.mark.parametrize("codontable", sorted(CodonTable.unambiguous_dna_by_id))
def test_msa_summary_all_iupac_triplets_match_independent_oracle(codontable):
    codons = ["".join(bases) for bases in product("ACGTRYSWKMBDHVNX", repeat=3)]
    codons.extend(["---", "???", "...", "A-?", "!AA", "AUG", "123"])
    records = _records(["".join(codons).lower()])
    assert plot._build_msa_summary(records, codontable) == _reference_summary(
        records, codontable
    )


@pytest.mark.parametrize("codontable", sorted(CodonTable.unambiguous_dna_by_id))
@pytest.mark.parametrize(
    "sequences",
    [
        [],
        ["", "ATG"],
        ["AT"],
        ["atgaaataatagtga---nnn", "ATGAACTAGTAATGA??.RYN", "ATGAAGTAGTGATAA...TAR"],
        ["NNN---ATGAAAAC", "RYN??.AA", "NNN...TTTTAAATGCCCGGG"],
        ["A--N?NNNN", "-A-NN?RYN", "--ANNNNRY"],
    ],
)
def test_msa_summary_preserves_ties_missing_partial_and_unequal_inputs(
    codontable, sequences
):
    records = _records(sequences)
    actual = plot._build_msa_summary(records, codontable)
    assert actual == _reference_summary(records, codontable)
    assert list(actual) == [
        "seq_strings",
        "seq_codons",
        "seq_aas",
        "consensus_codons",
        "consensus_aas",
        "aa_logo_frequencies",
    ]
    assert all(type(value) is list for value in actual.values())
    assert all(
        type(pair) is tuple and type(pair[1]) is float
        for column in actual["aa_logo_frequencies"]
        for pair in column
    )


@pytest.mark.parametrize("codontable", [1, 27, 28, 31])
def test_msa_svg_identical_to_independent_summary(monkeypatch, codontable):
    records = _records(
        ["ATGTAATGATARNNN---", "ATGTAATGGTAYNNN???", "ATGTAATGATAGNNN..."]
    )
    args = SimpleNamespace(
        codontable=codontable,
        width=600,
        height=400,
        row_height=18,
        label_width=100,
        wrap=18,
    )
    with matplotlib.rc_context({"svg.hashsalt": "msa-equivalence"}):
        actual = plot._serialize_figure(plot._build_msa_figure(records, args), "svg")
        monkeypatch.setattr(plot, "_build_msa_summary", _reference_summary)
        expected = plot._serialize_figure(plot._build_msa_figure(records, args), "svg")
    assert actual == expected

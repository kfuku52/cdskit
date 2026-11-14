"""
Tests for cdskit codonstats command.
"""

from itertools import product

import Bio.SeqIO
import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.codonstats import codonstats_main, summarize_record


@pytest.mark.parametrize(
    "sequence,expected_complete,expected_missing,expected_ambiguous",
    [
        ("?", 0, 1, 0),
        ("N", 0, 0, 0),
        ("ATG?", 1, 1, 0),
        ("ATGN", 1, 0, 0),
    ],
)
def test_summarize_record_ignores_partial_codon_for_complete_counts(
    sequence,
    expected_complete,
    expected_missing,
    expected_ambiguous,
):
    summary = summarize_record(SeqRecord(Seq(sequence), id="seq1"), codontable=1)

    assert summary["codons_complete"] == expected_complete
    assert summary["codons_missing"] == expected_missing
    assert summary["codons_ambiguous"] == expected_ambiguous


@pytest.mark.parametrize("code,tga,taa", [(1, "*", "*"), (28, "W", "Q")])
def test_usage_retains_only_concrete_codons_across_input_lengths(
    code, tga, taa, tmp_path, mock_args, write_fasta, capsys
):
    source = write_fasta(
        tmp_path / "source.fa",
        [("dup", "tgaGcn---" * 23_000), ("dup", "TgATAATAR???")],
    )
    codonstats_main(mock_args(seqfile=str(source), mode="usage", codontable=code))
    assert capsys.readouterr().out.splitlines() == [
        "codon\taa\tcount\tfraction",
        f"TAA\t{taa}\t1\t0.000043",
        f"TGA\t{tga}\t23001\t0.999957",
    ]


def test_usage_lists_every_concrete_codon_once(
    tmp_path, mock_args, write_fasta, capsys
):
    codons = ["".join(bases) for bases in product("ACGT", repeat=3)]
    source = write_fasta(
        tmp_path / "source.fa", [("r", "".join(reversed(codons)) * 300)]
    )
    codonstats_main(mock_args(seqfile=str(source), mode="usage", codontable=1))
    rows = [line.split("\t") for line in capsys.readouterr().out.splitlines()[1:]]
    assert [row[0] for row in rows] == codons
    assert all(row[2:] == ["300", "0.015625"] for row in rows)


class TestCodonstatsMain:
    """Tests for codonstats_main function."""

    def test_codonstats_summary_mode(self, temp_dir, mock_args, capsys):
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("GCTGCTGCT"), id="seq1", description=""),
            SeqRecord(Seq("ATGNNNTAA"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            mode="summary",
        )

        codonstats_main(args)
        captured = capsys.readouterr()

        assert captured.out.startswith("seq_id\tnt_length\tcodons_total")
        assert (
            "seq1\t9\t3\t3\t0\t0\t0\t66.666667\t100.000000\t100.000000\t0.000000"
            in captured.out
        )
        assert "seq2\t9\t3\t3\t0\t1\t1" in captured.out

    def test_codonstats_usage_mode(self, temp_dir, mock_args, capsys):
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATGATGTTT"), id="seq1", description=""),
            SeqRecord(Seq("ATGTTTTTT"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            mode="usage",
        )

        codonstats_main(args)
        captured = capsys.readouterr()

        assert captured.out.startswith("codon\taa\tcount\tfraction")
        assert "ATG\tM\t3\t0.500000" in captured.out
        assert "TTT\tF\t3\t0.500000" in captured.out

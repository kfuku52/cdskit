"""
Tests for cdskit aggregate command.
"""

import pytest

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.aggregate import aggregate_main, aggregate_name, select_aggregate_record


class TestAggregateHelpers:
    """Tests for aggregate helper functions."""

    def test_aggregate_name_applies_expressions_in_order(self):
        name = "prefix_gene_suffix.1"
        expressions = [r"^prefix_", r"_suffix", r"\.[0-9]+$"]
        assert aggregate_name(name, expressions) == "gene"

    def test_select_aggregate_record_longest_mode(self):
        rec_short = SeqRecord(Seq("ATG"), id="short", name="short", description="")
        rec_long = SeqRecord(Seq("ATGAAA"), id="long", name="long", description="")
        selected = select_aggregate_record(rec_short, rec_long, mode="longest")
        assert selected is rec_long


class TestAggregateMain:
    """Tests for aggregate_main function."""

    def test_aggregate_by_suffix(self, temp_dir, mock_args):
        """Test aggregating sequences by removing suffix."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # gene_A.1 and gene_A.2 should aggregate to gene_A
        records = [
            SeqRecord(
                Seq("ATGAAA"), id="gene_A.1", name="gene_A.1", description=""
            ),  # 6 nt
            SeqRecord(
                Seq("ATGAAACCC"), id="gene_A.2", name="gene_A.2", description=""
            ),  # 9 nt - longer
            SeqRecord(Seq("ATGCCC"), id="gene_B.1", name="gene_B.1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            expression=[r"\.[0-9]+$"],  # Remove .N suffix
            mode="longest",
        )

        aggregate_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Should have 2 sequences: gene_A (longest) and gene_B
        assert len(result) == 2
        # Find gene_A entry - should be the longer one
        gene_a = next(r for r in result if "gene_A" in r.id)
        assert len(gene_a.seq) == 9

    def test_aggregate_keep_longest(self, temp_dir, mock_args):
        """Test that longest sequence is kept for each group."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATG"), id="seq_1", name="seq_1", description=""),  # 3 nt
            SeqRecord(Seq("ATGAAA"), id="seq_2", name="seq_2", description=""),  # 6 nt
            SeqRecord(
                Seq("ATGAAACCC"), id="seq_3", name="seq_3", description=""
            ),  # 9 nt - longest
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            expression=[r"_[0-9]+$"],  # All become "seq"
            mode="longest",
        )

        aggregate_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 1
        assert len(result[0].seq) == 9

    def test_aggregate_no_matches(self, temp_dir, mock_args):
        """Test when regex doesn't match any sequence names."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1", name="seq1", description=""),
            SeqRecord(Seq("ATGCCC"), id="seq2", name="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            expression=[r"XXXX"],  # Won't match anything
            mode="longest",
        )

        aggregate_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # All sequences should remain as no aggregation happened
        assert len(result) == 2

    def test_aggregate_without_expression_does_not_aggregate(self, temp_dir, mock_args):
        """No --expression should keep all records unchanged."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="A-1", name="A-1", description=""),
            SeqRecord(Seq("ATGCCC"), id="A1", name="A1", description=""),
            SeqRecord(Seq("ATG"), id="dup", name="dup", description=""),
            SeqRecord(Seq("CCC"), id="dup", name="dup", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            expression=[],
            mode="longest",
        )

        aggregate_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [r.id for r in result] == ["A-1", "A1", "dup", "dup"]
        assert [str(r.seq) for r in result] == ["ATGAAA", "ATGCCC", "ATG", "CCC"]

    def test_aggregate_wiki_example_colon_pipe(self, temp_dir, mock_args):
        """Test aggregate with wiki example: remove :N and |N suffixes.

        Wiki example: cdskit aggregate --expression ":.*" "\\|.*"
        Input: seq1:1, seq1:2, seq1:3, seq2|1, seq2|2
        Output: longest of seq1 and longest of seq2
        """
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(
                Seq("ATGAAA"), id="seq1:1", name="seq1:1", description=""
            ),  # 6 nt
            SeqRecord(
                Seq("ATGAAACCC"), id="seq1:2", name="seq1:2", description=""
            ),  # 9 nt
            SeqRecord(
                Seq("ATGAAACCCGGGAAATTTCCCGGGAAATTTCCC"),
                id="seq1:3",
                name="seq1:3",
                description="",
            ),  # 33 nt - longest
            SeqRecord(
                Seq("ATGCCC"), id="seq2|1", name="seq2|1", description=""
            ),  # 6 nt
            SeqRecord(
                Seq("ATGCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCC"),
                id="seq2|2",
                name="seq2|2",
                description="",
            ),  # 54 nt - longest
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            expression=[r":.*", r"\|.*"],  # Wiki example expressions
            mode="longest",
        )

        aggregate_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Should have 2 sequences: seq1 (33nt) and seq2 (54nt)
        assert len(result) == 2

        # Verify longest were kept
        # seq1:3 becomes seq1 after removing :3
        seq1_result = next(r for r in result if "seq1" in r.id)
        assert len(seq1_result.seq) == 33
        # seq2|2 becomes seq2 after removing |2
        seq2_result = next(r for r in result if "seq2" in r.id)
        assert len(seq2_result.seq) == 54

    def test_aggregate_rejects_invalid_regex(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("ATGAAA"), id="seq1", name="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            expression=["["],
            mode="longest",
        )
        with pytest.raises(ValueError) as exc_info:
            aggregate_main(args)
        assert "Invalid regex in --expression" in str(exc_info.value)

    def test_aggregate_rejects_non_dna_input(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("PPP"), id="prot1", name="prot1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            expression=[r"prot"],
            mode="longest",
            seqtype="dna",
        )
        with pytest.raises(ValueError) as exc_info:
            aggregate_main(args)
        assert "DNA-only input is required" in str(exc_info.value)

    def test_aggregate_accepts_protein_input_when_seqtype_protein(
        self, temp_dir, mock_args
    ):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [
            SeqRecord(Seq("MKT"), id="geneA.1", name="geneA.1", description=""),
            SeqRecord(Seq("MKTA"), id="geneA.2", name="geneA.2", description=""),
            SeqRecord(Seq("QQQ"), id="geneB.1", name="geneB.1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            expression=[r"\.[0-9]+$"],
            mode="longest",
            seqtype="protein",
        )
        aggregate_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 2
        assert any(record.id == "geneA.2" for record in result)

"""
Tests for cdskit aggregate command.
"""

import pytest
from pathlib import Path

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.aggregate import aggregate_main


class TestAggregateMain:
    """Tests for aggregate_main function."""

    def test_aggregate_by_suffix(self, temp_dir, mock_args):
        """Test aggregating sequences by removing suffix."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # gene_A.1 and gene_A.2 should aggregate to gene_A
        records = [
            SeqRecord(Seq("ATGAAA"), id="gene_A.1", name="gene_A.1", description=""),  # 6 nt
            SeqRecord(Seq("ATGAAACCC"), id="gene_A.2", name="gene_A.2", description=""),  # 9 nt - longer
            SeqRecord(Seq("ATGCCC"), id="gene_B.1", name="gene_B.1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            expression=[r'\.[0-9]+$'],  # Remove .N suffix
            mode='longest',
        )

        aggregate_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Should have 2 sequences: gene_A (longest) and gene_B
        assert len(result) == 2
        # Find gene_A entry - should be the longer one
        gene_a = [r for r in result if "gene_A" in r.id][0]
        assert len(gene_a.seq) == 9

    def test_aggregate_keep_longest(self, temp_dir, mock_args):
        """Test that longest sequence is kept for each group."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATG"), id="seq_1", name="seq_1", description=""),  # 3 nt
            SeqRecord(Seq("ATGAAA"), id="seq_2", name="seq_2", description=""),  # 6 nt
            SeqRecord(Seq("ATGAAACCC"), id="seq_3", name="seq_3", description=""),  # 9 nt - longest
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            expression=[r'_[0-9]+$'],  # All become "seq"
            mode='longest',
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
            expression=[r'XXXX'],  # Won't match anything
            mode='longest',
        )

        aggregate_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # All sequences should remain as no aggregation happened
        assert len(result) == 2

    def test_aggregate_multiple_expressions(self, temp_dir, mock_args):
        """Test aggregating with multiple regex expressions."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Names like "prefix_gene_A_suffix.1"
        records = [
            SeqRecord(Seq("ATGAAA"), id="prefix_gene_suffix.1", name="prefix_gene_suffix.1", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="prefix_gene_suffix.2", name="prefix_gene_suffix.2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            expression=[r'^prefix_', r'_suffix', r'\.[0-9]+$'],  # Remove prefix, suffix, and version
            mode='longest',
        )

        aggregate_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Both should aggregate to "gene"
        assert len(result) == 1

    def test_aggregate_preserves_unique(self, temp_dir, mock_args):
        """Test that unique sequences are preserved."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="gene_A.1", name="gene_A.1", description=""),
            SeqRecord(Seq("ATGCCC"), id="gene_B.1", name="gene_B.1", description=""),
            SeqRecord(Seq("ATGGGG"), id="gene_C.1", name="gene_C.1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            expression=[r'\.[0-9]+$'],
            mode='longest',
        )

        aggregate_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # All 3 are unique after removing suffix
        assert len(result) == 3

    def test_aggregate_with_example_data(self, data_dir, temp_dir, mock_args):
        """Test aggregate with example data if available."""
        input_path = data_dir / "example_aggregate.fasta"
        output_path = temp_dir / "output.fasta"

        if not input_path.exists():
            pytest.skip("example_aggregate.fasta not found")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            expression=[r'_[0-9]+$'],
            mode='longest',
        )

        aggregate_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) > 0

    def test_aggregate_wiki_example_colon_pipe(self, temp_dir, mock_args):
        """Test aggregate with wiki example: remove :N and |N suffixes.

        Wiki example: cdskit aggregate --expression ":.*" "\\|.*"
        Input: seq1:1, seq1:2, seq1:3, seq2|1, seq2|2
        Output: longest of seq1 and longest of seq2
        """
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1:1", name="seq1:1", description=""),  # 6 nt
            SeqRecord(Seq("ATGAAACCC"), id="seq1:2", name="seq1:2", description=""),  # 9 nt
            SeqRecord(Seq("ATGAAACCCGGGAAATTTCCCGGGAAATTTCCC"), id="seq1:3", name="seq1:3", description=""),  # 33 nt - longest
            SeqRecord(Seq("ATGCCC"), id="seq2|1", name="seq2|1", description=""),  # 6 nt
            SeqRecord(Seq("ATGCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCC"), id="seq2|2", name="seq2|2", description=""),  # 54 nt - longest
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            expression=[r':.*', r'\|.*'],  # Wiki example expressions
            mode='longest',
        )

        aggregate_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Should have 2 sequences: seq1 (33nt) and seq2 (54nt)
        assert len(result) == 2

        # Verify longest were kept
        lengths = {r.id: len(r.seq) for r in result}
        # seq1:3 becomes seq1 after removing :3
        seq1_result = [r for r in result if 'seq1' in r.id][0]
        assert len(seq1_result.seq) == 33
        # seq2|2 becomes seq2 after removing |2
        seq2_result = [r for r in result if 'seq2' in r.id][0]
        assert len(seq2_result.seq) == 54

    def test_aggregate_species_isoforms(self, temp_dir, mock_args):
        """Test aggregating species isoforms - common bioinformatics use case."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Multiple isoforms per species
        records = [
            SeqRecord(Seq("ATGAAA"), id="Homo_sapiens_isoform1", name="Homo_sapiens_isoform1", description=""),
            SeqRecord(Seq("ATGAAACCCGGG"), id="Homo_sapiens_isoform2", name="Homo_sapiens_isoform2", description=""),  # longest
            SeqRecord(Seq("ATGCCC"), id="Mus_musculus_isoform1", name="Mus_musculus_isoform1", description=""),  # longest
            SeqRecord(Seq("ATG"), id="Mus_musculus_isoform2", name="Mus_musculus_isoform2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            expression=[r'_isoform[0-9]+$'],  # Remove isoform suffix
            mode='longest',
        )

        aggregate_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 2

        # Homo_sapiens should be 12nt (longest isoform)
        human = [r for r in result if 'Homo' in r.id][0]
        assert len(human.seq) == 12

        # Mus_musculus should be 6nt (longest isoform)
        mouse = [r for r in result if 'Mus' in r.id][0]
        assert len(mouse.seq) == 6

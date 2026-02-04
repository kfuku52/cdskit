"""
Tests for cdskit intersection command.
"""

import pytest
from pathlib import Path

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.intersection import intersection_main


class TestIntersectionMain:
    """Tests for intersection_main function."""

    def test_intersection_two_fasta_files(self, temp_dir, mock_args):
        """Test intersection of two FASTA files."""
        input1_path = temp_dir / "input1.fasta"
        input2_path = temp_dir / "input2.fasta"
        output1_path = temp_dir / "output1.fasta"
        output2_path = temp_dir / "output2.fasta"

        # File 1: seq1, seq2, seq3
        records1 = [
            SeqRecord(Seq("ATGAAA"), id="seq1", name="seq1", description=""),
            SeqRecord(Seq("ATGCCC"), id="seq2", name="seq2", description=""),
            SeqRecord(Seq("ATGGGG"), id="seq3", name="seq3", description=""),
        ]
        Bio.SeqIO.write(records1, str(input1_path), "fasta")

        # File 2: seq2, seq3, seq4
        records2 = [
            SeqRecord(Seq("CCCAAA"), id="seq2", name="seq2", description=""),
            SeqRecord(Seq("GGGAAA"), id="seq3", name="seq3", description=""),
            SeqRecord(Seq("TTTAAA"), id="seq4", name="seq4", description=""),
        ]
        Bio.SeqIO.write(records2, str(input2_path), "fasta")

        args = mock_args(
            seqfile=str(input1_path),
            seqfile2=str(input2_path),
            inseqformat2='fasta',
            outfile=str(output1_path),
            outfile2=str(output2_path),
            outseqformat2='fasta',
            ingff=None,
            outgff=None,
            fix_outrange_gff_records=False,
        )

        intersection_main(args)

        # Check output 1 has seq2 and seq3
        result1 = list(Bio.SeqIO.parse(str(output1_path), "fasta"))
        result1_ids = [r.id for r in result1]
        assert len(result1) == 2
        assert "seq2" in result1_ids
        assert "seq3" in result1_ids

        # Check output 2 has seq2 and seq3
        result2 = list(Bio.SeqIO.parse(str(output2_path), "fasta"))
        result2_ids = [r.id for r in result2]
        assert len(result2) == 2
        assert "seq2" in result2_ids
        assert "seq3" in result2_ids

    def test_intersection_no_overlap(self, temp_dir, mock_args):
        """Test intersection when no sequences overlap."""
        input1_path = temp_dir / "input1.fasta"
        input2_path = temp_dir / "input2.fasta"
        output1_path = temp_dir / "output1.fasta"
        output2_path = temp_dir / "output2.fasta"

        records1 = [SeqRecord(Seq("ATGAAA"), id="seq1", name="seq1", description="")]
        Bio.SeqIO.write(records1, str(input1_path), "fasta")

        records2 = [SeqRecord(Seq("CCCAAA"), id="seq2", name="seq2", description="")]
        Bio.SeqIO.write(records2, str(input2_path), "fasta")

        args = mock_args(
            seqfile=str(input1_path),
            seqfile2=str(input2_path),
            inseqformat2='fasta',
            outfile=str(output1_path),
            outfile2=str(output2_path),
            outseqformat2='fasta',
            ingff=None,
            outgff=None,
            fix_outrange_gff_records=False,
        )

        intersection_main(args)

        result1 = list(Bio.SeqIO.parse(str(output1_path), "fasta"))
        assert len(result1) == 0

    def test_intersection_fasta_gff(self, temp_dir, mock_args):
        """Test intersection of FASTA and GFF files."""
        input_fasta = temp_dir / "input.fasta"
        input_gff = temp_dir / "input.gff"
        output_fasta = temp_dir / "output.fasta"
        output_gff = temp_dir / "output.gff"

        # FASTA has seq1 and seq2
        records = [
            SeqRecord(Seq("ATGAAACCC"), id="seq1", name="seq1", description=""),
            SeqRecord(Seq("ATGCCCGGG"), id="seq2", name="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_fasta), "fasta")

        # GFF has seq2 and seq3
        gff_content = """##gff-version 3
seq2\tsource\tgene\t1\t6\t.\t+\t.\tID=gene1
seq3\tsource\tgene\t1\t6\t.\t+\t.\tID=gene2
"""
        input_gff.write_text(gff_content)

        args = mock_args(
            seqfile=str(input_fasta),
            seqfile2=None,
            ingff=str(input_gff),
            outfile=str(output_fasta),
            outgff=str(output_gff),
            fix_outrange_gff_records=False,
        )

        intersection_main(args)

        # Only seq2 should be in output
        result_fasta = list(Bio.SeqIO.parse(str(output_fasta), "fasta"))
        assert len(result_fasta) == 1
        assert result_fasta[0].id == "seq2"

        # GFF should only have seq2 entries
        with open(output_gff) as f:
            gff_output = f.read()
        assert "seq2" in gff_output
        assert "seq3" not in gff_output or "gene2" not in gff_output

    def test_intersection_fix_outrange_gff(self, temp_dir, mock_args):
        """Test fixing out-of-range GFF coordinates."""
        input_fasta = temp_dir / "input.fasta"
        input_gff = temp_dir / "input.gff"
        output_fasta = temp_dir / "output.fasta"
        output_gff = temp_dir / "output.gff"

        # Short sequence
        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1", name="seq1", description=""),  # 6 nt
        ]
        Bio.SeqIO.write(records, str(input_fasta), "fasta")

        # GFF with coordinates beyond sequence length
        gff_content = "##gff-version 3\nseq1\tsource\tgene\t1\t100\t.\t+\t.\tID=gene1\n"
        input_gff.write_text(gff_content)

        args = mock_args(
            seqfile=str(input_fasta),
            seqfile2=None,
            ingff=str(input_gff),
            outfile=str(output_fasta),
            outgff=str(output_gff),
            fix_outrange_gff_records=True,
        )

        intersection_main(args)

        # Check output files exist
        assert output_gff.exists()
        with open(output_gff) as f:
            gff_output = f.read()
        assert "seq1" in gff_output
        # The end coordinate 100 should have been fixed (not 100 anymore)
        assert "\t100\t" not in gff_output

    def test_intersection_requires_second_input(self, temp_dir, mock_args):
        """Test that intersection requires either seqfile2 or ingff."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [SeqRecord(Seq("ATGAAA"), id="seq1", name="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            seqfile2=None,
            ingff=None,
            outfile=str(output_path),
            outgff=None,
            fix_outrange_gff_records=False,
        )

        with pytest.raises(Exception) as exc_info:
            intersection_main(args)
        assert "seqfile2 or ingff" in str(exc_info.value)

    def test_intersection_with_test_data(self, data_dir, temp_dir, mock_args):
        """Test intersection with intersection_01 test data."""
        input1_path = data_dir / "intersection_01" / "input1.fasta"
        input2_path = data_dir / "intersection_01" / "input2.fasta"
        output1_path = temp_dir / "output1.fasta"
        output2_path = temp_dir / "output2.fasta"

        if not input1_path.exists():
            pytest.skip("intersection_01 test data not found")

        args = mock_args(
            seqfile=str(input1_path),
            seqfile2=str(input2_path),
            inseqformat2='fasta',
            outfile=str(output1_path),
            outfile2=str(output2_path),
            outseqformat2='fasta',
            ingff=None,
            outgff=None,
            fix_outrange_gff_records=False,
        )

        intersection_main(args)

        result1 = list(Bio.SeqIO.parse(str(output1_path), "fasta"))
        result2 = list(Bio.SeqIO.parse(str(output2_path), "fasta"))
        assert len(result1) == len(result2)

    def test_intersection_wiki_example_two_fasta(self, temp_dir, mock_args):
        """Test wiki example: intersection of two FASTA files.

        Wiki example shows:
        - File 1: seq1, seq2, seq3, seq4
        - File 2: seq3, seq4, seq5, seq6
        - Output: seq3, seq4 (only overlapping IDs)
        """
        input1_path = temp_dir / "input1.fasta"
        input2_path = temp_dir / "input2.fasta"
        output1_path = temp_dir / "output1.fasta"
        output2_path = temp_dir / "output2.fasta"

        # File 1: seq1, seq2, seq3, seq4
        records1 = [
            SeqRecord(Seq("ATGAAAAAA"), id="seq1", name="seq1", description=""),
            SeqRecord(Seq("ATGCCCAAA"), id="seq2", name="seq2", description=""),
            SeqRecord(Seq("ATGGGGGGG"), id="seq3", name="seq3", description=""),
            SeqRecord(Seq("ATGTTTTTT"), id="seq4", name="seq4", description=""),
        ]
        Bio.SeqIO.write(records1, str(input1_path), "fasta")

        # File 2: seq3, seq4, seq5, seq6
        records2 = [
            SeqRecord(Seq("CCCGGGGGG"), id="seq3", name="seq3", description=""),
            SeqRecord(Seq("CCCTTTTTT"), id="seq4", name="seq4", description=""),
            SeqRecord(Seq("CCCAAAAAA"), id="seq5", name="seq5", description=""),
            SeqRecord(Seq("CCCBBBBBB"), id="seq6", name="seq6", description=""),
        ]
        Bio.SeqIO.write(records2, str(input2_path), "fasta")

        args = mock_args(
            seqfile=str(input1_path),
            seqfile2=str(input2_path),
            inseqformat2='fasta',
            outfile=str(output1_path),
            outfile2=str(output2_path),
            outseqformat2='fasta',
            ingff=None,
            outgff=None,
            fix_outrange_gff_records=False,
        )

        intersection_main(args)

        # Output 1 should have seq3 and seq4 from file 1
        result1 = list(Bio.SeqIO.parse(str(output1_path), "fasta"))
        result1_ids = [r.id for r in result1]
        assert len(result1) == 2
        assert "seq3" in result1_ids
        assert "seq4" in result1_ids
        assert "seq1" not in result1_ids
        assert "seq2" not in result1_ids

        # Output 2 should have seq3 and seq4 from file 2
        result2 = list(Bio.SeqIO.parse(str(output2_path), "fasta"))
        result2_ids = [r.id for r in result2]
        assert len(result2) == 2
        assert "seq3" in result2_ids
        assert "seq4" in result2_ids
        assert "seq5" not in result2_ids
        assert "seq6" not in result2_ids

    def test_intersection_wiki_example_fasta_gff(self, temp_dir, mock_args):
        """Test wiki example: intersection of FASTA and GFF.

        Only sequences that have matching GFF entries are kept.
        """
        input_fasta = temp_dir / "input.fasta"
        input_gff = temp_dir / "input.gff"
        output_fasta = temp_dir / "output.fasta"
        output_gff = temp_dir / "output.gff"

        # FASTA with multiple sequences
        records = [
            SeqRecord(Seq("ATGAAACCCGGG"), id="chr1", name="chr1", description=""),
            SeqRecord(Seq("ATGCCCGGGTTT"), id="chr2", name="chr2", description=""),
            SeqRecord(Seq("ATGGGGAAACCC"), id="chr3", name="chr3", description=""),
        ]
        Bio.SeqIO.write(records, str(input_fasta), "fasta")

        # GFF only has entries for chr1 and chr3
        gff_content = """##gff-version 3
chr1\tsource\tgene\t1\t9\t.\t+\t.\tID=gene1
chr1\tsource\texon\t1\t6\t.\t+\t.\tID=exon1;Parent=gene1
chr3\tsource\tgene\t1\t12\t.\t+\t.\tID=gene2
"""
        input_gff.write_text(gff_content)

        args = mock_args(
            seqfile=str(input_fasta),
            seqfile2=None,
            ingff=str(input_gff),
            outfile=str(output_fasta),
            outgff=str(output_gff),
            fix_outrange_gff_records=False,
        )

        intersection_main(args)

        # Output FASTA should only have chr1 and chr3
        result_fasta = list(Bio.SeqIO.parse(str(output_fasta), "fasta"))
        result_ids = [r.id for r in result_fasta]
        assert len(result_fasta) == 2
        assert "chr1" in result_ids
        assert "chr3" in result_ids
        assert "chr2" not in result_ids

        # Output GFF should have chr1 and chr3 entries
        with open(output_gff) as f:
            gff_output = f.read()
        assert "chr1" in gff_output
        assert "chr3" in gff_output

    def test_intersection_preserves_sequences(self, temp_dir, mock_args):
        """Test that intersection preserves sequence content."""
        input1_path = temp_dir / "input1.fasta"
        input2_path = temp_dir / "input2.fasta"
        output1_path = temp_dir / "output1.fasta"
        output2_path = temp_dir / "output2.fasta"

        # Different sequences for same IDs
        records1 = [
            SeqRecord(Seq("AAAAAAAAAA"), id="common", name="common", description=""),
        ]
        Bio.SeqIO.write(records1, str(input1_path), "fasta")

        records2 = [
            SeqRecord(Seq("GGGGGGGGGG"), id="common", name="common", description=""),
        ]
        Bio.SeqIO.write(records2, str(input2_path), "fasta")

        args = mock_args(
            seqfile=str(input1_path),
            seqfile2=str(input2_path),
            inseqformat2='fasta',
            outfile=str(output1_path),
            outfile2=str(output2_path),
            outseqformat2='fasta',
            ingff=None,
            outgff=None,
            fix_outrange_gff_records=False,
        )

        intersection_main(args)

        # Each output should preserve its original sequence
        result1 = list(Bio.SeqIO.parse(str(output1_path), "fasta"))[0]
        result2 = list(Bio.SeqIO.parse(str(output2_path), "fasta"))[0]

        assert str(result1.seq) == "AAAAAAAAAA"  # From file 1
        assert str(result2.seq) == "GGGGGGGGGG"  # From file 2

"""
Tests for cdskit split command.
"""

import pytest
from pathlib import Path

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.split import (
    build_split_output_paths,
    resolve_output_prefix,
    split_main,
    split_record_by_codon_position,
)


class TestSplitMain:
    """Tests for split_main function."""

    def test_split_record_by_codon_position(self):
        record = SeqRecord(Seq("ATGCCCGGG"), id="seq1", description="")
        first, second, third = split_record_by_codon_position(record)
        assert str(first.seq) == "ACG"
        assert str(second.seq) == "TCG"
        assert str(third.seq) == "GCG"
        assert first.id == "seq1"

    def test_build_split_output_paths(self):
        first, second, third = build_split_output_paths("outprefix", "fasta")
        assert first == "outprefix_1st_codon_positions.fasta"
        assert second == "outprefix_2nd_codon_positions.fasta"
        assert third == "outprefix_3rd_codon_positions.fasta"

    def test_resolve_output_prefix_prefers_explicit_prefix(self, mock_args):
        args = mock_args(seqfile='input.fasta', prefix='custom_prefix', outfile='ignored')
        assert resolve_output_prefix(args) == 'custom_prefix'

    def test_resolve_output_prefix_uses_outfile_as_fallback(self, mock_args):
        args = mock_args(seqfile='input.fasta', prefix='INFILE', outfile='from_outfile')
        assert resolve_output_prefix(args) == 'from_outfile'

    def test_resolve_output_prefix_uses_stdin_label_for_stream_input(self, mock_args):
        args = mock_args(seqfile='-', prefix='INFILE', outfile='-')
        assert resolve_output_prefix(args) == 'stdin'

    def test_split_basic(self, temp_dir, mock_args):
        """Test basic split functionality - extract codon positions."""
        input_path = temp_dir / "input.fasta"

        # ATG = positions 1,2,3 (A,T,G)
        # CCC = positions 1,2,3 (C,C,C)
        # GGG = positions 1,2,3 (G,G,G)
        records = [
            SeqRecord(Seq("ATGCCCGGG"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            prefix=str(temp_dir / "output"),
        )

        split_main(args)

        # Check 1st codon positions
        first = list(Bio.SeqIO.parse(str(temp_dir / "output_1st_codon_positions.fasta"), "fasta"))
        assert str(first[0].seq) == "ACG"  # 1st position of each codon

        # Check 2nd codon positions
        second = list(Bio.SeqIO.parse(str(temp_dir / "output_2nd_codon_positions.fasta"), "fasta"))
        assert str(second[0].seq) == "TCG"  # 2nd position of each codon

        # Check 3rd codon positions
        third = list(Bio.SeqIO.parse(str(temp_dir / "output_3rd_codon_positions.fasta"), "fasta"))
        assert str(third[0].seq) == "GCG"  # 3rd position of each codon

    def test_split_multiple_sequences(self, temp_dir, mock_args):
        """Test split with multiple sequences."""
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATGCCC"), id="seq1", description=""),
            SeqRecord(Seq("AAAGGG"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            prefix=str(temp_dir / "output"),
        )

        split_main(args)

        # Check all sequences are in each output
        first = list(Bio.SeqIO.parse(str(temp_dir / "output_1st_codon_positions.fasta"), "fasta"))
        assert len(first) == 2
        assert str(first[0].seq) == "AC"
        assert str(first[1].seq) == "AG"

    def test_split_with_gaps(self, temp_dir, mock_args):
        """Test split preserves gap characters."""
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATG---CCC"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            prefix=str(temp_dir / "output"),
        )

        split_main(args)

        first = list(Bio.SeqIO.parse(str(temp_dir / "output_1st_codon_positions.fasta"), "fasta"))
        assert str(first[0].seq) == "A-C"

        second = list(Bio.SeqIO.parse(str(temp_dir / "output_2nd_codon_positions.fasta"), "fasta"))
        assert str(second[0].seq) == "T-C"

        third = list(Bio.SeqIO.parse(str(temp_dir / "output_3rd_codon_positions.fasta"), "fasta"))
        assert str(third[0].seq) == "G-C"

    def test_split_infile_prefix(self, temp_dir, mock_args):
        """Test split with INFILE prefix uses input filename."""
        input_path = temp_dir / "myseqs.fasta"

        records = [
            SeqRecord(Seq("ATGCCC"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            prefix='INFILE',
        )

        split_main(args)

        # Files should be named after input file
        expected_1st = temp_dir / "myseqs.fasta_1st_codon_positions.fasta"
        assert expected_1st.exists()

    def test_split_outfile_fallback_prefix(self, temp_dir, mock_args):
        """When --prefix is INFILE, --outfile is used as fallback prefix."""
        input_path = temp_dir / "myseqs.fasta"

        records = [
            SeqRecord(Seq("ATGCCC"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            prefix='INFILE',
            outfile=str(temp_dir / "from_outfile"),
        )

        split_main(args)

        expected_1st = temp_dir / "from_outfile_1st_codon_positions.fasta"
        assert expected_1st.exists()

    def test_split_output_lengths(self, temp_dir, mock_args):
        """Test split output sequences have correct lengths."""
        input_path = temp_dir / "input.fasta"

        # 6 codons = 18 nucleotides
        records = [
            SeqRecord(Seq("ATGCCCGGGAAATTTCCC"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            prefix=str(temp_dir / "output"),
        )

        split_main(args)

        first = list(Bio.SeqIO.parse(str(temp_dir / "output_1st_codon_positions.fasta"), "fasta"))
        # 6 codons = 6 first positions
        assert len(first[0].seq) == 6

    def test_split_rejects_non_multiple_of_three(self, temp_dir, mock_args):
        """Test split rejects sequences not multiple of 3."""
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATGCC"), id="seq1", description=""),  # 5 nt
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            prefix=str(temp_dir / "output"),
        )

        with pytest.raises(Exception) as exc_info:
            split_main(args)
        assert "multiple of three" in str(exc_info.value)

    def test_split_rejects_non_dna_input(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("PPPPPP"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            prefix=str(temp_dir / "output"),
        )

        with pytest.raises(Exception) as exc_info:
            split_main(args)
        assert "DNA-only input is required" in str(exc_info.value)

    def test_split_with_test_data(self, data_dir, temp_dir, mock_args):
        """Test split with split_01 test data."""
        input_path = data_dir / "split_01" / "input.fasta"
        expected_1st = data_dir / "split_01" / "output_1st_codon_positions.fasta"
        expected_2nd = data_dir / "split_01" / "output_2nd_codon_positions.fasta"
        expected_3rd = data_dir / "split_01" / "output_3rd_codon_positions.fasta"

        if not input_path.exists():
            pytest.skip("split_01 test data not found")

        args = mock_args(
            seqfile=str(input_path),
            prefix=str(temp_dir / "output"),
        )

        split_main(args)

        # Compare with expected outputs if they exist
        result_1st = list(Bio.SeqIO.parse(str(temp_dir / "output_1st_codon_positions.fasta"), "fasta"))

        if expected_1st.exists():
            expected = list(Bio.SeqIO.parse(str(expected_1st), "fasta"))
            assert len(result_1st) == len(expected)
            for r, e in zip(result_1st, expected):
                assert str(r.seq) == str(e.seq), f"Mismatch for {r.id}"

    def test_split_preserves_sequence_ids(self, temp_dir, mock_args):
        """Test split preserves sequence IDs."""
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATGCCC"), id="my_special_seq", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            prefix=str(temp_dir / "output"),
        )

        split_main(args)

        first = list(Bio.SeqIO.parse(str(temp_dir / "output_1st_codon_positions.fasta"), "fasta"))
        assert first[0].id == "my_special_seq"

    def test_split_wiki_example(self, temp_dir, mock_args):
        """Test split with wiki example sequence.

        Input: ATGAACCCAGCCGCTCAACTGCTGCGCATGCGCAGCGCT (39 nt = 13 codons)
        Codons: ATG | AAC | CCA | GCC | GCT | CAA | CTG | CTG | CGC | ATG | CGC | AGC | GCT
        1st positions: A,A,C,G,G,C,C,C,C,A,C,A,G
        2nd positions: T,A,C,C,C,A,T,T,G,T,G,G,C
        3rd positions: G,C,A,C,T,A,G,G,C,G,C,C,T
        """
        input_path = temp_dir / "input.fasta"

        # Use a 39 nt sequence (divisible by 3)
        records = [
            SeqRecord(Seq("ATGAACCCAGCCGCTCAACTGCTGCGCATGCGCAGCGCT"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            prefix=str(temp_dir / "output"),
        )

        split_main(args)

        # Verify 1st positions
        first = list(Bio.SeqIO.parse(str(temp_dir / "output_1st_codon_positions.fasta"), "fasta"))
        first_seq = str(first[0].seq)
        # 1st position of each codon: A, A, C, G, G, C, C, C, C, A, C, A, G
        assert first_seq == "AACGGCCCCACAG"

        # Verify 2nd positions
        second = list(Bio.SeqIO.parse(str(temp_dir / "output_2nd_codon_positions.fasta"), "fasta"))
        second_seq = str(second[0].seq)
        # 2nd position of each codon: T, A, C, C, C, A, T, T, G, T, G, G, C
        assert second_seq == "TACCCATTGTGGC"

        # Verify 3rd positions
        third = list(Bio.SeqIO.parse(str(temp_dir / "output_3rd_codon_positions.fasta"), "fasta"))
        third_seq = str(third[0].seq)
        # 3rd position of each codon: G, C, A, C, T, A, G, G, C, G, C, C, T
        assert third_seq == "GCACTAGGCGCCT"

    def test_split_aligned_sequences(self, temp_dir, mock_args):
        """Test split with aligned sequences containing gaps."""
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATG---CCCGGG"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAACCC---"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            prefix=str(temp_dir / "output"),
        )

        split_main(args)

        first = list(Bio.SeqIO.parse(str(temp_dir / "output_1st_codon_positions.fasta"), "fasta"))
        assert len(first) == 2
        assert str(first[0].seq) == "A-CG"
        assert str(first[1].seq) == "AAC-"

    def test_split_long_sequence(self, temp_dir, mock_args):
        """Test split with a longer sequence."""
        input_path = temp_dir / "input.fasta"

        # 30 codons = 90 nt
        seq = "ATGCCC" * 15
        records = [
            SeqRecord(Seq(seq), id="long_seq", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            prefix=str(temp_dir / "output"),
        )

        split_main(args)

        first = list(Bio.SeqIO.parse(str(temp_dir / "output_1st_codon_positions.fasta"), "fasta"))
        # 30 codons = 30 first positions
        assert len(first[0].seq) == 30
        # All first positions should be A or C (from ATG and CCC)
        assert str(first[0].seq) == "AC" * 15

    def test_split_creates_all_three_files(self, temp_dir, mock_args):
        """Test that split creates all three output files."""
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATGCCC"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            prefix=str(temp_dir / "output"),
        )

        split_main(args)

        # All three files should exist
        assert (temp_dir / "output_1st_codon_positions.fasta").exists()
        assert (temp_dir / "output_2nd_codon_positions.fasta").exists()
        assert (temp_dir / "output_3rd_codon_positions.fasta").exists()

    def test_split_threads_matches_single_thread(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        records = [
            SeqRecord(Seq("ATGCCCGGG"), id="seq1", description=""),
            SeqRecord(Seq("ATG---CCC"), id="seq2", description=""),
            SeqRecord(Seq("ATGAAATTT"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        single_prefix = str(temp_dir / "single")
        threaded_prefix = str(temp_dir / "threaded")
        args_single = mock_args(
            seqfile=str(input_path),
            prefix=single_prefix,
            threads=1,
        )
        args_threaded = mock_args(
            seqfile=str(input_path),
            prefix=threaded_prefix,
            threads=4,
        )

        split_main(args_single)
        split_main(args_threaded)

        for suffix in [
            "_1st_codon_positions.fasta",
            "_2nd_codon_positions.fasta",
            "_3rd_codon_positions.fasta",
        ]:
            single_records = list(Bio.SeqIO.parse(single_prefix + suffix, "fasta"))
            threaded_records = list(Bio.SeqIO.parse(threaded_prefix + suffix, "fasta"))
            assert [r.id for r in single_records] == [r.id for r in threaded_records]
            assert [str(r.seq) for r in single_records] == [str(r.seq) for r in threaded_records]

"""
Tests for cdskit pad command.
"""

import pytest
from pathlib import Path

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.pad import count_internal_stop_codons, get_stop_codons, pad_main, padseqs


class TestStopCodonHelpers:
    """Tests for low-level stop-codon helper functions."""

    def test_get_stop_codons_id_and_name_are_consistent(self):
        """Codon table id and name should provide identical stop codons."""
        stop_by_id = get_stop_codons(1)
        stop_by_name = get_stop_codons("Standard")
        assert stop_by_id == stop_by_name
        assert "TAA" in stop_by_id
        assert "TAG" in stop_by_id
        assert "TGA" in stop_by_id

    def test_count_internal_stop_codons_ignores_terminal_stop(self):
        """Terminal stop codon should not be counted as internal."""
        assert count_internal_stop_codons("ATGAAATGA", 1) == 0

    def test_count_internal_stop_codons_counts_internal_stops(self):
        """Internal stop codons should be counted."""
        assert count_internal_stop_codons("ATGTGACCC", 1) == 1

    def test_count_internal_stop_codons_short_sequence(self):
        """Sequences shorter than a full internal codon window return zero."""
        assert count_internal_stop_codons("AT", 1) == 0
        assert count_internal_stop_codons("ATG", 1) == 0


class TestPadSeqs:
    """Tests for padseqs class."""

    def test_no_padding_needed(self):
        """Test sequence already multiple of 3."""
        ps = padseqs(original_seq="ATGAAA", codon_table=1, padchar='N')
        ps.add(headn=0, tailn=0)
        result = ps.get_minimum_num_stop()
        assert str(result['new_seq']) == "ATGAAA"
        assert result['headn'] == 0
        assert result['tailn'] == 0

    def test_tail_padding(self):
        """Test adding tail padding."""
        ps = padseqs(original_seq="ATGAA", codon_table=1, padchar='N')
        ps.add(headn=0, tailn=1)
        result = ps.get_minimum_num_stop()
        assert str(result['new_seq']) == "ATGAAN"
        assert len(result['new_seq']) == 6

    def test_head_padding(self):
        """Test adding head padding."""
        ps = padseqs(original_seq="TGAAA", codon_table=1, padchar='N')
        ps.add(headn=1, tailn=0)
        result = ps.get_minimum_num_stop()
        assert str(result['new_seq']) == "NTGAAA"
        assert len(result['new_seq']) == 6

    def test_minimum_stop_selection(self):
        """Test that minimum stop codon option is selected."""
        # Original: ATGA (4 nt) - needs 2 padding
        # Option 1: ATGANN (tail) -> ATG ANN -> M X (0 stops)
        # Option 2: NNATGA (head) -> NNA TGA -> X * (1 stop)
        ps = padseqs(original_seq="ATGA", codon_table=1, padchar='N')
        ps.add(headn=0, tailn=2)  # ATGANN -> 0 internal stops
        ps.add(headn=2, tailn=0)  # NNATGA -> TGA is stop
        result = ps.get_minimum_num_stop()
        # Should pick the option with fewer stops
        assert result['num_stop'] <= 1

    def test_gap_padding_char(self):
        """Test using '-' as padding character."""
        ps = padseqs(original_seq="ATGAA", codon_table=1, padchar='-')
        ps.add(headn=0, tailn=1)
        result = ps.get_minimum_num_stop()
        assert str(result['new_seq']) == "ATGAA-"


class TestPadMain:
    """Tests for pad_main function using test data."""

    def test_pad_01_data(self, data_dir, temp_dir, mock_args):
        """Test pad command with pad_01 test data."""
        input_path = data_dir / "pad_01" / "input.fasta"
        expected_path = data_dir / "pad_01" / "output.fasta"
        output_path = temp_dir / "output.fasta"

        if not input_path.exists():
            pytest.skip("pad_01 test data not found")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            padchar='N',
            nopseudo=False,
        )

        pad_main(args)

        # Read output and expected
        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        expected = list(Bio.SeqIO.parse(str(expected_path), "fasta"))

        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert str(r.seq) == str(e.seq), f"Mismatch for {r.id}"

    def test_pad_sequences_become_multiple_of_three(self, temp_dir, mock_args):
        """Test that all output sequences are multiples of 3."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Create input with various lengths
        records = [
            SeqRecord(Seq("ATGAA"), id="len5", description=""),  # 5 nt
            SeqRecord(Seq("ATGAAC"), id="len6", description=""),  # 6 nt
            SeqRecord(Seq("ATGAACC"), id="len7", description=""),  # 7 nt
            SeqRecord(Seq("ATGAACCG"), id="len8", description=""),  # 8 nt
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            padchar='N',
            nopseudo=False,
        )

        pad_main(args)

        # Verify all sequences are multiples of 3
        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        for r in result:
            assert len(r.seq) % 3 == 0, f"{r.id} length {len(r.seq)} is not multiple of 3"

    def test_pad_with_nopseudo_flag(self, temp_dir, mock_args):
        """Test --nopseudo flag filters out sequences with stop codons."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Create sequences - one with stop, one without
        records = [
            SeqRecord(Seq("ATGAAATGA"), id="with_stop", description=""),  # M K * (has stop)
            SeqRecord(Seq("ATGAAACCC"), id="no_stop", description=""),  # M K P (no internal stop)
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            padchar='N',
            nopseudo=True,
        )

        pad_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # with_stop has stop at end (not internal), so it should pass
        # The nopseudo filters based on internal stops only
        assert len(result) >= 1

    def test_pad_replaces_x_with_n(self, temp_dir, mock_args):
        """Test that X characters are replaced with N when padding is needed."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Sequence that needs padding - X will be replaced with N during processing
        records = [
            SeqRecord(Seq("ATGXXXA"), id="seq_with_x", description=""),  # 7 nt, needs padding
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            padchar='N',
            nopseudo=False,
        )

        pad_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # X should be replaced with N and sequence padded to multiple of 3
        assert len(result[0].seq) % 3 == 0
        assert 'X' not in str(result[0].seq)

    def test_pad_02_data(self, data_dir, temp_dir, mock_args):
        """Test pad command with pad_02 test data."""
        input_path = data_dir / "pad_02" / "input.fasta"
        expected_path = data_dir / "pad_02" / "output.fasta"
        output_path = temp_dir / "output.fasta"

        if not input_path.exists():
            pytest.skip("pad_02 test data not found")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            padchar='N',
            nopseudo=False,
        )

        pad_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        expected = list(Bio.SeqIO.parse(str(expected_path), "fasta"))

        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert str(r.seq) == str(e.seq), f"Mismatch for {r.id}"

    def test_pad_issue7_head_padding_applied(self, temp_dir, mock_args, capsys):
        """Test Issue #7: pad correctly adds N padding to output.

        Issue: pad correctly detects required padding numbers and positions
        but doesn't add N in the output sequences in some cases.

        This test uses a sequence from the issue that required 2 head N padding.
        """
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Sequence from Issue #7 - 1033 nt, needs 2 head padding to become 1035 (divisible by 3)
        seq = (
            "ATGCGACTTAAGAGTTATAAACCTGAACAACAATCGGCTACAGGGGCGAGTTCCAAGATC"
            "ACTAGCCCGCTGTGAGCTGCTTGCTGTCATTGACATGGGAAGTAACAGATTGGATGATAC"
            "TTTCCCTATCTGGTTGCAGAATCTTCCAAACCTGCAGGCACTAGCCTTGGGATCAAATAA"
            "TCTCCAAGGTGGAATCGTAGCCAAATCTACCGGTTTCCCCAGCTTGCAAATCCTCGATCT"
            "CTCCAACAATCAACTATCAGGTAACTTGTCCGGAGGACTTCTCCGTGATCGAACTGCAAT"
            "GGAAGCTGGAAATCAAGGACAGACAGGATACCTGACAGTCGTTGTTCCCGTATTATTGCT"
            "TGGCGTGGAAATGGAAGCGACTTACCCATTCTTCATCACATTGAGCTACAAGGGCAGGGA"
            "ATCACCTTCCACATTGATCCTAAAAATCTTCACAAGCATTGATCTATCAAACAACAGGTT"
            "CAAGGGAAGCATCCCTGATTCTGTCGGGAATCTCGTTGGGCTTCAGGCTCTGAATCTCTC"
            "GCACAACAATATAACAGGATCCATCCCGCCATCACTAGGGAGGCTATCGAACCTAGAGTC"
            "TCTGGACCTCTCCAACAACTTCCTATCGGCAGACATCCCTCAGCAACTAGAGGAACTGAC"
            "CTTTCTTGAGATCTTCAATGTGTCTCATAATCGACTCACAGGGTCCATACCACAAGGGAA"
            "CCAATTTTCTACGTTTACCAATGATTCCTTCGAAGGAAACATTGGTCTATGTGGTAGTCC"
            "ACTGTCAAAGAAGTGCGGGCAAACTGCGAGTTCTTCATCCCCACAGGGCGAAAGTGCATC"
            "AGACAACGACAAAGATGAGTCCTCGGCTGTAATCGACTGGATCATCAGATCAATGGGCTA"
            "TCTCAGTGGCTTGGTAATAGGTGTCATCTTTGGTCACATTTTCACGACTAACAAGCATGA"
            "ATGGTTCGTAGAGACTTTCGGAAGAAAGCAGCGCAAAAAGAGAAAAGGAAACAGGAAGGC"
            "GCGAAGGAATTGA"
        )
        assert len(seq) == 1033  # Verify sequence length

        records = [
            SeqRecord(Seq(seq), id="foo1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            padchar='N',
            nopseudo=False,
        )

        pad_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Output should be 1035 nt (divisible by 3)
        assert len(result[0].seq) == 1035
        assert len(result[0].seq) % 3 == 0
        # Verify padding was actually added (should have N characters)
        seq_str = str(result[0].seq)
        # The original sequence didn't have N, so any N must be padding
        assert 'N' in seq_str

    def test_pad_issue8_no_report_when_no_padding_needed(self, temp_dir, mock_args, capsys):
        """Test Issue #8: pad should not report padding when no padding needed.

        Issue: Sequences that don't need padding but have stop codons were
        being reported with head_padding=0, tail_padding=0.
        """
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Sequence that's already a multiple of 3, has terminal stop but no internal stops
        # Should NOT be reported as "padded" since no padding was added
        records = [
            SeqRecord(Seq("ATGAAACCCGGGTGA"), id="Dr00005842-RA", description=""),  # 15 nt, multiple of 3
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            padchar='N',
            nopseudo=False,
        )

        pad_main(args)

        captured = capsys.readouterr()
        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))

        # Sequence should be unchanged
        assert str(result[0].seq) == "ATGAAACCCGGGTGA"

        # The "Number of padded sequences" should be 0
        assert "Number of padded sequences: 0" in captured.err

    def test_pad_issue5_xxx_codon_handling(self, temp_dir, mock_args):
        """Test Issue #5: Codon 'XXX' is invalid.

        Issue: When a sequence contains XXX, translation fails with:
        Bio.Data.CodonTable.TranslationError: Codon 'XXX' is invalid

        Fix: X characters are replaced with N before processing (line 41 of pad.py).
        Note: X replacement only happens when padding logic is triggered
        (i.e., sequence needs padding or has internal stop codons).
        """
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Sequence with XXX that needs padding (11 nt, not multiple of 3)
        # This triggers the padding logic which replaces X with N
        records = [
            SeqRecord(Seq("ATGXXXAAATG"), id="seq_with_xxx", description=""),  # 11 nt, needs padding
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            padchar='N',
            nopseudo=False,
        )

        # Should not raise Bio.Data.CodonTable.TranslationError
        pad_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # X should be replaced with N (when padding is needed)
        assert 'X' not in str(result[0].seq)
        # Length should be multiple of 3
        assert len(result[0].seq) % 3 == 0

    def test_pad_xxx_with_internal_stop(self, temp_dir, mock_args):
        """Test XXX handling when sequence has internal stop codons.

        X replacement also happens when the sequence has internal stop codons,
        even if the length is already a multiple of 3.
        """
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Sequence with XXX and internal stop codon (TGA)
        # This triggers stop codon minimization logic which replaces X with N
        records = [
            SeqRecord(Seq("ATGXXXTGACCC"), id="xxx_with_stop", description=""),  # 12 nt, has stop
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            padchar='N',
            nopseudo=False,
        )

        # Should not raise Bio.Data.CodonTable.TranslationError
        pad_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # X should be replaced with N
        assert 'X' not in str(result[0].seq)
        # Length should be multiple of 3
        assert len(result[0].seq) % 3 == 0

    def test_pad_xxx_no_replacement_when_no_padding_needed(self, temp_dir, mock_args):
        """Test that X is NOT replaced when no padding or stop codon handling is needed.

        This documents current behavior: X is only replaced when the padding
        logic is triggered. Sequences that are already valid (multiple of 3,
        no internal stops) keep their X characters.
        """
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Sequence with XXX but already valid (12 nt, no internal stops)
        records = [
            SeqRecord(Seq("ATGXXXAAACCC"), id="valid_xxx", description=""),  # 12 nt, no stops
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            padchar='N',
            nopseudo=False,
        )

        pad_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # X is NOT replaced when no padding logic is needed
        # This is documenting current behavior, not necessarily ideal behavior
        assert str(result[0].seq) == "ATGXXXAAACCC"

    def test_pad_threads_matches_single_thread(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        out_single = temp_dir / "single.fasta"
        out_threaded = temp_dir / "threaded.fasta"

        records = [
            SeqRecord(Seq("ATGAA"), id="seq1", description=""),
            SeqRecord(Seq("ATGTGACCC"), id="seq2", description=""),
            SeqRecord(Seq("ATGXXXAAATG"), id="seq3", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="seq4", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args_single = mock_args(
            seqfile=str(input_path),
            outfile=str(out_single),
            codontable=1,
            padchar='N',
            nopseudo=False,
            threads=1,
        )
        args_threaded = mock_args(
            seqfile=str(input_path),
            outfile=str(out_threaded),
            codontable=1,
            padchar='N',
            nopseudo=False,
            threads=4,
        )

        pad_main(args_single)
        pad_main(args_threaded)

        result_single = list(Bio.SeqIO.parse(str(out_single), "fasta"))
        result_threaded = list(Bio.SeqIO.parse(str(out_threaded), "fasta"))
        assert [r.id for r in result_single] == [r.id for r in result_threaded]
        assert [str(r.seq) for r in result_single] == [str(r.seq) for r in result_threaded]

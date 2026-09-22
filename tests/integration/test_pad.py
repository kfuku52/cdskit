"""
Tests for cdskit pad command.
"""

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.pad import get_stop_codons, pad_main, padseqs


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


class TestPadSeqs:
    """Tests for padseqs class."""

    def test_tail_padding(self):
        """Test adding tail padding."""
        ps = padseqs(original_seq="ATGAA", codon_table=1, padchar="N")
        ps.add(headn=0, tailn=1)
        result = ps.get_minimum_num_stop()
        assert str(result["new_seq"]) == "ATGAAN"
        assert len(result["new_seq"]) == 6

    def test_head_padding(self):
        """Test adding head padding."""
        ps = padseqs(original_seq="TGAAA", codon_table=1, padchar="N")
        ps.add(headn=1, tailn=0)
        result = ps.get_minimum_num_stop()
        assert str(result["new_seq"]) == "NTGAAA"
        assert len(result["new_seq"]) == 6

    def test_gap_padding_char(self):
        """Test using '-' as padding character."""
        ps = padseqs(original_seq="ATGAA", codon_table=1, padchar="-")
        ps.add(headn=0, tailn=1)
        result = ps.get_minimum_num_stop()
        assert str(result["new_seq"]) == "ATGAA-"


class TestPadMain:
    """Tests for pad_main function using test data."""

    def test_pad_01_data(self, data_dir, temp_dir, mock_args):
        """Test pad command with pad_01 test data."""
        input_path = data_dir / "pad_01" / "input.fasta"
        expected_path = data_dir / "pad_01" / "output.fasta"
        output_path = temp_dir / "output.fasta"

        assert input_path.exists(), "required tracked fixture pad_01 is missing"

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            padchar="N",
            nopseudo=False,
        )

        pad_main(args)

        # Read output and expected
        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        expected = list(Bio.SeqIO.parse(str(expected_path), "fasta"))

        assert len(result) == len(expected)
        for r, e in zip(result, expected, strict=False):
            assert str(r.seq) == str(e.seq), f"Mismatch for {r.id}"

    def test_pad_replaces_x_with_n(self, temp_dir, mock_args):
        """Normalize X for both incomplete codons and internal-stop repair."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Sequence that needs padding - X will be replaced with N during processing
        records = [
            SeqRecord(
                Seq("ATGXXXA"), id="seq_with_x", description=""
            ),  # 7 nt, needs padding
            SeqRecord(Seq("ATGXXXTGACCC"), id="internal_stop", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            padchar="N",
            nopseudo=False,
        )

        pad_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # X should be replaced with N and sequence padded to multiple of 3
        assert [record.id for record in result] == ["seq_with_x", "internal_stop"]
        for record in result:
            assert len(record.seq) % 3 == 0
            assert "X" not in str(record.seq)

    def test_pad_02_data(self, data_dir, temp_dir, mock_args):
        """Test pad command with pad_02 test data."""
        input_path = data_dir / "pad_02" / "input.fasta"
        expected_path = data_dir / "pad_02" / "output.fasta"
        output_path = temp_dir / "output.fasta"

        assert input_path.exists(), "required tracked fixture pad_02 is missing"

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            padchar="N",
            nopseudo=False,
        )

        pad_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        expected = list(Bio.SeqIO.parse(str(expected_path), "fasta"))

        assert len(result) == len(expected)
        for r, e in zip(result, expected, strict=False):
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
            padchar="N",
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
        assert "N" in seq_str

    def test_pad_issue8_no_report_when_no_padding_needed(
        self, temp_dir, mock_args, capsys
    ):
        """Test Issue #8: pad should not report padding when no padding needed.

        Issue: Sequences that don't need padding but have stop codons were
        being reported with head_padding=0, tail_padding=0.
        """
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Sequence that's already a multiple of 3, has terminal stop but no internal stops
        # Should NOT be reported as "padded" since no padding was added
        records = [
            SeqRecord(
                Seq("ATGAAACCCGGGTGA"), id="Dr00005842-RA", description=""
            ),  # 15 nt, multiple of 3
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            padchar="N",
            nopseudo=False,
        )

        pad_main(args)

        captured = capsys.readouterr()
        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))

        # Sequence should be unchanged
        assert str(result[0].seq) == "ATGAAACCCGGGTGA"

        # The "Number of padded sequences" should be 0
        assert "Number of padded sequences: 0" in captured.err

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
            SeqRecord(
                Seq("ATGXXXAAACCC"), id="valid_xxx", description=""
            ),  # 12 nt, no stops
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            padchar="N",
            nopseudo=False,
        )

        pad_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # X is NOT replaced when no padding logic is needed
        # This is documenting current behavior, not necessarily ideal behavior
        assert str(result[0].seq) == "ATGXXXAAACCC"

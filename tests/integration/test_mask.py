"""
Tests for cdskit mask command.
"""

import pytest

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.mask import (
    mask_main,
    mask_partial_gap_codons,
    mask_sequence_string,
)


class TestMaskHelpers:
    """Tests for mask helper functions."""

    @pytest.mark.parametrize("ambiguous", [False, True])
    def test_dot_and_mixed_gap_codons(self, ambiguous):
        codons = ["atg", "A.G", ".--", "...", "-.-", "A-G", "???"]
        expected = "atgNNN.--...-.-NNN" + ("NNN" if ambiguous else "???")
        assert (
            mask_sequence_string("".join(codons), 1, "NNN", ambiguous, False)
            == expected
        )
        assert mask_partial_gap_codons(codons, "NNN") is True
        assert codons == ["atg", "NNN", ".--", "...", "-.-", "NNN", "???"]

    def test_iupac_codons_use_translation_semantics(self):
        assert (
            mask_sequence_string(
                "TGY",
                codontable=1,
                mask_triplet="NNN",
                mask_ambiguous=True,
                mask_stop=False,
            )
            == "TGY"
        )
        assert (
            mask_sequence_string(
                "TAR",
                codontable=1,
                mask_triplet="NNN",
                mask_ambiguous=False,
                mask_stop=True,
            )
            == "NNN"
        )


class TestMaskMain:
    """Tests for mask_main function."""

    def test_mask_partial_gap_codons(self, temp_dir, mock_args):
        """Test masking codons with partial gaps (1 or 2 gaps)."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # ATG = no gap, A-G = partial gap (should be masked), --- = full gap (no mask)
        records = [
            SeqRecord(Seq("ATGA-G---"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            maskchar="N",
            ambiguouscodon="no",
            stopcodon="no",
        )

        mask_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        seq = str(result[0].seq)
        # First codon (ATG) unchanged, second codon (A-G) masked to NNN, third (---) unchanged
        assert seq == "ATGNNN---"

    def test_mask_ambiguous_codons(self, temp_dir, mock_args):
        """Test masking ambiguous codons (translate to X)."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # ANN translates to X (ambiguous)
        records = [
            SeqRecord(Seq("ATGANNTGA"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            maskchar="N",
            ambiguouscodon="yes",
            stopcodon="no",
        )

        mask_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        seq = str(result[0].seq)
        # ANN should be masked to NNN
        assert seq == "ATGNNNTGA"

    def test_mask_both_ambiguous_and_stop(self, temp_dir, mock_args):
        """Test masking both ambiguous and stop codons."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # ANN = X (ambiguous), TGA = * (stop) - 12 nt total
        records = [
            SeqRecord(Seq("ANNANNTGAAAA"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            maskchar="N",
            ambiguouscodon="yes",
            stopcodon="yes",
        )

        mask_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Ambiguous and stop codons should be masked
        seq = str(result[0].seq)
        assert seq == "NNNNNNNNNAAA"

    def test_mask_with_gap_character(self, temp_dir, mock_args):
        """Test masking with '-' as mask character."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGANNTGA"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            maskchar="-",
            ambiguouscodon="yes",
            stopcodon="no",
        )

        mask_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        seq = str(result[0].seq)
        assert seq == "ATG---TGA"

    def test_mask_consecutive_stop_codons(self, temp_dir, mock_args):
        """Test masking consecutive stop codons."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Two consecutive stop codons
        records = [
            SeqRecord(Seq("ATGTGATGAAAA"), id="seq1", description=""),  # M * * K
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            maskchar="N",
            ambiguouscodon="no",
            stopcodon="yes",
        )

        mask_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        seq = str(result[0].seq)
        # Both TGA codons should be masked
        assert seq == "ATGNNNNNN" + "AAA"

    def test_mask_multiple_sequences(self, temp_dir, mock_args):
        """Test masking multiple sequences at once."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGTGAAAA"), id="seq1", description=""),  # Has stop
            SeqRecord(Seq("ATGNNNAAA"), id="seq2", description=""),  # Has ambiguous
            SeqRecord(Seq("ATGAAACCC"), id="seq3", description=""),  # Clean
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            maskchar="N",
            ambiguouscodon="yes",
            stopcodon="yes",
        )

        mask_main(args)

        result = {r.id: str(r.seq) for r in Bio.SeqIO.parse(str(output_path), "fasta")}
        assert len(result) == 3
        assert result["seq1"] == "ATGNNNAAA"  # TGA masked
        assert result["seq2"] == "ATGNNNAAA"  # NNN already masked
        assert result["seq3"] == "ATGAAACCC"  # No change

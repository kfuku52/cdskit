"""
Tests for cdskit mask command.
"""

import pytest
from pathlib import Path

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.mask import (
    mask_main,
    mask_partial_gap_codons,
    should_mask_amino_acid,
)


class TestMaskHelpers:
    """Tests for mask helper functions."""

    def test_mask_partial_gap_codons(self):
        codons = ["ATG", "A-G", "---", "T-A"]
        changed = mask_partial_gap_codons(codons, "NNN")
        assert changed is True
        assert codons == ["ATG", "NNN", "---", "NNN"]

    def test_should_mask_amino_acid(self):
        assert should_mask_amino_acid("X", mask_ambiguous=True, mask_stop=False) is True
        assert should_mask_amino_acid("*", mask_ambiguous=False, mask_stop=True) is True
        assert should_mask_amino_acid("X", mask_ambiguous=False, mask_stop=True) is False
        assert should_mask_amino_acid("M", mask_ambiguous=True, mask_stop=True) is False


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
            maskchar='N',
            ambiguouscodon='no',
            stopcodon='no',
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
            maskchar='N',
            ambiguouscodon='yes',
            stopcodon='no',
        )

        mask_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        seq = str(result[0].seq)
        # ANN should be masked to NNN
        assert seq == "ATGNNNTGA"

    def test_mask_stop_codons(self, temp_dir, mock_args):
        """Test masking internal stop codons."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # TGA is stop codon
        records = [
            SeqRecord(Seq("ATGTGAAAA"), id="seq1", description=""),  # M * K
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            maskchar='N',
            ambiguouscodon='no',
            stopcodon='yes',
        )

        mask_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        seq = str(result[0].seq)
        # TGA should be masked to NNN
        assert seq == "ATGNNNAAA"

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
            maskchar='N',
            ambiguouscodon='yes',
            stopcodon='yes',
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
            maskchar='-',
            ambiguouscodon='yes',
            stopcodon='no',
        )

        mask_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        seq = str(result[0].seq)
        assert seq == "ATG---TGA"

    def test_mask_no_changes_needed(self, temp_dir, mock_args):
        """Test sequence that needs no masking."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="seq1", description=""),  # M K P - no issues
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            maskchar='N',
            ambiguouscodon='yes',
            stopcodon='yes',
        )

        mask_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert str(result[0].seq) == "ATGAAACCC"

    def test_mask_rejects_non_multiple_of_three(self, temp_dir, mock_args):
        """Test mask rejects sequences not multiple of 3."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAA"), id="seq1", description=""),  # 5 nt
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            maskchar='N',
            ambiguouscodon='yes',
            stopcodon='yes',
        )

        with pytest.raises(Exception) as exc_info:
            mask_main(args)
        assert "multiple of three" in str(exc_info.value)

    def test_mask_rejects_invalid_codontable(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=999,
            maskchar='N',
            ambiguouscodon='yes',
            stopcodon='yes',
        )

        with pytest.raises(Exception) as exc_info:
            mask_main(args)
        assert "Invalid --codontable" in str(exc_info.value)

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
            maskchar='N',
            ambiguouscodon='no',
            stopcodon='yes',
        )

        mask_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        seq = str(result[0].seq)
        # Both TGA codons should be masked
        assert seq == "ATGNNNNNN" + "AAA"

    def test_mask_with_example_data(self, data_dir, temp_dir, mock_args):
        """Test mask with example data if available."""
        input_path = data_dir / "example_mask.fasta"
        output_path = temp_dir / "output.fasta"

        if not input_path.exists():
            pytest.skip("example_mask.fasta not found")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            maskchar='N',
            ambiguouscodon='yes',
            stopcodon='yes',
        )

        mask_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) > 0
        # All output sequences should be multiple of 3
        for r in result:
            assert len(r.seq) % 3 == 0

    def test_mask_wiki_example_stop_codon(self, temp_dir, mock_args):
        """Test wiki example: masking stop codons.

        Wiki example shows TAA (stop) being replaced with NNN.
        """
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Wiki example input
        records = [
            SeqRecord(Seq("---ATGTAAATTATGTTGAAG---"), id="stop", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            maskchar='N',
            ambiguouscodon='no',
            stopcodon='yes',
        )

        mask_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        seq = str(result[0].seq)
        # TAA (positions 7-9) should be masked to NNN
        assert "TAA" not in seq or seq.count("TAA") < "---ATGTAAATTATGTTGAAG---".count("TAA")
        assert seq == "---ATGNNNATTATGTTGAAG---"

    def test_mask_wiki_example_ambiguous(self, temp_dir, mock_args):
        """Test wiki example: masking ambiguous codons with N."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("---ATGTNAATTATGTTGAAG---"), id="ambiguous1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            maskchar='N',
            ambiguouscodon='yes',
            stopcodon='no',
        )

        mask_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        seq = str(result[0].seq)
        # TNA codon should be masked
        assert "TNA" not in seq

    def test_mask_wiki_example_partial_gap(self, temp_dir, mock_args):
        """Test wiki example: masking partial gaps."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Partial gap: T-A should be masked
        records = [
            SeqRecord(Seq("---ATGT-AATTATGTTGAAG---"), id="ambiguous2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            maskchar='N',
            ambiguouscodon='no',
            stopcodon='no',
        )

        mask_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        seq = str(result[0].seq)
        # T-A partial gap should be masked to NNN
        assert "T-A" not in seq

    def test_mask_example_mask_fasta_detailed(self, data_dir, temp_dir, mock_args):
        """Test mask with example_mask.fasta - verify specific outputs."""
        input_path = data_dir / "example_mask.fasta"
        output_path = temp_dir / "output.fasta"

        if not input_path.exists():
            pytest.skip("example_mask.fasta not found")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            maskchar='N',
            ambiguouscodon='yes',
            stopcodon='yes',
        )

        mask_main(args)

        result = {r.id: str(r.seq) for r in Bio.SeqIO.parse(str(output_path), "fasta")}

        # 'stop' sequence: TAA should be masked
        if "stop" in result:
            assert "TAA" not in result["stop"]

        # 'ambiguous1' sequence: TNA should be masked
        if "ambiguous1" in result:
            assert "TNA" not in result["ambiguous1"]

        # 'ambiguous2' sequence: T-A partial gap should be masked
        if "ambiguous2" in result:
            assert "T-A" not in result["ambiguous2"]

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
            maskchar='N',
            ambiguouscodon='yes',
            stopcodon='yes',
        )

        mask_main(args)

        result = {r.id: str(r.seq) for r in Bio.SeqIO.parse(str(output_path), "fasta")}
        assert len(result) == 3
        assert result["seq1"] == "ATGNNNAAA"  # TGA masked
        assert result["seq2"] == "ATGNNNAAA"  # NNN already masked
        assert result["seq3"] == "ATGAAACCC"  # No change

    def test_mask_threads_matches_single_thread(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        out_single = temp_dir / "single.fasta"
        out_threaded = temp_dir / "threaded.fasta"
        records = [
            SeqRecord(Seq("ATGTGAAAA"), id="seq1", description=""),
            SeqRecord(Seq("ATGNNNAAA"), id="seq2", description=""),
            SeqRecord(Seq("ATGA-GAAA"), id="seq3", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="seq4", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args_single = mock_args(
            seqfile=str(input_path),
            outfile=str(out_single),
            codontable=1,
            maskchar='N',
            ambiguouscodon='yes',
            stopcodon='yes',
            threads=1,
        )
        args_threaded = mock_args(
            seqfile=str(input_path),
            outfile=str(out_threaded),
            codontable=1,
            maskchar='N',
            ambiguouscodon='yes',
            stopcodon='yes',
            threads=4,
        )

        mask_main(args_single)
        mask_main(args_threaded)

        result_single = list(Bio.SeqIO.parse(str(out_single), "fasta"))
        result_threaded = list(Bio.SeqIO.parse(str(out_threaded), "fasta"))
        assert [r.id for r in result_single] == [r.id for r in result_threaded]
        assert [str(r.seq) for r in result_single] == [str(r.seq) for r in result_threaded]

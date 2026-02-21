"""
Tests for cdskit backalign command.
"""

import pytest
from pathlib import Path

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.backalign import amino_acid_matches
from cdskit.backalign import backalign_main
from cdskit.backalign import backalign_record
from cdskit.backalign import get_record_map
from cdskit.backalign import remove_gap_chars
from cdskit.backalign import split_codons
from cdskit.backalign import stop_if_not_multiple_of_three_after_gap_removal
from cdskit.backalign import stop_if_sequence_ids_do_not_match
from cdskit.backalign import translate_codons


class TestBackalignHelpers:
    """Tests for helper functions used by backalign."""

    def test_remove_gap_chars(self):
        assert remove_gap_chars("ATG---AAA..CCC", {'-', '.'}) == "ATGAAACCC"

    def test_split_codons(self):
        assert split_codons("ATGAAACCC") == ["ATG", "AAA", "CCC"]

    def test_translate_codons_standard_table(self):
        codons = ["ATG", "AAA", "TGA"]
        assert translate_codons(codons, codontable=1) == ["M", "K", "*"]

    def test_translate_codons_alternative_table(self):
        # TGA is W in vertebrate mitochondrial table.
        codons = ["ATG", "TGA"]
        assert translate_codons(codons, codontable=2) == ["M", "W"]

    def test_amino_acid_matches_exact_and_case_insensitive(self):
        assert amino_acid_matches("M", "M")
        assert amino_acid_matches("m", "M")
        assert not amino_acid_matches("K", "Q")

    def test_amino_acid_matches_wildcards(self):
        assert amino_acid_matches("X", "K")
        assert amino_acid_matches("?", "Q")

    def test_stop_if_not_multiple_of_three_after_gap_removal_valid(self):
        records = [SeqRecord(Seq("ATG---AAA"), id="seq1")]
        stop_if_not_multiple_of_three_after_gap_removal(records)

    def test_stop_if_not_multiple_of_three_after_gap_removal_invalid(self):
        records = [SeqRecord(Seq("ATG--AA"), id="seq1")]
        with pytest.raises(Exception) as exc_info:
            stop_if_not_multiple_of_three_after_gap_removal(records)
        assert "multiple of three" in str(exc_info.value)

    def test_get_record_map_returns_by_id(self):
        records = [
            SeqRecord(Seq("ATG"), id="seq1"),
            SeqRecord(Seq("AAA"), id="seq2"),
        ]
        m = get_record_map(records, "--seqfile")
        assert set(m.keys()) == {"seq1", "seq2"}
        assert str(m["seq2"].seq) == "AAA"

    def test_get_record_map_rejects_duplicate_ids(self):
        records = [
            SeqRecord(Seq("ATG"), id="seq1"),
            SeqRecord(Seq("AAA"), id="seq1"),
        ]
        with pytest.raises(Exception) as exc_info:
            get_record_map(records, "--seqfile")
        assert "Duplicated ID" in str(exc_info.value)

    def test_stop_if_sequence_ids_do_not_match_success(self):
        cdn_records = [SeqRecord(Seq("ATG"), id="seq1"), SeqRecord(Seq("AAA"), id="seq2")]
        pep_records = [SeqRecord(Seq("M"), id="seq2"), SeqRecord(Seq("K"), id="seq1")]
        stop_if_sequence_ids_do_not_match(cdn_records, pep_records)

    def test_stop_if_sequence_ids_do_not_match_reports_missing_in_both(self):
        cdn_records = [SeqRecord(Seq("ATG"), id="seq1"), SeqRecord(Seq("AAA"), id="seq2")]
        pep_records = [SeqRecord(Seq("M"), id="seq2"), SeqRecord(Seq("K"), id="seq3")]
        with pytest.raises(Exception) as exc_info:
            stop_if_sequence_ids_do_not_match(cdn_records, pep_records)
        message = str(exc_info.value)
        assert "Missing in CDS: seq3" in message
        assert "Missing in amino acid alignment: seq1" in message


class TestBackalignRecord:
    """Tests for per-record backalignment behavior."""

    def test_backalign_record_basic(self):
        cdn_record = SeqRecord(Seq("ATGAAACCC"), id="seq1")
        pep_record = SeqRecord(Seq("MK-P"), id="seq1")
        result = backalign_record(cdn_record, pep_record, codontable=1)
        assert str(result.seq) == "ATGAAA---CCC"
        assert result.id == "seq1"

    def test_backalign_record_accepts_dot_as_gap(self):
        cdn_record = SeqRecord(Seq("ATGAAACCC"), id="seq1")
        pep_record = SeqRecord(Seq("MK.P"), id="seq1")
        result = backalign_record(cdn_record, pep_record, codontable=1)
        assert str(result.seq) == "ATGAAA---CCC"

    def test_backalign_record_accepts_wildcards(self):
        cdn_record = SeqRecord(Seq("ATGAAACCC"), id="seq1")  # MKP
        pep_record = SeqRecord(Seq("MX?"), id="seq1")
        result = backalign_record(cdn_record, pep_record, codontable=1)
        assert str(result.seq) == "ATGAAACCC"

    def test_backalign_record_accepts_lowercase_amino_acids(self):
        cdn_record = SeqRecord(Seq("ATGAAACCC"), id="seq1")  # MKP
        pep_record = SeqRecord(Seq("mk-p"), id="seq1")
        result = backalign_record(cdn_record, pep_record, codontable=1)
        assert str(result.seq) == "ATGAAA---CCC"

    def test_backalign_record_rejects_too_many_non_gap_sites(self):
        cdn_record = SeqRecord(Seq("ATGAAA"), id="seq1")  # MK
        pep_record = SeqRecord(Seq("MKA"), id="seq1")
        with pytest.raises(Exception) as exc_info:
            backalign_record(cdn_record, pep_record, codontable=1)
        assert "too many non-gap sites" in str(exc_info.value)

    def test_backalign_record_rejects_amino_acid_mismatch(self):
        cdn_record = SeqRecord(Seq("ATGAAA"), id="seq1")  # MK
        pep_record = SeqRecord(Seq("MQ"), id="seq1")
        with pytest.raises(Exception) as exc_info:
            backalign_record(cdn_record, pep_record, codontable=1)
        assert "mismatch" in str(exc_info.value)

    def test_backalign_record_rejects_invalid_codon(self):
        cdn_record = SeqRecord(Seq("ATG@@@"), id="seq1")
        pep_record = SeqRecord(Seq("MX"), id="seq1")
        with pytest.raises(Exception) as exc_info:
            backalign_record(cdn_record, pep_record, codontable=1)
        assert "Invalid codon" in str(exc_info.value)

    def test_backalign_record_rejects_unmatched_single_nonstop_codon(self):
        cdn_record = SeqRecord(Seq("ATGAAA"), id="seq1")  # MK
        pep_record = SeqRecord(Seq("M"), id="seq1")
        with pytest.raises(Exception) as exc_info:
            backalign_record(cdn_record, pep_record, codontable=1)
        assert "Unmatched codon remained" in str(exc_info.value)

    def test_backalign_record_rejects_unmatched_multiple_codons(self):
        cdn_record = SeqRecord(Seq("ATGAAACCC"), id="seq1")  # MKP
        pep_record = SeqRecord(Seq("M"), id="seq1")
        with pytest.raises(Exception) as exc_info:
            backalign_record(cdn_record, pep_record, codontable=1)
        assert "codons remained unmatched" in str(exc_info.value)

    def test_backalign_record_accepts_terminal_stop_omitted(self):
        cdn_record = SeqRecord(Seq("ATGAAATAA"), id="seq1")  # MK*
        pep_record = SeqRecord(Seq("MK"), id="seq1")
        result = backalign_record(cdn_record, pep_record, codontable=1)
        assert str(result.seq) == "ATGAAA"

    def test_backalign_record_keeps_terminal_stop_if_present_in_aa(self):
        cdn_record = SeqRecord(Seq("ATGAAATAA"), id="seq1")  # MK*
        pep_record = SeqRecord(Seq("MK*"), id="seq1")
        result = backalign_record(cdn_record, pep_record, codontable=1)
        assert str(result.seq) == "ATGAAATAA"


class TestBackalignMain:
    """Tests for backalign_main function."""

    def test_backalign_basic_with_id_matching(self, temp_dir, mock_args):
        """Back-align codons from unaligned CDS + aligned amino acids."""
        cdn_path = temp_dir / "cds.fasta"
        pep_path = temp_dir / "aa_aln.fasta"
        out_path = temp_dir / "out.fasta"

        # Intentionally use different order between CDS and protein files.
        cdn_records = [
            SeqRecord(Seq("ATGAAAGGG"), id="seq2", description=""),  # MKG
            SeqRecord(Seq("ATGAAACCC"), id="seq1", description=""),  # MKP
        ]
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")

        pep_records = [
            SeqRecord(Seq("MK-P"), id="seq1", description=""),
            SeqRecord(Seq("MKG-"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_path),
            aa_aln=str(pep_path),
            codontable=1,
        )

        backalign_main(args)

        result = list(Bio.SeqIO.parse(str(out_path), "fasta"))
        assert len(result) == 2
        # Output order follows CDS input order.
        assert result[0].id == "seq2"
        assert str(result[0].seq) == "ATGAAAGGG---"
        assert result[1].id == "seq1"
        assert str(result[1].seq) == "ATGAAA---CCC"

    def test_backalign_accepts_terminal_stop_omitted_in_aa(self, temp_dir, mock_args):
        """Allow dropping terminal stop codon when protein alignment has no trailing '*'."""
        cdn_path = temp_dir / "cds.fasta"
        pep_path = temp_dir / "aa_aln.fasta"
        out_path = temp_dir / "out.fasta"

        cdn_records = [SeqRecord(Seq("ATGAAATAA"), id="seq1", description="")]  # MK*
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")
        pep_records = [SeqRecord(Seq("MK"), id="seq1", description="")]
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_path),
            aa_aln=str(pep_path),
            codontable=1,
        )

        backalign_main(args)

        result = list(Bio.SeqIO.parse(str(out_path), "fasta"))
        assert len(result) == 1
        assert str(result[0].seq) == "ATGAAA"

    def test_backalign_rejects_sequence_id_mismatch(self, temp_dir, mock_args):
        """Raise when sequence IDs differ between CDS and amino acid alignment."""
        cdn_path = temp_dir / "cds.fasta"
        pep_path = temp_dir / "aa_aln.fasta"
        out_path = temp_dir / "out.fasta"

        cdn_records = [SeqRecord(Seq("ATGAAA"), id="seq1", description="")]
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")
        pep_records = [SeqRecord(Seq("MK"), id="seqX", description="")]
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_path),
            aa_aln=str(pep_path),
            codontable=1,
        )

        with pytest.raises(Exception) as exc_info:
            backalign_main(args)
        assert "did not match" in str(exc_info.value)

    def test_backalign_empty_inputs_produce_empty_output(self, temp_dir, mock_args):
        cdn_path = temp_dir / "cds.fasta"
        pep_path = temp_dir / "aa_aln.fasta"
        out_path = temp_dir / "out.fasta"

        cdn_path.write_text("")
        pep_path.write_text("")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_path),
            aa_aln=str(pep_path),
            codontable=1,
        )
        backalign_main(args)
        result = list(Bio.SeqIO.parse(str(out_path), "fasta"))
        assert len(result) == 0

    def test_backalign_rejects_non_multiple_of_three(self, temp_dir, mock_args):
        """Reject CDS where ungapped length is not multiple of 3."""
        cdn_path = temp_dir / "cds.fasta"
        pep_path = temp_dir / "aa_aln.fasta"
        out_path = temp_dir / "out.fasta"

        cdn_records = [SeqRecord(Seq("ATGAA"), id="seq1", description="")]  # 5 nt
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")
        pep_records = [SeqRecord(Seq("M"), id="seq1", description="")]
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_path),
            aa_aln=str(pep_path),
            codontable=1,
        )

        with pytest.raises(Exception) as exc_info:
            backalign_main(args)
        assert "multiple of three" in str(exc_info.value)

    def test_backalign_rejects_invalid_codontable(self, temp_dir, mock_args):
        cdn_path = temp_dir / "cds.fasta"
        pep_path = temp_dir / "aa_aln.fasta"
        out_path = temp_dir / "out.fasta"

        cdn_records = [SeqRecord(Seq("ATGAAA"), id="seq1", description="")]
        pep_records = [SeqRecord(Seq("MK"), id="seq1", description="")]
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_path),
            aa_aln=str(pep_path),
            codontable=999,
        )
        with pytest.raises(Exception) as exc_info:
            backalign_main(args)
        assert "Invalid --codontable" in str(exc_info.value)

    def test_backalign_rejects_rna_cds_input(self, temp_dir, mock_args):
        cdn_path = temp_dir / "cds.fasta"
        pep_path = temp_dir / "aa_aln.fasta"
        out_path = temp_dir / "out.fasta"

        cdn_records = [SeqRecord(Seq("AUGAAA"), id="seq1", description="")]
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")
        pep_records = [SeqRecord(Seq("MK"), id="seq1", description="")]
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_path),
            aa_aln=str(pep_path),
            codontable=1,
        )
        with pytest.raises(Exception) as exc_info:
            backalign_main(args)
        assert "DNA-only input is required" in str(exc_info.value)

    def test_backalign_rejects_translation_mismatch(self, temp_dir, mock_args):
        """Reject when amino acid alignment and CDS translation disagree."""
        cdn_path = temp_dir / "cds.fasta"
        pep_path = temp_dir / "aa_aln.fasta"
        out_path = temp_dir / "out.fasta"

        cdn_records = [SeqRecord(Seq("ATGAAA"), id="seq1", description="")]  # MK
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")
        pep_records = [SeqRecord(Seq("MQ"), id="seq1", description="")]  # mismatch at position 2
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_path),
            aa_aln=str(pep_path),
            codontable=1,
        )

        with pytest.raises(Exception) as exc_info:
            backalign_main(args)
        assert "mismatch" in str(exc_info.value)

    def test_backalign_rejects_unaligned_amino_acid_input(self, temp_dir, mock_args):
        """Reject when amino acid alignment is not aligned."""
        cdn_path = temp_dir / "cds.fasta"
        pep_path = temp_dir / "aa_aln.fasta"
        out_path = temp_dir / "out.fasta"

        cdn_records = [
            SeqRecord(Seq("ATGAAA"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAA"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")
        pep_records = [
            SeqRecord(Seq("MK"), id="seq1", description=""),
            SeqRecord(Seq("M"), id="seq2", description=""),  # different length
        ]
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_path),
            aa_aln=str(pep_path),
            codontable=1,
        )

        with pytest.raises(Exception) as exc_info:
            backalign_main(args)
        assert "not identical" in str(exc_info.value)

    def test_backalign_threads_matches_single_thread(self, temp_dir, mock_args):
        cdn_path = temp_dir / "cds.fasta"
        pep_path = temp_dir / "aa_aln.fasta"
        out_single = temp_dir / "single.fasta"
        out_threaded = temp_dir / "threaded.fasta"

        cdn_records = [
            SeqRecord(Seq("ATGAAAGGG"), id="seq2", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAATAA"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")

        pep_records = [
            SeqRecord(Seq("MK-P"), id="seq1", description=""),
            SeqRecord(Seq("MKG-"), id="seq2", description=""),
            SeqRecord(Seq("MK--"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args_single = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_single),
            aa_aln=str(pep_path),
            codontable=1,
            threads=1,
        )
        args_threaded = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_threaded),
            aa_aln=str(pep_path),
            codontable=1,
            threads=4,
        )

        backalign_main(args_single)
        backalign_main(args_threaded)

        result_single = list(Bio.SeqIO.parse(str(out_single), "fasta"))
        result_threaded = list(Bio.SeqIO.parse(str(out_threaded), "fasta"))
        assert [r.id for r in result_single] == [r.id for r in result_threaded]
        assert [str(r.seq) for r in result_single] == [str(r.seq) for r in result_threaded]

    def test_backalign_rejects_duplicate_ids_in_cds(self, temp_dir, mock_args):
        """Reject duplicate IDs in CDS input."""
        cdn_path = temp_dir / "cds.fasta"
        pep_path = temp_dir / "aa_aln.fasta"
        out_path = temp_dir / "out.fasta"

        cdn_records = [
            SeqRecord(Seq("ATGAAA"), id="seq1", description=""),
            SeqRecord(Seq("ATGCCC"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")
        pep_records = [SeqRecord(Seq("MK"), id="seq1", description="")]
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_path),
            aa_aln=str(pep_path),
            codontable=1,
        )

        with pytest.raises(Exception) as exc_info:
            backalign_main(args)
        assert "Duplicated ID" in str(exc_info.value)

    def test_backalign_rejects_duplicate_ids_in_amino_acids(self, temp_dir, mock_args):
        """Reject duplicate IDs in amino acid alignment input."""
        cdn_path = temp_dir / "cds.fasta"
        pep_path = temp_dir / "aa_aln.fasta"
        out_path = temp_dir / "out.fasta"

        cdn_records = [SeqRecord(Seq("ATGAAA"), id="seq1", description="")]
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")
        pep_records = [
            SeqRecord(Seq("MK"), id="seq1", description=""),
            SeqRecord(Seq("MK"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_path),
            aa_aln=str(pep_path),
            codontable=1,
        )

        with pytest.raises(Exception) as exc_info:
            backalign_main(args)
        assert "Duplicated ID" in str(exc_info.value)

    def test_backalign_accepts_gapped_cds_input(self, temp_dir, mock_args):
        """Allow gaps in CDS input by removing them before codon matching."""
        cdn_path = temp_dir / "cds.fasta"
        pep_path = temp_dir / "aa_aln.fasta"
        out_path = temp_dir / "out.fasta"

        cdn_records = [SeqRecord(Seq("ATG---AAA...CCC"), id="seq1", description="")]  # MKP
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")
        pep_records = [SeqRecord(Seq("MK-P"), id="seq1", description="")]
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_path),
            aa_aln=str(pep_path),
            codontable=1,
        )

        backalign_main(args)
        result = list(Bio.SeqIO.parse(str(out_path), "fasta"))
        assert str(result[0].seq) == "ATGAAA---CCC"

    def test_backalign_uses_selected_codon_table(self, temp_dir, mock_args):
        """Codon table selection should affect translation checks."""
        cdn_path = temp_dir / "cds.fasta"
        pep_path = temp_dir / "aa_aln.fasta"
        out_path = temp_dir / "out.fasta"

        # ATG TGA -> MW under table 2, M* under table 1.
        cdn_records = [SeqRecord(Seq("ATGTGA"), id="seq1", description="")]
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")
        pep_records = [SeqRecord(Seq("MW"), id="seq1", description="")]
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_path),
            aa_aln=str(pep_path),
            codontable=2,
        )
        backalign_main(args)
        result = list(Bio.SeqIO.parse(str(out_path), "fasta"))
        assert str(result[0].seq) == "ATGTGA"

        # Same input should fail under the standard table.
        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_path),
            aa_aln=str(pep_path),
            codontable=1,
        )
        with pytest.raises(Exception) as exc_info:
            backalign_main(args)
        assert "mismatch" in str(exc_info.value)

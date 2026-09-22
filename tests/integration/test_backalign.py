"""
Tests for cdskit backalign command.
"""

import pytest
import Bio.Data.CodonTable
import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.backalign import backalign_main
from cdskit.backalign import backalign_record


@pytest.mark.parametrize("table_id", sorted(Bio.Data.CodonTable.unambiguous_dna_by_id))
def test_translate_backalign_roundtrip_all_genetic_codes(table_id):
    import itertools
    import warnings

    from Bio import BiopythonWarning
    from cdskit.backalign import backalign_sequence_strings
    from cdskit.translate import translate_sequence_string

    sequence = "".join("".join(codon) for codon in itertools.product("ACGT", repeat=3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiopythonWarning)
        protein = translate_sequence_string(sequence, table_id, False)
    assert (
        backalign_sequence_strings(sequence, protein, table_id, "all-codons", False)
        == sequence
    )


@pytest.mark.parametrize("table_id,terminal", [(27, "TGA"), (28, "TAA"), (31, "TAG")])
def test_dual_coding_terminal_stop_may_be_omitted(table_id, terminal):
    from cdskit.backalign import backalign_sequence_strings

    assert (
        backalign_sequence_strings("ATG" + terminal, "M", table_id, "seq", False)
        == "ATG"
    )
    assert (
        backalign_sequence_strings("ATG" + terminal, "M*", table_id, "seq", False)
        == "ATG" + terminal
    )
    with pytest.raises(ValueError, match="Amino acid mismatch"):
        backalign_sequence_strings(
            "ATG" + terminal + "ATG", "M*M", table_id, "seq", False
        )


class TestBackalignRecord:
    """Tests for per-record backalignment behavior."""

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
        with pytest.raises(ValueError) as exc_info:
            backalign_record(cdn_record, pep_record, codontable=1)
        assert "too many non-gap sites" in str(exc_info.value)

    def test_backalign_record_rejects_invalid_codon(self):
        cdn_record = SeqRecord(Seq("ATG@@@"), id="seq1")
        pep_record = SeqRecord(Seq("MX"), id="seq1")
        with pytest.raises(ValueError) as exc_info:
            backalign_record(cdn_record, pep_record, codontable=1)
        assert "Invalid codon" in str(exc_info.value)

    def test_backalign_record_rejects_unmatched_single_nonstop_codon(self):
        cdn_record = SeqRecord(Seq("ATGAAA"), id="seq1")  # MK
        pep_record = SeqRecord(Seq("M"), id="seq1")
        with pytest.raises(ValueError) as exc_info:
            backalign_record(cdn_record, pep_record, codontable=1)
        assert "Unmatched codon remained" in str(exc_info.value)

    def test_backalign_record_rejects_unmatched_multiple_codons(self):
        cdn_record = SeqRecord(Seq("ATGAAACCC"), id="seq1")  # MKP
        pep_record = SeqRecord(Seq("M"), id="seq1")
        with pytest.raises(ValueError) as exc_info:
            backalign_record(cdn_record, pep_record, codontable=1)
        assert "codons remained unmatched" in str(exc_info.value)

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

        with pytest.raises(ValueError) as exc_info:
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

        with pytest.raises(ValueError) as exc_info:
            backalign_main(args)
        assert "multiple of three" in str(exc_info.value)

    def test_backalign_rejects_translation_mismatch(self, temp_dir, mock_args):
        """Reject when amino acid alignment and CDS translation disagree."""
        cdn_path = temp_dir / "cds.fasta"
        pep_path = temp_dir / "aa_aln.fasta"
        out_path = temp_dir / "out.fasta"

        cdn_records = [SeqRecord(Seq("ATGAAA"), id="seq1", description="")]  # MK
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")
        pep_records = [
            SeqRecord(Seq("MQ"), id="seq1", description="")
        ]  # mismatch at position 2
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_path),
            aa_aln=str(pep_path),
            codontable=1,
        )

        with pytest.raises(ValueError) as exc_info:
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

        with pytest.raises(ValueError) as exc_info:
            backalign_main(args)
        assert "not identical" in str(exc_info.value)

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

        with pytest.raises(ValueError) as exc_info:
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

        with pytest.raises(ValueError) as exc_info:
            backalign_main(args)
        assert "Duplicated ID" in str(exc_info.value)

    def test_backalign_accepts_gapped_cds_input(self, temp_dir, mock_args):
        """Allow gaps in CDS input by removing them before codon matching."""
        cdn_path = temp_dir / "cds.fasta"
        pep_path = temp_dir / "aa_aln.fasta"
        out_path = temp_dir / "out.fasta"

        cdn_records = [
            SeqRecord(Seq("ATG---AAA...CCC"), id="seq1", description="")
        ]  # MKP
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
        with pytest.raises(ValueError) as exc_info:
            backalign_main(args)
        assert "mismatch" in str(exc_info.value)

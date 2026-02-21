"""
Tests for cdskit parsegb command.
"""

import pytest
from pathlib import Path
from io import StringIO

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.parsegb import parsegb_main, parsegb_record


def create_genbank_record(seq, record_id, organism="Test organism", accession="TEST001"):
    """Helper to create a minimal GenBank record for testing."""
    record = SeqRecord(
        Seq(seq),
        id=record_id,
        name=record_id,
        description=f"{organism} test gene",
    )
    record.annotations["organism"] = organism
    record.annotations["accessions"] = [accession]
    record.annotations["molecule_type"] = "mRNA"
    return record


def create_genbank_with_cds(seq, cds_start, cds_end, record_id, organism="Test organism"):
    """Helper to create a GenBank record with CDS feature."""
    record = create_genbank_record(seq, record_id, organism)
    # Add CDS feature
    cds_feature = SeqFeature(
        FeatureLocation(cds_start, cds_end),
        type="CDS",
        qualifiers={"product": ["test protein"]}
    )
    record.features.append(cds_feature)
    return record


class TestParsegbMain:
    """Tests for parsegb_main function."""

    def test_parsegb_basic(self, temp_dir, mock_args):
        """Test basic parsegb - convert GenBank to FASTA."""
        input_path = temp_dir / "input.gb"
        output_path = temp_dir / "output.fasta"

        # Create a simple GenBank record
        record = create_genbank_record("ATGAAACCC", "TEST001", "Homo sapiens")
        Bio.SeqIO.write([record], str(input_path), "genbank")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            inseqformat='genbank',
            seqnamefmt='organism_accessions',
            extract_cds=False,
            list_seqname_keys=False,
        )

        parsegb_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 1
        assert str(result[0].seq) == "ATGAAACCC"
        assert "Homo" in result[0].id or "sapiens" in result[0].id

    def test_parsegb_extract_cds(self, temp_dir, mock_args):
        """Test parsegb with CDS extraction."""
        input_path = temp_dir / "input.gb"
        output_path = temp_dir / "output.fasta"

        # Full sequence with CDS at positions 10-19 (0-based)
        full_seq = "AAAAAAAAAA" + "ATGCCCGGG" + "AAAAAAAAAA"  # CDS is ATGCCCGGG
        record = create_genbank_with_cds(full_seq, 10, 19, "TEST001")
        Bio.SeqIO.write([record], str(input_path), "genbank")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            inseqformat='genbank',
            seqnamefmt='organism_accessions',
            extract_cds=True,
            list_seqname_keys=False,
        )

        parsegb_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 1
        # CDS should be extracted
        assert str(result[0].seq) == "ATGCCCGGG"

    def test_parsegb_no_cds_filtered(self, temp_dir, mock_args):
        """Test that records without CDS are filtered when extract_cds=True."""
        input_path = temp_dir / "input.gb"
        output_path = temp_dir / "output.fasta"

        # Record without CDS
        record = create_genbank_record("ATGAAACCC", "TEST001")
        Bio.SeqIO.write([record], str(input_path), "genbank")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            inseqformat='genbank',
            seqnamefmt='organism_accessions',
            extract_cds=True,
            list_seqname_keys=False,
        )

        parsegb_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Should be empty because no CDS found
        assert len(result) == 0

    def test_parsegb_multiple_records(self, temp_dir, mock_args):
        """Test parsegb with multiple GenBank records."""
        input_path = temp_dir / "input.gb"
        output_path = temp_dir / "output.fasta"

        records = [
            create_genbank_record("ATGAAA", "REC1", "Homo sapiens", "HS001"),
            create_genbank_record("ATGCCC", "REC2", "Mus musculus", "MM001"),
            create_genbank_record("ATGGGG", "REC3", "Rattus norvegicus", "RN001"),
        ]
        Bio.SeqIO.write(records, str(input_path), "genbank")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            inseqformat='genbank',
            seqnamefmt='organism_accessions',
            extract_cds=False,
            list_seqname_keys=False,
        )

        parsegb_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 3

    def test_parsegb_seqname_format(self, temp_dir, mock_args):
        """Test different sequence name formats."""
        input_path = temp_dir / "input.gb"
        output_path = temp_dir / "output.fasta"

        record = create_genbank_record("ATGAAACCC", "TEST001", "Homo sapiens", "HSA12345")
        Bio.SeqIO.write([record], str(input_path), "genbank")

        # Test organism_accessions format
        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            inseqformat='genbank',
            seqnamefmt='organism_accessions',
            extract_cds=False,
            list_seqname_keys=False,
        )

        parsegb_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Should contain both organism and accession info
        assert "Homo" in result[0].id or "HSA12345" in result[0].id

    def test_parsegb_with_test_data(self, data_dir, temp_dir, mock_args):
        """Test parsegb with parsegb_01 test data."""
        input_path = data_dir / "parsegb_01" / "input.gb"
        expected_path = data_dir / "parsegb_01" / "output.fasta"
        output_path = temp_dir / "output.fasta"

        if not input_path.exists():
            pytest.skip("parsegb_01 test data not found")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            inseqformat='genbank',
            seqnamefmt='organism_accessions',
            extract_cds=True,
            list_seqname_keys=False,
        )

        parsegb_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) > 0

        # If expected output exists, compare
        if expected_path.exists():
            expected = list(Bio.SeqIO.parse(str(expected_path), "fasta"))
            assert len(result) == len(expected)
            # Check sequences match
            for r, e in zip(result, expected):
                assert str(r.seq) == str(e.seq)

    def test_parsegb_preserves_sequence_content(self, temp_dir, mock_args):
        """Test that parsegb preserves exact sequence content."""
        input_path = temp_dir / "input.gb"
        output_path = temp_dir / "output.fasta"

        test_seq = "ATGCGATCGATCGATCGATCG"
        record = create_genbank_record(test_seq, "TEST001")
        Bio.SeqIO.write([record], str(input_path), "genbank")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            inseqformat='genbank',
            seqnamefmt='organism_accessions',
            extract_cds=False,
            list_seqname_keys=False,
        )

        parsegb_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert str(result[0].seq) == test_seq

    def test_parsegb_spaces_in_organism_name(self, temp_dir, mock_args):
        """Test handling of spaces in organism names."""
        input_path = temp_dir / "input.gb"
        output_path = temp_dir / "output.fasta"

        record = create_genbank_record("ATGAAA", "TEST001", "Homo sapiens neanderthalensis")
        Bio.SeqIO.write([record], str(input_path), "genbank")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            inseqformat='genbank',
            seqnamefmt='organism_accessions',
            extract_cds=False,
            list_seqname_keys=False,
        )

        parsegb_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Spaces should be replaced with underscores
        assert " " not in result[0].id

    def test_parsegb_threads_matches_single_thread(self, temp_dir, mock_args):
        input_path = temp_dir / "input.gb"
        out_single = temp_dir / "single.fasta"
        out_threaded = temp_dir / "threaded.fasta"

        records = [
            create_genbank_record("ATGAAA", "REC1", "Homo sapiens", "HS001"),
            create_genbank_record("ATGCCC", "REC2", "Mus musculus", "MM001"),
            create_genbank_record("ATGGGG", "REC3", "Rattus norvegicus", "RN001"),
        ]
        Bio.SeqIO.write(records, str(input_path), "genbank")

        args_single = mock_args(
            seqfile=str(input_path),
            outfile=str(out_single),
            inseqformat='genbank',
            seqnamefmt='organism_accessions',
            extract_cds=False,
            list_seqname_keys=False,
            threads=1,
        )
        args_threaded = mock_args(
            seqfile=str(input_path),
            outfile=str(out_threaded),
            inseqformat='genbank',
            seqnamefmt='organism_accessions',
            extract_cds=False,
            list_seqname_keys=False,
            threads=4,
        )

        parsegb_main(args_single)
        parsegb_main(args_threaded)

        result_single = list(Bio.SeqIO.parse(str(out_single), "fasta"))
        result_threaded = list(Bio.SeqIO.parse(str(out_threaded), "fasta"))
        assert [r.id for r in result_single] == [r.id for r in result_threaded]
        assert [str(r.seq) for r in result_single] == [str(r.seq) for r in result_threaded]

    def test_parsegb_rejects_non_genbank_inseqformat(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("ATGAAA"), id="REC1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            inseqformat='fasta',
            seqnamefmt='organism_accessions',
            extract_cds=False,
            list_seqname_keys=False,
        )

        with pytest.raises(Exception) as exc_info:
            parsegb_main(args)
        assert 'parsegb requires --inseqformat genbank' in str(exc_info.value)


class TestParsegbHelpers:
    """Tests for parsegb helper functions."""

    def test_parsegb_record_updates_id_and_clears_name_description(self):
        record = create_genbank_record("ATGAAA", "REC1", "Homo sapiens", "HS001")
        parsed = parsegb_record(
            record=record,
            seqnamefmt="organism_accessions",
            extract_cds=False,
            list_seqname_keys=False,
        )
        assert parsed is not None
        assert "Homo" in parsed.id or "HS001" in parsed.id
        assert parsed.name == ""
        assert parsed.description == ""

    def test_parsegb_record_extract_cds_returns_none_without_cds(self):
        record = create_genbank_record("ATGAAA", "REC1", "Homo sapiens", "HS001")
        parsed = parsegb_record(
            record=record,
            seqnamefmt="organism_accessions",
            extract_cds=True,
            list_seqname_keys=False,
        )
        assert parsed is None

"""
Tests for cdskit/util.py utility functions.
"""

import pytest
import numpy as np

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation, SeqFeature

from cdskit import util


class TestReadSeqs:
    """Tests for read_seqs function."""

    def test_rejects_excessively_long_sequence_identifier(
        self,
        temp_dir,
        monkeypatch,
    ):
        fasta_path = temp_dir / "long-id.fasta"
        fasta_path.write_text(">identifier-too-long\nATG\n", encoding="utf-8")
        monkeypatch.setenv("CDSKIT_MAX_SEQUENCE_ID_LENGTH", "8")

        with pytest.raises(ValueError, match="identifier exceeds 8 characters"):
            util.read_seqs(str(fasta_path), "fasta")


class TestThreadHelpers:
    """Tests for thread-related utility helpers."""

    def test_resolve_threads_default_and_auto(self, monkeypatch):
        assert util.resolve_threads(None) == 1
        monkeypatch.delattr(util.os, "sched_getaffinity", raising=False)
        monkeypatch.delattr(util.os, "process_cpu_count", raising=False)
        monkeypatch.setattr(util.os, "cpu_count", lambda: 7)
        assert util.resolve_threads(0) == 7

    def test_resolve_threads_rejects_negative(self):
        with pytest.raises(ValueError) as exc_info:
            util.resolve_threads(-1)
        assert "--threads should be >= 0" in str(exc_info.value)

    def test_parallel_map_ordered_keeps_input_order(self):
        items = [5, 3, 1, 4, 2]
        result = util.parallel_map_ordered(
            items=items, worker=lambda x: x * 2, threads=3
        )
        assert result == [10, 6, 2, 8, 4]

    def test_process_pool_selection_uses_total_residues_not_record_count(self):
        few_long = [SeqRecord(Seq("A" * 80), id="a"), SeqRecord(Seq("C" * 80), id="b")]
        many_short = [SeqRecord(Seq("ATG"), id=str(i)) for i in range(100)]

        assert util.should_use_process_pool(
            few_long,
            threads=4,
            min_total_residues=100,
        )
        assert not util.should_use_process_pool(
            many_short,
            threads=4,
            min_total_residues=1_000,
        )

    def test_iter_seq_chunks_preserves_order_and_bounds_memory(self, temp_dir):
        fasta_path = temp_dir / "chunked.fasta"
        Bio.SeqIO.write(
            [SeqRecord(Seq("ATG" * 4), id=f"seq{i}", description="") for i in range(5)],
            str(fasta_path),
            "fasta",
        )

        chunks = list(
            util.iter_seq_chunks(
                str(fasta_path),
                "fasta",
                max_chunk_records=2,
                max_chunk_residues=24,
            )
        )

        assert [len(chunk) for chunk in chunks] == [2, 2, 1]
        assert [record.id for chunk in chunks for record in chunk] == [
            "seq0",
            "seq1",
            "seq2",
            "seq3",
            "seq4",
        ]


class TestSafeRegex:
    def test_accepts_common_grouping_and_alternation(self):
        compiled = util.compile_safe_regex(r"^(?:alpha|beta)-\d+$")
        assert compiled.search("alpha-42")

    @pytest.mark.parametrize(
        "pattern",
        [
            r"(a+)+$",
            r"(a|aa)+$",
            r"(a?)*$",
            r"(.*)\1$",
        ],
    )
    def test_rejects_backtracking_prone_patterns(self, pattern):
        with pytest.raises(ValueError, match="potentially unsafe"):
            util.compile_safe_regex(pattern)


class TestReadItemPerLineFile:
    """Tests for read_item_per_line_file function."""

    def test_strips_whitespace_around_each_item(self, temp_dir):
        path = temp_dir / "items_whitespace.txt"
        path.write_text(" alpha \n\tbeta\t\n\n gamma\n")
        assert util.read_item_per_line_file(str(path)) == ["alpha", "beta", "gamma"]


class TestWriteSeqs:
    """Tests for write_seqs function."""

    def test_failed_write_preserves_existing_output(self, temp_dir, monkeypatch):
        fasta_path = temp_dir / "output.fasta"
        fasta_path.write_text("sentinel\n", encoding="utf-8")
        records = [SeqRecord(Seq("ATGAAATGA"), id="seq1", description="")]

        def fail_after_partial_write(records, path, seqformat):
            del records, seqformat
            with open(path, "w", encoding="utf-8") as output:
                output.write(">partial\nATG\n")
            raise RuntimeError("simulated writer failure")

        monkeypatch.setattr(util.Bio.SeqIO, "write", fail_after_partial_write)

        with pytest.raises(RuntimeError, match="simulated writer failure"):
            util.write_seqs(records, str(fasta_path), "fasta")

        assert fasta_path.read_text(encoding="utf-8") == "sentinel\n"


class TestStopIfNotMultipleOfThree:
    """Tests for stop_if_not_multiple_of_three function."""

    def test_mixed_sequences(self):
        """Test with mix of valid and invalid sequences."""
        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1"),  # 6 nt - valid
            SeqRecord(Seq("ATGAA"), id="seq2"),  # 5 nt - invalid
        ]
        with pytest.raises(ValueError):
            util.stop_if_not_multiple_of_three(records)


class TestStopIfNotAligned:
    """Tests for stop_if_not_aligned function."""

    def test_unaligned_sequences(self):
        """Test with sequences of different lengths."""
        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1"),
            SeqRecord(Seq("ATG"), id="seq2"),
        ]
        with pytest.raises(ValueError) as exc_info:
            util.stop_if_not_aligned(records)
        assert "not identical" in str(exc_info.value)


class TestStopIfNotDna:
    """Tests for stop_if_not_dna function."""

    def test_rejects_rna_sequences(self):
        records = [
            SeqRecord(Seq("AUGAAATGA"), id="seq1"),
            SeqRecord(Seq("ATGaaauaa"), id="seq2"),
        ]
        with pytest.raises(ValueError) as exc_info:
            util.stop_if_not_dna(records, label="--seqfile")
        assert "DNA-only input is required" in str(exc_info.value)
        assert "seq1,seq2" in str(exc_info.value)

    def test_rejects_non_dna_letters(self):
        records = [
            SeqRecord(Seq("ATGPPP"), id="seq_bad"),
            SeqRecord(Seq("ATGAAA"), id="seq_ok"),
        ]
        with pytest.raises(ValueError) as exc_info:
            util.stop_if_not_dna(records, label="--seqfile")
        assert "DNA-only input is required" in str(exc_info.value)
        assert "seq_bad" in str(exc_info.value)
        assert "P" in str(exc_info.value)


class TestStopIfNotProtein:
    """Tests for stop_if_not_protein function."""

    def test_rejects_invalid_protein_letters(self):
        records = [
            SeqRecord(Seq("MK1"), id="bad1"),
            SeqRecord(Seq("QQQ"), id="ok1"),
        ]
        with pytest.raises(ValueError) as exc_info:
            util.stop_if_not_protein(records, label="--seqfile")
        assert "Protein-only input is required" in str(exc_info.value)
        assert "bad1" in str(exc_info.value)
        assert "1" in str(exc_info.value)


class TestStopIfNotSeqtype:
    """Tests for stop_if_not_seqtype function."""

    def test_rejects_unknown_seqtype(self):
        records = [SeqRecord(Seq("ATG"), id="seq1")]
        with pytest.raises(ValueError) as exc_info:
            util.stop_if_not_seqtype(records=records, seqtype="rna", label="--seqfile")
        assert "Invalid --seq_type" in str(exc_info.value)


class TestStopIfInvalidCodontable:
    def test_rejects_invalid_codontable(self):
        with pytest.raises(ValueError) as exc_info:
            util.stop_if_invalid_codontable(999)
        assert "Invalid --codon_table" in str(exc_info.value)


class TestGetSeqname:
    """Tests for get_seqname function."""

    def test_builds_name_from_multiple_annotation_fields(self):
        record = SeqRecord(Seq("ATG"), id="r1")
        record.annotations["organism"] = "Homo sapiens"
        record.annotations["accessions"] = ["ABC123", "DEF456"]
        result = util.get_seqname(record, "organism_accessions")
        assert result == "Homo_sapiens_ABC123"

    def test_raises_for_unknown_annotation_key(self):
        record = SeqRecord(Seq("ATG"), id="r2")
        record.annotations["organism"] = "Homo sapiens"
        with pytest.raises(ValueError) as exc_info:
            util.get_seqname(record, "organism_unknown")
        assert "Invalid --seq_name_format element (unknown)" in str(exc_info.value)


class TestReplaceSeq2Cds:
    """Tests for replace_seq2cds function."""

    def test_replaces_sequence_with_cds_feature(self):
        record = SeqRecord(Seq("AAATGCCCCTTT"), id="cds_record")
        record.features = [
            SeqFeature(FeatureLocation(3, 9), type="CDS"),
        ]
        result = util.replace_seq2cds(record)
        assert result is not None
        assert str(result.seq) == "TGCCCC"

    def test_returns_none_when_no_cds_feature(self):
        record = SeqRecord(Seq("AAATGCCCCTTT"), id="no_cds_record")
        record.features = []
        result = util.replace_seq2cds(record)
        assert result is None

    @pytest.mark.parametrize(
        "strand,exception,expected",
        [
            (1, "(pos:7..9,aa:Sec)", "(pos:4..6,aa:Sec)"),
            (-1, "(pos:complement(7..9),aa:Sec)", "(pos:1..3,aa:Sec)"),
        ],
    )
    def test_cds_coordinate_qualifiers_and_references_are_not_left_on_genome(
        self, strand, exception, expected
    ):
        from Bio.SeqFeature import Reference

        record = SeqRecord(Seq("AAATGCCCCTTT"), id="cds_record")
        reference = Reference()
        reference.title = "Original reference"
        reference.location = [FeatureLocation(0, 12)]
        record.annotations = {
            "molecule_type": "DNA",
            "references": [reference],
            "contig": "join(old:1..12)",
        }
        record.features = [
            SeqFeature(
                FeatureLocation(3, 9, strand=strand),
                type="CDS",
                qualifiers={"transl_except": [exception]},
            )
        ]
        extracted = util.replace_seq2cds(record)
        assert extracted.features[0].qualifiers["transl_except"] == [expected]
        assert "contig" not in extracted.annotations
        assert extracted.annotations["references"][0].title == "Original reference"
        assert extracted.annotations["references"][0].location == []
        assert reference.location == [FeatureLocation(0, 12)]

    @pytest.mark.parametrize("strand", [1, -1])
    @pytest.mark.parametrize("joined", [False, True])
    def test_extraction_rebases_features_and_letter_annotations(self, strand, joined):
        from Bio.SeqFeature import CompoundLocation

        record = SeqRecord(Seq("AAATGCCCCTTT"), id="cds_record")
        record.annotations = {"molecule_type": "DNA", "topology": "circular"}
        record.letter_annotations["phred_quality"] = list(range(len(record)))
        location = FeatureLocation(3, 9, strand=strand)
        if joined:
            parts = [
                FeatureLocation(0, 3, strand=strand),
                FeatureLocation(6, 9, strand=strand),
            ]
            location = CompoundLocation(parts if strand == 1 else parts[::-1])
        cds = SeqFeature(location, type="CDS", qualifiers={"gene": ["example"]})
        record.features = [SeqFeature(FeatureLocation(0, 12), type="source"), cds]
        expected = cds.extract(record)
        extracted = util.replace_seq2cds(record)
        assert extracted.seq == expected.seq
        assert extracted.letter_annotations == expected.letter_annotations
        assert len(extracted.features) == 1
        assert extracted.features[0].location == FeatureLocation(
            0, len(expected), strand=1
        )
        assert extracted.features[0].extract(extracted).seq == expected.seq
        assert extracted.features[0].qualifiers == cds.qualifiers
        assert extracted.annotations["topology"] == "linear"
        assert str(record.seq) == "AAATGCCCCTTT"
        assert record.features[1].location == location


class TestReadGff:
    """Tests for read_gff function."""

    def test_single_record_gff_is_returned_as_1d_array(self, temp_dir):
        """Single non-header line should still produce length-1 structured array."""
        path = temp_dir / "single.gff"
        path.write_text("##gff-version 3\nseq1\tsource\tgene\t1\t10\t.\t+\t.\tID=g1\n")
        result = util.read_gff(str(path))
        assert len(result["data"]) == 1
        assert result["data"][0]["seqid"] == "seq1"

    def test_read_gff_preserves_long_attributes(self, temp_dir):
        path = temp_dir / "long_attr.gff"
        long_attr = "ID=" + ("A" * 700)
        path.write_text(
            f"##gff-version 3\nseq1\tsource\tgene\t1\t10\t.\t+\t.\t{long_attr}\n"
        )
        result = util.read_gff(str(path))
        assert len(result["data"]) == 1
        assert result["data"][0]["attributes"] == long_attr


class TestWriteGff:
    """Tests for write_gff function."""

    def test_write_and_read_roundtrip(self, temp_dir):
        out_path = temp_dir / "roundtrip.gff"
        dtype = [
            ("seqid", "U100"),
            ("source", "U100"),
            ("type", "U100"),
            ("start", "i4"),
            ("end", "i4"),
            ("score", "U100"),
            ("strand", "U10"),
            ("phase", "U10"),
            ("attributes", "U500"),
        ]
        data = np.array(
            [
                ("seq1", "src", "gene", 1, 100, ".", "+", ".", "ID=g1"),
                ("seq1", "src", "CDS", 10, 90, ".", "+", "0", "ID=c1"),
            ],
            dtype=dtype,
        )
        gff = {"header": ["##gff-version 3"], "data": data}

        util.write_gff(gff, str(out_path))
        reread = util.read_gff(str(out_path))
        assert reread["header"] == ["##gff-version 3"]
        assert len(reread["data"]) == 2
        assert reread["data"].tolist() == data.tolist()


class TestCoordinates2Ranges:
    """Tests for coordinates2ranges function."""

    def test_empty_coordinates(self):
        """Test with empty list."""
        coords = []
        result = util.coordinates2ranges(coords)
        assert result == []

    def test_multiple_ranges(self):
        """Test with multiple separate ranges."""
        coords = [1, 5, 6, 7, 20]
        result = util.coordinates2ranges(coords)
        assert result == [(1, 1), (5, 7), (20, 20)]

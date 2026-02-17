"""
Tests for cdskit accession2fasta command.
"""

from pathlib import Path

import pytest
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import cdskit.accession2fasta as accession_module
from cdskit.accession2fasta import (
    accession2seq_record,
    accession_batch_ranges,
    prepare_accession_record,
)


class TestAccession2FastaHelpers:
    """Tests for helper functions in accession2fasta."""

    def test_accession_batch_ranges(self):
        assert list(accession_batch_ranges(0, 1000)) == []
        assert list(accession_batch_ranges(3, 2)) == [(0, 2), (2, 3)]
        assert list(accession_batch_ranges(4, 2)) == [(0, 2), (2, 4)]

    def test_prepare_accession_record_updates_metadata(self):
        record = SeqRecord(Seq("ATGAAA"), id="ACC1", name="ACC1", description="desc")
        record.annotations["organism"] = "Homo sapiens"
        record.annotations["accessions"] = ["ACC1"]

        prepared = prepare_accession_record(
            record=record,
            seqnamefmt="organism_accessions",
            extract_cds=False,
            list_seqname_keys=False,
        )
        assert prepared is not None
        assert "Homo" in prepared.id or "ACC1" in prepared.id
        assert prepared.name == ""
        assert prepared.description == ""

    def test_prepare_accession_record_extract_cds(self):
        record = SeqRecord(Seq("AAATGCCCCTTT"), id="ACC1", name="ACC1", description="desc")
        record.annotations["organism"] = "Homo sapiens"
        record.annotations["accessions"] = ["ACC1"]
        record.features = [
            SeqFeature(FeatureLocation(3, 9), type="CDS"),
        ]

        prepared = prepare_accession_record(
            record=record,
            seqnamefmt="organism_accessions",
            extract_cds=True,
            list_seqname_keys=False,
        )
        assert prepared is not None
        assert str(prepared.seq) == "TGCCCC"


class TestAccession2SeqRecord:
    """Tests for accession retrieval batching logic."""

    def test_accession2seq_record_batches_without_empty_calls(self, monkeypatch):
        calls = []

        def fake_efetch(db, id, rettype, retmode, retmax):
            calls.append(list(id))
            return {"ids": list(id)}

        def fake_parse(handle, fmt):
            return [SeqRecord(Seq("ATG"), id=acc) for acc in handle["ids"]]

        monkeypatch.setattr(accession_module.Entrez, "efetch", fake_efetch)
        monkeypatch.setattr(accession_module.SeqIO, "parse", fake_parse)

        accessions = ["ACC1", "ACC2", "ACC3"]
        records = accession2seq_record(accessions, database="nuccore", batch_size=2)

        assert calls == [["ACC1", "ACC2"], ["ACC3"]]
        assert [r.id for r in records] == accessions

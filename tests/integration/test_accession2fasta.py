"""
Tests for cdskit accession2fasta command.
"""

import pytest
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

import cdskit.accession2fasta as accession_module
from cdskit.accession2fasta import (
    accession2fasta_main,
    accession_matches_record_id,
    accession2seq_record,
    accession_batch_ranges,
    find_missing_accessions,
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
        record = SeqRecord(
            Seq("AAATGCCCCTTT"), id="ACC1", name="ACC1", description="desc"
        )
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

    def test_accession_matches_record_id_handles_version_and_pipe(self):
        assert accession_matches_record_id("AB1", "AB1.2")
        assert accession_matches_record_id("AB1", "gi|1|ref|AB1.3|")
        assert not accession_matches_record_id("AB1", "AB12.1")

    def test_find_missing_accessions_does_not_use_substring_match(self):
        seq_records = [SeqRecord(Seq("ATG"), id="AB12.1")]
        missing = find_missing_accessions(
            accessions=["AB1", "AB12"], seq_records=seq_records
        )
        assert missing == ["AB1"]


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

    def test_accession2seq_record_closes_efetch_handles(self, monkeypatch):
        close_calls = []

        class FakeHandle:
            def __init__(self, ids):
                self.ids = ids

            def close(self):
                close_calls.append(tuple(self.ids))

        def fake_efetch(db, id, rettype, retmode, retmax):
            return FakeHandle(list(id))

        def fake_parse(handle, fmt):
            return [SeqRecord(Seq("ATG"), id=acc) for acc in handle.ids]

        monkeypatch.setattr(accession_module.Entrez, "efetch", fake_efetch)
        monkeypatch.setattr(accession_module.SeqIO, "parse", fake_parse)

        accessions = ["ACC1", "ACC2", "ACC3"]
        records = accession2seq_record(accessions, database="nuccore", batch_size=2)

        assert [r.id for r in records] == accessions
        assert close_calls == [("ACC1", "ACC2"), ("ACC3",)]


class TestAccession2FastaMain:
    """Tests for accession2fasta_main function."""

    def test_accession2fasta_threads_matches_single_thread(
        self, temp_dir, mock_args, monkeypatch
    ):
        accession_file = temp_dir / "acc.txt"
        accession_file.write_text("ACC1\nACC2\nACC3\n")
        out_single = temp_dir / "single.fasta"
        out_threaded = temp_dir / "threaded.fasta"

        def make_records():
            records = []
            for acc in ["ACC1", "ACC2", "ACC3"]:
                rec = SeqRecord(Seq("ATGAAA"), id=acc, name=acc, description="")
                rec.annotations["organism"] = "Homo sapiens"
                rec.annotations["accessions"] = [acc]
                records.append(rec)
            return records

        monkeypatch.setattr(
            accession_module,
            "accession2seq_record",
            lambda accessions, database, **kwargs: make_records(),
        )

        args_single = mock_args(
            accession_file=str(accession_file),
            outfile=str(out_single),
            outseqformat="fasta",
            email="",
            extract_cds=False,
            ncbi_database="nucleotide",
            seqnamefmt="organism_accessions",
            list_seqname_keys=False,
            threads=1,
        )
        args_threaded = mock_args(
            accession_file=str(accession_file),
            outfile=str(out_threaded),
            outseqformat="fasta",
            email="",
            extract_cds=False,
            ncbi_database="nucleotide",
            seqnamefmt="organism_accessions",
            list_seqname_keys=False,
            threads=4,
        )

        accession2fasta_main(args_single)
        accession2fasta_main(args_threaded)

        result_single = list(accession_module.SeqIO.parse(str(out_single), "fasta"))
        result_threaded = list(accession_module.SeqIO.parse(str(out_threaded), "fasta"))
        assert [r.id for r in result_single] == [r.id for r in result_threaded]
        assert [str(r.seq) for r in result_single] == [
            str(r.seq) for r in result_threaded
        ]

    def test_accession2fasta_requires_accession_file(self, mock_args):
        args = mock_args(
            accession_file="",
            outfile="-",
            outseqformat="fasta",
            email="",
            extract_cds=False,
            ncbi_database="nucleotide",
            seqnamefmt="organism_accessions",
            list_seqname_keys=False,
            threads=1,
        )

        with pytest.raises(ValueError) as exc_info:
            accession2fasta_main(args)
        assert "--accession_file is required" in str(exc_info.value)


@pytest.mark.parametrize(
    "returned", [["ACC1.1"], ["ACC1.1", "ACC1.1"], ["ACC1.1", "OTHER.1"]]
)
@pytest.mark.parametrize("strict", [True, False])
def test_incomplete_retrieval_is_explicit_and_preserves_output(
    tmp_path, monkeypatch, capsys, returned, strict
):
    from io import StringIO
    from Bio import SeqIO
    from cdskit.cli import main

    source, output = tmp_path / "ids.txt", tmp_path / "output.fa"
    source.write_text("ACC1\nACC2\n")
    output.write_text("original")
    records = [
        SeqRecord(
            Seq("ATGAAA"),
            id=acc,
            annotations={"molecule_type": "DNA", "accessions": [acc.split(".")[0]]},
        )
        for acc in returned
    ]
    response = StringIO()
    SeqIO.write(records, response, "genbank")
    monkeypatch.setattr(
        accession_module.Entrez,
        "efetch",
        lambda **kwargs: StringIO(response.getvalue()),
    )
    result = main(
        [
            "accession2fasta",
            "--accession_file",
            str(source),
            "--out_file",
            str(output),
            "--extract_cds",
            "no",
            "--strict",
            "yes" if strict else "no",
        ]
    )
    assert result == (1 if strict else 0)
    assert "missing=ACC2" in capsys.readouterr().err
    if strict:
        assert output.read_text() == "original"
    else:
        assert len(list(SeqIO.parse(output, "fasta"))) == len(returned)


def test_accession_aliases_keep_explicit_versions_and_request_duplicates():
    from cdskit.accession2fasta import accession_retrieval_issues

    record = SeqRecord(Seq("ATG"), id="gi|1|ref|ACC1.2|")
    assert not any(
        accession_retrieval_issues(["ACC1", "ACC1", "ACC1.2"], [record]).values()
    )
    assert accession_retrieval_issues(["ACC1.1"], [record])["missing"] == ["ACC1.1"]

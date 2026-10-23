"""Bounded codon-position copies and multi-output compatibility."""

import weakref

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
import pytest

from cdskit import split


def annotated_record(sequence="ATGAAACCC", seq_id="record"):
    return SeqRecord(
        Seq(sequence),
        id=seq_id,
        name=seq_id,
        description="annotated DNA",
        dbxrefs=["example:record"],
        features=[SeqFeature(SimpleLocation(0, len(sequence)), type="CDS")],
        annotations={"molecule_type": "DNA", "date": "01-JAN-2000"},
        letter_annotations={"phred_quality": [30] * len(sequence)},
    )


@pytest.mark.parametrize("sequence", ["ATGAAACCC", ""])
def test_public_split_preserves_metadata_and_source_sequence(sequence):
    original = annotated_record(sequence)

    result = split.split_record_by_codon_position(original)

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert len({id(record) for record in result}) == 3
    for position, record in enumerate(result):
        assert str(record.seq) == sequence[position::3]
        assert record.id == original.id
        assert record.name == original.name
        assert record.description == original.description
        assert record.dbxrefs == original.dbxrefs
        assert record.annotations == original.annotations
        if sequence:
            assert record.features == []
            assert record.letter_annotations == {}
        else:
            assert record.features == original.features
            assert record.letter_annotations == original.letter_annotations
    assert str(original.seq) == sequence
    assert len(original.features) == 1


def test_main_releases_previous_position_records_before_next_write(
    tmp_path, mock_args, monkeypatch
):
    originals = [annotated_record(), annotated_record(seq_id="second")]
    monkeypatch.setattr(split, "read_seqs", lambda **kwargs: originals)
    previous_refs = []
    calls = []
    write_seqs = split.write_seqs

    def inspect_write(records, outfile, outseqformat):
        assert all(reference() is None for reference in previous_refs)
        assert isinstance(records, list)
        position = len(calls)
        assert [str(record.seq) for record in records] == [
            str(record.seq)[position::3] for record in originals
        ]
        for original, projected in zip(originals, records, strict=True):
            assert projected.annotations == original.annotations
            assert projected.description == original.description
            assert projected.features == []
            assert projected.letter_annotations == {}
        previous_refs[:] = [weakref.ref(record) for record in records]
        calls.append(position)
        write_seqs(records=records, outfile=outfile, outseqformat=outseqformat)

    monkeypatch.setattr(split, "write_seqs", inspect_write)
    split.split_main(
        mock_args(seqfile=str(tmp_path / "input.fa"), prefix=str(tmp_path / "out"))
    )

    assert calls == [0, 1, 2]
    assert all(reference() is None for reference in previous_refs)
    assert all(len(record.features) == 1 for record in originals)
    assert all(record.annotations["molecule_type"] == "DNA" for record in originals)


@pytest.mark.parametrize("outseqformat", ["fasta", "phylip", "clustal", "genbank"])
def test_main_output_bytes_match_public_split_for_sequence_and_alignment_formats(
    tmp_path, mock_args, capsys, outseqformat
):
    source = tmp_path / "input.gb"
    originals = [annotated_record(), annotated_record("CCC---ATG", "second")]
    Bio.SeqIO.write(originals, source, "genbank")
    prefix = str(tmp_path / "result")

    split.split_main(
        mock_args(
            seqfile=str(source),
            inseqformat="genbank",
            outseqformat=outseqformat,
            prefix=prefix,
        )
    )

    loaded = list(Bio.SeqIO.parse(source, "genbank"))
    public_results = [split.split_record_by_codon_position(record) for record in loaded]
    for position, path in enumerate(
        split.build_split_output_paths(prefix, outseqformat)
    ):
        expected = tmp_path / f"expected_{position}.{outseqformat}"
        Bio.SeqIO.write(
            [result[position] for result in public_results], expected, outseqformat
        )
        with open(path, "rb") as actual:
            assert actual.read() == expected.read_bytes()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Number of input sequences: 2\n"
        "Writing first codon positions.\nNumber of output sequences: 2\n"
        "Writing second codon positions.\nNumber of output sequences: 2\n"
        "Writing third codon positions.\nNumber of output sequences: 2\n"
    )


@pytest.mark.parametrize("existing", [False, True])
def test_later_write_failure_rolls_back_all_positions(
    tmp_path, mock_args, write_fasta, monkeypatch, existing
):
    source = write_fasta(tmp_path / "input.fa", [("record", "ATGAAACCC")])
    prefix = str(tmp_path / "out")
    outputs = split.build_split_output_paths(prefix, "fasta")
    if existing:
        for output in outputs:
            with open(output, "w") as handle:
                handle.write("original " + output)
    write_seqs = split.write_seqs
    calls = 0

    def fail_second_write(**kwargs):
        nonlocal calls
        calls += 1
        write_seqs(**kwargs)
        if calls == 2:
            raise OSError("second output failed")

    monkeypatch.setattr(split, "write_seqs", fail_second_write)
    with pytest.raises(OSError, match="second output failed"):
        split.split_main(mock_args(seqfile=str(source), prefix=prefix))

    for output in outputs:
        path = tmp_path / output
        if existing:
            assert path.read_text() == "original " + output
        else:
            assert not path.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [source.name]
        + ([tmp_path.joinpath(path).name for path in outputs] if existing else [])
    )


def test_fastq_missing_transformed_qualities_preserves_all_outputs(tmp_path, mock_args):
    source = tmp_path / "input.fastq"
    Bio.SeqIO.write([annotated_record()], source, "fastq")
    prefix = str(tmp_path / "out")
    outputs = split.build_split_output_paths(prefix, "fastq")
    for output in outputs:
        with open(output, "w") as handle:
            handle.write("original")

    with pytest.raises(ValueError, match="requires per-base quality scores"):
        split.split_main(
            mock_args(
                seqfile=str(source),
                inseqformat="fastq",
                outseqformat="fastq",
                prefix=prefix,
            )
        )

    for output in outputs:
        with open(output) as handle:
            assert handle.read() == "original"


@pytest.mark.parametrize("position", [0, 1, 2])
def test_each_output_path_is_checked_against_input(
    tmp_path, mock_args, write_fasta, position
):
    prefix = str(tmp_path / "out")
    source = split.build_split_output_paths(prefix, "fasta")[position]
    write_fasta(source, [("record", "ATGAAACCC")])
    with open(source, "rb") as handle:
        original = handle.read()

    with pytest.raises(ValueError, match="Input and output paths"):
        split.split_main(mock_args(seqfile=source, prefix=prefix))

    with open(source, "rb") as handle:
        assert handle.read() == original
    assert len(list(tmp_path.iterdir())) == 1

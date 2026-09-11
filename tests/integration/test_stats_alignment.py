"""Compare against independently generated AMAS output and hand-counted sites."""

import csv
from io import StringIO
from pathlib import Path

import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.stats import summarize_alignment
from cdskit.cli import main


FIXTURES = Path(__file__).parents[1] / "fixtures" / "alignment_stats"


@pytest.mark.parametrize(
    "name",
    ["dna_edges", "dna_missing", "dna_rounding", "dna_random", "aa_edges", "aa_random"],
)
def test_matches_amas_reference(name, tmp_path):
    output = tmp_path / "summary.tsv"
    assert (
        main(
            [
                "stats",
                "--mode",
                "alignment",
                "--seq_file",
                str(FIXTURES / f"{name}.fasta"),
                "--seq_type",
                name.split("_")[0],
                "--out_file",
                str(output),
            ]
        )
        == 0
    )
    with (FIXTURES / f"{name}.tsv").open() as handle:
        expected = next(csv.DictReader(handle, delimiter="\t"))
    with output.open() as handle:
        actual = next(csv.DictReader(handle, delimiter="\t"))
    assert {key: actual[key] for key in expected} == expected
    if name.startswith("aa"):
        assert actual["GC_content"] == actual["AT_content"] == "NA"


def test_hand_counted_sites():
    records = [
        SeqRecord(Seq(s), id=str(i)) for i, s in enumerate(["AAN", "ACN", "GCN", "GCN"])
    ]
    row = summarize_alignment(records, "dna")
    assert row["No_variable_sites"] == 2
    assert row["Parsimony_informative_sites"] == 1
    assert row["Undetermined_characters"] == 4
    assert row["Missing_percent"] == 33.333
    assert row["GC_content"] == 0.625


@pytest.mark.parametrize(
    "sequences,ids,message",
    [
        ([], [], "nonempty"),
        ([""], ["a"], "nonempty"),
        (["AC", "A"], ["a", "b"], "equal lengths"),
        (["AC", "AC"], ["a", "a"], "Duplicate"),
        (["AC"], [""], "IDs must not be empty"),
        (["AC", "A."], ["a", "b"], "Invalid dna"),
    ],
)
def test_invalid_alignment(sequences, ids, message):
    records = [
        SeqRecord(Seq(seq), id=label) for seq, label in zip(sequences, ids, strict=True)
    ]
    with pytest.raises(ValueError, match=message):
        summarize_alignment(records, "dna")


def test_stdin_lowercase_and_nontriplet(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", StringIO(">a\nac\n>b\ngc\n"))
    assert main(["stats", "--mode", "alignment", "--seq_type", "dna"]) == 0
    rows = list(csv.DictReader(StringIO(capsys.readouterr().out), delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["No_variable_sites"] == "1"
    assert rows[0]["Alignment_length"] == "2"


def test_invalid_input_preserves_output(tmp_path):
    source = tmp_path / "bad.fasta"
    source.write_text(">a\nAC\n>b\nA\n")
    output = tmp_path / "out.tsv"
    output.write_text("keep me")
    assert (
        main(
            [
                "stats",
                "--mode",
                "alignment",
                "--seq_type",
                "dna",
                "--seq_file",
                str(source),
                "--out_file",
                str(output),
            ]
        )
        != 0
    )
    assert output.read_text() == "keep me"


def test_input_output_collision(tmp_path):
    source = tmp_path / "input.fasta"
    content = ">a\nAC\n"
    source.write_text(content)
    assert (
        main(
            [
                "stats",
                "--mode",
                "alignment",
                "--seq_type",
                "dna",
                "--seq_file",
                str(source),
                "--out_file",
                str(source),
            ]
        )
        != 0
    )
    assert source.read_text() == content


def test_default_and_explicit_sequence_mode_match(tmp_path, capsys):
    source = tmp_path / "input.fasta"
    source.write_text(">a\nACgtNN--\n>b\nGC\n")
    assert main(["stats", "--seq_file", str(source)]) == 0
    expected = capsys.readouterr().out
    output = tmp_path / "summary.txt"
    assert (
        main(
            [
                "stats",
                "--mode",
                "sequence",
                "--seq_file",
                str(source),
                "--out_file",
                str(output),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == ""
    assert output.read_text() == expected
    assert "Total length: 10" in expected


def test_sequence_mode_rejects_protein_option(tmp_path):
    source = tmp_path / "input.fasta"
    source.write_text(">a\nAC\n")
    output = tmp_path / "summary.txt"
    output.write_text("existing")
    assert (
        main(
            [
                "stats",
                "--seq_type",
                "aa",
                "--seq_file",
                str(source),
                "--out_file",
                str(output),
            ]
        )
        != 0
    )
    assert output.read_text() == "existing"


@pytest.mark.parametrize("mode", ["sequence", "alignment"])
def test_failed_file_write_preserves_existing_output(mode, tmp_path, monkeypatch):
    from contextlib import contextmanager
    from cdskit.atomicio import atomic_text_writer

    source = tmp_path / "input.fasta"
    source.write_text(">a\nAC\n")
    output = tmp_path / "out.txt"
    output.write_text("original")

    @contextmanager
    def fail_commit(*args, **kwargs):
        with atomic_text_writer(*args, **kwargs) as handle:
            yield handle
            raise OSError("simulated disk failure")

    module = "cdskit.stats" if mode == "sequence" else "cdskit.tsvio"
    monkeypatch.setattr(f"{module}.atomic_text_writer", fail_commit)
    assert (
        main(
            [
                "stats",
                "--mode",
                mode,
                "--seq_file",
                str(source),
                "--out_file",
                str(output),
            ]
        )
        != 0
    )
    assert output.read_text() == "original"


@pytest.mark.parametrize("mode", ["sequence", "alignment"])
def test_rejects_invalid_threads(mode, tmp_path):
    source = tmp_path / "input.fasta"
    source.write_text(">a\nAC\n")
    assert (
        main(["stats", "--mode", mode, "--threads", "-1", "--seq_file", str(source)])
        != 0
    )


@pytest.mark.parametrize("seq_format", ["phylip", "nexus"])
def test_non_fasta_alignment(seq_format, tmp_path):
    from Bio import SeqIO

    source = tmp_path / "input.aln"
    records = [
        SeqRecord(Seq(s), id=str(i), annotations={"molecule_type": "DNA"})
        for i, s in enumerate(["AAN", "ACN", "GCN", "GCN"])
    ]
    SeqIO.write(records, source, seq_format)
    output = tmp_path / "out.tsv"
    assert (
        main(
            [
                "stats",
                "--mode",
                "alignment",
                "--in_seq_format",
                seq_format,
                "--seq_file",
                str(source),
                "--out_file",
                str(output),
            ]
        )
        == 0
    )
    with output.open() as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["No_variable_sites"] == "2"
    assert row["Parsimony_informative_sites"] == "1"

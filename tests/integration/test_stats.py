"""Sequence statistics: all reported fields, empty input, and worker equivalence."""

import io

import pytest

from cdskit.stats import stats_main


def test_stats_mixed_content(tmp_path, mock_args, write_fasta, capsys):
    source = write_fasta(
        tmp_path / "input.fasta",
        [("mixed", "ATG---NNNcccGGG"), ("lower", "gcnn")],
    )
    stats_main(mock_args(seqfile=str(source)))
    output = capsys.readouterr().out
    for line in (
        "Number of sequences: 2",
        "Total length: 19",
        "Total gap (-) length: 3",
        "Total N length: 5",
        "Total softmasked length: 7",
        "GC content: 47.4%",
    ):
        assert line in output


def test_stats_empty_file(tmp_path, mock_args, capsys):
    source = tmp_path / "empty.fasta"
    source.write_text("")
    stats_main(mock_args(seqfile=str(source)))
    output = capsys.readouterr().out
    assert "Number of sequences: 0" in output
    assert "Total length: 0" in output
    assert "GC content: 0.0%" in output


def test_stats_threads_matches_single_thread(tmp_path, mock_args, write_fasta, capsys):
    source = write_fasta(
        tmp_path / "input.fasta",
        [("mixed", "ATG---NNNcccGGG"), ("clean", "ATGCCC"), ("gc", "GGGTTT")],
    )
    stats_main(mock_args(seqfile=str(source), threads=1))
    single = capsys.readouterr().out
    stats_main(mock_args(seqfile=str(source), threads=4))
    assert capsys.readouterr().out == single


def test_stats_aggregates_all_batches_from_stdin(monkeypatch, mock_args, capsys):
    content = ">repeated\nGC\n" * 10_002 + ">last\nccn-?\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(content))
    stats_main(mock_args(seqfile="-", threads=4))
    output = capsys.readouterr()
    assert output.out.splitlines() == [
        "Number of sequences: 10,003",
        "Total length: 20,009",
        "Total softmasked length: 3",
        "Total N length: 1",
        "Total gap (-) length: 1",
        "GC content: 100.0%",
    ]
    assert output.err == "Number of input sequences: 10,003\n"


def test_stats_invalid_diagnostics_cover_all_batches(tmp_path, mock_args, capsys):
    source = tmp_path / "input.fasta"
    source.write_text(
        "".join(f">bad{i}\nU\n" for i in range(11))
        + ">valid\nAC\n" * 10_001
        + ">late\nE\n"
    )
    output_path = tmp_path / "summary.txt"
    output_path.write_text("existing")
    expected_ids = ",".join(f"bad{i}" for i in range(10))
    with pytest.raises(ValueError) as raised:
        stats_main(mock_args(seqfile=str(source), outfile=str(output_path)))
    assert f"({expected_ids},...) [chars: EU]" in str(raised.value)
    assert output_path.read_text() == "existing"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Number of input sequences: 10,013\n"

"""Sequence statistics: all reported fields, empty input, and worker equivalence."""

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

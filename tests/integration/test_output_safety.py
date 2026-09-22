import json
import os

import pytest

from cdskit.cli import main, subparsers
from cdskit.command_paths import COMMAND_PATHS


def test_all_public_commands_declare_their_file_roles():
    assert set(COMMAND_PATHS) == set(subparsers.choices)


@pytest.mark.parametrize("alias_is_input", [False, True])
@pytest.mark.parametrize("alias_kind", ["literal_tilde", "symlink_parent"])
def test_input_alias_cannot_bypass_collision_check(
    tmp_path, monkeypatch, capsys, alias_is_input, alias_kind
):
    monkeypatch.chdir(tmp_path)
    if alias_kind == "literal_tilde":
        directory = tmp_path / "~"
        directory.mkdir()
        alias = "~/input.fa"
    else:
        if os.name == "nt":
            pytest.skip("Windows resolves parent components before directory symlinks")
        directory = tmp_path / "actual"
        child = directory / "child"
        child.mkdir(parents=True)
        try:
            (tmp_path / "link").symlink_to(child, target_is_directory=True)
        except OSError:
            pytest.skip("Directory symlinks are unavailable")
        alias = "link/../input.fa"
    source = directory / "input.fa"
    contents = ">sequence\nATGAAA\n"
    source.write_text(contents)
    input_path, output_path = (alias, str(source))
    if not alias_is_input:
        input_path, output_path = output_path, input_path
    assert main(["translate", "--seq_file", input_path, "--out_file", output_path]) == 1
    assert "Input and output paths" in capsys.readouterr().err
    assert source.read_text() == contents


@pytest.mark.parametrize("command", ["gapjust", "intersection"])
def test_gff_stdout_does_not_overwrite_literal_dash(
    tmp_path, monkeypatch, capsys, command
):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "input.fa"
    source.write_text(">seq1\nATGAAA\n")
    gff = tmp_path / "input.gff"
    contents = "##gff-version 3\nseq1\ttest\tgene\t1\t6\t.\t+\t.\tID=gene1\n"
    gff.write_text(contents)
    dash = tmp_path / "-"
    dash.write_text("keep this file")
    assert (
        main(
            [
                command,
                "--seq_file",
                str(source),
                "--in_gff",
                str(gff),
                "--out_file",
                str(tmp_path / "output.fa"),
                "--out_gff",
                "-",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == contents
    assert dash.read_text() == "keep this file"


@pytest.mark.parametrize(
    "command,input_option,contents",
    [
        ("backalign", "--aa_aln", ">seq1\nMK\n"),
        ("backtrim", "--trimmed_aa_aln", ">seq1\nMK\n"),
        ("accession2fasta", "--accession_file", "TEST0001\n"),
    ],
)
def test_auxiliary_input_cannot_be_overwritten(
    tmp_path, capsys, command, input_option, contents
):
    source = tmp_path / "input.txt"
    source.write_text(contents)
    assert main([command, input_option, str(source), "--out_file", str(source)]) == 1
    assert "Input and output paths should be different" in capsys.readouterr().err
    assert source.read_text() == contents


@pytest.mark.parametrize("command", ["filter", "trimcodon"])
@pytest.mark.parametrize("report_suffix", ["json", "tsv"])
def test_failed_sequence_output_does_not_replace_report(
    tmp_path, command, report_suffix
):
    source = tmp_path / "in.fa"
    source.write_text(">seq1\nATGAAA\n")
    output = tmp_path / "out.fastq"
    output.write_text("original sequence output")
    report = tmp_path / f"report.{report_suffix}"
    report.write_text("original report")
    assert (
        main(
            [
                command,
                "--seq_file",
                str(source),
                "--out_file",
                str(output),
                "--out_seq_format",
                "fastq",
                "--report",
                str(report),
            ]
        )
        == 1
    )
    assert output.read_text() == "original sequence output"
    assert report.read_text() == "original report"


@pytest.mark.parametrize("command", ["filter", "trimcodon"])
def test_sequence_and_json_report_commit_together(tmp_path, command):
    source = tmp_path / "in.fa"
    source.write_text(">seq1\nATGAAA\n")
    output = tmp_path / "out.fa"
    report = tmp_path / "report.json"
    assert (
        main(
            [
                command,
                "--seq_file",
                str(source),
                "--out_file",
                str(output),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    assert "ATGAAA" in output.read_text()
    assert isinstance(json.loads(report.read_text()), dict)


@pytest.mark.parametrize(
    "command,suffix",
    [("split", "1st_codon_positions"), ("degeneracy", "0fold_positions")],
)
def test_generated_output_filenames_cannot_replace_input(
    tmp_path, capsys, command, suffix
):
    prefix = tmp_path / "sequences"
    source = tmp_path / f"sequences_{suffix}.fasta"
    source.write_text(">sequence\nATGAAA\n")
    assert main([command, "--seq_file", str(source), "--prefix", str(prefix)]) == 1
    assert "Input and output paths" in capsys.readouterr().err
    assert source.read_text() == ">sequence\nATGAAA\n"


def test_resolved_model_alias_is_protected_before_loading(
    tmp_path, monkeypatch, capsys
):
    from cdskit import localize

    source = tmp_path / "sequences.fa"
    source.write_text(">sequence\nATGAAA\n")
    model = tmp_path / "cached-model.pt"
    model.write_bytes(b"existing model")
    monkeypatch.setattr(
        localize, "resolve_localize_model_path", lambda **kwargs: str(model)
    )
    monkeypatch.setattr(
        localize,
        "load_localize_model",
        lambda **kwargs: pytest.fail("Must reject before loading"),
    )
    assert (
        main(
            [
                "localize",
                "--seq_file",
                str(source),
                "--model",
                "plant",
                "--report",
                str(model),
            ]
        )
        == 1
    )
    assert "Input and output paths" in capsys.readouterr().err
    assert model.read_bytes() == b"existing model"


def test_expanded_model_path_is_protected(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    model = tmp_path / "model.json"
    model.write_text("existing model")
    assert main(["localize", "--model", "~/model.json", "--report", str(model)]) == 1
    assert "Input and output paths" in capsys.readouterr().err
    assert model.read_text() == "existing model"


@pytest.mark.parametrize(
    "command", ["filter", "trimcodon", "maxalign", "pad", "longestorf"]
)
def test_sequence_and_report_cannot_share_stdout(tmp_path, capsys, command):
    source = tmp_path / "input.fa"
    source.write_text(">s\nATGAAA\n")
    assert (
        main([command, "--seq_file", str(source), "--out_file", "-", "--report", "-"])
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "standard output" in captured.err


@pytest.mark.parametrize("command", ["intersection", "gapjust"])
def test_sequence_and_gff_cannot_share_stdout(tmp_path, capsys, command):
    source = tmp_path / "input.fa"
    source.write_text(">s\nATGAAA\n")
    gff = tmp_path / "input.gff"
    gff.write_text("s\t.\tgene\t1\t6\t.\t+\t.\tID=g\n")
    assert (
        main(
            [command, "--seq_file", str(source), "--in_gff", str(gff), "--out_gff", "-"]
        )
        == 1
    )
    assert capsys.readouterr().out == ""


def test_intersection_rejects_consuming_stdin_twice(tmp_path, capsys):
    assert (
        main(
            [
                "intersection",
                "--seq_file",
                "-",
                "--seq_file_2",
                "-",
                "--out_file_2",
                str(tmp_path / "other.fa"),
            ]
        )
        == 1
    )
    assert "standard input" in capsys.readouterr().err
    assert not (tmp_path / "other.fa").exists()

import json

import pytest

from cdskit.cli import main, subparsers
from cdskit.command_paths import COMMAND_PATHS


def test_all_public_commands_declare_their_file_roles():
    assert set(COMMAND_PATHS) == set(subparsers.choices)


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

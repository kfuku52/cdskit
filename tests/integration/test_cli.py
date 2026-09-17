"""
Tests for cdskit CLI (command line interface).

These tests verify that argparse help strings are properly formatted
and don't cause errors (e.g., Issue #10 with % characters).
"""

import pytest
import sys
import os
import subprocess
from pathlib import Path

from cdskit.cli import main as cli_main
from cdskit.cliutil import CdskitArgumentParser, parse_bool, resolve_threads


@pytest.mark.parametrize("model", [None, "targeting5", "custom.pt"])
def test_localize_model_default_and_override(model):
    from cdskit.cli import psr

    argv = ["localize", "--seq_file", "proteins.faa"]
    if model is not None:
        argv.extend(["--model", model])
    assert psr.parse_args(argv).model == (model or "esm2-localization-v1")


def test_localize_default_offline_reports_actionable_error(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("CDSKIT_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("CDSKIT_OFFLINE", "1")
    seq_file = tmp_path / "protein.faa"
    seq_file.write_text(">protein\nMAAAA\n")
    assert (
        cli_main(["localize", "--seq_file", str(seq_file), "--seq_type", "protein"])
        == 1
    )
    error = capsys.readouterr().err
    assert "esm2-localization-v1" in error
    assert "model download is disabled" in error


class TestCLIHelpStrings:
    """Tests for CLI help string formatting.

    Issue #10: ValueError: badly formed help string
    This occurred in Python 3.14 because of unescaped % in help text.
    The help string for --replace_chars contained special characters including %
    which was interpreted as a format specifier.
    """

    def test_label_replace_chars_help_format(self, capsys):
        """Test the specific help string format from Issue #10.

        The --replace_chars argument has a help string containing special characters
        including %. This must be properly escaped or it causes errors in Python 3.14+.
        """
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["label", "--help"])
        captured = capsys.readouterr()

        assert exc_info.value.code == 0
        assert "--replace_chars" in captured.out
        assert "!@#$%^&*+=/?<>|--_" in captured.out
        assert captured.err == ""


class TestCLIModuleImport:
    """Tests that CLI-related modules can be imported without errors."""

    def test_import_all_command_modules(self):
        """Test that all command modules can be imported."""
        modules = [
            "cdskit.accession2fasta",
            "cdskit.aggregate",
            "cdskit.backalign",
            "cdskit.backtrim",
            "cdskit.codonstats",
            "cdskit.codonutil",
            "cdskit.deeploc_benchmark",
            "cdskit.degeneracy",
            "cdskit.filter",
            "cdskit.hammer",
            "cdskit.intersection",
            "cdskit.label",
            "cdskit.longestcds",
            "cdskit.longestorf",
            "cdskit.localize",
            "cdskit.localize_bilstm",
            "cdskit.localize_learn",
            "cdskit.localize_model",
            "cdskit.localize_models",
            "cdskit.localize_multilabel_cnn",
            "cdskit.maxalign",
            "cdskit.mask",
            "cdskit.pad",
            "cdskit.parsegb",
            "cdskit.plot",
            "cdskit.printseq",
            "cdskit.rmseq",
            "cdskit.split",
            "cdskit.stats",
            "cdskit.translate",
            "cdskit.trimcodon",
            "cdskit.gapjust",
            "cdskit.util",
            "cdskit.validate",
        ]

        for module_name in modules:
            __import__(module_name)

    @pytest.mark.subprocess
    def test_root_version_option(self):
        """Test that cdskit --version works without a subcommand."""
        from cdskit import __version__

        cli_path = Path(__file__).resolve().parents[2] / "cdskit" / "cdskit"
        result = subprocess.run(
            [sys.executable, str(cli_path), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )

        assert result.stdout.strip() == f"cdskit version {__version__}"
        assert result.stderr == ""


class TestCLIConsistency:
    def test_automatic_help_describes_representative_argument_roles(self):
        parser = CdskitArgumentParser()
        parser.add_argument("--training_tsv")
        parser.add_argument("--verbose", action="store_true")

        help_text = parser.format_help()

        assert "Path to the training TSV file." in help_text
        assert "Enable detailed progress output." in help_text
        assert "Set training tsv." not in help_text

    def test_deprecated_long_option_is_accepted_with_warning(self, capsys):
        parser = CdskitArgumentParser()
        parser.add_argument("--seq_file")
        parser.add_deprecated_alias("--seqfile", "--seq_file")

        args = parser.parse_args(["--seqfile", "input.fasta"])

        assert args.seq_file == "input.fasta"
        assert "--seqfile is deprecated; use --seq_file" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("yes", True),
            ("true", True),
            ("1", True),
            ("no", False),
            ("off", False),
            ("0", False),
        ],
    )
    def test_boolean_spellings_are_shared(self, text, expected):
        assert parse_bool(text) is expected

    def test_threads_zero_uses_detected_cpu_count(self, monkeypatch):
        monkeypatch.delattr("cdskit.cliutil.os.sched_getaffinity", raising=False)
        monkeypatch.delattr("cdskit.cliutil.os.process_cpu_count", raising=False)
        monkeypatch.setattr("cdskit.cliutil.os.cpu_count", lambda: 6)
        assert resolve_threads(0) == 6

    def test_threads_zero_respects_allocation_and_safety_limit(self, monkeypatch):
        monkeypatch.setattr("cdskit.cliutil.os.cpu_count", lambda: 128)
        monkeypatch.setattr(
            "cdskit.cliutil.os.sched_getaffinity",
            lambda pid: {16, 17, 18, 19},
            raising=False,
        )
        monkeypatch.setenv("CDSKIT_MAX_THREADS", "64")
        assert resolve_threads(0) == 4
        assert resolve_threads(8) == 8  # Explicit requests keep their meaning.
        monkeypatch.setenv("CDSKIT_MAX_THREADS", "2")
        assert resolve_threads(0) == 2

    def test_threads_zero_uses_process_count_without_affinity_api(self, monkeypatch):
        monkeypatch.delattr("cdskit.cliutil.os.sched_getaffinity", raising=False)
        monkeypatch.setattr("cdskit.cliutil.os.cpu_count", lambda: 128)
        monkeypatch.setattr(
            "cdskit.cliutil.os.process_cpu_count", lambda: 3, raising=False
        )
        assert resolve_threads(0) == 3

    def test_threads_above_safety_limit_are_rejected(self):
        with pytest.raises(ValueError, match="--threads should be <= 64"):
            resolve_threads(65)

    def test_public_cli_legacy_option_warns_and_still_runs(self, tmp_path, capsys):
        fasta = tmp_path / "input.fasta"
        fasta.write_text(">seq1\nATGAAA\n", encoding="utf-8")

        return_code = cli_main(["stats", "--seqfile", str(fasta)])
        captured = capsys.readouterr()

        assert return_code == 0
        assert "--seqfile is deprecated; use --seq_file" in captured.err
        assert "Number of sequences: 1" in captured.out

    def test_cli_rejects_input_output_collision_without_modifying_input(
        self, tmp_path, capsys
    ):
        fasta = tmp_path / "input.fasta"
        original = ">seq1\nATGAAA\n"
        fasta.write_text(original, encoding="utf-8")

        return_code = cli_main(
            [
                "translate",
                "--seq_file",
                str(fasta),
                "--out_file",
                str(fasta),
            ]
        )
        captured = capsys.readouterr()

        assert return_code == 1
        assert "Input and output paths should be different" in captured.err
        assert fasta.read_text(encoding="utf-8") == original

    def test_threaded_checkout_cli_runs_command_once(self, tmp_path, capsys):
        fasta = tmp_path / "input.fasta"
        output = tmp_path / "output.fasta"
        fasta.write_text(
            "".join(f">seq{i}\nATGAAA\n" for i in range(2_000)),
            encoding="utf-8",
        )

        return_code = cli_main(
            [
                "translate",
                "--seq_file",
                str(fasta),
                "--out_file",
                str(output),
                "--threads",
                "2",
            ]
        )
        captured = capsys.readouterr()

        assert return_code == 0
        assert captured.err.count("cdskit translate: started") == 1
        assert (
            sum(line.startswith(">") for line in output.read_text().splitlines())
            == 2_000
        )

    @pytest.mark.subprocess
    def test_checkout_script_wrapper_runs_directly(self):
        script_path = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "split_eukaryota_presets.py"
        )
        command = [str(script_path), "--help"]
        if os.name == "nt":
            command.insert(0, sys.executable)

        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

        assert "--eukaryota_tsv" in result.stdout
        assert result.stderr == ""

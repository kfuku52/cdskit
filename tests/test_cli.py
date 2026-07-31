"""
Tests for cdskit CLI (command line interface).

These tests verify that argparse help strings are properly formatted
and don't cause errors (e.g., Issue #10 with % characters).
"""

import pytest
import sys
import argparse
import subprocess
from pathlib import Path
from io import StringIO
from unittest.mock import patch

from cdskit.cliutil import CdskitArgumentParser, parse_bool, resolve_threads

class TestCLIHelpStrings:
    """Tests for CLI help string formatting.

    Issue #10: ValueError: badly formed help string
    This occurred in Python 3.14 because of unescaped % in help text.
    The help string for --replace_chars contained special characters including %
    which was interpreted as a format specifier.
    """

    def test_argparse_help_with_percent_character(self):
        """Test that help strings with % character are properly escaped.

        This tests the root cause of Issue #10: argparse help strings with
        %(default)s format specifiers can fail if the help text contains
        unescaped % characters.
        """
        # Test that a properly escaped help string works
        parser = argparse.ArgumentParser()
        # This should NOT raise ValueError
        parser.add_argument(
            '--test',
            default='',
            help='default=%(default)s: Special chars like !@#$%%^&* are OK'  # %% escapes %
        )

        # Get help without error
        help_output = StringIO()
        with patch('sys.stdout', help_output):
            try:
                parser.parse_args(['--help'])
            except SystemExit:
                pass  # --help causes sys.exit

        # Verify % was properly handled
        output = help_output.getvalue()
        assert "ValueError" not in output
        assert "badly formed" not in output

    def test_label_replace_chars_help_format(self):
        """Test the specific help string format from Issue #10.

        The --replace_chars argument has a help string containing special characters
        including %. This must be properly escaped or it causes errors in Python 3.14+.
        """
        cli_path = Path(__file__).parent.parent / "cdskit" / "cdskit"
        result = subprocess.run(
            [sys.executable, str(cli_path), "label", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        assert "--replace_chars" in result.stdout
        assert "!@#$%^&*+=/?<>|--_" in result.stdout
        assert result.stderr == ""

    def test_argparse_special_chars_in_metavar(self):
        """Test that special characters in metavar don't cause issues."""
        parser = argparse.ArgumentParser()
        # Metavar with special characters (like the one in Issue #10)
        parser.add_argument(
            '--test',
            metavar='FROM1FROM2...--TO',
            default='',
            help='default=%(default)s: Replace characters'
        )

        # Should not raise any errors
        help_output = StringIO()
        with patch('sys.stdout', help_output):
            try:
                parser.parse_args(['--help'])
            except SystemExit:
                pass


class TestCLIModuleImport:
    """Tests that CLI-related modules can be imported without errors."""

    def test_import_all_command_modules(self):
        """Test that all command modules can be imported."""
        modules = [
            'cdskit.accession2fasta',
            'cdskit.aggregate',
            'cdskit.backalign',
            'cdskit.backtrim',
            'cdskit.codonstats',
            'cdskit.codonutil',
            'cdskit.deeploc_benchmark',
            'cdskit.degeneracy',
            'cdskit.filter',
            'cdskit.hammer',
            'cdskit.intersection',
            'cdskit.label',
            'cdskit.longestcds',
            'cdskit.longestorf',
            'cdskit.localize',
            'cdskit.localize_bilstm',
            'cdskit.localize_learn',
            'cdskit.localize_model',
            'cdskit.localize_models',
            'cdskit.localize_multilabel_cnn',
            'cdskit.maxalign',
            'cdskit.mask',
            'cdskit.pad',
            'cdskit.parsegb',
            'cdskit.plot',
            'cdskit.printseq',
            'cdskit.rmseq',
            'cdskit.split',
            'cdskit.stats',
            'cdskit.translate',
            'cdskit.trimcodon',
            'cdskit.gapjust',
            'cdskit.util',
            'cdskit.validate',
        ]

        for module_name in modules:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    def test_import_util_functions(self):
        """Test that commonly used utility functions are available."""
        from cdskit.util import read_seqs, write_seqs, stop_if_not_multiple_of_three

        # These should be callable
        assert callable(read_seqs)
        assert callable(write_seqs)
        assert callable(stop_if_not_multiple_of_three)

    def test_root_version_option(self):
        """Test that cdskit --version works without a subcommand."""
        from cdskit import __version__

        cli_path = Path(__file__).parent.parent / "cdskit" / "cdskit"
        result = subprocess.run(
            [sys.executable, str(cli_path), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )

        assert result.stdout.strip() == f"cdskit version {__version__}"
        assert result.stderr == ""


class TestCLIEdgeCases:
    """Tests for edge cases in CLI behavior."""

    def test_empty_default_value_with_format(self):
        """Test that empty default values work with %(default)s format."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--empty',
            default='',
            help='default=%(default)s: An empty default'
        )

        # Should handle empty default
        help_output = StringIO()
        with patch('sys.stdout', help_output):
            try:
                parser.parse_args(['--help'])
            except SystemExit:
                pass

        output = help_output.getvalue()
        assert 'default=' in output

    def test_numeric_default_with_format(self):
        """Test that numeric default values work with %(default)s format."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--number',
            type=int,
            default=42,
            help='default=%(default)s: A numeric default'
        )

        # Should handle numeric default
        help_output = StringIO()
        with patch('sys.stdout', help_output):
            try:
                parser.parse_args(['--help'])
            except SystemExit:
                pass

        output = help_output.getvalue()
        assert '42' in output


class TestCLIConsistency:
    def test_automatic_help_describes_common_argument_roles(self):
        parser = CdskitArgumentParser()
        parser.add_argument('--training_tsv')
        parser.add_argument('--out_json')
        parser.add_argument('--threshold_grid')
        parser.add_argument('--verbose', action='store_true')

        help_text = parser.format_help()

        assert 'Path to the training TSV file.' in help_text
        assert 'Path for the output result JSON file.' in help_text
        assert 'Decision threshold or candidate thresholds' in help_text
        assert 'Enable detailed progress output.' in help_text
        assert 'Set training tsv.' not in help_text

    def test_deprecated_long_option_is_accepted_with_warning(self, capsys):
        parser = CdskitArgumentParser()
        parser.add_argument('--seq_file')
        parser.add_deprecated_alias('--seqfile', '--seq_file')

        args = parser.parse_args(['--seqfile', 'input.fasta'])

        assert args.seq_file == 'input.fasta'
        assert '--seqfile is deprecated; use --seq_file' in capsys.readouterr().err

    @pytest.mark.parametrize(
        ('text', 'expected'),
        [('yes', True), ('true', True), ('1', True), ('no', False), ('off', False), ('0', False)],
    )
    def test_boolean_spellings_are_shared(self, text, expected):
        assert parse_bool(text) is expected

    def test_threads_zero_uses_detected_cpu_count(self, monkeypatch):
        monkeypatch.setattr('cdskit.cliutil.os.cpu_count', lambda: 6)
        assert resolve_threads(0) == 6

    def test_threads_above_safety_limit_are_rejected(self):
        with pytest.raises(ValueError, match='--threads should be <= 64'):
            resolve_threads(65)

    def test_public_cli_legacy_option_warns_and_still_runs(self, tmp_path):
        cli_path = Path(__file__).parent.parent / 'cdskit' / 'cdskit'
        fasta = tmp_path / 'input.fasta'
        fasta.write_text('>seq1\nATGAAA\n', encoding='utf-8')

        result = subprocess.run(
            [sys.executable, str(cli_path), 'stats', '--seqfile', str(fasta)],
            check=True,
            capture_output=True,
            text=True,
        )

        assert '--seqfile is deprecated; use --seq_file' in result.stderr
        assert 'Number of sequences: 1' in result.stdout

    def test_cli_rejects_input_output_collision_without_modifying_input(self, tmp_path):
        cli_path = Path(__file__).parent.parent / 'cdskit' / 'cdskit'
        fasta = tmp_path / 'input.fasta'
        original = '>seq1\nATGAAA\n'
        fasta.write_text(original, encoding='utf-8')

        result = subprocess.run(
            [
                sys.executable,
                str(cli_path),
                'translate',
                '--seq_file',
                str(fasta),
                '--out_file',
                str(fasta),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert 'Input and output paths should be different' in result.stderr
        assert fasta.read_text(encoding='utf-8') == original

    def test_threaded_checkout_cli_runs_command_once(self, tmp_path):
        cli_path = Path(__file__).parent.parent / 'cdskit' / 'cdskit'
        fasta = tmp_path / 'input.fasta'
        output = tmp_path / 'output.fasta'
        fasta.write_text(
            ''.join(f'>seq{i}\nATGAAA\n' for i in range(2_000)),
            encoding='utf-8',
        )

        result = subprocess.run(
            [
                sys.executable,
                str(cli_path),
                'translate',
                '--seq_file',
                str(fasta),
                '--out_file',
                str(output),
                '--threads',
                '2',
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.stderr.count('cdskit translate: started') == 1
        assert sum(line.startswith('>') for line in output.read_text().splitlines()) == 2_000

    def test_checkout_script_wrapper_runs_directly(self):
        script_path = (
            Path(__file__).parent.parent / 'scripts' / 'split_eukaryota_presets.py'
        )

        result = subprocess.run(
            [str(script_path), '--help'],
            check=True,
            capture_output=True,
            text=True,
        )

        assert '--eukaryota_tsv' in result.stdout
        assert result.stderr == ''

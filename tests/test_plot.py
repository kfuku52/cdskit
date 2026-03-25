"""
Tests for cdskit plot command.
"""

from pathlib import Path

import Bio.SeqIO
import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.plot import plot_main


class TestPlotMain:
    """Tests for plot_main function."""

    def test_plot_writes_svg_file_with_expected_sections(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.svg"

        records = [
            SeqRecord(Seq("ATGAAATGA"), id="seq1", description=""),
            SeqRecord(Seq("ATGNNNCCC"), id="seq2", description=""),
            SeqRecord(Seq("ATG---CCC"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            mode="summary",
            plotformat="svg",
            width=1100,
            height=680,
            min_occupancy=0.8,
            max_ambiguous_fraction=0.2,
            drop_stop_codon=False,
            title="My Plot",
            top_n=3,
        )

        svg = plot_main(args)
        assert isinstance(svg, str)
        assert "<svg" in svg

        txt = output_path.read_text()
        assert "<svg" in txt
        assert "My Plot" in txt
        assert "occupancy threshold" in txt
        assert "ambiguity threshold" in txt
        assert "keep/remove strip" in txt
        assert "Top ambiguous sequences" in txt
        assert "seq1" in txt
        assert "seq2" in txt
        assert "seq3" in txt
        assert "stop" in txt
        assert "occupancy" in txt
        assert "ambiguity" in txt

    def test_plot_writes_svg_to_stdout(self, temp_dir, mock_args, capsys):
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATGAAATGA"), id="seq1", description=""),
            SeqRecord(Seq("ATGNNNCCC"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile="-",
            mode="summary",
            plotformat="svg",
            width=900,
            height=620,
            min_occupancy=0.5,
            max_ambiguous_fraction=1.0,
            drop_stop_codon=False,
            title="Stdout Plot",
            top_n=2,
        )

        svg = plot_main(args)
        captured = capsys.readouterr()
        assert isinstance(svg, str)
        assert "<svg" in captured.out
        assert "<svg" in svg
        assert "Stdout Plot" in captured.out

    def test_plot_defaults_to_pdf_file_output(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.pdf"

        records = [
            SeqRecord(Seq("ATGAAATGA"), id="seq1", description=""),
            SeqRecord(Seq("ATGNNNCCC"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            mode="summary",
            width=900,
            height=620,
            min_occupancy=0.5,
            max_ambiguous_fraction=1.0,
            drop_stop_codon=False,
            title="PDF Plot",
            top_n=2,
        )

        payload = plot_main(args)
        pdf_bytes = output_path.read_bytes()
        assert isinstance(payload, bytes)
        assert payload.startswith(b"%PDF")
        assert pdf_bytes.startswith(b"%PDF")

    def test_plot_rejects_unaligned_input(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAATGA"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(temp_dir / "output.svg"),
            mode="summary",
            width=900,
            height=620,
            min_occupancy=0.5,
            max_ambiguous_fraction=1.0,
            drop_stop_codon=False,
            title="Broken",
            top_n=1,
        )

        with pytest.raises(Exception) as exc_info:
            plot_main(args)
        assert "correctly aligned" in str(exc_info.value)

    def test_plot_rejects_invalid_fraction(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"

        records = [SeqRecord(Seq("ATGAAATGA"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(temp_dir / "output.svg"),
            mode="summary",
            width=900,
            height=620,
            min_occupancy=1.2,
            max_ambiguous_fraction=1.0,
            drop_stop_codon=False,
            title="Broken",
            top_n=1,
        )

        with pytest.raises(Exception) as exc_info:
            plot_main(args)
        assert "between 0 and 1 inclusive" in str(exc_info.value)

    def test_plot_map_mode_writes_svg_to_stdout(self, temp_dir, mock_args, capsys):
        input_path = temp_dir / "input.fasta"
        records = [
            SeqRecord(Seq("ATGAAATAA---"), id="beta", description=""),
            SeqRecord(Seq("ATGAANTAA---"), id="alpha", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile="-",
            mode="map",
            plotformat="svg",
            width=1000,
            height=420,
            row_height=26,
            label_width=120,
            title="Example map",
            top_n=2,
            min_occupancy=0.75,
            max_ambiguous_fraction=0.25,
            drop_stop_codon=False,
        )

        svg = plot_main(args)
        captured = capsys.readouterr()
        assert isinstance(svg, str)
        assert "<svg" in svg
        assert "<svg" in captured.out
        assert "Example map" in captured.out
        assert "complete" in captured.out
        assert "ambiguous" in captured.out
        assert "stop" in captured.out
        assert "missing" in captured.out
        assert "keep" in captured.out
        assert "remove" in captured.out
        assert "Top ambiguous codon counts" in captured.out

    def test_plot_map_mode_writes_file_and_preserves_order(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "out.svg"
        records = [
            SeqRecord(Seq("ATGAAATAA---"), id="zeta", description=""),
            SeqRecord(Seq("ATGAANTAA---"), id="alpha", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            mode="map",
            plotformat="svg",
            width=900,
            height=360,
            row_height=24,
            label_width=140,
            title="File output",
            top_n=0,
            min_occupancy=0.5,
            max_ambiguous_fraction=1.0,
            drop_stop_codon=False,
        )

        plot_main(args)
        svg = output_path.read_text()
        assert "<svg" in svg
        assert "zeta" in svg
        assert "alpha" in svg

    def test_plot_msa_mode_writes_svg_to_stdout(self, temp_dir, mock_args, capsys):
        input_path = temp_dir / "input.fasta"
        records = [
            SeqRecord(Seq("ATGAAATGA---CCCGGG"), id="beta", description=""),
            SeqRecord(Seq("ATGAANTGA---CCCGGG"), id="alpha", description=""),
            SeqRecord(Seq("ATG---TAA---CCCNNN"), id="gamma", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile="-",
            mode="msa",
            plotformat="svg",
            width=1200,
            height=500,
            row_height=24,
            label_width=150,
            wrap=12,
            title="Example msa",
            top_n=2,
        )

        svg = plot_main(args)
        captured = capsys.readouterr()
        assert isinstance(svg, str)
        assert "<svg" in svg
        assert "<svg" in captured.out
        assert "Example msa" in captured.out
        assert "AA site" in captured.out
        assert "NT site" in captured.out
        assert "Consensus" in captured.out
        assert "AA frequency" in captured.out
        assert "beta" in captured.out
        assert "alpha" in captured.out
        assert "gamma" in captured.out
        assert "Top ambiguous sequences" not in captured.out

    def test_plot_msa_mode_writes_svg_file(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "msa.svg"
        records = [
            SeqRecord(Seq("ATGAAATGA---CCCGGG"), id="zeta", description=""),
            SeqRecord(Seq("ATGAANTGA---CCCGGG"), id="alpha", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            mode="msa",
            plotformat="svg",
            width=1000,
            height=420,
            row_height=22,
            label_width=140,
            wrap=9,
            title="MSA file",
            top_n=0,
        )

        plot_main(args)
        svg = output_path.read_text()
        assert "<svg" in svg
        assert "MSA file" in svg
        assert "AA site" in svg
        assert "NT site" in svg
        assert "Consensus" in svg
        assert "AA frequency" in svg
        assert "zeta" in svg
        assert "alpha" in svg
        assert "Top ambiguous sequences" not in svg

    def test_plot_msa_mode_rejects_wrap_not_multiple_of_three(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        records = [
            SeqRecord(Seq("ATGAAATGA---"), id="seq1", description=""),
            SeqRecord(Seq("ATGAANTGA---"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile="-",
            mode="msa",
            plotformat="svg",
            wrap=10,
        )

        with pytest.raises(Exception) as excinfo:
            plot_main(args)
        assert "multiple of three" in str(excinfo.value)

    @pytest.mark.parametrize(
        "records, message",
        [
            ([SeqRecord(Seq("ATGAAAT"), id="short", description="")], "multiple of three"),
            (
                [
                    SeqRecord(Seq("ATGAAA"), id="seq1", description=""),
                    SeqRecord(Seq("ATGAAATGA"), id="seq2", description=""),
                ],
                "identical",
            ),
            ([SeqRecord(Seq("ATGAAAUAA"), id="rna", description="")], "DNA-only input"),
        ],
    )
    def test_plot_map_mode_validates_input(self, temp_dir, mock_args, records, message):
        input_path = temp_dir / "input.fasta"
        Bio.SeqIO.write(records, str(input_path), "fasta")
        args = mock_args(seqfile=str(input_path), outfile="-", mode="map", plotformat="svg")

        with pytest.raises(Exception) as excinfo:
            plot_main(args)
        assert message in str(excinfo.value)

    def test_plot_rejects_invalid_mode(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        records = [SeqRecord(Seq("ATGAAATGA"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile="-",
            mode="weird",
        )

        with pytest.raises(Exception) as exc_info:
            plot_main(args)
        assert "Invalid --mode" in str(exc_info.value)

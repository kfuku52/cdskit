"""
Tests for cdskit degeneracy command.
"""

import json
from pathlib import Path

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.degeneracy import degeneracy_main


class TestDegeneracyMain:
    """Tests for degeneracy_main function."""

    def test_degeneracy_writes_expected_fold_outputs(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        prefix = temp_dir / "deg"

        records = [
            SeqRecord(Seq("GCTTTTATA"), id="seq1", description=""),
            SeqRecord(Seq("GCCTTCATC"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            prefix=str(prefix),
            fold=[0, 2, 3, 4],
            report='',
        )

        degeneracy_main(args)

        zero = list(Bio.SeqIO.parse(str(temp_dir / "deg_0fold_positions.fasta"), "fasta"))
        two = list(Bio.SeqIO.parse(str(temp_dir / "deg_2fold_positions.fasta"), "fasta"))
        three = list(Bio.SeqIO.parse(str(temp_dir / "deg_3fold_positions.fasta"), "fasta"))
        four = list(Bio.SeqIO.parse(str(temp_dir / "deg_4fold_positions.fasta"), "fasta"))

        assert [str(record.seq) for record in zero] == ["GCTTAT", "GCTTAT"]
        assert [str(record.seq) for record in two] == ["T", "C"]
        assert [str(record.seq) for record in three] == ["A", "C"]
        assert [str(record.seq) for record in four] == ["T", "C"]

    def test_degeneracy_excludes_conflicting_sites(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        prefix = temp_dir / "deg"

        records = [
            SeqRecord(Seq("GCT"), id="seq1", description=""),
            SeqRecord(Seq("GAT"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            prefix=str(prefix),
            fold=[0, 2, 4],
            report='',
        )

        degeneracy_main(args)

        zero = list(Bio.SeqIO.parse(str(temp_dir / "deg_0fold_positions.fasta"), "fasta"))
        two = list(Bio.SeqIO.parse(str(temp_dir / "deg_2fold_positions.fasta"), "fasta"))
        four = list(Bio.SeqIO.parse(str(temp_dir / "deg_4fold_positions.fasta"), "fasta"))

        assert [str(record.seq) for record in zero] == ["GC", "GA"]
        assert [str(record.seq) for record in two] == ["", ""]
        assert [str(record.seq) for record in four] == ["", ""]

    def test_degeneracy_ignores_missing_codons_and_writes_report(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        prefix = temp_dir / "deg"
        report_path = temp_dir / "report.json"

        records = [
            SeqRecord(Seq("GCT"), id="seq1", description=""),
            SeqRecord(Seq("---"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            prefix=str(prefix),
            fold=[4],
            report=str(report_path),
        )

        degeneracy_main(args)

        four = list(Bio.SeqIO.parse(str(temp_dir / "deg_4fold_positions.fasta"), "fasta"))
        report = json.loads(report_path.read_text())
        assert [str(record.seq) for record in four] == ["T", "-"]
        assert report["counts_by_fold"]["4"] == 1

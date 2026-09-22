"""Codon-position output and the supported output-prefix precedence rules."""

import io

import Bio.SeqIO
import pytest

from cdskit.split import split_main


@pytest.mark.parametrize("prefix_source", ["explicit", "outfile", "infile", "stdin"])
def test_split_outputs_preserve_positions_gaps_and_record_order(
    tmp_path, mock_args, write_fasta, monkeypatch, prefix_source
):
    source = write_fasta(
        tmp_path / "input.fasta",
        [("zebra", "ATG---CCCGGG"), ("apple", "ATGAAACCC---")],
    )
    options = {"seqfile": str(source), "prefix": "INFILE"}
    if prefix_source == "explicit":
        prefix = tmp_path / "explicit"
        options.update(prefix=str(prefix), outfile=str(tmp_path / "ignored"))
    elif prefix_source == "outfile":
        prefix = tmp_path / "fallback"
        options["outfile"] = str(prefix)
    elif prefix_source == "infile":
        prefix = source
    else:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.stdin", io.StringIO(source.read_text()))
        options["seqfile"] = "-"
        prefix = tmp_path / "stdin"

    split_main(mock_args(**options))

    for position, expected in (
        ("1st", [("zebra", "A-CG"), ("apple", "AAC-")]),
        ("2nd", [("zebra", "T-CG"), ("apple", "TAC-")]),
        ("3rd", [("zebra", "G-CG"), ("apple", "GAC-")]),
    ):
        output = prefix.with_name(prefix.name + f"_{position}_codon_positions.fasta")
        assert [
            (r.id, str(r.seq)) for r in Bio.SeqIO.parse(output, "fasta")
        ] == expected

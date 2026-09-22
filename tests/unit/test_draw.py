from types import SimpleNamespace

import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.draw import (
    build_svg,
    draw_main,
    nonnegative_int_arg,
    positive_int_arg,
    summarize_draw,
)


def _draw_args(**overrides):
    values = {
        "width": 1200,
        "height": 300,
        "row_height": 24,
        "label_width": 180,
        "title": "Unsafe <title> & summary",
        "top_n": 2,
        "codontable": 1,
        "min_clean_fraction": 0.5,
        "outfile": "-",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _records():
    return [
        SeqRecord(Seq("ATGNNN---TAA"), id="ambiguous<&"),
        SeqRecord(Seq("ATGAAA---TGG"), id="complete"),
    ]


def test_build_svg_escapes_labels_and_renders_all_sections():
    records = _records()
    args = _draw_args()
    summary = summarize_draw(
        records=records,
        codontable=args.codontable,
        min_clean_fraction=args.min_clean_fraction,
    )

    svg = build_svg(records=records, args=args, summary=summary)

    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert svg.endswith("</svg>")
    assert "Unsafe &lt;title&gt; &amp; summary" in svg
    assert 'data-seq="ambiguous&lt;&amp;"' in svg
    assert "tile complete" in svg
    assert "tile missing" in svg
    assert "tile stop" in svg
    assert "Top ambiguous codon counts" in svg
    assert "Sequences: 2 | Codon sites: 4" in svg


def test_build_svg_handles_empty_input():
    args = _draw_args(width=320, top_n=0, title="")
    summary = summarize_draw(
        records=[],
        codontable=args.codontable,
        min_clean_fraction=args.min_clean_fraction,
    )

    svg = build_svg(records=[], args=args, summary=summary)

    assert "No sequences to draw" in svg
    assert "Sequences: 0 | Codon sites: 0" in svg


def test_draw_main_writes_atomic_svg_output(tmp_path, write_fasta):
    input_path = write_fasta(
        tmp_path / "input.fasta",
        [("ambiguous", "ATGNNNTAA"), ("complete", "ATGAAATGG")],
    )
    output_path = tmp_path / "output.svg"
    args = _draw_args(
        seqfile=str(input_path),
        inseqformat="fasta",
        outfile=str(output_path),
    )

    draw_main(args)

    svg = output_path.read_text(encoding="utf-8")
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>\n")
    assert "Sequences: 2 | Codon sites: 3" in svg


@pytest.mark.parametrize(
    ("worker", "value", "message"),
    [
        (positive_int_arg, 0, "greater than zero"),
        (positive_int_arg, "bad", "must be an integer"),
        (nonnegative_int_arg, -1, "greater than or equal to zero"),
        (nonnegative_int_arg, "bad", "must be an integer"),
    ],
)
def test_draw_integer_argument_validation(worker, value, message):
    with pytest.raises(ValueError, match=message):
        worker("--value", value, 1)

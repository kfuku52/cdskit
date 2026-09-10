"""Codon preservation and failure semantics of explicit and strict mapping."""

import json

import Bio.SeqIO
import pytest

from cdskit.backtrim import backtrim_main
from cdskit.cli import main


def inputs(tmp_path, write_fasta, mock_args, cds=None, aa=None, **kwargs):
    cds = cds if cds is not None else [("s1", "GCTGCC"), ("s2", "GCTGCT")]
    aa = aa if aa is not None else [("s2", "A"), ("s1", "A")]
    return mock_args(
        seqfile=str(write_fasta(tmp_path / "cds.fa", cds)),
        trimmed_aa_aln=str(write_fasta(tmp_path / "aa.fa", aa)),
        outfile=str(tmp_path / "out.fa"),
        mapping_report=str(tmp_path / "map.json"),
        **kwargs,
    )


@pytest.mark.parametrize("site,expected", [(0, ["GCT", "GCT"]), (1, ["GCC", "GCT"])])
@pytest.mark.parametrize("threads", [1, 2])
def test_explicit_sites_preserve_synonymous_variation(
    tmp_path, write_fasta, mock_args, site, expected, threads
):
    path = tmp_path / "sites"
    path.write_text(str(site))
    args = inputs(
        tmp_path,
        write_fasta,
        mock_args,
        kept_sites=str(path),
        kept_sites_format="indices0",
        threads=threads,
    )
    backtrim_main(args)
    assert [
        str(record.seq) for record in Bio.SeqIO.parse(args.outfile, "fasta")
    ] == expected
    report = json.loads((tmp_path / "map.json").read_text())
    assert report["source"] == "provided"
    assert report["inference_status"] == "ambiguous"
    assert report["output_complete"]
    assert report["selected_sites"] == [site]
    assert len(report["kept_sites_sha256"]) == 64


@pytest.mark.parametrize(
    "cds,aa,status",
    [
        ("GCTGCC", "A", "ambiguous"),
        ("GCTCCT", "PA", "unmatched"),
    ],
)
def test_strict_failure_preserves_output(
    tmp_path, write_fasta, mock_args, cds, aa, status, capsys
):
    args = inputs(
        tmp_path,
        write_fasta,
        mock_args,
        [("s", cds)],
        [("s", aa)],
        mapping_policy="strict",
    )
    (tmp_path / "out.fa").write_text("existing output")
    with pytest.raises(ValueError, match=status):
        backtrim_main(args)
    assert (tmp_path / "out.fa").read_text() == "existing output"
    report = json.loads((tmp_path / "map.json").read_text())
    assert report["status"] == "failed"
    assert report["selected_sites"] == []
    assert report["inference_status"] == status
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "cds,aa,expected",
    [
        ("GCTGCC", "AA", "GCTGCC"),
        ("GCTCCTGCC", "AP", "GCTCCT"),
        ("atg---aaa", "m.k", "atg---aaa"),
        ("ATG???CCC", "MXP", "ATG???CCC"),
        ("GCTGCC", "", ""),
        ("", "", ""),
    ],
)
def test_strict_unique_and_empty(
    tmp_path, write_fasta, mock_args, cds, aa, expected, capsys
):
    args = inputs(
        tmp_path,
        write_fasta,
        mock_args,
        [("s", cds)],
        [("s", aa)],
        mapping_policy="strict",
    )
    backtrim_main(args)
    assert str(Bio.SeqIO.read(args.outfile, "fasta").seq) == expected
    assert "multiple matches" not in capsys.readouterr().err


@pytest.mark.parametrize("content", ["0\n1", "2", "1\n0", "0"])
def test_invalid_explicit_mapping_never_falls_back(
    tmp_path, write_fasta, mock_args, content
):
    path = tmp_path / "sites"
    path.write_text(content)
    args = inputs(
        tmp_path,
        write_fasta,
        mock_args,
        [("s", "GCTCCT")],
        [("s", "P")],
        kept_sites=str(path),
        kept_sites_format="indices0",
    )
    with pytest.raises(ValueError):
        backtrim_main(args)
    assert not (tmp_path / "out.fa").exists()
    assert json.loads((tmp_path / "map.json").read_text())["status"] == "failed"


def test_legacy_partial_report(tmp_path, write_fasta, mock_args):
    args = inputs(tmp_path, write_fasta, mock_args, [("s", "GCTCCT")], [("s", "PA")])
    backtrim_main(args)
    report = json.loads((tmp_path / "map.json").read_text())
    assert report["source"] == "legacy"
    assert report["unmatched_target_sites"] == [1]
    assert report["selected_sites"] == [1]
    assert not report["output_complete"]


def test_cli_strict_exit_and_explicit_success(tmp_path, write_fasta, mock_args):
    args = inputs(tmp_path, write_fasta, mock_args)
    command = [
        "backtrim",
        "--seq_file",
        args.seqfile,
        "--trimmed_aa_aln",
        args.trimmed_aa_aln,
        "--out_file",
        args.outfile,
    ]
    assert main([*command, "--mapping_policy", "strict"]) == 1
    assert not (tmp_path / "out.fa").exists()
    path = tmp_path / "sites"
    path.write_text("2\n")
    assert (
        main([*command, "--kept_sites", str(path), "--kept_sites_format", "indices1"])
        == 0
    )
    assert str(next(Bio.SeqIO.parse(args.outfile, "fasta")).seq) == "GCC"
    assert main([*command, "--mapping_report", args.seqfile]) == 1


@pytest.mark.parametrize(
    "options",
    [
        dict(kept_sites="sites"),
        dict(kept_sites_format="indices0"),
        dict(mapping_policy="bad"),
        dict(mapping_report="-"),
    ],
)
def test_option_validation(tmp_path, write_fasta, mock_args, options):
    args = inputs(tmp_path, write_fasta, mock_args)
    for name, value in options.items():
        setattr(args, name, value)
    with pytest.raises(ValueError):
        backtrim_main(args)


def test_empty_inputs_report(tmp_path, write_fasta, mock_args):
    args = inputs(tmp_path, write_fasta, mock_args, [], [], mapping_policy="strict")
    backtrim_main(args)
    assert (tmp_path / "out.fa").read_text() == ""
    assert (
        json.loads((tmp_path / "map.json").read_text())["inference_status"] == "unique"
    )


@pytest.mark.parametrize(
    "prefix,format_name",
    [("clipkit", "clipkit-log"), ("trimal", "trimal-colnumbering")],
)
def test_real_trimmer_outputs(data_dir, tmp_path, mock_args, prefix, format_name):
    fixture = data_dir / "backtrim_mapping"
    map_name = "clipkit.aa.fa.log" if prefix == "clipkit" else "trimal.columns"
    args = mock_args(
        seqfile=str(fixture / "source.cds.fa"),
        trimmed_aa_aln=str(fixture / f"{prefix}.aa.fa"),
        outfile=str(tmp_path / "out.fa"),
        kept_sites=str(fixture / map_name),
        kept_sites_format=format_name,
    )
    backtrim_main(args)
    # Independent known mask, not an AA-translation-only comparison.
    original = list(Bio.SeqIO.parse(args.seqfile, "fasta"))
    output = list(Bio.SeqIO.parse(args.outfile, "fasta"))
    assert [(record.id, str(record.seq)) for record in output] == [
        (record.id, str(record.seq)[3:6]) for record in original
    ]


@pytest.mark.parametrize(
    "table,cds,aa",
    [
        (27, "ATGTGAAAA", "MWK"),
        (28, "ATGTAATAGTGAAAA", "MQQWK"),
        (31, "ATGTAATAGAAA", "MEEK"),
        (1, "ATGTARAAA", "M*K"),
    ],
)
def test_strict_translation_semantics(tmp_path, write_fasta, mock_args, table, cds, aa):
    args = inputs(
        tmp_path,
        write_fasta,
        mock_args,
        [("s", cds)],
        [("s", aa)],
        mapping_policy="strict",
        codontable=table,
    )
    backtrim_main(args)
    assert str(Bio.SeqIO.read(args.outfile, "fasta").seq) == cds


def test_strict_failure_writes_no_stdout(tmp_path, write_fasta, mock_args, capsys):
    args = inputs(tmp_path, write_fasta, mock_args, mapping_policy="strict")
    args.outfile = "-"
    with pytest.raises(ValueError, match="ambiguous"):
        backtrim_main(args)
    assert capsys.readouterr().out == ""


def test_successful_stdout_and_report(tmp_path, write_fasta, mock_args, capsys):
    args = inputs(
        tmp_path,
        write_fasta,
        mock_args,
        [("s", "GCTGCC")],
        [("s", "AA")],
        mapping_policy="strict",
    )
    args.outfile = "-"
    backtrim_main(args)
    assert capsys.readouterr().out == ">s\nGCTGCC\n"
    assert json.loads((tmp_path / "map.json").read_text())["output_complete"]


def test_success_report_failure_preserves_sequence_file(
    tmp_path, write_fasta, mock_args, monkeypatch
):
    import cdskit.backtrim as module

    args = inputs(
        tmp_path,
        write_fasta,
        mock_args,
        [("s", "GCTGCC")],
        [("s", "AA")],
        mapping_policy="strict",
    )
    (tmp_path / "out.fa").write_text("previous alignment")
    (tmp_path / "map.json").write_text("previous report")

    def fail_report(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(module, "atomic_write_json", fail_report)
    with pytest.raises(OSError, match="disk full"):
        backtrim_main(args)
    assert (tmp_path / "out.fa").read_text() == "previous alignment"
    assert (tmp_path / "map.json").read_text() == "previous report"

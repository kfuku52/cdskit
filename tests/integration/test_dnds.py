import csv
from types import SimpleNamespace

import pytest

from cdskit.cli import main
from cdskit import dnds
from cdskit.dnds import dnds_main, estimate_pairs


def write_pairs(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("pair_id", "sequence_1", "sequence_2"))
        writer.writerows(rows)


def test_dnds_cli_tsv_contract_and_stdout_stderr(tmp_path, capsys):
    pairs = tmp_path / "pairs.tsv"
    sequence = "GCTGCCGGTTTTATGACG" * 20
    write_pairs(pairs, [("same", sequence, sequence), ("missing", "NNN", "NNN")])
    assert main(["dnds", "--pairs_file", str(pairs), "--out_file", "-"]) == 0
    output = capsys.readouterr()
    rows = list(csv.DictReader(output.out.splitlines(), delimiter="\t"))
    assert len(rows) == 2
    assert all(
        row["schema_version"] == row["codon_semantics_version"] == "2" for row in rows
    )
    assert rows[0]["dS"] == "0.0"
    assert rows[1]["dS"] == ""
    assert rows[1]["status"] == "no_aligned_sense_codons"
    assert "cdskit dnds: started" in output.err


@pytest.mark.parametrize(
    "rows",
    [
        [("duplicate", "GCT", "GCT"), ("duplicate", "GCT", "GCT")],
        [("", "GCT", "GCT")],
        [("bad", "GCTG", "GCT")],
    ],
)
def test_dnds_errors_preserve_existing_output_atomically(tmp_path, rows, capsys):
    pairs, output = tmp_path / "pairs.tsv", tmp_path / "result.tsv"
    write_pairs(pairs, rows)
    output.write_text("preserve this\n")
    assert main(["dnds", "--pairs_file", str(pairs), "-o", str(output)]) == 1
    assert output.read_text() == "preserve this\n"
    assert capsys.readouterr().err


def test_dnds_api_matches_headered_report(tmp_path):
    pairs, output = tmp_path / "pairs.tsv", tmp_path / "result.tsv"
    sequence = "GCTGCCGGTTTTATGACG" * 20
    write_pairs(pairs, [("a", sequence, sequence)])
    assert main(["dnds", "--pairs_file", str(pairs), "-o", str(output)]) == 0
    row = next(csv.DictReader(output.read_text().splitlines(), delimiter="\t"))
    expected = estimate_pairs([(sequence, sequence)])[0]
    assert float(row["S"]) == expected["S"]
    assert row["method"] == "YN00_weighting0_F3x4"


def test_dnds_input_output_overlap_is_rejected(tmp_path, capsys):
    pairs = tmp_path / "pairs.tsv"
    write_pairs(pairs, [("a", "GCTGCC" * 30, "GCTGCC" * 30)])
    original = pairs.read_bytes()
    assert main(["dnds", "--pairs_file", str(pairs), "-o", str(pairs)]) == 1
    assert pairs.read_bytes() == original
    assert capsys.readouterr().err


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_direct_dnds_handler_protects_inputs_without_cli_preflight(tmp_path, alias):
    pairs, output = tmp_path / "pairs.tsv", tmp_path / "result.tsv"
    write_pairs(pairs, [("a", "GCTGCC", "GCTGCC")])
    original = pairs.read_bytes()
    try:
        if alias == "same":
            output = pairs
        elif alias == "symlink":
            output.symlink_to(pairs)
        else:
            output.hardlink_to(pairs)
    except OSError:
        pytest.skip("Filesystem links unavailable")
    with pytest.raises(ValueError, match="Input and output paths"):
        dnds_main(
            SimpleNamespace(
                pairs_file=str(pairs), outfile=str(output), codontable=1, threads=1
            )
        )
    assert pairs.read_bytes() == original
    assert output.read_bytes() == original


def test_direct_dnds_handler_rejects_directory_outputs(tmp_path):
    pairs, output = tmp_path / "pairs.tsv", tmp_path / "output"
    write_pairs(pairs, [("a", "GCTGCC", "GCTGCC")])
    output.mkdir()
    with pytest.raises(ValueError):
        dnds_main(
            SimpleNamespace(
                pairs_file=str(pairs), outfile=str(output), codontable=1, threads=1
            )
        )
    assert output.is_dir()


def test_direct_dnds_handler_rechecks_output_alias_before_writing(
    tmp_path, monkeypatch
):
    pairs, output = tmp_path / "pairs.tsv", tmp_path / "result.tsv"
    write_pairs(pairs, [("a", "GCTGCC", "GCTGCC")])
    original = pairs.read_bytes()
    estimate = dnds.estimate_pairs

    def changed_destination(*args):
        result = estimate(*args)
        try:
            output.hardlink_to(pairs)
        except OSError:
            pytest.skip("Filesystem hardlinks unavailable")
        return result

    monkeypatch.setattr(dnds, "estimate_pairs", changed_destination)
    with pytest.raises(ValueError, match="Input and output paths"):
        dnds_main(
            SimpleNamespace(
                pairs_file=str(pairs), outfile=str(output), codontable=1, threads=1
            )
        )
    assert pairs.read_bytes() == original


def test_dnds_cli_audits_ambiguous_internal_and_terminal_stops(tmp_path, capsys):
    pairs = tmp_path / "pairs.tsv"
    sequence = "GCTGCCGGTTTTATGACG" * 20
    write_pairs(
        pairs,
        [
            ("internal", sequence + "TARGCT", sequence + "AAAGCT"),
            ("terminal", sequence + "TAR---", sequence + "AAA---"),
        ],
    )
    assert main(["dnds", "--pairs_file", str(pairs), "--out_file", "-"]) == 0
    output = capsys.readouterr()
    rows = list(csv.DictReader(output.out.splitlines(), delimiter="\t"))
    assert rows[0]["status"] == "internal_stop"
    assert rows[0]["dS"] == rows[0]["dN"] == ""
    assert rows[1]["status"] == "ok"
    assert rows[1]["excluded_codons"] == "2"

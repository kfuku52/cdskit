import gzip
import json
import sys

import pytest

from scripts import benchmark_dnds
from scripts.benchmark_dnds import benchmark_cli, measure, paml_agreement, read_pairs


@pytest.mark.parametrize("compressed", [False, True])
def test_dnds_benchmark_reads_audited_pairs_and_uses_joint_codon_deletion(
    tmp_path, compressed
):
    path = tmp_path / ("pairs.tsv.gz" if compressed else "pairs.tsv")
    text = "pair_id\tsequence_1\tsequence_2\nfirst\tATGNNNGCT---\tATGGCCGCGGCC\n"
    if compressed:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    assert read_pairs(path) == [("first", "ATGGCT", "ATGGCG")]


@pytest.mark.parametrize(
    "sequences", [("ATGG", "ATGG"), ("ATG", "ATGAAA"), ("NNN", "---")]
)
def test_dnds_benchmark_rejects_unusable_alignments(tmp_path, sequences):
    path = tmp_path / "pairs.tsv"
    path.write_text(
        "pair_id\tsequence_1\tsequence_2\nfirst\t" + "\t".join(sequences) + "\n"
    )
    with pytest.raises(ValueError):
        read_pairs(path)


def test_dnds_benchmark_normalizes_rna_without_silently_dropping_codon_columns(
    tmp_path,
):
    pairs = tmp_path / "pairs.tsv"
    pairs.write_text("pair_id\tsequence_1\tsequence_2\na\tauggcu\tATGGCC\n")
    assert read_pairs(pairs) == [("a", "ATGGCT", "ATGGCC")]


@pytest.mark.parametrize("sequence", ["ATGZAA", "ATGéAA", "ATG!AA", "ATG\u017fAA"])
def test_dnds_benchmark_rejects_invalid_dna_instead_of_deleting_it(tmp_path, sequence):
    pairs = tmp_path / "pairs.tsv"
    pairs.write_text(
        "pair_id\tsequence_1\tsequence_2\na\t" + sequence + "\tATGGCC\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid DNA alphabet"):
        read_pairs(pairs)


def test_dnds_benchmark_rejects_input_report_collisions_before_running(
    tmp_path, monkeypatch
):
    pairs = tmp_path / "pairs.tsv"
    pairs.write_text("pair_id\tsequence_1\tsequence_2\na\tATGGCT\tATGGCC\n")
    original = pairs.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_dnds.py",
            "--pairs",
            str(pairs),
            "--output",
            str(pairs),
            "--native-only",
            "--repeats",
            "1",
        ],
    )
    with pytest.raises(ValueError, match="Input and output paths"):
        benchmark_dnds.main()
    assert pairs.read_bytes() == original


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_dnds_benchmark_rejects_report_aliases_of_input(tmp_path, monkeypatch, alias):
    pairs, output = tmp_path / "pairs.tsv", tmp_path / "report.json"
    pairs.write_text("pair_id\tsequence_1\tsequence_2\na\tATGGCT\tATGGCC\n")
    try:
        if alias == "symlink":
            output.symlink_to(pairs)
        else:
            output.hardlink_to(pairs)
    except OSError:
        pytest.skip("Filesystem links unavailable")
    original = pairs.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_dnds.py",
            "--pairs",
            str(pairs),
            "--output",
            str(output),
            "--native-only",
        ],
    )
    with pytest.raises(ValueError, match="Input and output paths"):
        benchmark_dnds.main()
    assert pairs.read_bytes() == output.read_bytes() == original


@pytest.mark.parametrize(
    "text",
    [
        "pair_id\tsequence_1\tsequence_1\n",
        "pair_id\tsequence_1\n",
        "pair_id\tsequence_1\tsequence_2\n",
        "pair_id\tsequence_1\tsequence_2\na\tATG\n",
        "pair_id\tsequence_1\tsequence_2\na\tATG\tATG\textra\n",
        "pair_id\tsequence_1\tsequence_2\na\tATG\tATG\na\tGCT\tGCT\n",
        "pair_id\tsequence_1\tsequence_2\n \tATG\tATG\n",
    ],
)
def test_dnds_benchmark_validates_the_complete_tsv_contract(tmp_path, text):
    pairs = tmp_path / "pairs.tsv"
    pairs.write_text(text)
    with pytest.raises(ValueError):
        read_pairs(pairs)


def test_dnds_benchmark_accepts_bom_and_extra_columns(tmp_path):
    pairs = tmp_path / "pairs.tsv"
    pairs.write_text(
        "\ufeffsequence_2\textra\tpair_id\tsequence_1\nATGGCC\tnote\ta\tATGGCT\n",
        encoding="utf-8",
    )
    assert read_pairs(pairs) == [("a", "ATGGCT", "ATGGCC")]


@pytest.mark.parametrize("code,stop", [(1, "TAR"), (1, "TRA"), (2, "AGR")])
@pytest.mark.parametrize("tail", ["NNN", "GCT"])
def test_dnds_benchmark_never_deletes_definite_internal_stops(
    tmp_path, code, stop, tail
):
    pairs = tmp_path / "pairs.tsv"
    pairs.write_text(
        f"pair_id\tsequence_1\tsequence_2\na\tATG{stop}{tail}\tATGGCT{tail}\n"
    )
    with pytest.raises(ValueError, match="definite internal stop"):
        read_pairs(pairs, code)


@pytest.mark.parametrize("code,stop", [(1, "TAR"), (2, "AGR"), (1, "TAA")])
def test_dnds_benchmark_excludes_terminal_stops_jointly(tmp_path, code, stop):
    pairs = tmp_path / "pairs.tsv"
    pairs.write_text(f"pair_id\tsequence_1\tsequence_2\na\tATG{stop}---\tATGGCT---\n")
    assert read_pairs(pairs, code) == [("a", "ATG", "ATG")]


@pytest.mark.parametrize("code", [27, 28, 31])
def test_native_benchmark_uses_ordinary_dual_coding_semantics(tmp_path, code):
    pairs = tmp_path / "pairs.tsv"
    pairs.write_text("pair_id\tsequence_1\tsequence_2\na\tATGTAGTGA\tATGTAGTGA\n")
    assert read_pairs(pairs, code) == [("a", "ATGTAGTGA", "ATGTAGTGA")]


@pytest.mark.parametrize(
    "flags",
    [
        ["--native-only", "--paml-only"],
        ["--paml-only", "--cli"],
        ["--native-only", "--individual"],
        ["--codon-table", "28"],
        ["--threads", "0"],
        ["--repeats", "0"],
    ],
)
def test_dnds_benchmark_rejects_incompatible_arguments_before_reading(
    tmp_path, monkeypatch, flags
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_dnds.py",
            "--pairs",
            str(tmp_path / "absent.tsv"),
            "--output",
            str(tmp_path / "report.json"),
            *flags,
        ],
    )
    with pytest.raises(SystemExit) as error:
        benchmark_dnds.main()
    assert error.value.code == 2


def test_dnds_benchmark_measures_repeats_and_rejects_output_drift():
    report, values = measure(lambda: [{"dS": 0.1}], 3)
    assert values == [{"dS": 0.1}]
    assert len(report["samples_seconds"]) == 3
    assert report["warmup_runs"] == 1
    outputs = iter([{"dS": 0.1}, {"dS": 0.2}])
    with pytest.raises(ValueError, match="changed its output"):
        measure(lambda: next(outputs), 1)


@pytest.mark.subprocess
def test_dnds_cli_benchmark_includes_fresh_process_and_complete_output():
    result = benchmark_cli([("same", "GCTGCC" * 30, "GCTGCC" * 30)], 1, 1, 2)
    assert len(result["samples_seconds"]) == 2
    assert len(result["output_sha256"]) == 64
    assert result["import_and_tsv_io_in_timing"] is True
    assert result["input_preparation_in_timing"] is False


def test_native_only_dnds_benchmark_does_not_require_paml(
    tmp_path, monkeypatch, capsys
):
    pairs, output = tmp_path / "pairs.tsv", tmp_path / "benchmark.json"
    pairs.write_text("pair_id\tsequence_1\tsequence_2\nsame\tGCTGCC\tGCTGCC\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_dnds.py",
            "--pairs",
            str(pairs),
            "--output",
            str(output),
            "--native-only",
            "--repeats",
            "1",
        ],
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Native-only benchmarking must not call PAML")

    monkeypatch.setattr(benchmark_dnds, "run_paml", forbidden)
    monkeypatch.setattr(benchmark_dnds.shutil, "which", forbidden)
    benchmark_dnds.main()
    report = json.loads(output.read_text())
    assert report["pair_count"] == 1
    assert set(report["benchmarks"]) == {"cdskit_batch"}
    assert "paml_values" not in report
    assert report["codon_table"] == 1
    assert capsys.readouterr().out


def test_dnds_benchmark_compares_saturated_diagnostics_and_detects_disagreement():
    native = [{"dN_diagnostic": 0.1, "dS_diagnostic": 4.987654, "kappa": 4.6}]
    paml = [{"dN": 0.1, "dS": 4.9877, "kappa": 4.6}]
    report = paml_agreement(native, paml)
    assert report["agreement_compared_pairs"] == {"dN": 1, "dS": 1, "kappa": 1}
    assert report["agreement_includes_saturated_diagnostics"] is True
    paml[0]["dS"] = 4.0
    with pytest.raises(ValueError, match="printed precision"):
        paml_agreement(native, paml)


def test_paml_parser_preserves_nonfinite_values_as_json_missing():
    result = benchmark_dnds.parse_paml("(B)\n2 1 0 3 0 4.6 0 nan +- 0 inf +- 0\n(C)")
    assert result == [{"dN": None, "dS": None, "kappa": 4.6}]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_paml_agreement_reports_partial_coverage_instead_of_claiming_all_pairs():
    native = [{"dN_diagnostic": 0.0, "dS_diagnostic": None, "kappa": 4.6}]
    paml = [{"dN": 0.0, "dS": 0.0, "kappa": 4.6}]
    report = paml_agreement(native, paml)
    assert report["agreement_all_pairs_compared"] is False
    assert report["agreement_compared_pairs"]["dS"] == 0
    assert report["agreement_max_absolute_error"]["dS"] is None
    assert report["agreement_missing_pairs"]["dS"]["native_missing_only"] == 1
    paml[0]["dN"] = None
    with pytest.raises(ValueError, match="no finite PAML reference"):
        paml_agreement(native, paml)


def test_failed_benchmark_never_replaces_existing_report(tmp_path, monkeypatch):
    pairs, output = tmp_path / "pairs.tsv", tmp_path / "report.json"
    pairs.write_text("pair_id\tsequence_1\tsequence_2\na\tATG\tATG\n")
    output.write_text("preserve this report\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_dnds.py",
            "--pairs",
            str(pairs),
            "--output",
            str(output),
            "--repeats",
            "1",
        ],
    )
    monkeypatch.setattr(
        benchmark_dnds, "run_paml", lambda *args: [{"dN": 99, "dS": 99, "kappa": 99}]
    )
    monkeypatch.setattr(benchmark_dnds.shutil, "which", lambda *args: str(pairs))
    with pytest.raises(ValueError, match="printed precision"):
        benchmark_dnds.main()
    assert output.read_text() == "preserve this report\n"

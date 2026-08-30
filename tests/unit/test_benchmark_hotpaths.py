import json

import pytest

from cdskit import benchmark_hotpaths
from cdskit.benchmark_hotpaths import measure, run_benchmarks
from cdskit.benchmarking import compare_reports, output_fingerprint


def test_hotpath_benchmark_measure_reports_all_samples():
    report = measure(lambda: None, repeats=2)

    assert len(report["samples_seconds"]) == 2
    assert report["minimum_seconds"] >= 0.0
    assert report["median_seconds"] >= report["minimum_seconds"]
    assert report["output_sha256"] == output_fingerprint(None)


def test_small_hotpath_benchmark_covers_every_tracked_workload():
    report = run_benchmarks(scale=1, repeats=1)

    assert set(report) == {
        "translate",
        "filter",
        "degeneracy",
        "hammer",
        "maxalign_exact",
        "targetp_features",
        "read_gff",
    }
    assert all(result["samples_seconds"] for result in report.values())


def test_benchmark_main_serializes_results(monkeypatch, capsys):
    observed = {}

    def fake_run_benchmarks(scale, repeats):
        observed.update(scale=scale, repeats=repeats)
        return {"workload": {"median_seconds": 0.1}}

    monkeypatch.setattr(benchmark_hotpaths, "run_benchmarks", fake_run_benchmarks)

    report = benchmark_hotpaths.main(["--full", "--repeats", "2"])

    assert observed == {"scale": 10, "repeats": 2}
    assert json.loads(capsys.readouterr().out) == report


def test_benchmark_main_rejects_nonpositive_repeats():
    with pytest.raises(SystemExit):
        benchmark_hotpaths.main(["--repeats", "0"])


def test_benchmark_rejects_changing_output_between_repetitions():
    outputs = iter(["warmup", "different"])
    with pytest.raises(ValueError, match="output changed"):
        measure(lambda: next(outputs), 1)


def test_output_fingerprint_distinguishes_dictionary_nesting():
    assert output_fingerprint({"a": {}, "b": 1}) != output_fingerprint({"a": {"b": 1}})


@pytest.mark.parametrize(
    "change,expected",
    [("time", "slowdown"), ("output", "changed_output"), ("none", "ok")],
)
def test_comparison_separates_performance_and_output_changes(change, expected):
    import copy

    previous = {
        "schema_version": 2,
        "environment": {},
        "benchmarks": {
            "test": {
                "workload": {"records": 3},
                "median_seconds": 1.0,
                "output_sha256": "abc",
            },
        },
    }
    current = copy.deepcopy(previous)
    if change == "time":
        current["benchmarks"]["test"]["median_seconds"] = 1.5
    elif change == "output":
        current["benchmarks"]["test"]["output_sha256"] = "def"
    comparison = compare_reports(current, previous)
    assert comparison["workloads"]["test"]["status"] == expected


def test_comparison_does_not_claim_regressions_across_environments():
    previous = {"schema_version": 2, "environment": {"python": "3.12.1"}}
    current = {"schema_version": 2, "environment": {"python": "3.14.1"}}
    assert compare_reports(current, previous)["status"] == "incomparable"
    assert compare_reports(current, {})["status"] == "incomparable"


def test_comparison_does_not_claim_success_without_comparable_workloads():
    previous = {"schema_version": 2, "environment": {}, "benchmarks": {}}
    current = {
        "schema_version": 2,
        "environment": {},
        "benchmarks": {"new": {"workload": {"records": 3}}},
    }
    comparison = compare_reports(current, previous)
    assert comparison["status"] == "incomparable"
    assert comparison["workloads"]["new"]["status"] == "incomparable"

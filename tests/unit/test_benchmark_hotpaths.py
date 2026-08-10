import json

import pytest

from cdskit import benchmark_hotpaths
from cdskit.benchmark_hotpaths import measure, run_benchmarks


def test_hotpath_benchmark_measure_reports_all_samples():
    report = measure(lambda: None, repeats=2)

    assert len(report["samples_seconds"]) == 2
    assert report["minimum_seconds"] >= 0.0
    assert report["median_seconds"] >= report["minimum_seconds"]


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

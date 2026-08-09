from cdskit.benchmark_hotpaths import measure


def test_hotpath_benchmark_measure_reports_all_samples():
    report = measure(lambda: None, repeats=2)

    assert len(report['samples_seconds']) == 2
    assert report['minimum_seconds'] >= 0.0
    assert report['median_seconds'] >= report['minimum_seconds']

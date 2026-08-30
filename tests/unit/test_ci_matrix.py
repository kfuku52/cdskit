from scripts.ci_matrix import core_matrix


def test_platform_sensitive_changes_keep_all_supported_combinations():
    matrix = core_matrix("push", ["cdskit/atomicio.py"])
    assert len(matrix) == 9
    assert {row["os"] for row in matrix} == {
        "ubuntu-latest",
        "macos-latest",
        "windows-latest",
    }
    assert {row["python-version"] for row in matrix} == {
        "3.10",
        "3.11",
        "3.12",
        "3.13",
        "3.14",
    }


def test_ml_only_changes_keep_core_boundaries_without_platform_jobs():
    assert core_matrix(
        "push", ["cdskit/targetp_torch.py", "tests/ml/test_targetp_torch.py"]
    ) == [
        {"os": "ubuntu-latest", "python-version": "3.10"},
        {"os": "ubuntu-latest", "python-version": "3.14"},
    ]


def test_scheduled_and_manual_runs_never_lose_platform_coverage():
    for event in ["schedule", "workflow_dispatch"]:
        assert len(core_matrix(event, [])) == 9
    assert len(core_matrix("pull_request", ["uv.lock"])) == 4

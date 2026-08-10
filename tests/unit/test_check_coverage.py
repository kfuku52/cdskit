from scripts.check_coverage import MODULE_FLOORS, validate_coverage


def test_validate_coverage_reports_missing_and_low_modules():
    filenames = list(MODULE_FLOORS)
    payload = {
        "files": {
            filenames[0]: {"summary": {"percent_covered": MODULE_FLOORS[filenames[0]]}},
            filenames[1]: {"summary": {"percent_covered": 1.0}},
        }
    }

    failures = validate_coverage(payload)

    assert any(filenames[1] in failure and "below" in failure for failure in failures)
    assert any(filenames[2] in failure and "missing" in failure for failure in failures)

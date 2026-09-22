import csv
import io

import pytest

from cdskit.tsvio import (
    TSV_REPORT_SCHEMA_VERSION,
    read_tsv,
    write_sectioned_tsv,
    write_tsv,
)


def test_read_tsv_rejects_non_rectangular_rows(tmp_path):
    path = tmp_path / "broken.tsv"
    path.write_text("accession\tsequence\nA\tMAAA\textra\n", encoding="utf-8")

    with pytest.raises(ValueError, match="row width mismatch"):
        read_tsv(str(path))


def test_read_tsv_rejects_duplicate_and_missing_columns(tmp_path):
    duplicate = tmp_path / "duplicate.tsv"
    duplicate.write_text("accession\taccession\nA\tB\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate columns"):
        read_tsv(str(duplicate))

    missing = tmp_path / "missing.tsv"
    missing.write_text("accession\nA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sequence"):
        read_tsv(str(missing), required_columns=["accession", "sequence"])


def test_write_tsv_is_utf8_lf_and_rectangular(tmp_path):
    path = tmp_path / "out.tsv"
    write_tsv(
        path=str(path),
        rows=[{"accession": "α", "sequence": "MAAA", "kept": True}],
        fieldnames=["accession", "sequence", "kept"],
    )

    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")
    with path.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.reader(handle, delimiter="\t")) == [
            ["accession", "sequence", "kept"],
            ["α", "MAAA", "yes"],
        ]


def test_write_tsv_rejects_unexpected_columns(tmp_path):
    path = tmp_path / "unexpected.tsv"

    with pytest.raises(ValueError, match=r"data row 1: extra"):
        write_tsv(
            path=str(path),
            rows=[{"accession": "A", "sequence": "MAAA", "extra": "value"}],
            fieldnames=["accession", "sequence"],
        )


def test_write_tsv_rejects_non_mapping_rows(tmp_path):
    path = tmp_path / "not-a-mapping.tsv"

    with pytest.raises(TypeError, match=r"TSV row 1.*list"):
        write_tsv(
            path=str(path),
            rows=[["A", "MAAA"]],
            fieldnames=["accession", "sequence"],
        )


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_write_tsv_preserves_destination_when_rows_fail(tmp_path, existing, error_type):
    path = tmp_path / "report.tsv"
    if existing:
        path.write_bytes(b"original report\n")
    failure = error_type("row generation failed")

    def rows():
        yield {"accession": "A"}
        raise failure

    with pytest.raises(error_type) as exc_info:
        write_tsv(path, rows(), ["accession"])

    assert exc_info.value is failure
    if existing:
        assert path.read_bytes() == b"original report\n"
    else:
        assert not path.exists()
    assert list(tmp_path.iterdir()) == ([path] if existing else [])


@pytest.mark.parametrize("fail", [False, True])
def test_write_tsv_leaves_stdout_open(monkeypatch, fail):
    output = io.StringIO()
    monkeypatch.setattr("sys.stdout", output)
    failure = RuntimeError("row generation failed")

    def rows():
        yield {"accession": "α", "kept": True}
        if fail:
            raise failure

    if fail:
        with pytest.raises(RuntimeError) as exc_info:
            write_tsv("-", rows(), ["accession", "kept"])
        assert exc_info.value is failure
    else:
        write_tsv("-", rows(), ["accession", "kept"])

    assert not output.closed
    assert output.getvalue() == "accession\tkept\nα\tyes\n"


def test_write_sectioned_tsv_adds_schema_version_and_section(tmp_path):
    path = tmp_path / "report.tsv"
    write_sectioned_tsv(
        path=str(path),
        rows=[{"section": "summary", "metric": "num_sequences", "value": 3}],
        fieldnames=["section", "metric", "value"],
    )

    rows, fieldnames = read_tsv(str(path), return_fieldnames=True)
    assert fieldnames == ["schema_version", "section", "metric", "value"]
    assert rows == [
        {
            "schema_version": TSV_REPORT_SCHEMA_VERSION,
            "section": "summary",
            "metric": "num_sequences",
            "value": "3",
        }
    ]


def test_write_sectioned_tsv_rejects_conflicting_schema_version(tmp_path):
    path = tmp_path / "report.tsv"

    with pytest.raises(ValueError, match="Conflicting TSV schema_version"):
        write_sectioned_tsv(
            path=str(path),
            rows=[{"schema_version": "1", "section": "summary"}],
            fieldnames=["schema_version", "section"],
        )


def test_write_sectioned_tsv_requires_nonempty_section(tmp_path):
    path = tmp_path / "report.tsv"

    with pytest.raises(ValueError, match="Missing TSV section"):
        write_sectioned_tsv(
            path=str(path),
            rows=[{"metric": "num_sequences", "value": 3}],
            fieldnames=["metric", "value"],
        )

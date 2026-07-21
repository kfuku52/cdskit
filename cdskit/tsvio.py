import csv
import json
import os
import sys
from collections.abc import Mapping


TSV_ENCODING = 'utf-8'
TSV_LINE_TERMINATOR = '\n'
TSV_REPORT_SCHEMA_VERSION = '2'


def _column_list(values):
    return [str(value) for value in (values or [])]


def validate_fieldnames(fieldnames, path, required_columns=None):
    fieldnames = _column_list(fieldnames)
    if len(fieldnames) == 0:
        raise ValueError('TSV has no header: {}'.format(path))
    empty = [index + 1 for index, name in enumerate(fieldnames) if name.strip() == '']
    if empty:
        raise ValueError(
            'TSV has empty header name(s) at column(s) {}: {}'.format(
                ', '.join(str(value) for value in empty),
                path,
            )
        )
    duplicates = sorted({name for name in fieldnames if fieldnames.count(name) > 1})
    if duplicates:
        raise ValueError(
            'TSV has duplicate columns in {}: {}'.format(path, ', '.join(duplicates))
        )
    required = _column_list(required_columns)
    missing = [name for name in required if name not in fieldnames]
    if missing:
        raise ValueError(
            'Missing required columns in {}: {}'.format(path, ', '.join(missing))
        )
    return fieldnames


def read_tsv(path, required_columns=None, return_fieldnames=False):
    """Read a strict, rectangular, headered TSV file.

    UTF-8 BOM is accepted for interoperability, while writers always emit plain
    UTF-8 with LF line endings.
    """
    with open(path, 'r', encoding='utf-8-sig', newline='') as inp:
        reader = csv.reader(inp, delimiter='\t', strict=True)
        try:
            fieldnames = next(reader)
        except StopIteration:
            raise ValueError('TSV is empty: {}'.format(path))
        fieldnames = validate_fieldnames(
            fieldnames=fieldnames,
            path=path,
            required_columns=required_columns,
        )
        rows = []
        for line_number, values in enumerate(reader, start=2):
            if len(values) == 0:
                continue
            if len(values) != len(fieldnames):
                raise ValueError(
                    'TSV row width mismatch in {} at line {}: expected {} columns, got {}.'.format(
                        path,
                        line_number,
                        len(fieldnames),
                        len(values),
                    )
                )
            rows.append(dict(zip(fieldnames, values)))
    if return_fieldnames:
        return rows, fieldnames
    return rows


def _open_tsv_output(path):
    if path == '-':
        return sys.stdout, False
    out_dir = os.path.dirname(str(path))
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)
    return open(path, 'w', encoding=TSV_ENCODING, newline=''), True


def write_tsv(path, rows, fieldnames):
    fieldnames = validate_fieldnames(
        fieldnames=fieldnames,
        path=path,
        required_columns=None,
    )
    out, should_close = _open_tsv_output(path)
    try:
        writer = csv.DictWriter(
            out,
            fieldnames=fieldnames,
            delimiter='\t',
            lineterminator=TSV_LINE_TERMINATOR,
            extrasaction='raise',
        )
        writer.writeheader()
        expected = set(fieldnames)
        for row_number, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                raise TypeError(
                    'TSV row {} for {} should be a mapping, got {}.'.format(
                        row_number,
                        path,
                        type(row).__name__,
                    )
                )
            unexpected = sorted(str(name) for name in set(row) - expected)
            if unexpected:
                raise ValueError(
                    'Unexpected TSV columns in {} at data row {}: {}.'.format(
                        path,
                        row_number,
                        ', '.join(unexpected),
                    )
                )
            writer.writerow({name: json_cell(row.get(name, '')) for name in fieldnames})
    finally:
        if should_close:
            out.close()


def write_sectioned_tsv(
    path,
    fieldnames,
    rows,
    schema_version=TSV_REPORT_SCHEMA_VERSION,
):
    """Write a versioned rectangular report with ``section`` on every row."""
    fieldnames = validate_fieldnames(
        fieldnames=fieldnames,
        path=path,
        required_columns=None,
    )
    fieldnames = [
        'schema_version',
        'section',
    ] + [
        name for name in fieldnames
        if name not in {'schema_version', 'section'}
    ]
    schema_version = str(schema_version)
    if schema_version.strip() == '':
        raise ValueError('TSV schema_version should not be empty: {}'.format(path))

    def versioned_rows():
        for row_number, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                yield row
                continue
            section = str(row.get('section') or '').strip()
            if section == '':
                raise ValueError(
                    'Missing TSV section in {} at data row {}.'.format(
                        path,
                        row_number,
                    )
                )
            declared_version = row.get('schema_version')
            if declared_version not in (None, '', schema_version):
                raise ValueError(
                    'Conflicting TSV schema_version in {} at data row {}: '
                    'expected {}, got {}.'.format(
                        path,
                        row_number,
                        schema_version,
                        declared_version,
                    )
                )
            versioned = dict(row)
            versioned['schema_version'] = schema_version
            versioned['section'] = section
            yield versioned

    write_tsv(path=path, rows=versioned_rows(), fieldnames=fieldnames)


def json_cell(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if value is None:
        return ''
    return value

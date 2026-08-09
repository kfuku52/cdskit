# Testing CDSKIT

The test suite is organized by execution boundary:

- `tests/unit`: isolated helpers and pure transformations.
- `tests/integration`: command-level I/O and multi-component behavior.
- `tests/ml`: tests whose module requires optional machine-learning packages.
- `tests/fixtures`: immutable, repository-owned input and expected-output files.

Tests in the first three directories receive the matching `unit`, `integration`,
or `ml` pytest marker automatically. Individual optional tests in otherwise
dependency-light modules are marked explicitly. Additional `slow`, `subprocess`,
and `serial` markers document execution constraints.

## Install

Install only what the selected check needs:

```bash
python -m pip install -e ".[test]"
python -m pip install -e ".[test,coverage,ml]"
python -m pip install -e ".[quality]"
python -m pip install -e ".[build]"
python -m pip install -e ".[security]"
```

`.[dev]` installs all of these groups for a complete development environment.

## Common commands

Use the dependency-light suite for the normal edit/test loop:

```bash
python -m pytest -q tests/unit tests/integration -m "not ml and not subprocess"
```

Run one boundary or the optional ML tests:

```bash
python -m pytest -q tests/unit
python -m pytest -q tests/integration -m "not ml"
python -m pytest -q -m ml
```

Run every installed test:

```bash
python -m pytest -q
```

Run the same parallel coverage check used by CI:

```bash
python -m pytest -q -n 2 --dist=worksteal \
  --cov=cdskit --cov-report=term-missing --cov-fail-under=75
```

Run the security and complexity guards used by CI:

```bash
python -m pip_audit --local --skip-editable
python -m bandit -q -r cdskit -lll
python -m ruff check cdskit --select C901 \
  --config lint.mccabe.max-complexity=38
```

Dependency minimums express API compatibility, not a recommendation to retain
old releases indefinitely. Use a freshly resolved environment for production
work so patched versions satisfying the declared ranges are installed. GitHub
Actions repeats the dependency and source-security audit every Monday.

For a short feedback loop after a failure, use `--lf` to rerun failures or
`-f` to stop on the first failure and restart when a test file changes.

## Test-writing rules

- Prefer pytest's `tmp_path`; `temp_dir` remains as a compatibility alias.
- Reuse the `record_factory`, `write_fasta`, and `write_tsv` fixtures for small
  generated inputs.
- Keep behavior assertions in-process. Reserve subprocess tests for installed
  entry points and executable wrappers.
- Missing tracked fixtures are test failures, not reasons to skip.
- Mark optional dependency tests with `ml` and expensive training/E2E tests
  with `slow`.
- Do not add fixed timing assertions. Review `--durations` output in CI when a
  regression changes the slowest tests.

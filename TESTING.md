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

CI and local development use the committed `uv.lock`. Install only what the
selected check needs:

```bash
uv sync --locked --no-dev --group test
uv sync --locked --no-dev --group test --group coverage --extra ml
uv sync --locked --only-group quality
uv sync --locked --only-group build
uv sync --locked --only-group security
```

Use `uv sync --locked --extra ml` for the complete development environment.
The PEP 621 extras remain available to pip users, for example
`python -m pip install -e ".[test,coverage]"`.

Refresh the lock only as an intentional dependency update, then review and
test the diff:

```bash
uv lock --upgrade
uv lock --check
```

## Common commands

Use the dependency-light suite for the normal edit/test loop:

```bash
uv run --no-sync python -m pytest -q tests/unit tests/integration -m "not ml and not subprocess"
```

Run one boundary or the optional ML tests:

```bash
uv run --no-sync python -m pytest -q tests/unit
uv run --no-sync python -m pytest -q tests/integration -m "not ml"
uv run --no-sync python -m pytest -q -m ml
```

Run every installed test:

```bash
uv run --no-sync python -m pytest -q
```

Run the same parallel coverage check used by CI:

```bash
uv run --no-sync python -m pytest -q -n 2 --dist=worksteal \
  --cov=cdskit --cov-report=term-missing --cov-report=json:coverage.json
uv run --no-sync python scripts/check_coverage.py coverage.json
```

Coverage is branch-aware. The project-wide 74% floor is complemented by
higher module-specific floors for transactional I/O, command dispatch,
pretrained-model downloads, and TargetP training configuration.

Run the security and complexity guards used by CI:

```bash
uv run --no-sync python -m ruff format --check cdskit scripts tests
uv run --no-sync python -m ruff check cdskit scripts tests
uv run --no-sync python -m mypy cdskit
uv run --no-sync python -m pip_audit --local --skip-editable
uv run --no-sync python -m bandit -q -r cdskit -lll
uv run --no-sync python -m ruff check cdskit --select C901
```

The complexity ceiling is 35. A scheduled GitHub Actions workflow runs the
full CPU hot-path benchmark weekly and retains its JSON result for 90 days.

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

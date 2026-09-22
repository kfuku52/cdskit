# Testing CDSKIT

The committed `uv.lock` is shared by local checks and GitHub Actions. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) and use the same
entry point as CI from the repository root (Python 3.10+ to launch it):

```bash
python scripts/check.py quick
python scripts/check.py core --python 3.10
python scripts/check.py ml
python scripts/check.py quality
python scripts/check.py coverage
python scripts/check.py build
python scripts/check.py all
```

The script creates isolated environments under `.venvs/`, choosing Python 3.12
unless `--python` is supplied. Core, full, and build checks have separate
environments; they do not replace an existing `.venv` or remove another check's
dependencies. Subprocesses and executable script wrappers also inherit the
selected environment, rather than resolving Python from the caller's `PATH`.
Full checks install the test, coverage, quality, build, security
and ML dependencies, and explicitly check ML imports before running pytest.
The [uv cache key](https://docs.astral.sh/uv/concepts/cache/#dynamic-metadata)
includes `cdskit/__init__.py` so version changes also refresh installed metadata.

The full environment uses the `ml-cpu` extra. On Linux this selects the official
CPU-only PyTorch index through uv; other packages still come from PyPI.
The normal `ml` extra and library dependency ranges remain unchanged.
For GPU development, use `--ml-backend default` with the same script.
The two backend profiles use separate environments.
See [uv's PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/)
for index configuration. From a repository checkout, pip users can install
`python -m pip install -e '.[ml]'` and select the appropriate PyTorch index
separately; uv source settings are not wheel metadata. Outside a checkout, use
the GitHub URL shown in the README because CDSKIT is not published on PyPI.

## Fast feedback

`quick` runs unit and integration tests without ML or subprocess tests.
`core` also exercises subprocess entry points. Both compile every package and
script module under the selected Python version.

Pass focused pytest options after `--`:

```bash
python scripts/check.py quick -- -k backalign -x
python scripts/check.py quick -- --lf
python scripts/check.py ml -- -k targetp
```

For repeated runs without synchronization, use the selected environment directly:

```bash
.venvs/core-3.12/bin/python -m pytest -q tests/unit/test_util.py
```

On Windows, its interpreter is `.venvs/core-3.12/Scripts/python.exe`.
Use `-x` to stop on the first failure; pytest does not provide a built-in file
watcher.

## Choose checks by change

Start with the affected test selection, then run the relevant complete profile
before delivery. These are local checks; CI supplies its declared platform matrix.

| Change | Minimum affected verification |
| --- | --- |
| One sequence command | `quick -- -k COMMAND -x`; then `core` for CLI subprocess coverage |
| Shared codon semantics | `quick -- -k codon`; then `core`, since many commands share the helpers |
| CLI dispatch, sequence/TSV I/O, output paths | `core` (includes command and output-safety regressions) |
| Localization decisions or model loading | Relevant unit/integration tests plus `ml`; use `coverage` for changes spanning backends |
| Python implementation | Add `quality` to the affected behavior checks |
| Packaging or version | `build` for a fresh installed-wheel smoke check |
| Docs or examples | Compare help and execute affected examples per [documentation maintenance](docs/documentation.md); no text-matching tests |
| Check scripts or test organization | Exercise the affected profiles; `coverage` if test coverage changes |

The table uses profile names as shorthand for `python scripts/check.py PROFILE`.
`-k` matches test names and paths; inspect the collected selection and do not
count zero selected tests or optional-dependency skips as successful coverage.
For a particular test file, use the environment directly as above; the wrapper
already supplies suite paths, so appending another path does not narrow them.

## Setup cost, network access and outputs

- Every profile first runs `uv sync --locked`. An uncached setup can download
  Python and packages. `quick`/`core` need only core/test packages; `quality`,
  `ml`, `coverage` and `all` use the full environment, including large ML packages.
  Prefer an existing matching profile for repeated checks; do not rebuild or
  upgrade another task's environment to obtain it.
- `quick` excludes ML and subprocess tests, **not** the `slow` marker. For a
  deliberately smaller iteration use `python scripts/check.py quick -- -m
  "not ml and not subprocess and not slow"`; this is not the complete core suite.
- The test suites use small fixtures and local/mocked models and services.
  Published model downloads, ESM backbone downloads, accession retrieval against
  live services and full pipeline training are separate integration/experimental
  work. Check the relevant guide and required data/resources before running them;
  do not launch the default `localize` prediction as a lightweight smoke test.
- `all` calls the external vulnerability service through `pip-audit`. `build`
  installs a fresh wheel and dependencies into a temporary environment, which
  can also require network access. An unavailable service is a blocked check,
  not a reason to suppress errors or relax the check.
- Tests use pytest temporary directories; wheel outputs are temporary and removed
  on completion. Coverage writes ignored `.coverage*` and `coverage.json` in the
  repository root; do not run competing coverage jobs there. Compiler and tool
  caches are also local generated files, not source changes.

Success means exit status zero, the intended tests actually passing, and no
unexpected skips. Core profiles can skip the optional PyYAML pipeline-config
cases; use the full profile when that configuration changes. CUDA-only cases
can skip on CPU hosts; report this separately from CPU ML coverage. For missing
tools/packages, report the failed command and missing prerequisite. Static inspection or a reduced selection is not a pass
for an unexecuted profile.

## Quality and compatibility

`quality` runs Ruff lint/format checks, mypy, the complexity guard (ceiling 35),
and high-severity Bandit checks. Mypy checks against the running interpreter and
its installed dependency stubs, including ML dependencies. Hard-coding a Python
3.10 target while loading modern NumPy stubs under Python 3.12 is not supported.
Minimum Python compatibility is instead compiled and exercised by the 3.10
core check, alongside the newest supported Python.

`coverage` runs all tests with two workers and combined branch/statement
coverage. The project-wide 74% floor and the higher transactional I/O, dispatch,
model-download and training-configuration floors are enforced.
`all` additionally audits the installed dependencies, builds an sdist and wheel,
and imports and exercises the installed wheel outside the checkout in a fresh
environment (including sequence output and an SVG plot).
Editable distributions are excluded from the dependency audit. Official PyTorch
`+cpu` wheels are checked against their corresponding upstream release's
advisories because PyPI has no record for the local `+cpu` version. Other local
version suffixes are not rewritten. Any unresolved dependency fails the audit;
the script does not install or resolve a second set of packages.

GitHub Actions always runs the Linux Python boundaries and full CPU ML/quality/
package validation. Changes to core I/O, CLI, dependencies, tests or workflow
infrastructure also run macOS and Windows coverage. Localization download, model I/O, pipeline and OOF-cache changes also require
platform coverage. Pure ML computation changes,
tests restricted to `tests/ml`, or
documentation paths alone do not require those platform jobs. A version bump in
`cdskit/__init__.py` does trigger core/platform coverage, including when bundled
with documentation changes. Relevant pull requests use four core combinations;
relevant pushes, scheduled and manual runs cover all nine. Short validation jobs share a
single environment; superseded runs are cancelled.

The weekly security workflow also audits the normal GPU-capable ML resolution,
including dependencies absent from CPU-only CI.

## Benchmarks

```bash
.venvs/core-3.12/bin/python -m cdskit.benchmark_hotpaths --full --repeats 5 > baseline.json
.venvs/core-3.12/bin/python -m cdskit.benchmark_hotpaths --full --repeats 5 --baseline baseline.json > current.json
```

Reports use schema version 2: workload results are under `benchmarks`, with
`environment` metadata (commit, Python, package versions and CPU) and
`process_peak_rss_bytes`. RSS is the whole benchmark process high-water mark,
not an allocation measurement for each workload; it is unavailable on Windows.
Each workload warms once and hashes its complete output outside the timed
region, checking every repetition for consistency.

Comparisons require matching workloads, Python/dependencies and CPU metadata.
Changed outputs are reported separately from slowdowns; legacy reports and
different environments are marked incomparable. The weekly workflow compares
against the latest retained successful result, writes a job summary, and warns
on output changes or a median slowdown exceeding 30%. Timing warnings do not
fail CI: confirm them with repeated runs on the same hardware. JSON artifacts
are retained for 30 days. There are no fixed timing assertions in tests.

## Dependencies and test layout

Refresh the lock only as an intentional dependency update and review its diff:

```bash
uv lock --upgrade
uv lock --check
```

Dependency minimums describe API compatibility, not a recommendation to keep old
releases. Test newly resolved compatible packages when updating the lock; do not
add upper bounds merely to freeze the development environment.

- `tests/unit`: isolated helpers and pure transformations.
- `tests/integration`: command-level I/O and multi-component behavior.
- `tests/ml`: optional machine-learning packages.
- `tests/fixtures`: immutable repository-owned inputs and expected outputs.

Directories supply `unit`, `integration`, and `ml` markers automatically.
Mark individual optional tests in otherwise dependency-light modules explicitly.
Use `slow`, `subprocess`, and `serial` when their execution requires it.
Prefer pytest's `tmp_path` and the shared record/FASTA/TSV fixtures. Missing tracked
fixtures are failures, not reasons to skip. Keep assertions in-process except
when checking installed entry points or executable wrappers.

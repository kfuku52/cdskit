# Changelog

This project follows semantic versioning. Deprecated CLI spellings remain
available for at least the 0.24 release series and print their canonical
replacement to standard error.

## 0.28.3 — 2026-08-31

### Documentation

- Reconcile the complete published wiki with version-controlled page/image
  sources and document how to publish both repositories without losing edits.
- Correct Python/ML installation instructions and compensate for missing runtime
  dependencies in the external Bioconda recipe. Document a verified scikit-learn 1.5.2
  environment for the existing pretrained targeting5 artifacts; those artifacts
  fail to load under scikit-learn 1.9.0. Library dependency ranges are unchanged.
- Fix the unaligned `hammer` example and its occupancy boundary, remove obsolete
  `label --threads` guidance, and clarify ambiguous translation, CDS annotations,
  GC denominators, and the two-table `codonstats --mode both` output.
- Document safe model loading, offline caches, original targeting5's constant-zero
  peroxisome score, and current TargetP checkpoint provenance requirements.
- Keep the README concise, separate historical experiments from current guides,
  remove unsupported TargetP-equivalence claims, and clarify CI/release behavior.

See the [documentation review](docs/documentation-review-0.28.3.md) for the
verified examples and remaining external packaging/model-artifact limitations.

## 0.28.2 — 2026-08-31

### Security

- Audit the installed CPU PyTorch wheel against its corresponding upstream
  release's advisories; PyPI does not publish records for `+cpu` versions.
  The audit now fails on any unresolved non-editable dependency instead of
  allowing a skipped package to leave a coverage gap.

## 0.28.1 — 2026-08-31

### Fixed

- Activate the selected check environment for subprocesses as well as pytest.
  Direct executable wrappers now resolve their shebang interpreter inside that
  environment, including on clean Linux/macOS CI runners.
- Include the version source in uv's cache key so changing `__version__` refreshes
  the installed editable package metadata on the next sync.

## 0.28.0 — 2026-08-31

### Fixed

- Protect every CLI input, including auxiliary alignments, accession lists and
  resolved model aliases; validate generated output names and file aliases.
- Reject directory/special-file destinations and overlapping output paths before
  staging. Restore dangling symlinks correctly after a failed multi-file commit.
- Commit `filter` and `trimcodon` sequence outputs and reports together, preserving
  both existing files when either write fails.
- Rebase extracted GenBank CDS locations and translation exceptions, preserve
  per-letter annotations, and remove stale genomic feature/reference coordinates.
- Make backalignment agree with translation for context-dependent genetic codes
  27, 28 and 31 while retaining terminal-stop handling.
- Record data/configuration/fold provenance in every TargetP epoch checkpoint.
  Interrupted nested OOF runs can resume; extending the epoch budget is supported
  without accepting changed data or numerical training settings.
- Aggregate TargetP classification and cleavage losses with their respective
  denominators, independent of validation batch size and ordering.
- Apply offline prediction settings to every nested ESM stage and blend component.

### Performance and development

- Compute frozen ESM embeddings once for head training; share a bounded,
  label-independent cache across folds and classifier stages.
- Share bounded batch prediction between the CLI, training metrics and CV; honor
  the evaluation device and preserve organism constraints and stage-3 OOF details.
- Reuse the optimized specialist feature implementation in training and inference.
- Add `python scripts/check.py` with isolated, locked check environments. Mypy now
  targets its actual interpreter and installed dependency stubs; minimum Python
  support retains compile and core-test checks.
- Add a separate uv `ml-cpu` profile for Linux CI, without changing normal ML
  dependency ranges or GPU installation. Combine short validation jobs and gate
  platform runners on relevant changes, with full scheduled/manual coverage.
- Benchmark JSON schema 2 records environment, output hashes and process peak RSS.
  Scheduled runs compare retained results and flag output changes or slowdowns.
  See [TESTING.md](TESTING.md) for commands and the new report layout.

See the [review follow-up](docs/review-0.28.md) for regression evidence,
before/after measurements and compatibility limits.

## 0.27.0 — 2026-08-10

### Development

- Migrated package metadata, extras, entry points, and tool configuration to
  `pyproject.toml`; replaced the legacy setup and pytest files with one source
  of truth.
- Added a committed `uv.lock`, task-specific dependency groups, cached locked
  CI installs, and pull-request concurrency cancellation. Pull requests now
  run a representative four-environment matrix while pushes retain the full
  nine-environment compatibility matrix.
- Standardized the repository with Ruff formatting and enabled BugBear,
  performance, and Ruff-specific maintenance rules. Tightened the cyclomatic
  complexity ceiling from 38 to 35.

### Changed

- Extracted typed transactional file operations from the high-fan-in utility
  module while retaining compatibility re-exports.
- Replaced repeated CLI command wrappers with a typed, lazily imported command
  runtime, and split TargetP training configuration, resume options, metric
  selection, and scheduler decisions into independently tested pure helpers.
- Enabled checks inside previously untyped functions and require complete
  annotations for the new shared modules and selected common modules.
- Replaced generic user-input exceptions with `ValueError` and made tests
  assert the concrete failure type.

### Testing

- Enabled branch-aware coverage with a project-wide floor and higher floors
  for transactional I/O, CLI dispatch, pretrained-model downloads, and
  TargetP training helpers.
- Raised pretrained-model resolver coverage from 57% to 95%, including
  successful atomic downloads, declared and streamed size limits, network
  failures, cleanup, offline mode, checksums, and unpublished assets.
- Added installed-wheel smoke tests for both sequence processing and plotting,
  plus a scheduled full CPU benchmark whose JSON results are retained as CI
  artifacts.

### Security

- Run dependency audits from a fresh locked environment and retain the weekly
  high-severity source scan. Model downloads remain checksum-verified and now
  have explicit failure-path regression coverage.

## 0.26.0 — 2026-08-09

### Security

- Hardened the privileged release workflow by checking out only the trusted
  default branch, verifying it matches the successful test run, and pinning
  third-party actions to reviewed commit SHAs.
- Pinned DeepLoc 2.1 benchmark inputs by URL and SHA-256, and added per-change
  and weekly checks for known dependency vulnerabilities and new high-severity
  Bandit findings.

### Performance

- Batched composite TargetP localization models, sklearn feature ensembles,
  specialist heads, and peroxisome heads instead of invoking them per record.
- Vectorized codon translation and masking, shared cached codon classification,
  fused validation passes, and reduced repeated sequence conversion in
  degeneracy and drawing paths.
- Selected process parallelism by total sequence workload, streamed large
  translation jobs in bounded chunks, accelerated exact and greedy maxalign,
  and replaced generic GFF parsing and iterative coordinate shifts with typed
  and search-based implementations.
- Avoided duplicate transactional file synchronization and enabled explicit
  CPU worker counts for research tree-ensemble training commands.

### Testing

- Added batching and workload-selection regression coverage plus a reusable
  CPU hot-path benchmark runner.
- Split unit, integration, and optional-ML tests; centralized immutable
  fixtures; added strict pytest markers; and introduced a dependency-light
  development suite.
- Parallelized only the measured-beneficial coverage run, split CI quality and
  package checks into concurrent jobs, and enabled pip dependency caching.
- Added mocked ESM-head training and inference coverage, legacy SVG rendering
  coverage, reduced the maximum function complexity from 68 to 38, and added
  a regression guard at the new ceiling.

### Fixed

- Kept batched TargetP blends and specialist rerankers consistent with
  record-wise prediction for missing classes, non-plant constraints, and
  malformed class-specific thresholds.
- Preserved partial-codon helper behavior, stop-at-first-stop translation,
  legacy same-site gap edit ordering, and incomplete-codon statistics.
- Exposed the hot-path benchmark as the packaged
  `python -m cdskit.benchmark_hotpaths` module.
- Corrected GitHub development installation instructions so the optional ML
  extra no longer assumes that cdskit is published on PyPI.

## 0.25.2 — 2026-08-06

### Changed

- Restricted automated Git tags, GitHub Releases, and downstream Bioconda
  updates to major and minor versions whose patch component is zero.
- Made release creation wait for the full test workflow and use its exact
  tested commit.

## 0.25.1 — 2026-08-01

### Changed

- Removed preventive upper bounds from the optional PyTorch, scikit-learn, and
  Transformers dependencies. Compatibility is now expressed by the supported
  minimum versions instead of excluding untested future major releases.

### Testing

- Verified the full test suite with current stable ML dependencies, including
  Transformers 5.

## 0.25.0 — 2026-07-31

### Security

- Made localize model loading safe by default. Legacy pickle-backed PyTorch
  models now require the explicit `--allow_unsafe_model yes` opt-in; registered
  pretrained aliases are checksum-verified before legacy loading is allowed.
- Disabled pickle-backed NumPy archive loading, pinned remote TargetP and ESM
  resources to immutable revisions, verified TargetP checksums, and bounded
  model, dataset, UniProt, regex, image, sequence, thread, and gap inputs.
- Hardened UniProt pagination and response validation and added timeouts to
  external MMseqs invocations.

### Fixed

- Prevented commands from overwriting an input path or partially replacing
  sequence, GFF, report, and multi-file outputs after failures.
- Preserved or adjusted GFF coordinates correctly when gaps are shortened,
  rejected out-of-range transformed features, and cleared stale sequence
  annotations after length-changing operations.
- Corrected ambiguous-codon masking, duplicate label allocation, label clipping,
  padded-sequence counts, non-finite numeric arguments, malformed benchmark
  rows, and probability validation.
- Added provenance fingerprints to reusable TargetP and PyTorch caches so stale
  artifacts are not silently accepted.

### Changed

- Moved the public CLI into import-safe `cdskit.cli:main`; the checkout script is
  now a thin wrapper and installed packages use a console entry point.
- Limited automatic thread selection to 64 workers by default. The limit can be
  changed with `CDSKIT_MAX_THREADS`; options that did no parallel work were
  removed.
- Added `transformers` to the `ml` extra and recorded model/library provenance
  in trained artifacts.
- Raised the minimum supported Python version to 3.10 and declared support
  through Python 3.14.

### Testing

- Added clean-checkout fixtures and regression coverage for transactional
  outputs, unsafe model and archive rejection, cache provenance, path
  collisions, CLI startup, label/gap/codon edge cases, and finite values.
- Expanded CI across Linux, macOS, Windows, and Python 3.10–3.14, with Ruff,
  mypy, coverage enforcement, distribution builds, and wheel-install checks.

## 0.24.0 — 2026-07-21

### Changed

- Standardized long CLI options on `snake_case`, boolean spellings, thread
  handling, default display, and argument help across the public and research
  commands.
- Standardized TSV encoding, LF line endings, header validation, rectangular
  row validation, canonical biological column names, and boolean output.
- Replaced concatenated multi-table reports from `filter`, `trimcodon`,
  `validate`, and `maxalign` with rectangular report schema version 2. Each row
  now starts with `schema_version` and `section`.
- Changed TSV writing to reject undeclared row keys instead of silently
  discarding them.

### Compatibility

- All renamed long options identified by the compatibility audit remain
  accepted and print a deprecation warning containing the replacement option.
- Supported legacy biological input columns, including `kingdom` and
  `partition`, remain readable where they were accepted and print a warning in
  loaders that normalize them to `organism_group` and `fold_id`.
- JSON report formats are unchanged. TSV report consumers should migrate using
  [MIGRATION.md](MIGRATION.md).

### Testing

- Test imports are configured centrally through `pytest.ini`; individual test
  modules no longer mutate `sys.path` or depend on collection order.
- Checkout-local research wrappers now have consistent shebangs, executable
  modes, and import bootstrapping, so they can be invoked directly.
- Added common TSV contract tests for unknown keys, non-mapping rows, report
  schema versions, and conflicting schema declarations.

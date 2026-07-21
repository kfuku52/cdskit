# Changelog

This project follows semantic versioning. Deprecated CLI spellings remain
available for at least the 0.24 release series and print their canonical
replacement to standard error.

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

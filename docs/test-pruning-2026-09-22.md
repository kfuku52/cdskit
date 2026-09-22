# Test review, 2026-09-22

Reviewed all 79 test modules in the unit, integration and ML suites by the failures their assertions
can detect, including parametrized inputs, test doubles and shared fixtures.
The question was: what realistic regression becomes undetectable after removal?
Neither test count nor coverage preservation was a selection criterion.

## Removed or consolidated

| Area | Decision and remaining regression protection |
| --- | --- |
| `stats` | Replace separate field-by-field runs and masked-character helper tests with one mixed-case, multi-record output check. Keep empty-input division and worker equivalence. |
| `split` | Replace 17 overlapping tests with four output-prefix scenarios. Each checks all three files, exact bases, gaps, IDs and order. Drop repeated sequence lengths, slicing examples and filename-helper assertions. |
| `util` | Drop plain FASTA read/write, array conversion and translation wrappers already exercised by commands. Combine GFF field checks into the existing round trip. Keep failed-write rollback, chunk bounds, regex rejection, validation diagnostics and CDS metadata rebasing. |
| `codonutil` | Remove the thin `codon_is_clean` wrapper table. Independent IUPAC expansion, dual-coding semantics, filter, draw and trimcodon tests retain the biological distinctions. |
| `aggregate`, `printseq`, `rmseq` | Drop repeated regex examples, longest-record selection at multiple layers, and helper checks covered by command output. Keep threshold boundaries, case handling, duplicate problematic characters, ID-versus-name behavior, invalid input and combined filters. |
| `mask` | Remove repeated wiki examples, a fixture test with conditional assertions, the helper Boolean truth table and repeated no-change/stop cases. Retain independent options, simultaneous masking, consecutive stops, alternative mask characters and mixed-record output. |
| `backalign`, `backtrim`, `parsegb` | Remove helper/command duplication and repeated content-only checks. Retain genetic-code handling, coordinate rebasing, gaps/case, mismatched IDs, ambiguous mappings, explicit maps and output safety. |
| `gapjust` | Drop six repeated count-only or log-only tests. Existing threaded and threshold tests now assert complete sequences, covering multiple edits and unchanged bases. Keep coordinate, CDS-protection and transaction regressions. |
| `hammer` | Drop weak length-only duplicates and a separate ordering run. Remaining threshold tests assert exact ordered output; keep empty input, invalid alignment and gap-only prevention. |
| `label`, `maxalign` | Remove helper checks duplicating command behavior. Retain stale descriptions, suffix collisions, protected records, solver-specific ties, process fallback and constraints. |
| `pad` | Remove weak minimum-stop/keep-count assertions, repeated X conversion and stop-count examples. Retain saved outputs, head/tail padding, custom padding, no-padding behavior and scalar/fast-path comparisons. |
| CLI and draw | Remove the import-only suite, duplicate legacy-option parsing and CPU-count fallback, and trivial formatting-helper examples. Keep actual CLI execution, deprecated-option warnings, resource allocation, escaped SVG output and validation. |
| Model loading | Remove a copied published checksum/default assertion and the simpler cache-path test subsumed by alias-shadowing protection. Remove private ESM source-resolution tests covered by real loader wiring with offline factories. |
| TargetP | Remove parser-default copies, constant-map membership, feature-shape-only checks and repeated fit-only runs. Combine multiclass/binary runtime checks and retain normalized scores, taxonomy gating, foldwise evaluation and model round trips. |
| Fixtures | Remove four unused FASTA/GFF fixtures after checking references across the suite. |

## Retained after review

- Atomic I/O, output safety and model-download concurrency tests exercise distinct
  failure phases, filesystem aliases, interrupted writes and process locks.
  Their injected failures are inputs to the real transaction code, not tests of
  the mocks themselves.
- Column mapping, codon scan/translation oracles, degeneracy, filter, validate,
  trimcodon and alignment statistics protect sequence meaning and file formats.
  Independent expansion and scalar/vectorized comparisons remain: agreement
  between genuinely different paths is useful regression evidence.
- Localization decisions, scientific contracts, observed targets, evaluation
  boundaries, frozen evaluation and evidence filtering protect against label
  leakage, false certainty and misleading scores. Unknown, negative and absent
  labels are not interchangeable cases.
- Localization training, distillation, pipeline resume/export, pickle migration,
  embedding caches and window/batch invariance exercise different saved artifacts,
  architectures or data partitions. Runtime-cache checks protect bounded memory
  and repeated encoder loading; they are not arbitrary call-count expectations.
- DeepLoc/perox/TargetP benchmark and external-data tests retain calibration,
  overlap exclusion, organism rules, OOF alignment and classifier integration.
  Expensive training assertions remain when they cross a distinct boundary.
- Dependency-audit, coverage-check and CI-matrix tests protect their checks from
  silently reporting success. TSV tests retain malformed-input, UTF-8 and report
  schema guarantees. Plot tests retain separate renderers and output formats.

The concurrent integrity-fix task's new regression tests are preserved. No
production behavior, test selection rules or coverage thresholds were changed
for this cleanup. Execution time is not claimed as a benchmark improvement;
the concrete reduction is fewer redundant runs and less test code to maintain.

## Validation

Relative to the preceding integrity-fix commit, test functions decrease from
873 to 754; collected cases decrease by 128. These counts describe the result,
not a target used to choose deletions.

- Full `scripts/check.py all`: 1,137 passed, one CUDA-only skip; quality checks,
  dependency audit, coverage gates and installed-wheel checks passed.
- Python 3.10 and 3.14 core checks: each 984 passed, two optional YAML skips.
  The YAML cases ran in the full environment.
- Final 0.31.4 build and isolated installed-wheel smoke checks passed.

The full run preceded the metadata-only version bump; both Python boundary
runs and the final package build used 0.31.4. The final consolidation of the
internal-stop X-normalization input was checked again with the pad suite.

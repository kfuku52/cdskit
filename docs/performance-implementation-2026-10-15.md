# Efficiency implementation — 2026-10-15

## Scope and compatibility

Version 0.32.4 implements the nine candidates from the follow-up investigation:
bootstrap F1 aggregation, shared codon arrays, problematic-character counting,
alignment statistics, scalar probability normalization, streamed masking,
provenance hashing, duplicate-column diagnostics and frozen PLM feature reuse.
It builds on the 0.32.3 repository audit; it does not claim that every possible
workload has been optimized.

Public imports and command options, report schemas, genetic codes, unknown-label
handling, seeds, resample counts, thresholds, label order and saved model layouts
are preserved. Arbitrary bootstrap callbacks still receive the original sampled
rows. The optimized metric path requires the established metric, unique labels
and exact binary observed targets; legacy coercions retain the row-based path.
F1 uses the original precision/recall arithmetic and the identical random draws.

Named FASTA, FASTA-2line and tab masking output uses the existing transactional
writer with bounded input chunks. Stdout and other output formats retain whole
input validation. Late invalid input cannot replace an existing named output.
Duplicate-column warnings show the first/last candidate and count above 20
candidates; smaller candidate lists, selected indices, ordering and report counts
retain their meaning. This intentional diagnostic abbreviation is excluded from
sequence/result fingerprints.

## Reproduction and measurements

Baseline: `ab59d5ff4c492401773ca6b348af4eafe4ef5174` (0.32.3).
Current: the 0.32.4 implementation accompanying this report. The fresh-process
runner verifies source imports and hashes complete outputs outside the timed
region. Both snapshots use the same interpreter/dependencies, fixed inputs and
one numerical thread. One warmup precedes five repetitions; the final backtrim,
duplicate mapping, probability and rmseq comparisons use seven repetitions.

```sh
baseline_source=$(mktemp -d)
git archive ab59d5ff4c492401773ca6b348af4eafe4ef5174 | tar -x -C "$baseline_source"
.venvs/full-3.12-cpu/bin/python scripts/benchmark_efficiency_changes.py \
  --source before="$baseline_source" --source after=. \
  --repeats 5 --output /tmp/cdskit-efficiency-comparison.json
```

Host: Apple M2 Max, arm64 macOS, CPython 3.12.14; NumPy 2.5.2,
Biopython 1.88 and matplotlib 3.11.1. The ML environment uses the existing lock.
CPU medians are primary because elapsed samples can be affected by host load.
These synthetic workload measurements are not universal command speedups.

| Workload | Before CPU (s) | After CPU (s) | CPU ratio | Before elapsed (s) | After elapsed (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| DNA statistics: 200 × 10,000 sites | 0.160324 | 0.074203 | 2.16× | 0.174574 | 0.077706 |
| AA statistics: 200 × 10,000 sites | 0.186659 | 0.090364 | 2.07× | 0.208035 | 0.096093 |
| Bootstrap: 10,000 rows, 500 groups, 200 draws | 0.663210 | 0.013128 | 50.52× | 0.692956 | 0.013204 |
| Partial targets: same bootstrap, 30% unknown | 1.955993 | 0.013978 | 139.93× | 2.066709 | 0.014030 |
| Paired: 2,000 rows, 100 groups, 1,000 draws | 5.331220 | 0.039492 | 134.99× | 5.503009 | 0.039667 |
| Codon summary: 100 × 5,000 concrete codons | 0.186240 | 0.015419 | 12.08× | 0.190689 | 0.015513 |
| Codon summary: same size, mixed IUPAC/missing | 0.243571 | 0.016990 | 14.34× | 0.250469 | 0.017865 |
| Codon summary: same mixed input, genetic code 28 | 0.244765 | 0.019320 | 12.67× | 0.252024 | 0.021775 |
| Normalize 10,003 five-class probability rows | 0.169425 | 0.048253 | 3.51× | 0.171269 | 0.049236 |
| Problematic characters: 100 × 100,000 residues | 0.242947 | 0.022803 | 10.65× | 0.243533 | 0.024412 |
| Duplicate mapping: 20 taxa, 2,000 → 1,000 columns | 0.078294 | 0.001973 | 39.68× | 0.078741 | 0.001974 |
| Tiny local ESM: 12 proteins × 12 epochs | 0.256367 | 0.038744 | 6.62× | 0.256035 | 0.039878 |
| Record digest: 2,000 rows, about 40 MB sequence | 0.090393 | 0.085574 | 1.06× | 0.095515 | 0.090025 |
| Dataset digest: same sequence volume | 0.090723 | 0.089256 | 1.02× | 0.095077 | 0.093000 |
| Named FASTA masking: 10,000 × 3,000 residues | 0.973901 | 0.996278 | 0.98× | 1.058609 | 1.189444 |
| Backtrim without report: 200 × 3,000 codons | 0.060144 | 0.059817 | 1.01× | 0.060403 | 0.060091 |

All 16 output fingerprints match across snapshots. Large duplicate diagnostics
are intentionally abbreviated. The PLM cache workload uses scalar window batches
to verify bitwise feature equality; a separate real tiny-ESM regression compares
batched windows with scalar features within `rtol=atol=1e-6` and verifies fewer
forward calls. Synthetic model tests do not establish biological accuracy.

Peak RSS is the entire fresh worker's high-water mark, including imports, inputs,
warmup and output checks; it is not incremental allocation by the changed code.

| Workload | Before peak RSS (MiB) | After peak RSS (MiB) |
| --- | ---: | ---: |
| Record digest: 2,000 rows, about 40 MB sequence | 402.8 | 129.6 |
| Dataset digest: same sequence volume | 351.9 | 80.9 |
| Named FASTA masking: 10,000 × 3,000 residues | 125.4 | 73.8 |
| DNA statistics: 200 × 10,000 sites | 50.9 | 59.3 |
| AA statistics: 200 × 10,000 sites | 50.5 | 59.1 |
| Codon summary: same size, mixed IUPAC/missing | 54.6 | 56.7 |
| Tiny local ESM: 12 proteins × 12 epochs | 384.2 | 384.0 |

Masking trades approximately 2% more CPU in this run for 41% lower peak RSS.
The hash changes primarily reduce memory; the small timing differences are not
claimed as speedups. Ordinary backtrim CPU time is effectively unchanged.
The first duplicate implementation also allocated match objects for unique
columns and slowed ordinary backtrim; constructing those objects only when a
duplicate is encountered removed that overhead. The raw evidence retains the
initial trial and the final seven-repetition comparison.

The [raw evidence](performance-implementation-2026-10-15.json) contains all CPU
and elapsed samples, RSS, output fingerprints, environment metadata and source
hashes. Some measurements preceded the patch-version metadata bump; their
recorded version reflects their execution time.

## Resource bounds and PLM behavior

- Codon summaries use per-row encoded arrays and per-site totals. A bounded cache
  of 16 genetic codes fills lookup entries only for encountered triplets, using
  the scalar codon semantics. Exhaustive IUPAC/lowercase checks cover codes 1, 2,
  27, 28 and 31. Other existing semantic tests still cover every supported code.
- Alignment site counts use blocks of at most 4,096 columns, targeting at most
  one million cells with a minimum width of one column. NumPy reduces Python
  overhead but adds temporary arrays; this accounts for the modest RSS increase.
- JSON hashes retain the exact default JSON list separators, ASCII escaping,
  sorted dataset keys and row order. Temporary serialization scales with one row.
  Kept-site file hashing also reads bounded blocks. Backtrim does not calculate
  unused report hashes; model wrapping reuses its partition hashes locally.
- Duplicate matching uses binary search over ordered original indices. Removed
  choices are all at or before the global last choice; they cannot affect the
  next ordered candidate. Diagnostics avoid enumerating large remaining lists.
- Each PLM encoder owns a 256 MiB LRU for frozen, unlabelled residue arrays, with
  per-entry overhead included. Oversized entries are not retained. Hits return
  independent writable arrays; caller mutation cannot poison the cache. Disk
  cache keys/validation, encoder identity and artifact format remain unchanged.
- Missing PLM windows are bucketed by equal length and processed longest first,
  keeping each sequence's overlap additions in their scalar order. Each forward
  is limited by the caller's batch size, eight windows by default and 8,192
  tokens (with a minimum of one window). No additional padding is introduced.
  Direct encoder configurations accept `memory_cache_bytes=0` to disable the
  LRU and `window_batch_size=1` for scalar forwards. These execution controls
  are not new pipeline configuration-file keys or scientific hyperparameters.

The PLM benchmark creates a tiny local, frozen ESM checkpoint; it downloads
nothing. Its 6.62× CPU ratio measures repeated feature reuse, not large-model
throughput. GPU execution and memory behavior remain unmeasured locally.

## Verification and publication

- `python scripts/check.py core --python 3.10`, `core` (3.12), and
  `core --python 3.14`: each **1,254 passed**, two expected PyYAML skips and
  16 deselected ML tests. The full environment executes the YAML cases.
- `python scripts/check.py coverage`, with numerical threads set to one:
  **1,414 passed**, one CUDA-only skip, **78.36%** combined statement/branch
  coverage. The 74% floor and every critical-module floor passed, including
  CPU ML, pipeline/export, artifact roundtrip and subprocess checks.
- `python scripts/check.py quality`: Ruff lint/format, mypy, complexity and
  high-severity Bandit checks passed.
- `python scripts/check.py build`: fresh sdist/wheel and installed-wheel import,
  CLI, padding, alignment statistics, dN/dS and SVG smoke checks passed.
- New regressions compare bootstrap summaries with an independent row-resampling
  callback, exhaustive scalar codon summaries, hashes with established JSON,
  randomized legacy mapping selection, probability errors across NumPy versions,
  streamed output with buffered stdout, late-input rollback, cache eviction and
  feature ownership. Expected legacy empty-label NaN warnings remain visible.

Dependencies and workflows are unchanged. The external dependency audit in
`all`, pretrained model downloads, biological model evaluation and GPU tests were
not run. Local checks are separate from the GitHub platform matrix after push.

This project distributes a Python CLI/library through GitHub. Per
[RELEASING.md](../RELEASING.md), patch version 0.32.4 is published on `master`
without a numeric tag or GitHub Release. There is no production service deployment,
PyPI publisher or automatic conda publisher in this repository.

# Performance audit — 2026-10-15

## Scope and decisions

The survey covered the structure of all 82 package modules and 33 Python
scripts, the command dispatch and I/O contracts, existing CPU benchmarks,
localization feature/training/evaluation paths, and the check/release workflow.
Targeted source inspection and profiling selected three changes. This is not a
runtime measurement of every command or a biological accuracy evaluation.

| Area | Finding and action |
| --- | --- |
| Degeneracy | Python loops built three lists and sets for every codon column. Replace these with a genetic-code-specific lookup and per-record NumPy reductions. Auxiliary sequence storage now scales with alignment length rather than the full matrix. |
| MaxAlign exact search | Every candidate rescanned every distinct column-support mask. Aggregate support into compact candidate masks and compute superset sums once; retain the existing candidate ranking. |
| TargetP features | Eight composition windows also appeared in the later feature block. A local summary cache was prototyped and measured, then discarded because the modest CPU difference did not replicate consistently. |
| TSV I/O | Header validation called `list.count` for each column. Use one frequency table for duplicate and required-column lookup. Preserve diagnostics and header order. |
| Other sequence processing | Translation and dN/dS already use lookup arrays and batches. Backalignment uses cached codon translations; backtrimming coalesces retained ranges. No further change without a measured bottleneck. |
| Neural inference and training | Existing paths batch inference and cache runtime models/frozen embeddings. Keep encoder identity, masking, saved-model layouts, seeds, thresholds and training settings intact. |
| Statistical evaluation | Cluster resampling and partition audits enforce the scientific contract. No statistical shortcuts or reductions to production resample counts. |
| CLI, atomic output, packaging and CI | Retain lazy command imports, transactional output and the existing locked validation profiles. No dependency or workflow changes. |

## Compatibility and resource bounds

Degeneracy uses the existing forward-table semantics, including dual-coding
genetic codes. Ambiguous/missing codons and ordinary stops remain excluded;
conflicting folds remain unassigned. Only one record's encoded sequence and
per-column extrema are needed at a time. The small lookup cache is bounded.

The MaxAlign count table is limited to 2^20 signed 64-bit entries (8 MiB).
Larger searches and counts that cannot safely use that representation retain
the original arbitrary-integer scan. Protected sequences are accounted for
before aggregation. Candidate restrictions, removal limits and the ordering by
area, number retained and lexicographic indices are unchanged. Dense searches
avoid process startup after aggregation; the original process/thread path is
retained for larger searches. Exact search remains exponential in the number
of variable sequences.

## Reproducing the comparison

Baseline: `df5834ee983a5f69949b3815b87e5d9b15fa6c07` (0.32.2).
Current: the 0.32.3 implementation accompanying this report. Both snapshots use
the same interpreter and installed dependencies. The baseline is an extracted
Git archive, so its automatically detected Git commit is null in the raw data.

```sh
baseline_source=$(mktemp -d)
git archive df5834ee983a5f69949b3815b87e5d9b15fa6c07 | tar -x -C "$baseline_source"
.venvs/core-3.12/bin/python scripts/benchmark_recent_changes.py \
  --source baseline="$baseline_source" --source current=. \
  --cases degeneracy maxalign_exact maxalign_dense tsv_header \
  --repeats 5 --output /tmp/cdskit-performance-comparison.json
```

Inputs use the script's deterministic seed, with one warmup and five measured
runs per snapshot/workload in fresh processes. Snapshot order alternates by
workload. Complete outputs are fingerprinted outside the timed region for
every repetition. The driver fixes numerical thread counts to one and disables
model downloads. These workloads require no pretrained model or optional ML
framework.

The host had substantial concurrent load. The benchmark therefore records
process CPU time as well as elapsed time. Elapsed times are noisy and should
not be interpreted as unloaded latency. Process CPU comparisons and output
equivalence are the primary evidence. Peak RSS is the whole worker's high-water
mark, including input construction and imports, rather than per-operation
allocation. The data do not establish memory improvements from small RSS
differences.

## Measurements

Environment: macOS arm64, Apple M2 Max (12 CPUs), CPython 3.12.14,
NumPy 2.5.2, Biopython 1.88 and matplotlib 3.11.1, using the existing lock.
Times below are medians in seconds; CPU speedup is baseline CPU time divided
by current CPU time. These are synthetic workload results, not universal CLI
speedups.

| Workload | Baseline CPU | Current CPU | CPU speedup | Baseline elapsed | Current elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Degeneracy: 100 records × 10,000 mixed codons | 0.320706 | 0.014880 | 21.55× | 3.042136 | 0.131195 |
| MaxAlign: 17 sequences, 80 sites | 0.363736 | 0.038538 | 9.44× | 3.287540 | 0.040004 |
| MaxAlign: 16 sequences, 4,000 sites | 7.268150 | 0.028242 | 257.35× | 24.843485 | 0.030426 |
| TSV: 2,000 columns, 10 validations | 0.293272 | 0.001531 | 191.56× | 0.824611 | 0.001532 |

All four complete output fingerprints match between snapshots. The CPU
improvements are substantially larger than observed sample variation. Full
samples, RSS measurements, environment metadata and source hashes are in the
[raw benchmark evidence](performance-audit-2026-10-15.json).
The current MaxAlign samples were rerun after finalizing integer handling;
the fixed baseline samples were reused.

### Discarded TargetP experiment

The prototype reused counts, hydropathy sums and hydrophobic runs for overlapping
windows in 1,000 proteins of 500 residues. Current/legacy feature arrays were
bitwise identical. Five-repetition CPU medians suggested 1.11× and 1.10× speedups.
A second run reversed snapshot order and used seven repetitions:

| Feature schema | Baseline CPU median | Prototype CPU median | Ratio |
| --- | ---: | ---: | ---: |
| Current | 0.584430 | 0.576549 | 1.014× |
| Legacy | 0.646904 | 0.574420 | 1.126× |

The current-schema result is within measurement variation. No robust overall
speed improvement is claimed, and the prototype is excluded from 0.32.3.

## Local verification

- `python scripts/check.py core --python 3.10`, `core` (3.12), and
  `core --python 3.14`: each **1,207 passed**, two expected optional-PyYAML
  skips and 16 deselected ML cases.
- `python scripts/check.py coverage`, with numerical thread counts set to one:
  **1,365 passed**, one CUDA-only skip; **78.16%** combined statement/branch
  coverage, above the 74% floor. All critical-module floors passed. This runs
  the PyYAML and CPU ML cases, including subprocess tests.
- `python scripts/check.py quality`: Ruff lint/format, mypy, complexity and
  high-severity Bandit checks passed.
- `python scripts/check.py build`: fresh sdist/wheel build and installed-wheel
  import, CLI, padding, alignment statistics, dN/dS and SVG smoke checks passed.
- Independent new oracles check every supported genetic code and compare
  constrained exact search against direct column enumeration. Tests also cover
  large Python/NumPy integer counts, high sequence bits and empty support.
- A focused final alignment/helper regression run passed 72 tests. Core 3.12
  emitted a non-failing pytest temporary-directory cleanup warning; the tests
  and all delivery profiles exited successfully.

The initial `quick` baseline passed 1,156 tests with two optional-PyYAML skips.
The full profiles above supply the delivery validation. The external dependency audit
in `all` was not run because dependencies were unchanged. Published encoder
downloads, biological model evaluation, GPU execution and the Linux/Windows CI
matrix were not run locally. GitHub CI is a separate validation after push.

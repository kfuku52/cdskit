# Performance follow-up — 2026-10-23

## Decisions and scope

Ten candidates were investigated by ten dedicated subagents. All ten were
adopted after repeated measurements, output comparisons and additional review.
Two returning agents independently reviewed the other agents' implementations;
the parent repeated the principal benchmarks and runs the integrated checks.
The baseline is `31ba31a08d148bdf97c871538424bdb7948031eb` (0.32.5).

| # | Candidate and adopted change | Measured benefit and scope |
| --- | --- | --- |
| 1 | `trimcodon`: compute contiguous kept ranges once for all records | Approximately 2.2× CPU for long runs; 1.16× for alternating kept sites, including command I/O. |
| 2 | `split`: build and release one position's record list at a time | Approximately 44% lower peak RSS for 100,000 short records. CPU is essentially unchanged. |
| 3 | `plot`: share distinct-codon translations and existing codon rows | Approximately 7.6× CPU for MSA preparation; figure rendering is outside this timing. |
| 4 | `gapjust`: prepare edit arrays and prefix sums once for both endpoints | Approximately 1.27× CPU for the coordinate helper. An additional full FASTA/GFF comparison preserves both outputs. |
| 5 | `validate`: bounded ASCII ambiguity scan and allocation-free gap search | Approximately 50× CPU for the legacy ambiguity helper and 5.9× for long all-gap detection. The modern codon-summary algorithm is unchanged. |
| 6 | `backalign`: reuse uppercase slices and avoid exceptional checks on ordinary matches | Approximately 1.30× CPU for uppercase CDS. Case preservation, terminal stops and first-error diagnostics are unchanged. |
| 7 | `stats`: byte operations for short ASCII records and bounded NumPy reductions for long records | Approximately 6.1× CPU for the mixed sequence-count workload; large-record RSS is also reduced. |
| 8 | Calibration bins: share sparse selection indices between the two gathers | Approximately 1.8× CPU for one million values and ten bins; exact floating-point outputs are retained. |
| 9 | Grouped folds: count string labels once per group | Approximately 1.13× CPU with five labels and 2.2× with twenty. Every fold and RNG state is preserved. |
| 10 | Alphabet validation: scan a valid prefix and collect invalid characters in bounded suffix blocks | 5.53× CPU for DNA and 8.25× for protein validation; invalid-character diagnostics are unchanged. |

These are workload-specific CPU ratios, not universal CLI latency claims. The
machine has substantial concurrent load. Process CPU time and interleaved
comparisons provide the primary timing evidence; elapsed times remain in the raw
results. No dependencies, workflow settings, model weights, scientific
thresholds, splits, seeds, feature schemas or saved-model formats changed.
This is a local implementation commit; the package version remains 0.32.5.

The parent measurements below use eighteen interleaved samples per implementation.
Times are median process CPU milliseconds, including command I/O where stated.

| Workload | Before ms | After ms | CPU ratio |
| --- | ---: | ---: | ---: |
| Trim command: 100 × 10,000 codons, long kept runs | 131.696 | 59.270 | 2.22× |
| Trim command: same size, alternating kept sites | 94.628 | 81.711 | 1.16× |
| Split command: 100 × 300,000 nt, three outputs | 292.515 | 289.490 | 1.01× |
| MSA preparation: 100 × 1,000 codons, code 28 | 175.419 | 23.117 | 7.59× |
| Coordinate helper: 10,000 intervals and edits | 4.081 | 3.225 | 1.27× |
| Legacy ambiguity count: 3,300,000 nt | 124.073 | 2.473 | 50.18× |
| Gap-only predicate: 3,500,000 nt | 88.930 | 15.156 | 5.87× |
| Backalignment: 90,000 nt → 50,000 aligned AA columns | 6.419 | 4.957 | 1.29× |
| Sequence counts: 100 × 150,000 mixed nt | 58.138 | 9.471 | 6.14× |
| Calibration: 100,000 × 10 values, ten bins | 46.497 | 25.880 | 1.80× |
| Group folds: 100,000 rows, 2,000 groups, five labels | 59.794 | 53.085 | 1.13× |
| Group folds: same size, twenty labels | 122.066 | 55.234 | 2.21× |
| DNA validation: 100 × 100,000 residues | 64.824 | 11.719 | 5.53× |
| Protein validation: 100 × 100,000 residues | 87.356 | 10.594 | 8.25× |

Fresh-worker whole-process high-water RSS, including the measurement harness:

| Workload | Before MiB | After MiB | Reduction |
| --- | ---: | ---: | ---: |
| Split: 100,000 × 300 nt | 404.6 | 227.2 | 43.8% |
| Split: 100 × 300,000 nt | 142.5 | 125.3 | 12.0% |
| Sequence counts: one 30,000,000-nt record | 199.7 | 130.7 | 34.6% |

## Compatibility and resource bounds

- The public single-record trimming and splitting helpers retain their existing
  behavior. The command-specific range optimization only receives sorted kept
  sites. Unusual direct trimming inputs retain their original slicing semantics.
- Splitting retains the input record list and one projected list, rather than
  three projected lists. All three output files still commit or roll back
  together. FASTA, GenBank and alignment formats, quality errors, metadata,
  output bytes, stderr and input/output collision checks are covered.
- MSA translation still uses the existing genetic-code-aware helper, including
  codes 27/28/31. Consensus tie order, logo order, partial and unequal direct
  inputs, and the resulting SVG are unchanged. The cache is local to one summary.
- Coordinate preparation retains range-aware deleted-endpoint rejection and the
  legacy sequential path for non-monotonic edits. Values, dtype, shape, warnings,
  exceptions and input ownership were compared against the baseline.
- The legacy ambiguity helper uses its original scalar path below 384 nt and
  for non-ASCII or non-exact-string inputs. Its vector path processes at most
  786,432 nt per block. Missing codons take precedence over ambiguity, and
  partial tails remain excluded. Gap detection uses a simple negated character
  search. An initial `lstrip` prototype was discarded because a short gap prefix
  could force a copy of a long clean suffix.
- Sequence statistics retain Unicode uppercase behavior and custom string
  behavior on the original path. ASCII records below 8,192 nt use byte methods;
  larger records use blocks of 65,536 bytes. Full-record string/byte encoding and
  input storage still scale with record length; temporary comparisons are bounded.
  A full-size `bincount` prototype was discarded because of its larger allocation.
- Calibration means consume the same elements in the same C order. Sparse-bin
  index storage is limited to the original boolean mask's byte size. Dense bins,
  non-contiguous inputs and unusual shapes retain boolean selection. No rounded
  comparison or changed floating-point accumulation conceals output differences.
- Group counts use `Counter` only for at least four distinct labels whose exact
  types are `str` or `numpy.str_`. One to three classes, NaN, numeric labels and
  custom equality retain the original comparisons. The four-class gate is an
  execution choice supported by crossover measurements, not a scientific rule.
- Alphabet scans retain the complete sorted set of invalid characters, record
  order, duplicate IDs, the ten-ID display limit and all three alphabet modes.
  Invalid suffix slices are at most 65,536 characters. Independent review found
  a difference for strings with overridden length/iteration/slicing returned by
  `str(record.seq)`; an exact-string guard and nine regression cases restore the
  original set-based behavior for those subclasses.

Small or unfavorable workloads do not all improve. Empty statistics add tens of
nanoseconds, twenty-row fold assignment can add several microseconds, and
entirely invalid alphabet inputs cost approximately 2–3% more CPU in the detailed
agent comparison. Lowercase all-wildcard backalignment varied within about 3%
across comparisons. These tradeoffs are retained explicitly; the principal
benefits concern larger ordinary inputs. No timing assertions were added to tests.

## Reproduction and evidence

Use the same interpreter and dependencies for both source snapshots:

```sh
baseline_source=$(mktemp -d)
git archive 31ba31a08d148bdf97c871538424bdb7948031eb | tar -x -C "$baseline_source"
.venvs/core-3.12/bin/python scripts/benchmark_followup_changes.py \
  --source before="$baseline_source" --source after=. \
  --repeats 7 --output /tmp/cdskit-followup-fresh.json
.venvs/core-3.12/bin/python scripts/benchmark_followup_paired.py \
  --before "$baseline_source" --after . --rounds 9 \
  --cases trimcodon_runs trimcodon_fragmented split_short split_long plot_msa \
    gap_coordinates validate_ambiguity validate_gap_only backalign stats_counts \
    reliability_bins group_folds alphabet_dna alphabet_protein \
  --output /tmp/cdskit-followup-paired.json
```

The fresh runner verifies snapshot imports, warms once and records repeated CPU,
elapsed, process peak RSS, environment metadata and complete output fingerprints.
It rejects different fingerprints between snapshots. The paired runner loads
each implementation module against the same baseline dependencies, alternates
ABBA/BAAB order and verifies complete fingerprints after each timed sample.
Nine rounds yield eighteen samples per implementation. Short operations are
batched to target at least 20 ms of baseline CPU time, subject to a 1,000-call cap.

Both use deterministic synthetic inputs and fixed numerical thread counts;
neither downloads models. `split_many` is mainly a memory workload and is omitted
from the long paired command above. Additional selectable cases reproduce
lowercase wildcards, empty statistics, dense bins, short fold assignments and
twenty-label folds. The fresh memory comparison used three timed repetitions
inside one fresh worker per snapshot/case for `split_many`, `split_long` and
`stats_long`; other fresh cases used seven. Supplementary agent runs also
measured independent workers.

Host: Apple M2 Max, arm64 macOS, CPython 3.12.15, NumPy 2.5.2, Biopython 1.88 and
matplotlib 3.11.1. The [raw evidence](performance-followup-2026-10-23.json)
contains complete primary samples, memory measurements, environment and source
hashes, supplementary agent measurements, independent-review results and check
outcomes. Peak RSS is the complete fresh worker's high-water mark, including
imports, input construction, warmup and fingerprinting; it is not incremental
allocation by the optimized function. The paired process does not attribute RSS
separately to the two implementations.

## Local verification

- `python scripts/check.py core --python 3.10`, `core` (3.12), and
  `core --python 3.14`: each **2,171 passed**, two optional-PyYAML skips and
  sixteen deselected ML tests. These include compilation of package/scripts and
  CLI subprocess checks after the final string-subclass fix.
- `python scripts/check.py quality`: ML dependency imports, Ruff lint/format,
  mypy (84 source files), the complexity ceiling and high-severity Bandit passed.
- `python scripts/check.py coverage`: **2,334 passed**, one CUDA-only skip,
  **78.93%** combined statement/branch coverage. The 74% project floor and every
  critical-module floor passed. This executes the YAML cases, CPU ML,
  pipeline/export, model roundtrips and subprocess tests.
- Nineteen paired cases and twenty-one complete-snapshot fresh cases have matching
  output fingerprints. Paired module hashes match the final implementations.
- `git diff --check`, local report links and the final changed-file review passed.

Independent baseline comparisons include 452,505 backalignment outcomes across
all 27 genetic codes, 3,282 coordinate cases, all IUPAC triplets for MSA summaries,
and explicit bin boundaries, Unicode, partial sequences, metadata and rollback
cases. Additional reviewers independently compared 19,024 sequence/plot/output
cases, 3,360 calibration cases, 400 fold cases, 1,140 alphabet cases and eighteen
string-subclass cases after the compatibility fix.

These are local validations. GitHub CI, GPU execution, biological model accuracy,
published model downloads, packaging builds and the external dependency audit
were not run for this local implementation task.

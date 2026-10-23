# Additional performance improvements — 2026-10-23

## Decisions

Five additional candidates were investigated against
`f5810dc9c35c7ce315558fbd79d6801735f86afb` (0.32.5). Four were adopted after
repeated measurements, complete-output comparisons and independent review.
Three subagents investigated mask, labels/padding and reports; returning agents
reviewed other implementations, and the parent repeated the principal benchmarks
and ran the integrated checks. The previous ten-change report is unchanged.

| Candidate | Decision | Result on the measured workload |
| --- | --- | --- |
| Shared codon summaries: reuse the existing bounded histogram and scan backward in bounded blocks for the last evaluable codon | Adopt | Approximately 10–14× CPU with NumPy already loaded, for about 300,000 nt. |
| Mask: byte-valued gap comparisons, adjacent-position flags and reuse of the encoded input | Adopt | Approximately 2.0–3.6× CPU for 900,000 nt. |
| Clipped labels: share suffix ranges already proven reserved across colliding original IDs | Adopt | Approximately 73× helper CPU for 2,400 IDs; 3.0× including CLI startup and FASTA I/O. |
| Pad: bounded stop-pattern lookup for long inputs, retaining string scans for short inputs | Adopt | Approximately 9.6–23× helper CPU for 300,000 nt; 1.2–3.6× in the long-input CLI cases. |
| Stream sectioned TSV rows instead of building the complete output row list | Reject | Prototype peak RSS fell 20–52%, but independent review found differences in public helper behavior. All prototype source changes were reverted. |

These ratios describe the named synthetic workloads. They are not universal CLI
latency claims. No dependency, version, scientific parameter, genetic-code rule,
report schema or output transaction changed. This is a local implementation
commit on `master`; there is no push or version bump.

## Comparable measurements

The principal comparison uses eighteen interleaved ABBA/BAAB samples per
implementation after warmup. Times below are median process CPU milliseconds.
Both implementations use the same baseline dependencies, and every timed sample
has the same complete output fingerprint. Final implementation hashes were
checked against the measured modules.

| Workload | Before ms | After ms | CPU ratio |
| --- | ---: | ---: | ---: |
| Summary: 299,997 concrete nt | 10.652 | 0.835 | 12.76× |
| Summary: 300,006 mixed nt, code 1 | 10.645 | 0.788 | 13.51× |
| Summary: same mixed input, code 28 | 10.700 | 0.782 | 13.69× |
| Summary: 300,006 nt with a long missing suffix | 25.705 | 2.517 | 10.21× |
| Mask: 900,000 mixed nt, ambiguity and stop masking | 8.895 | 4.387 | 2.03× |
| Mask: 900,000 gappy nt, gap masking only | 7.028 | 2.278 | 3.08× |
| Mask: 900,000 clean nt, ambiguity and stop masking | 5.912 | 1.651 | 3.58× |
| Labels: 2,400 clipped IDs with shared-prefix suffix collisions | 295.634 | 4.035 | 73.26× |
| Labels: 2,400 unique IDs | 0.423 | 0.411 | 1.03× |
| Pad: 300,000 random A/C/G/T nt, code 1 | 5.572 | 0.583 | 9.56× |
| Pad: 300,006 stop-rich nt, code 1 | 14.703 | 0.634 | 23.21× |
| Pad: 299,997 nt with no frame-zero stops, code 1 | 6.355 | 0.572 | 11.11× |

Small summaries (90 nt), partial-tail scalar summaries and code-27 padding
remain approximately neutral in the paired comparison (ratios 1.01×, 0.98× and
0.99×). The mask's 36-nt paired case is 1.13×. Label measurements include resetting
IDs and descriptions between calls, equally for both implementations.

Fresh-process CLI comparisons additionally include imports, sequence I/O, logs
and process overhead. All output, stdout and stderr fingerprints match.

| CLI workload | CPU ms before → after | Wall ms before → after |
| --- | ---: | ---: |
| Labels: 2,400 records, 300 nt each | 465.4 → 154.0 | 546.4 → 208.6 |
| Pad: ten 300,000-nt records, stop-rich | 668.2 → 184.2 | 1503.6 → 312.8 |
| Pad: ten 300,000-nt records, random | 345.1 → 182.3 | 837.1 → 392.5 |
| Pad: ten 300,000-nt records, rare stops | 195.3 → 163.0 | 284.9 → 216.0 |

### Cold imports, memory and small-input costs

NumPy's first import made the initial summary prototype unsuitable for small
direct-library calls: approximately 35 ms could replace a sub-millisecond
summary. The final implementation uses the histogram only for exact strings,
complete codon lengths and at least 6,144 nt when NumPy is loaded. When it is not
loaded, the threshold is 3,000,000 nt. Invalid alphabets retain the scalar path.

Forty-eight fresh processes tested both concrete and mixed inputs at 6,144,
30,000, 300,000 and 3,000,000 nt, with NumPy absent before the call. At 3,000,000 nt,
CPU fell from 103.4 to 38.0 ms for concrete input and 117.6 to 41.0 ms for mixed
input, including NumPy import. Smaller inputs did not import NumPy. A 300,000-nt
mixed-input slowdown in the initial three-sample set did not reproduce in a
quiet ABBA/BAAB comparison with nine fresh processes per implementation: concrete
CPU was 10.70 → 10.33 ms and mixed CPU 10.79 → 10.74 ms, with overlapping ranges.

Whole-process high-water RSS includes input construction, imports, warmup,
fingerprinting and allocator retention; it is not a function's incremental
allocation. The summary trades additional NumPy/table memory for CPU: the warmed
300,000-nt workers used about 35.5 → 39.4 MiB, and the cold 3,000,000-nt concrete
case used 28.9 → 40.7 MiB. The mixed cold case used 26.0 → 39.0 MiB.

The independent 12,000,000-nt mask comparison reduced peak RSS by approximately
7.5–10.3% when translation or output replacement dominated. Gap-only clean cases
were approximately unchanged. The parent's smaller mixed-mask final fresh worker
used 59.4 → 55.9 MiB; its gap-only worker fell from 54.5 to 50.7 MiB. Earlier
worker measurements varied, so there is no universal mask RSS reduction claim.
Pad's 3,000,000-nt
lookup worker used approximately 0.72 MiB more RSS; lookup temporary arrays are
bounded by the block size.

Some small workloads add overhead. First-call clean masking of twelve codons,
with translation disabled, took 68 → 77 microseconds CPU (64.1 → 73.7 microseconds
wall). A 20,000-ID positive-clip unique-label comparison took 4.25 → 4.54 ms CPU;
the 2,400-ID parent case was neutral. Tiny CLI startup varied: the 48-record
label CLI was 123.2 → 138.7 ms CPU, while the 20-record short pad CLI was
149.9 → 156.4 ms. No small-input CLI speedup is claimed. Elapsed time is noisier
than process CPU under this host's concurrent load.

## Compatibility and resource bounds

- Shared summaries retain the independent ambiguity, definite-stop,
  possible-stop and context-dependent counts, Python result types and key order.
  Codes 27/28/31, selected code names/IDs, `physical` and `last_evaluable`, trailing
  missing codons and partial tails were compared. The existing histogram has
  bounded temporaries; backward lookup handles at most 196,608 nt per block.
  Non-exact strings retain the original scalar behavior.
- Mask retains partial-gap rules, all-gap codons, both masking switches, case,
  replacement semantics and error behavior. The ASCII encoding is reused for
  output, and the full gap-position temporary is released before translation.
  Independent review found that reusing the encoding changed stateful string
  subclasses. They retain the original second encode call, including its error
  precedence; two added regression cases cover this behavior.
- Label suffix jumps apply only to exact string IDs and exact integer clip
  lengths when multiple originals are duplicated. The cache key includes the
  clipped base and suffix width. Every skipped suffix is already reserved;
  original order, reserved singleton IDs, width transitions, exhaustion errors
  and mutations before an error are preserved. Independent review found custom
  slicing/equality and integer-subclass differences in an early prototype;
  guards and four regression cases restore their original suffix searches.
- Pad still obtains definite stop patterns through the selected genetic-code
  semantics. The bounded LRU has at most 128 pattern tables, each with a 32-KiB
  lookup. Sequence blocks contain at most 65,535 nt. Short inputs and unusual
  string subclasses retain string scans. Only frame-zero complete triplets
  count; only a complete physical terminal codon is exempt, and a partial tail
  exempts none. Uppercase handling, non-ASCII input, padding candidates, tie
  order, selected indexes and complete reports match the baseline.
- The rejected TSV prototype preserved ordinary files and reduced row-list RAM,
  but streaming changed observation order for serializable custom scalars that
  mutate another row. Extra path comparisons also called a custom `__ne__` that
  the old helper never called. Recursive guards or a new streaming API would add
  scope and complexity. `codonreport.py`, `filter.py` and `trimcodon.py` remain
  byte-for-byte equal to the baseline.

Final independent comparisons include 5,140 summary cases; 3,492 grouped mask
comparisons covering all 6,859 triplets over the 19-symbol alphabet, all 27 codes,
case and masking switches; and 4,320 complete padding-result dictionaries plus
648 threshold/block cases. A separate instrumented label review checked 3,020
cases and verified every one of 7,580,484 skipped suffixes was reserved. Final
label comparisons additionally cover ordered IDs, metadata, errors and partial
mutation. These establish software equivalence, not biological accuracy.

## Reproduction and evidence

Use one interpreter and dependency environment for both source snapshots:

```sh
baseline_source=$(mktemp -d)
git archive f5810dc9c35c7ce315558fbd79d6801735f86afb | tar -x -C "$baseline_source"
.venvs/core-3.12/bin/python scripts/benchmark_more_changes.py \
  --paired --before "$baseline_source" --after . --rounds 9 \
  --output /tmp/cdskit-additional-paired.json
.venvs/core-3.12/bin/python scripts/benchmark_more_changes.py \
  --source before="$baseline_source" --source after=. --repeats 7 \
  --output /tmp/cdskit-additional-fresh.json
```

Both modes use deterministic inputs, fixed numerical thread counts and complete
output fingerprints. The fresh mode rejects cross-snapshot differences and
records per-case CPU, wall time and peak RSS after one warmup and seven timed
repetitions. The paired mode alternates ABBA/BAAB, uses eighteen samples per
implementation, and batches short operations toward 20 ms baseline CPU with a
1,000-call cap. These runners import NumPy before summary timing; cold direct
summary measurements use separate workers without NumPy loaded.

Host: Apple M2 Max, arm64 macOS, CPython 3.12.15, NumPy 2.5.2, Biopython 1.88 and
matplotlib 3.11.1. The [raw evidence](performance-additional-2026-10-23.json)
contains principal samples, environment, commands, fingerprints, final source
hashes, supplementary CLI/memory/cold samples and review results. It also retains
the standalone cold worker source for reproduction, and summaries of rejected
prototype measurements. Summary benchmarks were repeated after the final import
gate, and mask benchmarks after the string-subclass guard. Supplementary mask
cold-start samples precede that final guard and are labelled with their source
hash. Retained label/pad hashes are unchanged.

## Local verification

- `python scripts/check.py core --python 3.10`, `core --python 3.12`, and
  `core --python 3.14`: each **2,572 passed**, two optional-PyYAML skips,
  sixteen deselected ML tests and nine warnings. These include compilation and
  CLI subprocess coverage.
- `python scripts/check.py quality`: ML imports, Ruff lint/format (230 files),
  mypy (84 source files), the complexity ceiling and high-severity Bandit passed.
- `python scripts/check.py coverage`: **2,735 passed**, one CUDA-only skip,
  thirteen warnings, **79.10%** combined statement/branch coverage. The 74%
  project floor and every critical-module floor passed. This includes CPU ML,
  YAML configuration, pipeline/export and subprocess tests.
- All sixteen principal paired and fresh workloads have matching complete output
  fingerprints. Final source hashes match the paired modules. No timing
  assertions were added; the four new test files add 401 behavior cases.
- `git diff --check`, local report links and the final file-list review passed.

Warnings include existing Biopython ambiguity, weak-label, legacy scikit-learn
artifact-version and NumPy empty-slice warnings. These are local macOS results;
GitHub CI, GPU execution, biological model evaluation, published model downloads,
packaging builds and the external dependency audit were not run for this local
implementation change.

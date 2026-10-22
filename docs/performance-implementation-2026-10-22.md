# Sequence performance implementation — 2026-10-22

## Scope and contracts

Version 0.32.5 adopts seven of the ten investigated candidates: compact/lazy
translation, hammer membership lookups, codon-statistics histograms, batched
dN/dS contractions, bounded MaxAlign subset evaluation, bounded ORF position
scans and BiLSTM inference mode. TargetP prefix changes, a lower-level FASTA
parser and padded PLM window buckets were not adopted: their measured benefits
did not justify compatibility risks or regressions. Existing equal-length PLM
batching remains unchanged.

Each accepted candidate has a separate implementation commit and regression
tests. Public imports, CLI options, atomic output, report schemas, genetic-code
selection and tie handling remain unchanged. No thresholds governing biological
decisions, model parameters, data splits, seeds, encoder revisions or saved-model
formats were retuned. Dependency ranges and workflows are unchanged. A targeted
security refresh of the lock is described below.

Translation initializes 64 concrete triplets and resolves encountered ambiguous
triplets through the authoritative scalar translator. The public lookup getter
still returns its complete five-field mutable dictionary. Explicit public lookup
use retains the original int32 indexing arithmetic, including customized table
metadata. Private fixed-alphabet indices fit in uint8 or uint16. Already resolved
amino-acid arrays are reduced directly before allocating a missing-entry mask.

Codon statistics use the shared scalar-backed codon features, preserving missing,
ambiguity, possible-stop and dual-coding overlap, positional GC denominators and
Counter insertion order/types. Records below 256 codons retain scalar handling;
invalid, partial and Unicode direct inputs retain their established path. The
256 boundary was selected conservatively from concrete/mixed length sweeps for
codes 1, 27, 28 and 31; it is a computational choice, not a scientific threshold.

MaxAlign scores at most 32,768 subsets per block, keeping the existing exact
objective and lexicographic index-list tie rule. Python integer checks precede
int64 multiplication; unusual direct API inputs retain scalar arithmetic and
errors. Dense count tables still use up to 8 MiB and the search remains
exponential. ORF scans use at most 65,538 encoded bytes per block, including two
overlap bases. Only canonical cached patterns, at least 16,384 nt and at least
eight start/definite-stop patterns select the vector path. Other patterns and
string subclasses retain literal `str.find` behavior. Codes 27/28/31 continue to
distinguish ordinary translation from complete-CDS termination.

BiLSTM inference mode surrounds only prediction forwards. Model construction,
cached tensors and training stay outside it. CPU scores were bitwise identical
across tested feature-fusion, model-size and thread configurations. Published
BiLSTM checkpoints were not available locally; local training/save/load and
safe-v1 gates were tested. These synthetic tests do not establish biological
accuracy, calibration or GPU performance.

## Measurements and reproduction

Baseline: `bb21450972cf8d3c5d1a1766deb05ec1c45f7324` (0.32.4).
The benchmark runner verifies snapshot imports, uses deterministic synthetic
inputs, warms once and compares complete output fingerprints outside timing.
Each case runs in a fresh process with nine timed repetitions. Numerical thread
counts are one; models are generated locally and no model is downloaded.

```sh
baseline_source=$(mktemp -d)
git archive bb21450972cf8d3c5d1a1766deb05ec1c45f7324 | tar -x -C "$baseline_source"
.venvs/full-3.12-cpu/bin/python scripts/benchmark_sequence_kernels.py \
  --source before="$baseline_source" --source after=. \
  --repeats 9 --output /tmp/cdskit-sequence-comparison.json
```

Host: Apple M2 Max, arm64 macOS 27.0.1, CPython 3.12.15, NumPy 2.5.2,
Biopython 1.88, PyTorch 2.13. CPU time is primary; elapsed time and process peak
RSS are also recorded. Peak RSS includes imports, input construction, warmup and
fingerprinting; it is not incremental allocation by the changed function.

| Workload | Before CPU ms | After CPU ms | CPU ratio | Before wall ms | After wall ms | Peak MiB before/after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hammer, 500 × 5,000 codons | 162.070 | 104.856 | 1.55× | 165.102 | 105.963 | 148.2/121.7 |
| Hammer, same, no gap guard | 85.847 | 95.441 | 0.90× | 87.872 | 96.451 | 148.1/121.9 |
| Translate, 1,000 × 90 nt | 7.296 | 3.875 | 1.88× | 7.305 | 3.876 | 42.9/42.5 |
| Translate, 299,997 concrete nt | 0.553 | 0.361 | 1.53× | 0.554 | 0.363 | 45.9/43.3 |
| Translate, 90,000 mixed nt | 0.168 | 0.173 | 0.97× | 0.169 | 0.173 | 43.4/44.2 |
| Codonstats, 300 × 90 nt | 6.837 | 6.499 | 1.05× | 6.878 | 6.503 | 42.6/42.2 |
| Codonstats, 100,000 concrete codons | 79.522 | 0.686 | 115.92× | 79.936 | 0.687 | 43.4/46.8 |
| Codonstats, 30 × 1,002 mixed codons | 17.763 | 1.607 | 11.05× | 17.855 | 1.610 | 42.4/44.2 |
| MaxAlign, 16 sequences × 80 sites | 16.926 | 8.414 | 2.01× | 17.050 | 8.417 | 47.0/48.8 |
| MaxAlign, 20 sequences × 80 sites | 274.671 | 135.981 | 2.02× | 275.072 | 136.245 | 90.7/92.5 |
| ORF, 32 × 150 nt/code 11 | 0.302 | 0.320 | 0.94× | 0.303 | 0.320 | 43.0/42.9 |
| ORF, 300,000 nt/code 11 | 8.207 | 4.054 | 2.02× | 8.230 | 4.056 | 47.9/49.0 |
| ORF, 300,000 nt/code 28 | 1.174 | 1.190 | 0.99× | 1.174 | 1.198 | 47.4/47.8 |
| dN/dS, 256 pairs × 300 codons | 9.002 | 6.360 | 1.42× | 9.084 | 6.371 | 77.0/75.9 |
| BiLSTM, 1 × 64 aa, hidden 128/layers 2 | 3.284 | 3.144 | 1.04× | 3.286 | 3.147 | 202.5/201.1 |

Fourteen integrated output fingerprints match exactly. dN/dS differs only in
floating-point values, as discussed below. Hammer includes FASTA parsing and
atomic output writes but excludes interpreter startup; other rows measure the
functions named in the runner. Codonstats CLI follow-ups including startup/I/O
showed only 1.11–1.12× for 30 records of 1,000 codons and 6.19–6.34× for a
million-codon record. The 115.92× warmed aggregation ratio is not a CLI speedup.
The [raw evidence](performance-implementation-2026-10-22.json) retains all
samples, workload definitions, environment metadata, fingerprints and source
hashes. Measurements preceded the security lock refresh and final metadata
commit; both snapshots used the same installed dependencies.

Process-block timing can vary substantially on this shared host. An initial ORF
comparison showed 7.912 to 9.343 ms CPU despite an earlier 4.1 ms after sample.
An exact-input follow-up alternated baseline/current ABBA/BAAB within three fresh
processes, using longer batches. All 36 paired rounds improved: process medians
were 8.856 to 4.427, 8.550 to 4.647 and 9.088 to 4.626 ms CPU (1.84–2.00×).
Output hashes matched the integrated run, and no cumulative slowing recurred.
Profiling independently confirmed less time collecting positions.

Fresh-process hammer blocks without the gap guard appeared 9–10% slower,
including the final table. Three fresh processes with paired ABBA/BAAB runs of
the final source found CPU
83.947 to 81.918, 92.106 to 87.733 and 91.174 to 88.835 ms. Elapsed medians
improved 3.0–4.5%; complete FASTA and stderr fingerprints matched. These controls
support adoption, together with the lower RSS and faster guarded case, but do
not identify a hardware scheduling or frequency cause,
which was not directly measured. Small cross-process differences are not treated
as universal speedups or regressions.

## Memory bounds and numerical limits

The original vector prototypes allocated arrays proportional to the full
candidate/sequence count. Follow-up implementations bound their temporary work:

| Workload | Baseline peak MiB | Full-vector prototype MiB | Adopted blocks MiB |
| --- | ---: | ---: | ---: |
| MaxAlign, 20 variables, NumPy 2.5/Accelerate | 92.98 | 112.81 | 94.59 |
| MaxAlign, 20 variables, NumPy 1.23/OpenBLAS | 95.38 | 116.20 | 96.95 |
| ORF selection, 3M concrete nt/code 11 | 114.28 | 136.23 | 118.42 |
| ORF selection, 3M start/stop-rich nt/code 11 | 121.69 | 149.05 | 127.22 |
| Codonstats CLI, 1M concrete codons | 53.6 | 71.7 | 56.6 |

MaxAlign RSS used one fresh sample per method/environment; ORF 3M values are
three-sample medians. A 12M-nt concrete ORF selection also used 333.92 baseline
versus 335.69 MiB adopted, compared with 409.56 MiB for the unbounded prototype
(one sample each). Returned positions, sequences, count tables and input parsing
still consume size-dependent memory; total memory is not constant.

dN/dS changes floating-point summation order. On NumPy 1.23.5/OpenBLAS 0.3.20
and NumPy 2.5.2/Apple Accelerate, 9,288 pairs per backend across all 27 codes,
plus correction-boundary sweeps, preserved status, kappa, model and NA decisions.
The largest observed distance delta was 2.19e-12; comparisons use absolute
1e-10/relative 1e-12 tolerances. Existing independent PAML references retain
their original 0.000051 tolerance. TSV numeric cells can differ in the last
digits because the writer emits raw floats; no rounding was added to conceal
differences. Four-, six- and eight-decimal diagnostic comparisons agreed in the
tested reports, but ten-decimal comparisons sometimes differed.

The contraction follow-up measured CPU ratios of 1.26–1.42× for 256/4,096-pair
single-thread workloads on both backends. Small batches were rechecked with
longer loops and showed no material regression. Four-worker elapsed gains were
only 1.01–1.05×; no broad parallel-speedup claim is made. BiLSTM gains likewise
depend on model/thread configuration; one large four-thread case was slightly
slower within noise. GPU execution and aggregate parallel-worker RSS remain
unmeasured locally.

## Verification

- `python scripts/check.py core --python 3.10`, `core` (3.12) and
  `core --python 3.14`, after the lock refresh: each **1,448 passed**, two
  expected optional-PyYAML skips and 16 deselected ML tests. Full checks execute
  the YAML cases.
- `python scripts/check.py all`, after fixing the translation dictionary's
  missing type annotation and refreshing the security lock: **1,611 passed**,
  one CUDA-only skip, **78.6%** combined statement/branch coverage. Ruff lint
  and format, mypy, complexity, high-severity Bandit and every critical coverage
  floor passed. The audit found no vulnerabilities in its 85-package Python
  3.12 environment. Fresh sdist/wheel builds and installed-wheel import, CLI,
  padding, alignment statistics, dN/dS and SVG smoke checks passed.
- The isolated Python 3.11.17/NumPy 1.23.5/Biopython 1.85 environment passed
  **262 tests**, with no skips, covering the six changed sequence kernels and
  independent codon semantics. Both BLAS backends also passed 142 focused
  dN/dS tests, including the existing PAML references.
- `uv lock --check`, `git diff --check` and the outgoing file/commit review
  passed. These are local results; GitHub's platform matrix runs after push.

No timing assertions were added to tests. GPU execution, full pretrained-model
biological evaluation and aggregate parallel-worker RSS were not run locally.
Patch version 0.32.5 is distributed from `master` without a numeric release tag,
according to [the release procedure](../RELEASING.md).

The first full dependency audit found advisories in Pillow 12.3 and the certifi
2026.7.22 certificate bundle. A targeted lock refresh updates only those package
versions to Pillow 13.0.0 (Python 3.11+) and certifi 2026.10.19; the resolver also
normalizes lock markers. Package minimums and supported Python versions are
unchanged. [Pillow's advisory](https://github.com/python-pillow/Pillow/security/advisories/GHSA-4rr5-89j8-mwwm)
identifies 13.0.0 as patched, and
[certifi's advisory](https://github.com/certifi/python-certifi/security/advisories/GHSA-6q48-6c7w-9gw2)
describes the compromised emSign roots removed from the bundle.

Pillow 13 [requires Python 3.11](https://pypi.org/project/pillow/13.0.0/).
Python 3.10 therefore retains its existing Pillow 12.3 resolution and its known
upstream advisories. The clean dependency audit applies to the full Python 3.12
environment, not that legacy resolution. Python 3.11+ is needed for the patched
Pillow release; dropping CDSKIT's Python 3.10 compatibility is outside this patch.

Independent expansion tests cover every supported triplet in all 27 codes.
Translation matched the archived baseline for 741,852 result/exception checks.
Shared codon features matched 1,111,158 triplet rows across numeric/string/name
code keys, sequence order and case, including dtype, shape and bytes. MaxAlign
direct API comparisons covered 312 cases per NumPy backend with no result or
exception mismatch. New regressions also exercise block boundaries, missing GC
denominators, invalid/partial/Unicode input, public-table mutation, tie handling,
large integers, cached-tensor backward use and saved-model roundtrips.

## Primary references

Algorithm ideas and API semantics informed the investigation; all speed claims
above come from CDSKIT measurements rather than published speedups.

- [NumPy indexing](https://numpy.org/doc/stable/user/basics.indexing.html),
  [bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html),
  [ufunc.at](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html)
  and [frombuffer](https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html)
  support fixed lookups and bounded histogram work.
- [NumPy 1.23 matmul](https://numpy.org/doc/1.23/reference/generated/numpy.matmul.html)
  documents batch broadcasting used by the contraction implementation.
- [OrfM](https://pmc.ncbi.nlm.nih.gov/articles/PMC5013905/) and the
  [orfipy author implementation](https://github.com/urmi-21/orfipy/blob/master/orfipy/orfipy_core.pyx)
  motivate separating position discovery from candidate selection. CDSKIT uses
  its own ambiguity-aware block lookup, not those programs' exact algorithms.
- [PyTorch inference_mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)
  explains the inference-only scope and restrictions tested here.
- [NCBI genetic codes](https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi?chapter=cgencodes)
  and the repository [codon contract](../wiki/codon-semantics.md) govern scientific
  interpretation, including context-dependent codes 27, 28 and 31.

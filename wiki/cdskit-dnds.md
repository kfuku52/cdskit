# cdskit dnds

Estimate pairwise nonsynonymous (`dN`) and synonymous (`dS`) distances from
already aligned CDS pairs. The default `--method yn00` uses a native NumPy
implementation of Yang–Nielsen (2000), with equal path weights (`weighting=0`),
pair-specific F3x4 codon frequencies and pair-specific kappa. All methods work
directly on the supplied codon alignment and run independently of PAML.

## Input and execution

The input is a tab-separated table with required columns `pair_id`, `sequence_1`
and `sequence_2`. Pair IDs must be unique and nonempty. Each pair must have equal
aligned lengths divisible by three; pairs can have different lengths. Additional
columns are allowed. For example:

```tsv
pair_id	sequence_1	sequence_2
example	ATGGCTGCT	ATGGCCGCT
```

```bash
cdskit dnds --pairs_file pairs.tsv --codon_table 1 --threads 4 --out_file ds.tsv
```

The default genetic code is 1. `--threads 0` follows CDSKIT resource limits;
an explicit positive integer limits batch workers. Output defaults to stdout and
uses the standard atomic TSV output contract when `--out_file` is supplied.

## Interpretation and missing values

Report schema 2 records method, genetic code, codon-semantics version, dN/dS,
kappa, synonymous/nonsynonymous sites, retained/excluded codons and status.
Missing or gapped codon columns are removed jointly within each pair. An ordinary
terminal stop is excluded; an internal stop yields `NA` with `internal_stop`.
This includes definite ambiguous stops such as `TAR` in code 1, but not merely
possible stops such as `TAN`. The terminal position is the last codon without
gaps (`-?.`); trailing `NNN` cannot disguise an earlier internal stop.
Codes 27/28/31 follow CDSKIT's context-dependent sense-codon semantics.
Lowercase and RNA `U` are normalized. Invalid DNA or frames are errors, not
automatically repaired.

`dS_model` and `dN_model` identify F84, K80 or JC69 corrections. Insufficient
information for kappa is reported with `kappa_defaulted`; the initialization is
4.6, matching PAML YN00's independent-data-set initialization. Saturated or
otherwise unestimable distances are `NA` (empty TSV cells), not zero. Valid zero distances remain
zero. `dS_diagnostic` and `dN_diagnostic` preserve PAML-compatible finite-sample
diagnostics for audit/comparison; saturated diagnostics are not usable distances.
Inspect status and retained-codon counts before biological interpretation.

## Optional likelihood methods

Install the `dnds-ml` extra to use the SciPy optimizer:

```bash
python -m pip install '.[dnds-ml]'
cdskit dnds --pairs_file pairs.tsv --method gy94 --out_file distances.tsv
```

`gy94` fits a pairwise Goldman–Yang model with CF3x4 sense-codon frequencies.
The TSV `method` field is `GY94_CF3x4_pairwise`. It uses the same paired
deletion and stop rules. `S` and `N` are computed from neutral-model
site fractions at fitted kappa. An identical retained pair has zero
distances and unestimated kappa. `rate_boundary` leaves a distance missing
when the numerical upper bound is reached. Its diagnostic field records
the fitted bound. `kappa_boundary` indicates that kappa reached an
optimization limit. Nonconverged fits leave model estimates missing.

Simulation showed lower error for both distances on longer alignments.
Ratios on short alignments could have much larger errors, and the
likelihood fit takes substantially more CPU time. [Evaluation and runtime](https://github.com/kfuku52/cdskit/blob/master/docs/dnds-gy94-evaluation-2026-11-11.md)

```bash
cdskit dnds --pairs_file pairs.tsv --method gy94-regularized --out_file distances-regularized.tsv
```

`gy94-regularized` adds a Gaussian penalty on the natural log of dN/dS,
centered at `log(0.5)` with a standard deviation of `1.8`. It shrinks
weakly determined ratios toward 0.5. Its `method` field is
`GY94_CF3x4_log_ratio_penalty`. The three-codon example retains all three
codons, with dS about 0.286, dN about 0.046, and status `kappa_boundary`.
The penalty improved average ratio error on held-out simulations and
increased error for the smallest simulated true ratio (0.005).
[Regularization evaluation](https://github.com/kfuku52/cdskit/blob/master/docs/dnds-gy94-evaluation-2026-11-11.md#regularization)

`gy94-regularized-kappa` adds a second Gaussian penalty on the natural log of
the transition/transversion rate parameter κ, centered at `log(3)` with a
standard deviation of `1`. On held-out simulations, the extra penalty improved
average accuracy most for short alignments. The log-κ error increased at the
most extreme simulated κ values.
[Log-κ regularization evaluation](https://github.com/kfuku52/cdskit/blob/master/docs/dnds-kappa-evaluation-2026-11-14.md)

```bash
cdskit dnds --pairs_file pairs.tsv --method gy94-regularized-kappa --out_file kappa.tsv
```

The three-codon example retains all three codons and reports
`GY94_CF3x4_log_ratio_and_kappa_penalty`, with dS about 0.242, dN about 0.044,
κ about 3.78, and status `ok`.

All GY94 methods accept `--frequency_model empirical-shrinkage`. This
blends the observed sense-codon frequencies with the CF3x4 estimates,
using more of the observed distribution when the alignment supports it.
With the three-codon example, the command below reports
`GY94_empirical_shrinkage_log_ratio_penalty`, retains three codons, and
reports dS about 0.276, dN about 0.054, and status `kappa_boundary`.

```bash
cdskit dnds --pairs_file pairs.tsv --method gy94-regularized \
  --frequency_model empirical-shrinkage --out_file distances-shrinkage.tsv
```

The largest gains were on long simulated alignments with equilibrium
frequencies outside the CF3x4 model. Average errors increased slightly at
60 codons. [Frequency evaluation](https://github.com/kfuku52/cdskit/blob/master/docs/dnds-gy94-evaluation-2026-11-11.md#empirical-codon-frequency-shrinkage)

The likelihood CLI uses processes when `--threads` allows at least two
workers and the input has at least 256 pairs. It uses at most eight
processes, with at least 128 pairs per process on average. Use `--threads 1`
to keep fits in one process. Set BLAS threads to one when running several
workers to limit nested parallelism.

## Numerical and speed comparison

`scripts/benchmark_dnds.py` compares identical paired-deletion alignments with
PAML YN00 (`weighting=0`, `commonkappa=0`, `commonf3x4=0`). It checks all finite
reported dN/dS/kappa values, including saturated diagnostics, against PAML's
four-decimal printed precision and checks deterministic output across repeats.
It rejects invalid alphabets and definite internal stops before paired deletion.
If an estimator has no finite value, the report records comparison coverage and
missing counts instead of claiming that every pair was compared. Nonfinite
PAML values remain JSON `null`. The report is written atomically and must not
overlap the input, including filesystem aliases.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/benchmark_dnds.py --pairs pairs.tsv --output benchmark.json \
  --threads 1 --repeats 5
```

Audited `.tsv.gz` input is also supported. Set BLAS/OpenMP threads to one to
avoid nested parallelism inside the requested batch workers.
Both engines warm once. PAML uses `ndata` batches by default; `--individual`
measures the distinct one-process-per-pair workflow. Timings exclude alignment
and native module import for the API comparison. Results record input/source
hashes, commands, runtime metadata and whole-process peak RSS; this RSS is not
a per-engine memory comparison.

To measure the command itself without requiring PAML:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/benchmark_dnds.py --pairs pairs.tsv --output cli-benchmark.json \
  --threads 4 --repeats 5 --native-only --cli
```

The additional `cdskit_cli` measurement launches a fresh process each time and
includes imports, input reading, estimation and atomic TSV writing. Preparing
the common paired-deletion input and hashing its complete output are outside
the timed region. On Unix, CLI peak RSS is the maximum child-process high-water
mark across these launches; it is unavailable on Windows. This is separate
from the benchmark process's own RSS. `--cli` can also accompany the normal
PAML comparison.

The native implementation shares a read-only 2 MiB ASCII-codon lookup, projects
only the distinct entries of symmetric kappa-class matrices, and converts
result columns in bulk. The full class matrices are restored before the original
normalization; scientific corrections and numerical precision are unchanged.
Parallel batches grow to amortize worker overhead while small inputs retain
multiple batches. Their dense arrays can be up to four times the serial
per-worker batch size, trading memory for throughput. No compiled extension,
additional dependency or reduced-precision arithmetic is required.

No universal speedup is implied. See the repository's `TESTING.md` for standard checks.

# cdskit dnds

Estimate pairwise nonsynonymous (`dN`) and synonymous (`dS`) distances from
already aligned CDS pairs. This command uses a native NumPy batch implementation
of Yang--Nielsen (2000), with equal path weights (`weighting=0`), pair-specific
F3x4 codon frequencies and pair-specific kappa. It does not require PAML and does
not run protein alignment. It is not weighted-path YN00 or codeml maximum likelihood.

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

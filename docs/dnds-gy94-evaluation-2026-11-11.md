# Pairwise dN/dS likelihood evaluation

`dnds --method gy94` uses the Goldman–Yang reversible codon model. The
likelihood estimates dS, dN, and kappa jointly. Codon frequencies remain
fixed during the likelihood fit. Iterative proportional fitting matches
the three observed nucleotide marginals after excluding stops, with
a half-count added to each nucleotide at each position. The model
normalizes the two distances using neutral substitution fractions.
The neutral-model site counts sum to three times the retained codon
count. [Goldman and Yang](https://doi.org/10.1093/oxfordjournals.molbev.a040153) [CF3x4](https://doi.org/10.1371/journal.pone.0011230)

The fit bounds the distances to `[1e-6, 10]` and kappa to `[0.05, 60]`.
A fit at the lower distance bound is reported as zero. The upper distance
bound produces a missing distance and a `rate_boundary` status. The
identical-pair status reports zero for both distances. Kappa and the site
counts are unestimated in that case.

## Simulation

The evaluation crossed three genetic codes (1, 2, and 6), two positional
frequency profiles, two kappa values, three true ratios, four path lengths,
three alignment lengths, and product versus perturbed nonproduct equilibrium
frequencies. There were 160 independently sampled pairs per setting and
138,240 pairs total. True distances follow from the stationary substitution
rates and the neutral-model site fractions. The generator calculates these
rates independently of the CDSKIT likelihood code.

Entries are mean absolute errors on pairs with finite values under both
methods. The ratio requires positive dS.

| Codons | dS YN00 → GY94 | dN YN00 → GY94 | dN/dS YN00 → GY94 |
| ---: | ---: | ---: | ---: |
| 60 | 0.2615 → 0.2618 | 0.0635 → 0.0611 | 0.3292 → 0.5755 |
| 240 | 0.2221 → 0.1270 | 0.0557 → 0.0279 | 0.2576 → 0.2479 |
| 900 | 0.2064 → 0.0716 | 0.0535 → 0.0168 | 0.2193 → 0.1526 |
| All | 0.2301 → 0.1537 | 0.0576 → 0.0353 | 0.2666 → 0.3168 |

YN00 and GY94 produced finite dS on 130,949 and 136,950 pairs, respectively.
Finite ratio counts were 125,287 and 130,806. GY94 reported 377 optimizer
failures and 920 upper rate-bound cases. A short, weakly identified dS can
make its ratio unstable. This evaluation uses stationary simulated
alignments and covers one kind of frequency misspecification. Performance
on alignments with errors, other evolutionary processes, or empirical
ground truth remains unmeasured.

A separate equal-frequency check matched PAML 4.10.10 `codeml` with
`runmode=-2` and `CodonFreq=0` to its displayed dN, dS, and kappa precision.
That check fixes frequencies to compare the same likelihood. The shipped
estimator uses CF3x4. [PAML 4.10.10](https://github.com/abacus-gene/paml/releases/tag/v4.10.10)

## Runtime and reproduction

For 400 simulated pairs of 900 codons on Apple M2 Max with Python 3.12,
the warmed single-worker wall-time medians over seven interleaved runs were
0.014 seconds for YN00 and 3.285 seconds for GY94. Each method produced identical
output across its repeats. Set the BLAS thread count to one when benchmarking
multiple CDSKIT workers.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venvs/full-3.12-cpu/bin/python scripts/evaluate_dnds_estimators.py \
  --replicates 160 --workers 6 --output /tmp/dnds-gy94-evaluation.json
```

On macOS 27 arm64 with Python 3.10, the SciPy 1.15.3 wheel tagged
`macosx_14_0_arm64` failed to import in the loader. The PyPI wheel of the
same release tagged `macosx_12_0_arm64` imported and completed a GY94
smoke test. [SciPy issue](https://github.com/scipy/scipy/issues/25635) [Verified wheel](https://files.pythonhosted.org/packages/a0/6e/0c3bf90fae0e910c274db43304ebe25a6b391327f3f10b5dcc638c090795/scipy-1.15.3-cp310-cp310-macosx_12_0_arm64.whl)

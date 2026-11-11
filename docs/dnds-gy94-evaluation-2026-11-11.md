# Pairwise dN/dS likelihood evaluation

`dnds --method gy94` uses the Goldman–Yang reversible codon model. The
likelihood estimates dS, dN, and kappa jointly. Codon frequencies remain
fixed during the likelihood fit. Iterative proportional fitting matches
the three observed nucleotide marginals after excluding stops, with
a half-count added to each nucleotide at each position. Near excluded stops,
the same marginal equations can converge slowly under iterative scaling.
The fit then solves those equations in log-frequency coordinates and checks
all 12 marginal errors against `1e-10`. In 5,310 constructed compositions
near excluded stops, all 5,310 matched the target after this change,
compared with 4 under the previous iteration limit. The model
normalizes the two distances using neutral substitution fractions.
The neutral-model site counts sum to three times the retained codon
count. [Goldman and Yang](https://doi.org/10.1093/oxfordjournals.molbev.a040153) [CF3x4](https://doi.org/10.1371/journal.pone.0011230) [Exponential-family moment matching](https://arxiv.org/abs/1301.7376)

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
| 60 | 0.2614 → 0.2615 | 0.0634 → 0.0611 | 0.3294 → 0.5756 |
| 240 | 0.2216 → 0.1268 | 0.0556 → 0.0278 | 0.2574 → 0.2478 |
| 900 | 0.2060 → 0.0714 | 0.0534 → 0.0168 | 0.2197 → 0.1532 |
| All | 0.2297 → 0.1534 | 0.0575 → 0.0352 | 0.2667 → 0.3168 |

YN00 and GY94 produced finite dS on 130,949 and 137,324 pairs, respectively.
Finite ratio counts were 125,287 and 131,124. GY94 reported 923 upper
rate-bound cases. Raising the L-BFGS-B line-search limit from 20 to 60
reduced failures from 377 to 7 on these same pairs. Reevaluation with
scaling and squaring resolved the remaining 7 failures on the full panel.
[SciPy L-BFGS-B options](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)

When the line search stops abnormally, CDSKIT reevaluates the same
likelihood and gradient using scaling and squaring, which avoids cancellation
in small transition probabilities from the spectral calculation. Starting
from the last finite parameters and requiring optimizer convergence recovered
all 41 prior line-search failures across this panel and the regularization
panel below. [Matrix exponential and Fréchet derivative](https://doi.org/10.1137/080716426)

A short, weakly identified dS can
make its ratio unstable. This evaluation uses stationary simulated
alignments and covers one kind of frequency misspecification. Performance
on alignments with errors, other evolutionary processes, or empirical
ground truth remains unmeasured.

A separate equal-frequency check matched PAML 4.10.10 `codeml` with
`runmode=-2` and `CodonFreq=0` to its displayed dN, dS, and kappa precision.
On 36 more simulated pairs of 300 or 900 codons under code 1, both distances
agreed within `0.00005` of PAML. Across another 72 pairs under codes 2 and
6, 70 met that threshold and the largest distance difference was `0.00020`,
on a near-zero dN fit. These checks fix frequencies to compare the same
likelihood. The default frequency model is CF3x4. [PAML 4.10.10](https://github.com/abacus-gene/paml/releases/tag/v4.10.10)

## Regularization

`--method gy94-regularized` minimizes the same negative log likelihood plus
`(log(dN/dS) - log(0.5))² / (2 × 1.8²)`. It optimizes both distances on the
log scale with the same numerical bounds. The penalty applies only to the
ratio. CDSKIT reports the fitted distances and no posterior intervals.
The Bayesian pairwise literature motivates regularizing dN/dS when data
weakly identify the ratio. The penalty here is fixed across pairs.
[Bayesian pairwise substitution-rate estimates](https://doi.org/10.1093/molbev/msu142)

The held-out design used two different composition profiles and crossed
codes 1, 2 and 6, two kappa values, seven true ratios (0.005–5), four path
lengths (0.01–2.2), four alignment lengths (40–1,200 codons), and three
simulation models. The models used product frequencies, nonproduct
frequencies, or two equally weighted site classes with a shared true dS
and ratios of 0.15 and 1.85 times the overall ratio. Each setting had 60
simulated pairs, for 241,920 total. The design was fixed before inspecting
the held-out errors.

Mean absolute errors below use the same finite pairs across all three
methods for each metric. The ratio requires a positive dS.

| Codons | dS YN00 / GY94 / regularized | dN YN00 / GY94 / regularized | dN/dS YN00 / GY94 / regularized |
| ---: | ---: | ---: | ---: |
| 40 | 0.1692 / 0.1985 / 0.1540 | 0.0619 / 0.0683 / 0.0603 | 0.6499 / 0.9008 / 0.6044 |
| 90 | 0.1493 / 0.1346 / 0.1143 | 0.0558 / 0.0453 / 0.0447 | 0.5926 / 0.6751 / 0.5371 |
| 360 | 0.1286 / 0.0721 / 0.0698 | 0.0520 / 0.0328 / 0.0329 | 0.5186 / 0.5061 / 0.4181 |
| 1,200 | 0.1212 / 0.0501 / 0.0502 | 0.0508 / 0.0283 / 0.0283 | 0.4512 / 0.3900 / 0.3548 |
| All | 0.1419 / 0.1134 / 0.0968 | 0.0551 / 0.0437 / 0.0415 | 0.5431 / 0.5925 / 0.4655 |

Finite ratio counts were 201,924, 203,519, and 224,962, with 200,241
pairs in common. At true dN/dS of 0.005 the ratio mean absolute errors
were 0.0082, 0.0061, and 0.0344. The penalty can pull strongly conserved
pairs away from their true ratios. The simulation does not establish
empirical accuracy or validate tests of selection.
Raising the line-search limit reduced failures from 880 to 29 for GY94
and from 125 to 5 for the regularized method on this panel.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venvs/full-3.12-cpu/bin/python scripts/evaluate_dnds_regularization.py \
  --replicates 60 --workers 6 --output /tmp/dnds-regularization-evaluation.json
```

## Empirical codon frequency shrinkage

`--frequency_model empirical-shrinkage` is available for both GY94 methods.
It estimates each sense-codon frequency as a blend of the empirical frequency
and the CF3x4 target. The target weight is the summed sampling variance of
the empirical frequencies divided by their squared Euclidean distance from
the target. It is clipped to `[1/(L+1), 1]` for `L` retained codons, keeping
frequencies positive even with zero observed variance. The variance treats
the two endpoints of a codon column as one observation, with values 0, ½,
or 1 for each codon. [Multinomial frequency shrinkage](https://jmlr.org/papers/v10/hausser09a.html)

The independent evaluation used 103,680 simulated pairs across codes 1,
2 and 11, two new nucleotide composition profiles, two kappa values, five
true ratios (0.008–4.5), four path lengths (0.05–2.2), and four alignment
lengths. The three generating models used product frequencies, nonproduct
frequencies, and two site classes with a shared dS and ratios of 0.3 and 1.7
times the overall ratio. Both estimators used the same Gaussian log-ratio
penalty. The table reports mean absolute errors on common finite pairs.

| Panel | dS CF3x4 → shrinkage | dN CF3x4 → shrinkage | dN/dS CF3x4 → shrinkage |
| --- | ---: | ---: | ---: |
| All | 0.1530 → 0.1473 | 0.0392 → 0.0381 | 0.4513 → 0.4435 |
| 60 codons | 0.2647 → 0.2707 | 0.0576 → 0.0585 | 0.6569 → 0.6589 |
| 520 codons | 0.1072 → 0.0975 | 0.0316 → 0.0299 | 0.3816 → 0.3718 |
| 1,500 codons | 0.0743 → 0.0565 | 0.0275 → 0.0247 | 0.2868 → 0.2653 |
| 1,500 codons, nonproduct | 0.1263 → 0.0728 | 0.0196 → 0.0112 | 0.2994 → 0.2349 |

The ratio was finite for 103,265 CF3x4 estimates and 103,239 shrinkage
estimates, with 103,228 in common. Effects on the product and site-mixture
panels were small. The simulation evaluates stationary alignments and uses
regularized GY94 throughout.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venvs/full-3.12-cpu/bin/python scripts/evaluate_dnds_frequencies.py \
  --replicates 36 --workers 6 --output /tmp/dnds-frequency-evaluation.json
```

## Runtime and reproduction

For 400 simulated pairs of 900 codons on Apple M2 Max with Python 3.12,
the warmed single-worker wall-time medians over seven interleaved runs were
0.014 seconds for YN00 and 3.285 seconds for GY94. Each method produced identical
output across its repeats. Set the BLAS thread count to one when benchmarking
multiple CDSKIT workers.

With 1,200 mixed simulated pairs (45–1,200 codons) and `--threads 4`,
four interleaved fresh CLI runs on the same host had median wall times of
17.97 seconds with threads and 5.46 seconds with processes for GY94.
Regularized GY94 took 11.02 and 3.61 seconds. Complete TSV bytes matched
within each method. Sampled peak RSS summed across the CLI process tree
rose from 119 to 507 MiB for GY94. This sum includes shared pages.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venvs/full-3.12-cpu/bin/python scripts/evaluate_dnds_estimators.py \
  --replicates 160 --workers 6 --output /tmp/dnds-gy94-evaluation.json
```

On macOS 27 arm64 with Python 3.10, the SciPy 1.15.3 wheel tagged
`macosx_14_0_arm64` failed to import in the loader. The PyPI wheel of the
same release tagged `macosx_12_0_arm64` imported and completed a GY94
smoke test. [SciPy issue](https://github.com/scipy/scipy/issues/25635) [Verified wheel](https://files.pythonhosted.org/packages/a0/6e/0c3bf90fae0e910c274db43304ebe25a6b391327f3f10b5dcc638c090795/scipy-1.15.3-cp310-cp310-macosx_12_0_arm64.whl)

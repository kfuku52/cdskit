# Pairwise dN/dS with log-κ regularization

`gy94-regularized-kappa` adds `(log(κ) − log(3))² / 2` to the
[existing log-ratio penalty](dnds-gy94-evaluation-2026-11-11.md#regularization).
The value 3 is the center of the penalty on the log scale. Each pair still
fits dS, dN, and κ jointly. The default codon frequency model is CF3x4.
The transition/transversion rate parameter κ is distinct from the observed
transition/transversion count ratio.
[Yang and Nielsen (2000)](https://doi.org/10.1093/oxfordjournals.molbev.a026236)

The held-out simulation crossed genetic codes 1, 2, 11, and 28, two new
position-specific nucleotide frequency profiles, five values of κ
(0.25–20), four dN/dS ratios (0.008–2.7), four path lengths (0.02–2.2),
and three alignment lengths. The generating models used product codon
frequencies, perturbed nonproduct frequencies, or two equally weighted site
classes that shared dS and had dN/dS equal to 0.2 and 1.8 times the overall
ratio. The design used a separate seed and frequency profiles from the pilot.
There were 16 independent alignments per setting and 92,160 pairs total.

Each entry is mean absolute error on pairs with finite estimates under
both methods for that metric. The ratio requires positive estimated dS.

| Codons | dS | dN | dN/dS | log(κ) |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 0.35015 → 0.33282 | 0.05586 → 0.05236 | 0.44895 → 0.44226 | 0.96697 → 0.73178 |
| 140 | 0.22447 → 0.21578 | 0.03657 → 0.03669 | 0.34998 → 0.34770 | 0.71698 → 0.53520 |
| 650 | 0.12549 → 0.12597 | 0.02749 → 0.02764 | 0.24385 → 0.24508 | 0.34381 → 0.30918 |
| All | 0.23331 → 0.22480 | 0.03997 → 0.03890 | 0.34354 → 0.34112 | 0.66429 → 0.51723 |

Log-κ error increased at κ = 0.25 (0.66763 to 0.75537) and κ = 20
(0.64964 to 0.72454). At the smallest true ratio, dN/dS mean absolute
error rose from 0.02666 to 0.02719. The extra penalty reduced the number
of pairs reaching an upper distance bound from 92 to 40. Identical
pairs continued to have unestimated κ. The simulation measures error
against its generating model for correctly aligned pairs.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venvs/full-3.12-cpu/bin/python scripts/evaluate_dnds_kappa_regularization.py \
  --replicates 16 --workers 5 --output /tmp/dnds-kappa-evaluation.json
```

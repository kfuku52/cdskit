# Additional review after 0.30.3

The review started from `90e28c098d88440fc469698c5d3b4656790db7a1` with a clean
worktree. It covered evaluation inputs and ranking, split/cluster identifiers,
calibration consumers, frozen evaluation and atomic output paths, and the
optimized ORF selection path. The published model weights were not modified.

## Reproduced and fixed

| Area | Reproduction before the fix | Corrected behavior |
|---|---|---|
| Incomplete cluster vectors | Three rows and two groups produced a fold vector ending in `None`; bootstrap could silently evaluate only the two grouped rows | Require one nonmissing group per row before splitting or resampling |
| Bootstrap input alignment | Extra prediction rows were silently ignored; nonpositive iteration counts could return misleading empty results | Validate full target/prediction dimensions, binary predictions and positive integer iterations before drawing samples |
| Zero-valued identifiers | Train and test rows with numeric cluster ID `0` passed the overlap audit; calibration could substitute sequence IDs for cluster `0` | Normalize identifiers consistently across audits and consumers; `0` and `"0"` identify the same group |
| AP score types | Perfect ranking with `uint8` scores `[1, 0]` returned AP 0.5; boolean scores raised a NumPy exception | Sort descending without negating scores; compare tie boundaries without subtraction, avoiding integer wrap and finite-float overflow |
| Unsupported bootstrap estimates | All-unknown targets returned `status="ok"` with both confidence intervals null | Return `insufficient_observations`, matching the paired-bootstrap convention |

AP now also rejects mismatched shapes, nonbinary targets and nonfinite scores
before the no-positive-label early return. Scores may be arbitrary finite real
ranking values; they need not be probabilities. Valid empty/no-positive targets
still return null. Large integer scores are not cast to float, preserving the
ordering of adjacent integers above the exact range of float64.

Missing partition IDs include null values, nonfinite numeric values, blank
strings and the established case-insensitive `none`/`nan`/`na`/`null` tokens.
Numeric zero remains a valid identifier. Optional absent IDs do not themselves
create an overlap; sequence overlap is still checked independently.

## Verification

- Regression tests exercise missing/extra groups, zero IDs, prediction-size
  mismatches, invalid iteration counts, unsupported bootstrap estimates,
  unsigned/boolean/extreme scores and calibration-group preservation.
- The 32 saved prediction files from the completed 0.30.2 study (30 fold/seed
  files plus two external evaluations) produce **exactly the same probability
  metrics** as the 0.30.3 evaluator. Their stored cluster vectors are complete.
  Thus these fixes do not change the previously reported study results.
- Five hundred randomized AP comparisons, including ties and multiple numeric
  dtypes, agree with scikit-learn within `1e-14`.
- Three thousand randomized ORF cases across supported genetic codes agree
  between optimized selection and exhaustive candidate enumeration.
- Full repository validation (`python scripts/check.py all`): 1,391 tests passed,
  one unchanged CUDA-only test skipped on the CPU host; coverage 78.0% and all
  critical module floors passed. Lint, formatting, types, security/dependency
  audit, build and fresh installed-wheel checks passed. No new model training
  or model publication is needed for these evaluation-input fixes.

The added regression suite is
[`test_localize_evaluation_boundaries.py`](../tests/unit/test_localize_evaluation_boundaries.py).

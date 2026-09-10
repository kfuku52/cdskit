# 0.30.3 release validation

The default model remains the frozen ESM2 650M / Light Attention baseline,
trained with ordinary BCE and validation-BCE epoch selection. Experimental
accuracy-improvement recipes remain opt-in. The trained head and thresholds
were not changed during this release.

## Fix and distribution

The training checkpoint retained an absolute residue-cache path and
`local_files_only=True`, which prevents first use on machines without the
backbone cache. `scripts/export_localize_plm.py` exports a safe-loaded checkpoint
with those two runtime settings cleared, while checking the pinned remote
encoder identity. Local encoders and mismatched identities are rejected.
The exported model is registered under the existing default alias with a
release URL and SHA-256. CLI help, prediction documentation and the model card
now describe first-use and offline operation.

The model release tag `localize-esm2-localization-v1` points to training commit
`a27f868e814351747b7dcb44bc74300051924b96` (0.30.2). The 0.30.3 source release
adds portable export and activates the downloadable alias.

## Checks

- Full required `python scripts/check.py all`: **1,354 passed**, one CUDA-only
  test skipped on the Mac; statement/branch coverage **78.0%**, with all critical
  module floors passed. Ruff lint/format/complexity, mypy, Bandit, dependency
  audit, wheel build and fresh installed-wheel checks passed.
- Python 3.10 core: **1,182 passed**, two optional-PyYAML tests skipped in the
  core-only environment, 16 ML tests deselected. The full environment exercises
  the YAML and ML paths.
- Real CUDA device: the CPU/CUDA batch equivalence tests both passed, covering
  the CUDA case omitted locally.
- Real checkpoint: safe load succeeded; every tensor and all metadata were
  compared recursively. Only the residue-cache path and local-only flag differ
  from the original. Weights and thresholds are exactly equal.
- Exported checkpoint on Linux CPU: all 32 sample proteins matched the original
  CPU probabilities exactly (maximum absolute error 0) and all labels matched.
- Public release, fresh cdskit model cache, omitted `--model`: all 32 proteins
  matched exactly after checksum-verified download. The independently cached
  Hugging Face backbone was reused; this does not benchmark a cold backbone
  download.
- `CDSKIT_OFFLINE=1`, populated caches, two-protein subset: all labels matched;
  maximum probability difference from the original 32-protein run was
  `5.96e-8`, within the established numeric tolerance. Subset batching can change
  floating-point rounding.

The CPU checks used Slurm allocations rather than the login shell. Model assets
contain the classifier, thresholds and settings, not raw training sequences or
backbone weights. See the [model card](../wiki/cdskit-localize-esm2-localization-v1.md)
for evaluation scope and model limitations.

## Worktree and commit audit

- The detached integration worktree's complete staged tree exactly matches
  existing commit `5ab20cd`; it has no additional unstaged changes. It was
  preserved unchanged.
- Local branch `codex/localize` has no commits absent from `master`.
- Stash `25d7e7d` (including its untracked files) contains the localization work
  integrated in `9157ccc`. A syntax-tree comparison and review of remaining
  differences found formatting, typing, strict-zip checks, runtime-device and
  offline handling refinements rather than omitted work. Relocated tests and
  dedicated documentation are present. The stash was preserved unchanged.
- The separate remote `ci/smoke` branch dates to September 2025; it is historical
  CI work, not an uncommitted local worktree. It was not merged into this release.

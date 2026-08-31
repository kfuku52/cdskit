# Documentation review: 0.28.3

Reviewed on 2026-08-31 against source commit `94ba687` (0.28.2), after a
fast-forward pull, and public wiki commit `3f40b3d` (2026-07-21). The review
covered the README, developer/release/migration guides, all 31 published wiki
pages, and the three newer but divergent repository wiki files.
The policy-only update `44c7d63` was pulled before publication; it does not
change the implementation reviewed here.

## Corrections

| Finding | Correction |
| --- | --- |
| Wiki installation said Python >=3.9; source requires >=3.10. Optional ML instructions omitted Transformers or used a nonexistent PyPI package. | Align requirements with `pyproject.toml`, use explicit GitHub pip URLs, and explain the complete ML extra. |
| Bioconda 0.27.0 still declares Python >=3.8/Biopython >=1.77 and omits Matplotlib. | Add a verified conda solve with explicit upstream minimums, explain the downstream recipe mismatch and when to remove the workaround. |
| Both real pretrained targeting5 artifacts fail under the current sklearn 1.9.0 environment with `No module named '_loss'`. | Read the serialized version (1.5.2), verify both artifacts in an isolated matching runtime, and document that model-specific environment without changing library dependency ranges. |
| Published wiki lacked the newer safe-loading, offline, and training-cache limitations. | Explain trusted pickle loading, checksum-verified aliases, exact cache paths, ESM offline behavior, and version/fold/data/configuration requirements for OOF resume. |
| The original targeting5 model's zero peroxisome score could be mistaken for a learned negative prediction. | Identify its constant-zero head and distinguish the experimental trained peroxisome model. |
| `hammer` sample records had different lengths, and the text described `> N` rather than the implemented `>= N` occupancy rule. | Align the example with terminal gaps, confirm the published output, and explain translation-based occupancy. |
| Obsolete worker options and imprecise codon/annotation descriptions. | Remove `label --threads`; document worker caps/serial execution, resolvable IUPAC codons, CDS coordinate rebasing, gap coordinate clamping, and ORF length reporting. |
| A blanket rectangular-TSV claim overlooked `codonstats --mode both`; GC denominators also differed between commands. | Identify the two-table exception and advise separate summary/usage outputs; explain each GC denominator. |
| Tiny learning examples could be read as sufficient for five-fold evaluation; illustrative metrics looked measured. | Separate smoke data from evaluation requirements, explain the missing `fold_id` column, and label illustrative metrics. |
| Different TargetP datasets were used to justify an equivalence claim; historical scores appeared current. | Remove the unsupported equivalence claim, retain provenance in a dated archive, and distinguish historical metrics from current reproducibility instructions. |
| The README duplicated long feature guides; repository/public wiki edits did not synchronize. | Shorten the README, keep all page/image sources in `wiki/`, and document the separate publication process. Add a deprecated-alias page for `longestcds`. |
| Release/CI descriptions overstated automatic publication and platform-job exclusions. | Describe tested-commit checks, manual recovery, patch-only tag behavior, downstream release delays, and platform checks triggered by version-file changes. |

## Verification

- 15 complete core CLI examples: run from their documented input and compare
  output records, headers, GFF rows, or stdout. All match after the hammer fix.
- README pipeline: seven protein records produced successfully from the tracked
  padding fixture, with stop residues masked before translation.
- 87 CLI/helper command blocks: parse against the actual argument definitions
  without running full downloads/training workflows.
- The exact six-row localization example: train a centroid JSON model and predict
  successfully. Random CV, fixed-fold CV, threshold tuning, and a 15-epoch CPU
  BiLSTM also train/export/predict successfully. CV wiring checks use duplicated
  synthetic rows, so their metric values are **not** accuracy evidence.
- Download both registered model assets, verify the registry SHA-256 values,
  reproduce failure with scikit-learn 1.9.0, and successfully predict with 1.5.2.
  The `targeting5` probabilities match the wiki example after rounding.
- Audit all 48 non-editable packages in the isolated pretrained-model environment:
  no skipped packages and no known vulnerabilities reported at review time.
- Check internal page/image/anchor targets and external URL responses; inspect
  all three plot previews. Label the previews as illustrative and match the MSA
  example's wrap setting to the image.
- Confirm the documented UniProt query is accepted by the live API using a
  one-row request; do not download a full training dataset.
- Resolve the documented conda environment successfully in a dry run:
  cdskit 0.27.0, Python 3.14.7, Biopython 1.88, NumPy 2.5.2, Matplotlib 3.11.1.
- Run `python scripts/check.py core`: **747 passed, 15 deselected**.

## Limits and follow-up

The Bioconda recipe is maintained in another repository and was not changed
here. Its runtime requirements and package smoke tests still need updating;
the explicit installation constraints compensate for the currently published
metadata. The conda check resolved dependencies but did not perform a full
installation.

The pretrained artifact check used macOS ARM64, Python 3.12.13, torch 2.13.0,
NumPy 2.5.2, and scikit-learn 1.5.2. It does not establish cross-version sklearn
pickle compatibility or validate other platforms for those legacy artifacts.
Replacement model releases need an explicitly recorded runtime and independent
prediction validation; no release assets were modified or reuploaded here.

Large UniProt/DeepLoc/TargetP training, historical accuracy benchmarks, GPU
training, and complete scientific reproduction were not rerun. Historical
tables are retained as labelled provenance, not newly validated model-quality
claims. The PyPI/package, API, and vulnerability observations are time-specific.

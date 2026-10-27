# ESM2 LoRA development pilot, 2026-10-27

Outcome: retain the frozen ESM2 default. The three-seed LoRA pilot gained 0.00282
macro-F1, below the predeclared 0.02 target, while micro-F1 fell 0.00573, exceeding
the 0.005 loss guardrail. Every per-seed macro-F1 difference interval includes
zero. This recipe is not promoted. This is a small development experiment, not
independent biological validation or an evaluation of the released checkpoint.

## Matched comparison

Both methods train a new ten-label Light Attention head on the same 512 proteins,
reserve 256 for epoch/threshold selection, and evaluate on 1,024 development-test
proteins. Proteins are at most 512 residues. Original official folds are retained:
fold 0 supplies test rows, fold 4 supplies validation, and folds 1–3 supply training.
Accession hashing with subset seed 20261027 selects rows before model fitting.

- Encoder: `facebook/esm2_t33_650M_UR50D`, immutable revision
  `08e4846e537177426273712802403f7ba8261b6c`.
- Control: frozen encoder, Light Attention, ordinary masked BCE.
- Candidate: the same encoder/head plus query/value LoRA in all 33 layers;
  rank 4, alpha 8, 675,840 adapter parameters. Base encoder parameters remain
  frozen and encoder dropout remains disabled. Adapter outputs start at zero;
  adapter initialization uses a private CPU RNG stream, preserving the head's
  initialization and dropout RNG on CPU, CUDA and MPS.
- AdamW: head learning rate 0.001, adapter learning rate 0.0001, weight decay
  0.0001, batch size 2, at most 6 epochs, patience 3, seeds 1/2/3.
- Minimum validation BCE chooses the epoch. Validation-only F1 tuning chooses
  each label's threshold. `safe-v1` and no forced label apply to both methods.
- Frozen residue features may be shared between seeds. Fitted LoRA embeddings
  bypass both memory and disk feature caches. Long-sequence overlap averaging
  retains gradients in the adapter path; synthetic tests exercise that behavior.

The source DeepLoc labels retain their original closed-world dataset convention.
Zero labels are dataset annotations, not experimentally established absence.
All three seeds use the same proteins; they are repeated fits, not independent
biological test cohorts. Cluster bootstrap resamples the 772 test components.
The peroxisome class has only 4 training, 6 validation, and 17 test positives.

## Additional partition audit and correction

The historical 28,303-row source has cached MMseqs cluster assignments. Their
dataset digest matched the source. Excluding the 540 rows in clusters crossing
official folds left 27,763 rows, but a fresh cross-partition search still found
homology between folds 0 and 1.

A new all-against-all search used MMseqs2 release 18-8cc5c, identity 0.30,
coverage 0.80, cov-mode 0, and 4 threads. Its 118,302 hits were joined with the
original cluster memberships into connected components. Entire components
crossing official folds were excluded, preserving original fold assignments.
This excludes 1,995 rows and retains 26,308. Fresh searches across all ten fold
pairs reported zero cross-partition hits under those criteria. This is a specified
homology audit, not proof that all remote homology is absent.

The original preliminary pilot passed direct pairwise searches, but the new
global components revealed two train/test and one validation/test shared
components. That pilot was interrupted and retained as obsolete evidence.
The corrected comparison uses the new component groups and fresh pairwise
searches; both audits pass. No test scores selected the partition correction.

Filtering changes the population: 595 of 4,187 original cell-membrane positives
and 29 of 304 peroxisome positives are excluded. The corrected full source has
275 peroxisome positives. Per-label exclusions are saved separately. Existing
data, historical experiment directories, and published weights were preserved.

## Results

The primary comparison ran on audrey1's RTX 6000 Ada 48 GB in one Slurm GPU
allocation. Both methods used the same CUDA runtime. Means below are over three
fits on the same data; they do not describe the released model's full-data recipe.

| Method | Macro-F1 | Micro-F1 | Macro-AP | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Frozen ESM2 | 0.511993 | 0.629979 | 0.559423 | 0.067145 |
| ESM2 + LoRA | 0.514814 | 0.624245 | 0.562676 | 0.068117 |
| LoRA minus frozen | +0.002820 | −0.005734 | +0.003253 | +0.000972 |

Higher Brier scores are worse. These are diagnostics under the source annotation
convention, not independently established probability calibration.

| Seed | Frozen macro-F1 | LoRA macro-F1 | Difference | Paired 95% interval | Micro-F1 difference | Macro-AP difference |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 0.508031 | 0.518156 | +0.010125 | [−0.002953, +0.022808] | +0.003831 | +0.007920 |
| 2 | 0.511708 | 0.505702 | −0.006006 | [−0.020300, +0.007982] | −0.019509 | +0.007284 |
| 3 | 0.516240 | 0.520582 | +0.004342 | [−0.013289, +0.022126] | −0.001523 | −0.005447 |

Intervals use 1,000 paired component-bootstrap draws, bootstrap seed 1. Seed 2's
micro-F1 difference interval is [−0.032929, −0.006953]. Selected epochs were
2/2/3 for frozen models and 2/2/2 for LoRA, using validation BCE throughout.

| Label | Frozen mean F1 | LoRA mean F1 |
| --- | ---: | ---: |
| nucleus | 0.694117 | 0.686239 |
| cytoplasm | 0.655606 | 0.636763 |
| extracellular | 0.869624 | 0.869002 |
| mitochondrion | 0.715265 | 0.736799 |
| cell_membrane | 0.490520 | 0.522914 |
| endoplasmic_reticulum | 0.466621 | 0.449877 |
| chloroplast | 0.835354 | 0.840377 |
| golgi_apparatus | 0.240236 | 0.214000 |
| lysosome_vacuole | 0.152587 | 0.192164 |
| peroxisome | 0.000000 | 0.000000 |

The ranking-score gain is small and not consistent across seeds. Thresholded
decisions also trade improvements in some locations for losses in others.
The four training peroxisome positives do not support a broad rare-label claim.

Predeclared adoption requirements are an absolute macro-F1 gain of at least
0.02, no macro-AP loss, micro-F1 loss at most 0.005, a positive paired-bootstrap
interval, five folds × three seeds, and a genuinely new independent external
test. This pilot fails the initial effect-size and micro-F1 guardrails. Full-data
five-fold × three-seed retraining and a new independent external evaluation were
not run. The result does not rule out other LoRA settings or benefits with more
training data. ProtT5 was not evaluated in this pilot.

## Reproducibility and software validation

The experiment helper is `scripts/localize_lora_experiment.py`.
The [published protocol, audit and result summary](localize-lora-pilot-2026-10-27.json)
retains the numeric evidence and hashes without requiring the local artifact
directory. Model binaries and per-protein outputs remain local experimental
artifacts. `scripts/localize_homology_components.py` now implements the full-source
component reconstruction, including original-row digest binding and fresh fold
audits. Its groups and retained-row digest exactly reproduce the 26,308-row result.

Local raw evidence is under `.artifacts/localize_lora_20261027/`:

- `pilot_cuda/`: primary protocol, full selected rows, source snapshot,
  safely loadable models, per-protein NPZ outputs, per-model metrics and comparisons.
- `pilot_cuda/analysis.json`: six-model independent metric/hash verification,
  per-seed differences and mean results.
- `cuda_seed1_portability/verification.json`: offline CPU CLI replay of four
  CUDA-scored proteins from the saved LoRA artifact.
- `full_partition_repair/clusters_source.json`: source-bound component assignments.
- `full_partition_repair/report.json`: full-source cross-partition audit.
- `full_partition_repair/selection_impact.json`: class counts before/after exclusions.
- `full_partition_repair/repair_partitions.py`: reconstruction method.
- `pilot_650m/interruption.json`: reason the earlier pilot was interrupted.
- `pilot_650m_components/interruption.json`: partial exploratory MPS run, excluded
  from the primary comparison. The adapter initialization RNG was corrected
  before the primary CUDA study; results from different runtimes are not pooled.

The component payload named `clusters_source.json` binds groups to the original
unmodified source rows; the initial intermediate `clusters.json` hashes rows
after cluster annotation. Use the source-bound payload with the experiment helper.

The primary run records Python 3.12.11, PyTorch 2.6.0+cu124, Transformers 5.16.1,
NumPy 2.5.3, and CUDA on Linux x86_64. The existing CUDA environment was reused
without modifying its packages; the desktop checks use the committed lock.
MMseqs reports commit
`8cc5ce367b5638c4306c2d7cfc652dd099a4643f`. Its dedicated binary and the encoder
download leave existing user environments and executable installations intact.
Slurm job 66992 completed with exit code 0 in 24m21s, requesting one GPU, four
CPUs and 8 GiB RAM. It used the historical `debug` partition; no other jobs or
scheduler settings were changed. kfauto's current fresh-native request policy
uses explicit `background`, `normal` or `urgent` for subsequent submissions.
Recorded elapsed times include training, validation, loading and prediction;
they are operational timings, not a controlled inference-speed benchmark.

Adapter models use `multilabel_plm_lora_v1`, so older readers reject them rather
than silently omit the adapter. Existing frozen models keep their format and
the default model alias remains unchanged. No dependencies were added.

Local validation: `python scripts/check.py coverage` passed 2,693 tests, with three
CUDA-only skips; combined coverage was 79.15% and critical-module floors passed.
`python scripts/check.py quality` passed lint, format, mypy, complexity and source
security checks. Tests cover initial frozen/LoRA embedding agreement, overlap
gradients, frozen base weights, adapter updates, safe checkpoint loading and
scalar/batch prediction. On audrey1, seven related CPU/CUDA tests passed with one
MPS-only skip. All six artifacts match their recorded SHA-256, all test arrays
align, and F1/AP/Brier agree with independent recalculation to within 5.1e-9.
The saved CUDA LoRA artifact also ran through the offline desktop CPU CLI:
labels and decision statuses agree on all four checked proteins; the maximum
absolute score difference is 1.59e-5. This is a limited portability check.
These checks do not establish biological accuracy. CI was not run.

Existing HPA evaluations remain previously inspected development/stress evidence.
A new external cohort with appropriate positive/negative evidence is required
before claiming an independent improvement or promoting new default weights.
PLM pretraining overlap is a separate, uncontrolled limitation. Future model
comparisons should use the repaired source partitions and retain a freshly fitted
frozen-encoder control when changing training-data volume.

## Release hardening and window assembly measurements

Before the 0.33.0 push, regressions reproduced two checkpoint-loading problems:
an invalid late tensor could leave earlier parameters changed, and an encoder
cached after failed loading could emit predictions on retry. Adapter weights are
now completely validated, including conversion to the receiving dtype, before
parameters change. A failed load clears the encoder so every retry remains an
error. Experiment arguments also reject duplicate seeds and existing output
directories before reading data or fitting models.

Window assembly now returns a single window directly, joins two windows without
whole-protein padding, and finalizes prefixes while retaining the overlapping
tail for longer sequences. Addition order, overlap semantics and gradients are
preserved. The [component benchmark](localize-lora-window-benchmark-2026-10-27.json)
uses cached synthetic 1,280-dimensional residue windows, window 1,000/overlap 128,
one CPU thread, three warmups and 21 alternating paired repetitions. It includes
assembly and scalar backward, with no protein-language-model forward pass.

| Residues | Previous assembly, median ms | Current assembly, median ms | Median paired speedup |
| --- | ---: | ---: | ---: |
| 128 | 0.451 | 0.353 | 1.29× |
| 1,024 | 4.582 | 4.144 | 1.11× |
| 4,096 | 22.923 | 19.240 | 1.20× |
| 16,384 | 154.924 | 76.711 | 2.01× |

Output hashes are identical in every benchmark case, and scalar gradients match
exactly. Separate fresh-process runs measured about 723 MiB versus 816 MiB peak
RSS at 16,384 residues: the speed improvement trades roughly 13% higher process
peak memory in this synthetic workload. RSS includes imports and allocator
high-water marks. Timings varied with system load; alternating order reduces
that effect. These are component measurements, not an end-to-end training claim.
Real encoder regressions cover CPU/CUDA gradients. Synthetic window regressions
cover short/two-window inputs and overlaps larger than the stride. The original pilot measurements above remain
unchanged; the default model and published weights are not replaced.

Release validation: `quality`, `coverage` and installed-wheel `build` passed.
Coverage was 79.19%, with 2,711 tests passing and three CUDA-only skips. Python
3.10 `core` passed 2,530 tests, with two optional-PyYAML skips; the full profile
covers those YAML cases. audrey1's normal-partition job 67003 passed 25 selected
CPU/CUDA checks, with one MPS-only skip. Replaying the saved 650M LoRA artifact
through the desktop CPU CLI retained identical scores, labels and statuses for
the same four proteins. The deployment changes do not constitute a new accuracy
experiment.

# Full-data localization, specialists and distillation

This experiment uses all 28,303 prepared DeepLoc Swiss-Prot proteins across the
official five outer folds. Each outer model reserves the highest remaining fold
for early stopping, label thresholds and specialist blend calibration. Training,
calibration and outer evaluation never share an accession or exact sequence.

Development results and checkpoints are written under
`data/localize_bench/full_localization_20260908`. `protocol.json` fixes selection
criteria and records the environment. The legacy CNN is compared with the terminal
CNN using the same validation protocol, rather than historical in-sample thresholds.
The initial seed selects a base architecture using macro F1 with an absolute 0.01
micro-F1 loss guardrail; the selected base is repeated with seeds 2 and 3.

## Specialized heads

Each localization has a binary histogram gradient boosting classifier fitted only
on sequence-derived broad localization features from the model's training folds.
Taxonomic indicators are omitted by extracting features with an empty organism
group. The specialists do not consume fitted base-model predictions during fitting,
so stacking training predictions is unnecessary. For each localization the inner
validation set chooses a specialist contribution from 0, 0.25, 0.5, 0.75 and 1,
and a decision threshold. HGB uses 100 iterations, 15 leaves, minimum leaf size 20
and L2 regularization 1; it does not use the validation set for its own fitting.

The checkpoint stores numeric tree nodes, not sklearn estimators. Export is checked
against sklearn's public `predict_proba` before acceptance. Normal safe model
loading remains available. A `specialist_head` and `specialist_weights` embedded in
the localization model activate the calibrated blend during prediction.

## Distillation and the control

The student is a CNN with 32 filters, separate N/C termini and sequence-feature
fusion. Its control has the same architecture, seed, optimizer and training budget,
but learns only hard labels. The distilled student minimizes BCE against
`0.5 * hard_label + 0.5 * teacher_probability` (temperature 1), using class weights
computed from hard labels. The teacher is the integrated model trained/calibrated
inside the same outer split. Only its predictions on **training rows** enter the
student loss. Student stopping and thresholds use hard validation labels.

The normal control is necessary: an improvement over the original CNN alone cannot
be attributed to distillation when the student also has different features and
capacity. `sequence_only_features` ensures the same feature convention at runtime.
An exported student contains no teacher or specialist ensemble.

## Commands

Use a fresh output directory for new configurations; the scripts reject conflicting
saved configurations and resume completed model files. Keep the original prepared
TSV unchanged between stages.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

python scripts/localize_full_experiment.py \
  --output data/localize_bench/full_localization_20260908 \
  --recipes cnn_legacy,cnn_termini --seeds 1 --epochs 12 --device mps

# After development selects cnn_legacy:
python scripts/localize_full_experiment.py \
  --output data/localize_bench/full_localization_20260908 \
  --recipes cnn_legacy --seeds 2,3 --epochs 12 --device mps

python scripts/localize_integrate_distill.py \
  --base_dir data/localize_bench/full_localization_20260908/cnn_legacy_seed1 \
  --output data/localize_bench/full_localization_20260908/integration_seed1 \
  --seed 1 --epochs 12 --device mps
```

Repeat integration using matching base directories and seeds 2 and 3. On CPU use
`--device cpu`. Training times in logs may include concurrent experimental jobs and
should not be interpreted as isolated throughput benchmarks.

Once development choices are frozen:

```bash
python scripts/localize_finalize_experiment.py \
  --experiment_dir data/localize_bench/full_localization_20260908 \
  --base_recipe cnn_legacy --epochs 12 --device mps

python scripts/localize_runtime_benchmark.py \
  --models data/localize_bench/full_localization_20260908/final/cnn_legacy.pt \
           data/localize_bench/full_localization_20260908/final/integrated.pt \
           data/localize_bench/full_localization_20260908/final/student_control.pt \
           data/localize_bench/full_localization_20260908/final/distilled.pt \
  --output data/localize_bench/full_localization_20260908/runtime
```

Final models train on four source partitions and calibrate on partition 4; the
calibration rows are not subsequently used for fitting. HPA is read for scoring
after models and thresholds have been fixed. This 1,717-row human dataset has been
inspected historically, so it is a regression/stress test, not a newly collected
final holdout. Its few rare-location positives require particular caution.

CPU inference measurements use identical, length-stratified workloads, one CPU
thread, a warmup, three repeats and one process per model. Report wall time,
process peak RSS and model-file size together with accuracy and output differences.
Distillation changes predictions; these measurements do not establish an
optimization with equivalent outputs.

`localize_experiment_report.py` combines OOF metrics, saved HPA metrics and timings.
Its paired uncertainty estimates resample MMseqs clusters (30% identity, 80%
coverage). If absent, `clusters.json` is generated automatically. Existing cluster
files must contain aligned `groups`, a successful `mmseqs_cluster_assignments` report
and the training dataset SHA-256. Cluster IDs are used for
uncertainty; they do not replace the fixed official folds mid-experiment.

```bash
python scripts/localize_experiment_report.py \
  --experiment_dir data/localize_bench/full_localization_20260908
```

The additional HPA homology audit can be reproduced after scoring:

```bash
python scripts/localize_external_homology.py \
  --experiment_dir data/localize_bench/full_localization_20260908
```

For this run, 149 MMseqs clusters (540 rows) crossed the official partitions.
The report therefore also scores the remaining 27,763 development rows with the
same fixed models, in addition to the complete official-fold result. This is a
sensitivity analysis, not a retroactive change to model-selection partitions.

## Adopted integrated model and audit (2026-09-08)

The adopted accuracy model is
`data/localize_bench/full_localization_20260908/final/integrated.pt`.
It contains the CNN, ten numeric specialist heads, validation-selected blend
weights and thresholds. It loads safely without enabling legacy pickle loading.
The file SHA-256 is
`e9b35ead3eca4dcdf833e469f18f1d67e05a3de274b00861cbf08bf138d01621`.
Use the audited source revision/environment; the checkpoint is a local experiment
artifact and is not bundled in the distribution or a remote pretrained alias.

```bash
cdskit localize --seq_file proteins.fasta --seq_type protein \
  --model data/localize_bench/full_localization_20260908/final/integrated.pt \
  --threads 1 --report localization.tsv
```

The multi-label CNN now honors CLI thread settings and canonicalizes peptide input
in the same way as the CLI, including terminal stop removal and internal stop
rejection. Saving converts NumPy scalar metadata to native Python values, including
inside tuples, so locally generated fold models remain safe-loadable. Loading
integrated heads rejects invalid blend weights, feature schemas and malformed or
cyclic trees. Experiment resumption checks both training and validation provenance.

The original CV checkpoints contained a NumPy string in their calibration-fold
metadata. All 45 affected integrated/control/distilled checkpoints were reserialized
without changing tensors or specialist parameters; originals and before/after
hashes remain under `audit/original_checkpoints` and
`audit/serialization_migration.json`. The adopted final model did not require
migration and its bytes are unchanged.

Reproduce the full integrated-model audit with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 python scripts/localize_audit_integrated.py \
  --experiment_dir data/localize_bench/full_localization_20260908
```

`audit/verification.json` records full OOF replay for all 15 integrated checkpoints,
independent sklearn metrics, training/calibration hashes, independently refitted
sklearn specialist comparisons, threshold/weight reproduction and complete HPA CLI
agreement. All 84,909 OOF row predictions and all 1,717 HPA CLI probabilities were
unchanged. Independent specialist predictions matched on all 7,179 calibration/HPA
rows for every label. `audit/report.md` records adoption findings and limitations.

### Limits of the adopted model

- CV performance describes the complete 28,303-row evaluation protocol. The final
  checkpoint fits 22,841 rows and reserves 5,462 for calibration; it was not fitted
  on calibration labels. Its exact future accuracy needs a new independent test set.
- HPA is a previously inspected human stress set, not a pristine final holdout.
  Its macro F1 is essentially unchanged; peroxisome F1 is zero on seven positives.
- Features are sequence-only; `--organism_group non_plant` does not suppress this
  model's multi-label `chloroplast` output (the existing constraint targets cTP/lTP
  in the separate targeting model). HPA produced eight chloroplast predictions.
  These are not evidence of human chloroplast localization. Apply biological review
  when interpreting outputs; no taxonomy filter was added after HPA inspection.
- Probabilities are independent classifier scores, not a normalized distribution
  or experimentally calibrated confidence. `ensure_one_label` forces at least one
  label; there is no abstention mechanism. Empty/all-unknown/low-information inputs
  can receive finite scores and should not be treated as meaningful predictions.
- The CNN uses 512 terminal residues for long sequences, including the original
  joined-terminus layout; global sequence features enter through specialists.
  Improved group averages do not establish improvement for every protein or class.
- The audit ran on local macOS/Python 3.10/PyTorch 2.2.2. CPU and MPS boundary checks
  agreed in labels, but the full Linux/Windows/Python-version CI matrix was not run.

## Expanded experimental peroxisome positives

`data/localize_bench/perox_expansion_20260908/report.md` records an additional
fixed-model evaluation on 115 experimentally annotated UniProt positives,
including 15 human proteins. Reviewed release 2026_03 contained 425 eligible
entries with ECO:0000269 attached directly to a peroxisomal location, after
excluding fragments and isoform-specific comments. Development/HPA identifiers,
secondary accessions, Ensembl protein cross-references and canonical exact
sequences were checked; duplicate new sequences were removed before scoring.

On these 115 new positives, baseline recall was 31/115 and integrated recall
25/115. Removing detected training-source homology left 57 positives, with
recall 6/57 and 9/57 respectively. This positive-only extension cannot estimate
precision or population F1. Its fixed thresholds were not retuned. Original HPA
negative-label results remain a separate reference, with incomplete annotation
as an additional limitation. Source snapshots, evidence citations, cohort
selection, clustering and predictions are retained alongside the report.

Run `python scripts/localize_perox_expansion.py` against the saved snapshots to
reproduce the comparison. New downloads should use a new experiment directory;
do not overwrite the frozen source files or tune against this panel and then
present it as an untouched evaluation set.


## Distribution status

Source version 0.29.0 includes inference and experiment code. The integrated
checkpoint is a separate local artifact; a source push and its automatic numeric
release do not upload model weights. The prepared model-asset bundle contains
`cdskit-localize-multilabel-integrated-v1.pt`, a model card, evaluation summary and
SHA-256 checksums. A model-specific GitHub Release and a verified download alias
must be published separately before users can obtain it by alias.

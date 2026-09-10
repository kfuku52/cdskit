# Localization feature and decision contract

This contract separates a sequence motif, a learned score, and a localization
decision. It does not establish improved biological accuracy or probability
calibration. Published weights remain unchanged.

## PTS2 and feature compatibility

The current detector uses `[RK][LIVQ].{5}[HQ][LA]`, a nine-residue motif entirely
within the first 40 residues. `MRLQVVLGHLAAAA` matches; the eight-residue control
`MRLQVVVHLAAAA` does not. The limited correction preserves the existing allowed
residues, wildcard behavior (including X), search window, and PTS1 precedence in
`signal_type`. If both motifs occur, both match flags are true but the summary
remains PTS1. Direct `detect_perox_signals` expects canonical uppercase amino
acids; feature extractors canonicalize their input.

The thiolase nonapeptide has experimental support in
[Flynn et al. (1998)](https://onlinelibrary.wiley.com/doi/full/10.1046/j.1365-313x.1998.00344.x).
Motif matching alone is not proof of targeting: the same motif did not mediate
import in the tested diatom system in
[A Single Peroxisomal Targeting Signal Mediates Matrix Protein Import in Diatoms (2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3178647/).
The central wildcard and 40-residue boundary remain heuristic limitations.

| Feature schema | PTS2 definition | Use |
| --- | --- | --- |
| `localize-pts2-8-v1` | historical eight-residue regex | Unversioned artifacts and explicitly legacy models |
| `localize-pts2-9-v2` | corrected nine-residue regex | New feature extraction and newly trained models |

The identifiers version semantics, not just names or dimensions. Basic, broad,
perox and derived TargetP features inherit this choice. Loading an unversioned
artifact explicitly assigns the legacy schema. Saving it preserves that schema.
Unknown schema identifiers and conflicting model/head declarations are rejected.
Saving a raw unversioned dictionary also preserves legacy semantics: external
model builders must explicitly declare the schema used for fitting.

A perox head or sequence specialist can have its own feature schema. Attaching a
new head never changes the base model's features. Numeric feature-matrix APIs
still require the caller to supply features matching the receiving head; an
unannotated array cannot reveal its feature semantics. Prefer sequence-level
prediction APIs, which select the extraction schema from the model.

New pipeline manifests and teacher-prediction identities include feature and
decision versions. Historical experiment configurations also include the feature
version, preventing silent reuse of old results in a corrected experiment. Use a
new run directory after a schema change; retain old runs for reproducibility.

## Input decisions

`--decision_policy model` uses the artifact's setting. Unversioned/published
artifacts retain `legacy`. New staged pipeline artifacts default to `safe-v1`
and `ensure_one_label=False`. Pipeline JSON can explicitly set
`"decision_policy": "legacy"`; existing pipeline run identities will differ.

`safe-v1` canonicalizes the sequence and skips inference for an empty sequence,
all-X sequence, or one-residue sequence. It does not merely switch off forced
selection: skipped rows never reach the CNN, PLM, centroid or single-label
backend. Empty batches return correctly shaped empty results. Internal stop
codons remain errors. CLI input-format checks still apply before prediction.

| Decision status | Meaning |
| --- | --- |
| `invalid_input` | Sequence is empty after canonicalization |
| `abstained` | All residues are unknown, or only one residue remains |
| `below_threshold` | Valid input; no label crosses its threshold |
| `taxonomy_excluded` | Threshold-positive labels were removed by a taxonomy rule, with no replacement selected |
| `forced_label` | Legacy `ensure_one_label` chose the highest score/threshold ratio among allowed labels |
| `predicted` | A normal decision was produced |

An empty label list is not verified non-localization. No arbitrary general
minimum protein length, entropy cutoff, or unknown-residue fraction has been
validated here. Homopolymers of length two or more and partial sequences may
still receive predictions. The gate is not a general out-of-distribution detector
and cannot infer fragment status or whether a terminus is complete.

## Taxonomy

`--taxonomy_id` is explicit organism metadata, separate from targeting5's
`--organism_group`. The initial rule `human-plastid-v1` excludes `chloroplast`
only for [Homo sapiens, NCBI 9606](https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=9606).
Other IDs and an omitted ID apply no mask. There is no lineage inference, network
lookup or blanket `non_plant` mask. Use this rule for human organism localization,
not a heterologous chloroplast import experiment. A file-wide ID is appropriate
only for a file with that organism context.

Taxonomy changes the multilabel decision, preserving all original scores without
renormalization. Forced selection cannot resurrect excluded labels; if all
classes are excluded, no label is forced. Input abstention takes precedence over
both taxonomy and forced selection. Five-class cTP/lTP constraints retain their
existing separate meaning.

## Reports and Python APIs

```sh
cdskit localize --seq_file proteins.faa --seq_type protein \
  --model integrated.pt --decision_policy safe-v1 --report_schema v2 \
  --report localization.json

# For human proteins only:
cdskit localize --seq_file human.faa --seq_type protein \
  --model integrated.pt --decision_policy safe-v1 --taxonomy_id 9606 \
  --report localization.tsv
```

`--report_schema model` selects legacy columns for legacy inference, and v2 for
safe inference or an explicit taxonomy ID. `legacy` output is rejected when it
would conceal a safe decision or taxonomy override. `--report_schema v2` can
also document legacy predictions without changing their numerical scores.

V2 retains the existing prediction and `p_*` columns, adding decision status,
quality reason, score availability, forced-label flag, source-checkpoint SHA-256,
feature schemas, score/head types, and calibration status. `label_head_status`
is a JSON mapping serialized as a string, for per-label centroid constants.
`source_model_sha256` identifies the loaded file, not an in-memory edit of it.
No artifact hash is available for a model constructed directly in Python.

`--include_features` reports the model's actual feature values. In v2,
`perox_signal_type` and `pts2_annotation_match` instead use the corrected detector,
identified by `signal_schema`. Thus a legacy model can have a zero `pts2_match`
feature alongside a true `pts2_annotation_match`. Legacy output retains its old
motif annotation as well as its numerical predictions.

Skipped rows have null `p_*` values in JSON and empty TSV score cells, empty
predicted labels/class, and `score_available=false`. Valid below-threshold rows
retain their scores. A legacy low-information prediction remains available but
its quality reason is exposed in v2.

Sequence-level Python predictors include `decision_status`, `quality_reason`,
`score_available`, and `forced_label`. Batch predictors return these per row.
Batch `prob_matrix` and single-label numeric fields use **zero placeholders** for
skipped rows to retain numeric array shapes; always consult `score_available`
before interpreting, exporting or evaluating scores. This also applies with
`apply_thresholds=False`. Feature-only centroid APIs have no sequence from which
to determine input quality; use their sequence-level wrapper for the gate.

```python
from cdskit.localize_model import load_localize_model, predict_multilabel_localization
from cdskit.localize_runtime import PredictionRuntime, prediction_runtime

model = load_localize_model("integrated.pt")
with prediction_runtime(PredictionRuntime(decision_policy="safe-v1", taxonomy_id="9606")):
    result = predict_multilabel_localization("M", model)
assert result["decision_status"] == "abstained"
assert not result["score_available"]
```

`p_*` can be sigmoid, centroid-derived, blended or constant scores. Independent
multilabel scores need not sum to one. The targeting5 constant perox head remains
zero for legacy compatibility, identified as `perox_head_status=constant`, and
is not evidence of confirmed non-localization. `calibration_status=unverified`
does not claim that a fitted temperature or F1 threshold establishes calibration.

## Model update and independent evaluation

The published integrated CNN has no feature input, but its specialists do.
Correcting a feature does not require retraining that feature-free CNN. Refit
affected feature-dependent heads, then reselect blend weights and thresholds
on development data. Regenerate teacher predictions and retrain dependent
students when the teacher changes. A constant head needs adequate labeled data,
not a new threshold, to become a trained detector.

Before publishing replacement weights, the separate data/evaluation work must
fix label meanings, unknown handling and independent partitions. Evaluate the
PTS2-only change and decision-only change separately, followed by their
combination. Report per-label and subgroup performance, abstention coverage,
performance on accepted rows and all-row accounting, plus per-label reliability
and Brier scores on a held-out set with suitable labels. Freeze any probability
calibrator separately from F1 threshold selection. Previously inspected HPA
results cannot become a new unknown holdout. No new fitted weights, biological
accuracy claim, or independent calibration result accompanies this code change.

Pipeline evaluation reports `score_coverage`, scored/unscored row counts and
`accepted_only` decision metrics. Overall decision metrics account for every row;
probability metrics use only available scores. Exported NPZ files include
`score_available` and `decision_status`. Calibration and distillation reject
unscored rows instead of learning from numeric placeholders.

Frozen external evaluation follows the same rule for CNN, PLM and centroid
models. Its NPZ files also preserve `quality_reason`; probability metrics and
reliability bins exclude unscored placeholders even when every row is rejected.
All model predictions and the final metrics are staged together, with rollback
on write/publication errors and Python interruptions. A completed evaluation
still cannot be overwritten. An uncatchable process kill can leave a run lock;
verify its recorded owner before removing that stale lock.

Window-based CNNs exclude completely empty batch-alignment windows from pooling
even with `mask_padding=False`. That option controls residue padding within real
windows, not the addition of artificial competing windows from other sequences.

Final single-stage targeting models must expose the five classes expected by
the inference CLI. If nearest-centroid fitting or a constant head lacks classes,
`localize-learn` rejects the final model before replacing the model/report files.
CV's internal estimators may still fit the classes observed in their own fold.

The safe pipeline rejects these low-information sequences in training and
validation, but retains them in the test partition to measure abstention coverage.
Decision metrics include every test row; probability and reliability metrics,
including each reported stratum, use only rows with available scores. Bootstrap
comparisons use all decisions, including abstentions.

Feature-based fitting APIs accept `feature_schema`; callers supplying precomputed
legacy features must declare the legacy schema. Inference validates score shape,
finiteness and range before reporting results. Model hashes identify the bytes
read from the same open file as the deserialized model.

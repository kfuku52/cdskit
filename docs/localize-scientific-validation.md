# Localization scientific validation contract

This contract separates localization observations, targeting-peptide evidence,
weak proxies, and unobserved labels. It also separates calibration performance
from evaluation of a complete fitted pipeline. No new biological performance
claim accompanies these code changes.

## Input and model compatibility

Existing model files remain loadable; these changes do not update their weights,
PTS2 features, score semantics, or inference policy. Models trained with the new
label contract must be trained afresh. Do not rename old checkpoints as new
models. Pipeline run fingerprints and experiment code hashes prevent resuming
changed experiments against old artifacts.

- Pipeline configuration `schema_version: 1` retains the original positive-list,
  closed-world dataset interpretation. Its zero labels are dataset labels, not
  necessarily experimentally established negatives.
- `schema_version: 2` requires `negative_labels`, `label_evidence`, and a complete
  `cluster_id` column in addition to the existing columns. Names can be mapped by
  `data.negative_col`, `data.evidence_col`, and `data.cluster_col`.
- In schema 2, `localization_labels` lists observed positives; `negative_labels`
  lists observed negatives. Both are semicolon-separated. All other labels are
  unknown. Empty positive and negative lists mean all labels are unknown.
- `label_evidence` is a JSON list. Every observed label requires a record with
  `label`, `state` (`positive`/`negative`), `evidence_type`, `source`,
  `source_version`, and `reference`. These are auditable declarations; the parser
  cannot prove that a paper supports a particular isoform, condition or assay.
- `data.evidence_policy: experimental` is the schema-2 default. `declared` also
  accepts `dataset`, `similarity`, `sequence_analysis`, and `localization_proxy`
  records, for explicitly separated development/weak-supervision experiments.
  Never call that experiment an experimental gold-standard evaluation.
- Schema 2 preserves optional `organism_group`, `taxonomy_id`, `fragment`,
  `compartment`, and `isoform` columns for stratified evaluation.

An evidence cell can contain:

```json
[{"label":"nucleus","state":"positive","evidence_type":"experimental","source":"curated study","source_version":"fixed snapshot identifier","reference":"actual publication and assay identifier"}]
```

Use a real reference and verify that it applies to the sequence and experimental
condition in that row. A positive at another location does not establish a
negative at this location. Do not manufacture negative records from missing
UniProt features or missing localization annotations.

For ordinary `localize-learn`:

- `--label_mode evidence` reads a curated `targeting_evidence` JSON column from a
  TSV. Use exactly one experimentally supported targeting class. SP requires
  `feature_type: SIGNAL`; mTP/cTP/lTP require `feature_type: TRANSIT`. Optional
  `start` and `end` are one-based inclusive integers or null. Presence may be
  known even if the cleavage coordinate is not. `noTP` requires experimental
  support for absence of the covered targeting signals, not an empty annotation.
- `--label_mode uniprot_cc` is a weak proxy. Free-text notes and negated clauses
  do not create positive labels; missing/unrecognized targeting text is skipped,
  and missing peroxisome annotation is unknown. This mode may no longer supply a
  noTP class. It is not a substitute for peptide evidence.
- `--label_mode legacy_uniprot_cc` explicitly reproduces the historical
  location-to-targeting and missing-as-negative conversion, with a warning.
- In `explicit` mode, targeting labels remain user-declared. Missing/unknown
  perox labels are excluded from that head's training and evaluation. This
  combined legacy model requires at least one observed perox target in each
  training partition and refuses to invent a negative head when none exist.
  Use the multilabel pipeline when the task has no perox observations.

UniProt SIGNAL/TRANSIT features must be checked at feature/evidence level:
reviewed status or protein-level existence is not evidence for cleavage.
UniProt distinguishes experimental, similarity-propagated and predicted TRANSIT
annotations, including predictions from TargetP itself. See
[UniProt TRANSIT](https://www.uniprot.org/help/transit).

## Losses, metrics, and distillation

Multilabel Python training APIs accept 0/1/NaN matrices; NaN means unknown. The
internal helper returns finite target values plus an observation mask. NPZ
outputs preserve targets and an explicit observation mask. TSV/JSON do not use
NaN literals.

CNN, PLM and specialist losses/targets, class balancing, validation selection,
thresholds and metrics exclude unknown cells. Training with no observations for
a configured head is rejected. All-unknown batches cause no optimizer step;
all-unknown validation is rejected. With only one observed outcome, thresholds
stay at their documented fixed value rather than optimizing an unidentified
threshold. A one-sided observed training label can still produce a constant
head; this does not establish calibrated biological confidence.

Distillation applies only on observed cells, using the existing weighted soft
label formulation there. Teacher outputs on unknown cells do not contribute to
the student loss. The model records `distillation_scope: observed_cells`.
Unknown-cell pseudo-label training would be a separate experimental policy, not
an implicit extension of ground truth.

Masked metrics report observation counts. Entire-row subset accuracy uses only
fully observed rows; sample metrics use observed cells and exclude empty rows.
No-observation metrics are null, not zero. Metrics remain conditional on the
annotation process: observed negatives may not represent the deployment
population. Positive-only panels do not establish population precision/F1.
AP, Brier and reliability bins are diagnostics, not proof of calibration.

## Splits and complete-pipeline evaluation

`--cv_split_method exact` groups canonical identical sequences by default.
`--cv_group_col` accepts complete provided groups. Provided `--cv_fold_col` is
also audited for exact/group overlap. Group balance never splits a group to
satisfy class counts. The number of effective folds may be smaller when too few
groups exist. Small inner partitions may lack classes; report coverage rather
than interpreting such results as robust rare-class estimates.

`--cv_split_method mmseqs` uses identity 0.30 and coverage 0.80 with the existing
MMseqs implementation, then rechecks cross-partition hits. Residual hits stop
evaluation; rebuild groups rather than relaxing the audit. These are explicit
search criteria, not proof that remote homology is absent. Missing or failed
MMseqs does not silently become random splitting. `random` remains an explicitly
requested diagnostic comparison and reports overlaps; overlapping random folds
cannot produce a nested evaluation.

Pipeline schema 2 requires supplied groups but does not silently claim they were
independently checked. Set `data.homology_audit: mmseqs` for a cross-partition
search before a new run. The default `provided` records the weaker audit scope.
Frozen external evaluation always performs the cross-development/test search.
See [SpanSeq](https://pmc.ncbi.nlm.nih.gov/articles/PMC11327874/) for the distinction
between similarity reduction and independent sequence partitioning.

When ordinary CV tunes temperature, class thresholds or the cTP/lTP gate/blend,
it now generates inner OOF predictions using only each outer training partition,
fits postprocessing on those predictions, and scores frozen decisions on the
outer test partition. The full-data deployment model may subsequently fit its
postprocessing on development OOF.

- `nested_postprocess` model metadata and `nested_postproc_*` report rows contain
  held-out final-pipeline metrics.
- `calibration_postprocess` contains development tuning metrics.
- Existing `cv_postproc_*` and `two_stage_ctp_ltp_oof_*` keys remain compatibility
  aliases for calibration/resubstitution metrics; their scope is explicitly
  recorded in `postprocess_evaluation_contract`. They are not independent CV.
- External augmented feature OOF includes `nested_threshold`. Its older
  `foldwise_threshold` is labeled `oof_meta_split_not_nested`: excluding a fold
  only from threshold fitting is insufficient when base OOF models have seen it.

Existing DeepLoc inner-validation and pipeline train/validation/test separation
are retained. External augmentation keeps exact/conflict exclusion and optional
similarity filtering; its localization-derived sources are explicitly weak.
External train/calibration splitting keeps clusters or identical sequences
together and reports the actual fraction after group assignment.

## Frozen external evaluation

After data curation and model selection are complete, prepare a schema-2 pipeline
configuration containing development train/validation rows and a genuinely new
test partition. Keep existing HPA and the inspected 115-positive panel in the
historical/development category. Freeze the external protocol before scoring:

```json
{
  "population": "predeclared intended deployment population",
  "annotation_protocol": "documented evidence/negative/isoform rules and snapshot",
  "selection_history": "documented prior access and exclusion history",
  "test_used_for_selection": false,
  "primary_metric": "macro_f1",
  "comparison": {"reference": "control", "candidate": "student"},
  "development_tsvs": ["additional-supervised-training-source.tsv"]
}
```

`development_tsvs` adds all supervised development sources beyond the configured
train/validation partitions for overlap checks. The truth of historical non-use
and completeness of that inventory is a protocol responsibility, explicitly
marked as declared in the result. PLM pretraining overlap remains a separate
limitation.

```bash
python scripts/localize_scientific_evaluation.py freeze \
  --config external-config.json --protocol external-protocol.json \
  --model control=control.pt --model student=student.pt --model teacher=teacher.pt \
  --output new-frozen-evaluation
python scripts/localize_scientific_evaluation.py evaluate \
  new-frozen-evaluation/protocol.json
```

Freeze creates a new directory and hashes data, model files, code, configuration
and protocol, and checks the PLM encoder identity (including local directory
contents). Inputs are rechecked after the audit and scoring. Evaluate rejects
changes and preserves a completed evaluation. The F1 protocol requires observed
positives and negatives for every configured label; positive-only panels need
a different, recall-oriented protocol.
The report includes per-label metrics, observation counts, taxonomic/length and
available compartment/fragment/isoform strata, calibration diagnostics, and a
paired cluster-bootstrap candidate-minus-reference interval. No automatic
winner is declared. Supplement with independent rare-localization observations,
predeclared effect sizes/non-inferiority margins and adequate cluster counts.
Do not count seeds as independent biological samples. Compare distillation with
a matched-capacity control; compare sequence layouts in a separate ablation.

## Remaining biological validation

The regression tests use synthetic examples and offline tiny models. No new
external performance estimate or full-data retraining has been produced by this
change. Before releasing new weights:

1. Audit real label conflicts (including the documented 12 perox annotation
   discrepancies), isoforms, negative evidence and snapshot dates.
2. Agree the PTS2 feature version with its separate implementation task, and
   freeze the inference abstention/taxonomy policy before evaluation.
3. Curate and freeze new independent external data, then train with the corrected
   labels/features/splits and regenerate teacher targets and calibration.
4. Run matched-control, rare-compartment and external comparisons. Publish
   inconclusive or negative results as such; preserve historical experiments.

Repository validation uses `python scripts/check.py quick`, `ml`, and `all`
(quality, coverage, dependency audit and wheel build/smoke tests).

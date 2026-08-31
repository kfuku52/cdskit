# cdskit localize-learn

`cdskit localize-learn` trains a custom model for `cdskit localize` from a TSV
table or from UniProt entries downloaded by query. It is useful when you have
your own targeting/localization labels, or when you want to evaluate a training
recipe with explicit cross-validation before using it for prediction.

The generated model can be used with:

```bash
cdskit localize \
  --seq_file proteins.faa \
  --seq_type protein \
  --model localize_model.json \
  --report localize.tsv
```

## Input TSV

For explicit labels, prepare a tab-separated table with at least:

- `sequence`: CDS or protein sequence.
- `localization`: one of `noTP`, `SP`, `mTP`, `cTP`, or `lTP`.
- `peroxisome`: `yes` or `no`.

Example:

```tsv
accession	sequence	localization	peroxisome
seq_noTP	MAAAAAAAAGGGGGGGG	noTP	no
seq_SP	MKKLLLLLLLLLLAVAVAASAASA	SP	no
seq_mTP	MRRKRRAARAKRRNQAAARRRAA	mTP	no
seq_cTP	MSTSTSTTSTASSSAATSTASSTT	cTP	no
seq_lTP	MARRVAAARRLLLLLVVVVVAAST	lTP	no
seq_perox	MGPVNQDEGPVNQDEGPVNQDESKL	noTP	yes
```

Train a lightweight JSON model:

```bash
cdskit localize-learn \
  --training_tsv train.tsv \
  --seq_col sequence \
  --seq_type protein \
  --label_mode explicit \
  --localization_col localization \
  --perox_col peroxisome \
  --model_out localize_model.json \
  --report localize_learn_report.tsv
```

If `--seq_type dna` is used, sequences are translated in frame before training.
With `--seq_type auto` (the default), CDS-like rows are translated and
protein-like rows are used directly. Prefer an explicit type when the source
is known: short proteins composed only of DNA alphabet letters can be
misclassified by automatic detection. The six-row example above is a smoke
test, not enough data for a reliable model or five-fold stratified CV.

## UniProt Download

`localize-learn` can also download UniProt rows and infer labels from
`cc_subcellular_location` text.

```bash
cdskit localize-learn \
  --uniprot_preset viridiplantae \
  --uniprot_query "keyword:Transit peptide" \
  --label_mode uniprot_cc \
  --seq_col sequence \
  --seq_type protein \
  --localization_col cc_subcellular_location \
  --uniprot_fields accession,sequence,cc_subcellular_location \
  --uniprot_exclude_fragments yes \
  --uniprot_out_tsv uniprot_download.tsv \
  --model_out localize_model.json \
  --report localize_learn_report.tsv
```

Useful UniProt options:

- `--uniprot_preset`: restricts the query scope, for example `viridiplantae`,
  `eukaryota`, `metazoa`, or `fungi`.
- `--uniprot_query`: additional UniProt query text; combined with the preset by
  `AND`.
- `--uniprot_reviewed yes`: keeps Swiss-Prot entries only.
- `--uniprot_out_tsv`: saves the downloaded table so the training set can be
  inspected and reused.
- `--uniprot_max_rows` and `--uniprot_sampling random`: limit large query
  results reproducibly.

UniProt-derived labels are weak labels. Review the saved TSV and run an external
or held-out evaluation before treating the model as production quality.

## Model Choices

The default model is `nearest_centroid`, which is fast and writes a lightweight
JSON model. It is a good starting point for small datasets and smoke tests.

```bash
cdskit localize-learn \
  --training_tsv train.tsv \
  --seq_col sequence \
  --seq_type protein \
  --label_mode explicit \
  --localization_col localization \
  --perox_col peroxisome \
  --model_arch nearest_centroid \
  --model_out localize_model.json
```

For larger datasets, `bilstm_attention` can train a PyTorch `.pt` model. Training
can use GPU (`cuda` or `mps`) when available, while `cdskit localize` loads and
runs the resulting model on CPU for user inference.

```bash
cdskit localize-learn \
  --training_tsv train.tsv \
  --seq_col sequence \
  --seq_type protein \
  --label_mode explicit \
  --localization_col localization \
  --perox_col peroxisome \
  --model_arch bilstm_attention \
  --dl_epochs 15 \
  --dl_device auto \
  --model_out localize_model.pt
```

Install `esm_head` dependencies with
`python -m pip install 'cdskit[ml] @ git+https://github.com/kfuku52/cdskit.git'`.
The extra already includes Transformers. Remote encoder
downloads are pinned to a Hugging Face commit by `--esm_model_revision`.
Inference honors `--model_download no` and `CDSKIT_OFFLINE`; pre-download the
encoder or set `--esm_model_local_dir` for offline use. `esm_head` remains
experimental, and a held-out evaluation is recommended before biological
interpretation.

Frozen ESM embeddings are computed once per sequence for head training, rather
than once per epoch. A bounded in-memory cache (256 MiB including estimated
entry overhead) shares these label-independent features across classifier
stages and CV folds within the command. Keys include sequence, encoder source
and revision, maximum length, pooling, and local-file size/mtime metadata.
The cache is not serialized into model artifacts; classifier heads are trained
independently in each fold. A separate per-fit embedding matrix can use
additional memory, so 256 MiB is not a cap on total process memory. Training
also honors `CDSKIT_OFFLINE` for ESM encoder downloads; it does not disable an
explicitly requested UniProt data download.

Training-set and cross-validation evaluation use bounded batch inference on
`--dl_device`, preserving per-sequence organism constraints and stage-3 OOF
details. `--model_download no` applies to all nested ESM predictors, including
multi-stage and blend models.

## Evaluation

For fair evaluation, use cross-validation or fixed fold IDs instead of judging
only the training set.

Random stratified CV (use a larger table with multiple independent examples
per class, not the six-row smoke input):

```bash
cdskit localize-learn \
  --training_tsv train.tsv \
  --seq_col sequence \
  --seq_type protein \
  --label_mode explicit \
  --localization_col localization \
  --perox_col peroxisome \
  --cv_folds 5 \
  --cv_seed 1 \
  --model_out localize_model.json \
  --report localize_learn_report.tsv
```

Fixed folds from a column (add a non-empty `fold_id` column to the training
table; the minimal example above does not contain one):

```bash
cdskit localize-learn \
  --training_tsv train.tsv \
  --seq_col sequence \
  --seq_type protein \
  --label_mode explicit \
  --localization_col localization \
  --perox_col peroxisome \
  --cv_fold_col fold_id \
  --model_out localize_model.json \
  --report localize_learn_report.tsv
```

For threshold-based postprocessing, use out-of-fold probabilities:

```bash
cdskit localize-learn \
  --training_tsv train.tsv \
  --seq_col sequence \
  --seq_type protein \
  --label_mode explicit \
  --localization_col localization \
  --perox_col peroxisome \
  --cv_folds 5 \
  --localize_threshold_tune yes \
  --localize_threshold_objective macro \
  --model_out localize_model.json \
  --report localize_learn_report.tsv
```

When possible, keep a separate external test set. For related proteins, split by
homology cluster, species, gene family, or publication source rather than by
random rows only.

## Output

The model file is written to `--model_out`. The report file contains training
counts and any requested CV metrics.

The following metric/value rows are illustrative, not the measured result of
the six-row training example:

```tsv
metric	value
class_train_accuracy	0.91
perox_train_accuracy	0.98
cv_class_accuracy_mean	0.84
cv_class_accuracy_std	0.04
cv_perox_accuracy_mean	0.96
cv_perox_accuracy_std	0.02
count_class_SP	120
count_perox_yes	18
```

Use the trained model with `cdskit localize`:

```bash
cdskit localize \
  --seq_file query.faa \
  --seq_type protein \
  --model localize_model.json \
  --report query_localize.tsv
```

## Notes

- The pretrained aliases such as `targeting5` and
  `targeting5-perox-deeploc21-et-v1` are already trained release models. Use
  `localize-learn` only when you want to train your own model.
- The `peroxisome` column is a separate binary label. It does not replace the
  main `localization` class.
- For plant data, `cTP` and `lTP` are meaningful; for non-plant data, use
  `cdskit localize --organism_group non_plant` during prediction to constrain
  plant-only predictions.
- Small or highly imbalanced datasets can give optimistic training accuracy.
  Prefer CV and external evaluation before publishing model performance.

## Related Pages

- [cdskit localize](https://github.com/kfuku52/cdskit/wiki/cdskit-localize)
- [targeting5-v1 model details](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-targeting5-v1)
- [experimental peroxisome head](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-peroxisome-head)
- [localize benchmarks and development notes](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-benchmarks-and-notes)

# Localization model improvement experiments

The new training paths predict **multiple mature-protein localizations** using
DeepLoc labels. They do not reinterpret `targeting5` labels as mature locations,
change the released targeting5/peroxisome weights, or claim improved biological
accuracy before a controlled comparison.

## Evaluation protocol

- Existing complete `fold_id` partitions are retained. An outer test fold is never
  used to fit model parameters, feature normalization, early stopping or thresholds.
- Within each training complement, the last available partition is reserved for
  early stopping and label thresholds. An exported model uses the same convention;
  its metadata reports the actual training and validation counts. It is **not**
  refitted on the calibration partition after selecting thresholds.
- Without an independent validation partition, thresholds remain 0.5. A label with
  no positive or no negative validation examples also retains 0.5.
- Missing CV partitions now raise an error instead of silently creating random
  row splits. For unpartitioned data use `--split_method mmseqs` (30% identity,
  80% coverage). A missing/failed MMseqs executable raises an error. Clustering is
  not a guarantee that every possible homolog is excluded.
- Shared accessions, exact sequences and supplied `cluster_id` values across
  partitions raise an error. External evaluation applies the same audit.
- `--external_test no` keeps HPA out of development runs. Already inspected HPA
  scores are regression/stress checks, not a fresh final holdout.
- Optional `--homology_check yes` adds external hit/no-hit metrics relative to the
  complete training/calibration source, using MMseqs at 30% identity/80% coverage.
- JSON reports include AP per label (undefined labels are null), observed-label
  macro AP, micro AP, Brier score, organism/length strata, input SHA-256 and fold
  calibration metadata. Cluster-bootstrap F1 intervals are computed only when
  real `cluster_id` values exist; fold IDs are not substituted for clusters.
- `<comparison_json>.oof.tsv` contains accession, sequence hash, fold/cluster IDs,
  truth, prediction and probability for each label, aligned with the dataset digest.

The prepared DeepLoc table uses its existing annotation convention. This change
**does not make unannotated locations confirmed negatives** or curate weak UniProt
CC annotations into experimental peptide labels. New annotations require a
separate evidence-aware curation step before entering these supervised datasets.

## CNN experiments

New CNN training defaults to `separate_termini` and masked max pooling. The N/C
segments are convolved independently and their pooled features concatenated in N/C order;
no convolution crosses a synthetic N/C join. `windows` retains the middle of long
proteins using overlapping half-stride windows. `legacy` retains the old sequence
layout for ablations. Old checkpoints without these fields retain their old
layout and unmasked pooling at inference.

```bash
python -m cdskit.deeploc_benchmark \
  --download no --prepare no --benchmark yes --task localization \
  --model_arch cnn --dl_sequence_layout separate_termini \
  --dl_mask_padding yes --dl_epochs 12 --dl_patience 3 \
  --external_test no \
  --comparison_json data/localize_bench/cnn_termini_v2/comparison.json \
  --comparison_md data/localize_bench/cnn_termini_v2/comparison.md \
  --model_out data/localize_bench/cnn_termini_v2/model.pt
```

Run one-change-at-a-time comparisons with identical outer partitions and
validation rules. Use a fresh output directory to avoid overwriting prior runs:

```bash
python scripts/localize_ablation.py \
  --out_dir data/localize_bench/localization_ablation_v2 \
  --recipes cnn_legacy,cnn_masked,cnn_termini,cnn_windows \
  --seeds 1,2,3 --epochs 12 --device cpu
```

`cnn_legacy` is an architectural baseline under the **new validation protocol**;
its score is not expected to reproduce historical training-threshold benchmarks.

## Frozen ESM residue heads

`--model_arch plm` uses a frozen ESM encoder with one of `mean`, `light_attention`
or `label_attention` pooling, a sigmoid multilabel head, validation BCE early
stopping, and validation-only label thresholds. No full encoder fine-tuning,
external specialist stacking or distillation is performed automatically.

Long sequences use overlapping encoder windows and average overlapping residue
embeddings at their original positions. Special tokens are removed. Padding is
masked for attention and max pooling. Cached `.npy` embeddings use a digest of
sequence, encoder identity and window settings; cached files never include fitted
localization labels or head parameters. Cross-validation can share these frozen
embeddings. Whole proteins are padded within each head batch, so reduce
`--plm_batch_size` for very long proteins or larger encoders.

Remote encoders require `--plm_revision` with an immutable 40-character commit SHA.
A local `save_pretrained` ESM directory with safetensors is also accepted; its
contents are hashed and checked again when loading the localization model.

```bash
# Set ESM_COMMIT to the verified commit of the encoder being evaluated.
python -m cdskit.deeploc_benchmark \
  --download no --prepare no --benchmark yes --task localization \
  --model_arch plm --plm_model_name facebook/esm2_t6_8M_UR50D \
  --plm_revision "$ESM_COMMIT" --plm_pooling light_attention \
  --plm_batch_size 8 --dl_epochs 12 --dl_patience 3 \
  --external_test no \
  --comparison_json data/localize_bench/plm_light_v2/comparison.json \
  --comparison_md data/localize_bench/plm_light_v2/comparison.md \
  --model_out data/localize_bench/plm_light_v2/model.pt

cdskit localize --seq_type protein --seq_file proteins.faa \
  --model data/localize_bench/plm_light_v2/model.pt --report localize.tsv
```

The head checkpoint records the encoder identity but does not bundle its weights.
The same encoder must be available at inference; use `--plm_local_files_only yes`
during training to persist offline-only encoder loading. A local encoder directory
must remain accessible. CPU inference still needs the encoder and is more costly
than the CNN. Cached embeddings occupy disk proportional to residues × hidden
size × four bytes. The first implementation supports the ESM model family;
ProtT5 requires its own tokenizer/encoder adapter.

Compare `plm_mean,plm_light,plm_label` with the ablation script and the same
`--plm_model_name`/`--plm_revision`. Choose a candidate on development folds before
running a frozen external evaluation. Specialist integration and distillation
should follow only after the base representation shows a reproducible gain.

## References

- [Light Attention reference implementation](https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py): feature and attention convolutions, weighted/max pooling. This implementation additionally masks the max pool and supports multilabel heads.
- [DeepLoc 2.1 official data and model description](https://services.healthtech.dtu.dk/services/DeepLoc-2.1/): localization ontology and provided partitions.
- [CellProfiling sequence-to-localization benchmark](https://github.com/CellProfiling/seq2loc_benchmark): evidence curation and aggregation comparisons for future dataset extensions.

For full-data comparisons, portable per-location specialist integration and controlled
distillation experiments, see [Full-data localization experiments](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-full-experiment).

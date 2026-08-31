# cdskit localize benchmarks and notes

This page keeps detailed benchmark commands, model-provenance notes, TargetP
comparison tables, and historical development snapshots for `cdskit localize`.
For everyday command usage, see [cdskit localize](https://github.com/kfuku52/cdskit/wiki/cdskit-localize).

## Reproducibility and scope

Install the optional ML dependencies before running neural benchmarks. Research
commands write datasets/models below their output directories; these generated
files are not included in the package. Some later experiments require supplied
UniProt snapshots and an `mmseqs` executable on `PATH`. A fresh live query is not
identical to a frozen historical snapshot.

Keep the CDSKIT commit/version, package environment, random seeds, dataset
hashes, fold definitions, commands, and predictions with each result. Separate
training/calibration metrics from held-out test metrics. Published TargetP and
DeepLoc references use their own datasets and protocols; numeric comparisons
alone do not demonstrate equivalent accuracy on a common test set.

## DeepLoc 2.1 benchmark models

The helper module can download and prepare the public DeepLoc 2.1 datasets, run
cross-validation, write comparison reports, and export models that are directly
usable by `cdskit localize --seq_type protein`.

```
python -m cdskit.deeploc_benchmark \
  --download yes \
  --prepare yes \
  --benchmark yes \
  --task localization \
  --model_out cdskit_deeploc_localization_model.json
```

Supported tasks:

| Task | Model choices illustrated by the historical snapshot |
| --- | --- |
| `localization` | `--model_arch cnn` for higher micro F1/Jaccard; `centroid` for simpler JSON-only models |
| `membrane` | `--model_arch cnn` |
| `sorting_signals` | default `centroid` |

Supported labels:

- `localization`: `nucleus`, `cytoplasm`, `extracellular`, `mitochondrion`,
  `cell_membrane`, `endoplasmic_reticulum`, `chloroplast`,
  `golgi_apparatus`, `lysosome_vacuole`, `peroxisome`.
- `membrane`: `peripheral`, `transmembrane`, `lipid_anchor`, `soluble`.
- `sorting_signals`: `SP`, `MT`, `CH`, `TH`, `GPI`, `NLS`, `NES`, `PTS`,
  `TM`.

CNN models require PyTorch during training and loading, but `cdskit localize`
runs them on CPU. Training can use `--dl_device auto`, `cpu`, `cuda`, or `mps`.

Example CNN training for the 10-label localization task:

```
python -m cdskit.deeploc_benchmark \
  --download no \
  --prepare no \
  --benchmark yes \
  --task localization \
  --out_dir data/localize_bench/deeploc21 \
  --model_arch cnn \
  --dl_epochs 6 \
  --dl_device auto \
  --model_out cdskit_deeploc_localization_cnn.pt \
  --comparison_md localization_cnn_comparison.md
```

Rare-label recall experiments can be run with a different threshold objective
for labels below a count or frequency cutoff. `f2` gives recall more weight than
precision.

```
python -m cdskit.deeploc_benchmark \
  --prepare no \
  --benchmark yes \
  --task sorting_signals \
  --model_arch centroid \
  --rare_label_threshold_objective f2 \
  --rare_label_max_count 150
```

For CNN training, `--dl_sample_weight_power` samples rows containing rare labels
more often. It is experimental and should be checked against the generated
comparison report before use.

## Local benchmark snapshot

The following metrics are preserved from earlier development on the public
DeepLoc 2.1 prepared data; they have not been rerun for the current version. The DeepLoc published references use
large protein language models and are not CPU-light baselines.

| Task | Model | Split | Rows | Jaccard | Micro F1 | Macro F1 | Subset acc. |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| localization | centroid | SwissProt CV | 28303 | 0.4218 | 0.5105 | 0.4152 | 0.1721 |
| localization | CNN | SwissProt CV | 28303 | 0.4470 | 0.5489 | 0.3980 | 0.1911 |
| localization | centroid | HPA test | 1717 | 0.3529 | 0.4568 | 0.2111 | 0.0996 |
| localization | CNN | HPA test | 1717 | 0.3690 | 0.4733 | 0.2066 | 0.1293 |
| membrane | centroid | SwissProt CV | 28026 | 0.5354 | 0.6275 | 0.4688 | 0.2507 |
| membrane | CNN | SwissProt CV | 28026 | 0.8007 | 0.8147 | 0.5124 | 0.7168 |
| sorting_signals | centroid | SwissProt CV | 1868 | 0.6463 | 0.6963 | 0.6001 | 0.4540 |
| sorting_signals | CNN | SwissProt CV | 1868 | 0.5116 | 0.5547 | 0.3474 | 0.4079 |

Published reference points from
[DeepLoc 2.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC9252801/) and
[DeepLoc 2.1, Table 2](https://academic.oup.com/nar/article/52/W1/W215/7642068)
are included in the generated reports:

- DeepLoc 2.0 localization, SwissProt CV: ESM1b micro F1 0.72, macro F1 0.64;
  ProtT5 micro F1 0.73, macro F1 0.66.
- DeepLoc 2.0 localization, HPA test: ESM1b micro F1 0.57, macro F1 0.44;
  ProtT5 micro F1 0.60, macro F1 0.46.
- DeepLoc 2.1 membrane held-out test: ESM1b micro F1 0.88, macro F1 0.74;
  ProtT5 micro F1 0.89, macro F1 0.75.

These lightweight models are therefore useful CPU baselines, but they do not
match the published transformer-based DeepLoc models.

## TargetP 2.0 comparison

The official [TargetP 2.0](https://services.healthtech.dtu.dk/services/TargetP-2.0/)
page publishes the FASTA sequences and tab-separated annotations
used for its nested cross-validation dataset. The benchmark helper also
downloads the fold metadata from the TargetP 2.0 source repository.

```
python -m cdskit.targetp_benchmark \
  --download yes \
  --run_cdskit_cv yes \
  --model_arch bilstm_attention \
  --localize_strategy single_stage \
  --comparison_md targetp2_cdskit_comparison.md
```

The TargetP task maps to cdskit labels as follows:

| TargetP label | cdskit label |
| --- | --- |
| `Other` | `noTP` |
| `SP` | `SP` |
| `MT` | `mTP` |
| `CH` | `cTP` |
| `TH` | `lTP` |

## TargetP checkpoint compatibility

For new nested OOF training with `python -m cdskit.targetp_torch`, use a fresh
`--model_dir`. Since 0.28, every epoch checkpoint records a fingerprint of the
input data, numerical training configuration, CDSKIT version, and outer and
validation folds. Resuming with `--reuse_oof_cache yes` requires matching
provenance. Extending `--epochs` is allowed; changing data, settings, folds, or
the CDSKIT version requires a new directory and regenerated outputs.

Legacy checkpoints without the fingerprint are not valid OOF resume inputs.
Do not rename or copy old caches into a new run to bypass these checks. This
restriction concerns training resumption, not the ability to run a trusted
published prediction model in its compatible runtime. See the
[0.28 review](https://github.com/kfuku52/cdskit/blob/master/docs/review-0.28.md)
for the checkpoint and validation-loss changes.

## Published models and historical experiments

- [targeting5-v1 model details](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-targeting5-v1)
  describe the release model, its training provenance, and limits of the reported comparisons.
- [Experimental peroxisome head](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-peroxisome-head)
  documents the separate signal-oriented score and external stress tests.
- [Historical development results](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-development-history)
  preserve older TargetP stacks, OOF probes, commands, and failed approaches.
  Those entries are not a claim of current reproducibility or release accuracy.

For CPU execution-time benchmarks rather than prediction accuracy, follow
[TESTING.md](https://github.com/kfuku52/cdskit/blob/master/TESTING.md#benchmarks).

# cdskit localize targeting5-v1

`targeting5-v1` is the pretrained model used by:

```bash
cdskit localize --seq_file proteins.faa --seq_type protein --model targeting5 --report localize.tsv
```

It predicts the TargetP-compatible five-class targeting-peptide labels:

| TargetP label | cdskit label |
| --- | --- |
| `Other` | `noTP` |
| `SP` | `SP` |
| `MT` | `mTP` |
| `CH` | `cTP` |
| `TH` | `lTP` |

The model file is `cdskit-localize-targeting5-v1.pt`. When it is not already
cached, `cdskit localize --model targeting5` downloads it from the cdskit
GitHub Release and verifies its SHA-256 checksum before loading it.

## Dependencies

The pretrained model runs on CPU, but it uses optional ML dependencies at
runtime:

- `torch`
- `scikit-learn`

The embedded sklearn estimators were serialized with scikit-learn 1.5.2.
Use the [dedicated runtime setup](https://github.com/kfuku52/cdskit/wiki/Installation-and-dependencies#pretrained-targeting5-runtime);
an unconstrained install of the latest `ml` extra is not sufficient for these
legacy artifacts. The alias verifies the release checksum before enabling
legacy pickle loading.

## Model cache

The cache root is `$CDSKIT_MODEL_DIR`, else `$XDG_CACHE_HOME/cdskit/models`,
else `~/.cache/cdskit/models`. This artifact is stored under
`localize/targeting5-v1/v1/cdskit-localize-targeting5-v1.pt` within that root.
Its SHA-256 is
`ddaeab7093533a213ee58117b70ad0f45b0c126cf82c77df32e369eaff2beeb2`.
See [offline and safety notes](https://github.com/kfuku52/cdskit/wiki/cdskit-localize#model-safety-and-offline-use).

## Training data

TargetP-2.0 and the cdskit `targeting5-v1` model use the same five target
labels, but they do not use identical training recipes. cdskit does not use
TargetP executable code, trained weights, or fitted parameters.

| Data source | Rows | Class counts | Role |
| --- | ---: | --- | --- |
| TargetP-2.0 public table, `targetp2_benchmark.tsv` | 13,005 | noTP 9,537; SP 2,697; mTP 499; cTP 227; lTP 45 | canonical TargetP-compatible supervised labels for base training |
| Strict external table, `targetp2_external_torch_strict_cleanholdout_seed1seed2_plus_thylum_h128_e6_external.tsv` | 13,983 | noTP 4,684; SP 3,986; mTP 2,914; cTP 1,872; lTP 527 | external augmentation for base PyTorch models |
| Reranker development holdout | 20,000 | 4,000 per class | trains the multiclass HGB reranker and class thresholds |
| mTP/noTP specialist train | 16,000 | mTP 8,000; noTP 8,000 | trains the binary mTP/noTP specialist |
| mTP/noTP specialist validation | 2,000 | mTP 1,000; noTP 1,000 | selects the specialist threshold, currently 0.41 |

## Model structure

| Component | cdskit targeting5-v1 |
| --- | --- |
| Base predictors | Two cdskit-trained TargetP2-style PyTorch models (`sqrt` and `logcw`) |
| Blend | Classwise alpha 0.4 |
| Thresholds | noTP 1.0, SP 0.65, mTP 1.0, cTP 0.8, lTP 1.0 |
| Reranker | scikit-learn HGB multiclass reranker |
| Specialist | scikit-learn HGB mTP/noTP specialist |
| Inference device | CPU |

## Performance snapshot

These are historical development results associated with the published model,
not an official TargetP-2.0 server rerun or a new evaluation of the current
CDSKIT version. The TargetP paper row and the CDSKIT holdouts use different
datasets, class distributions, and evaluation protocols.

| Model / evaluation | Rows | Macro F1 | Accuracy | noTP F1 | SP F1 | mTP F1 | cTP F1 | lTP F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TargetP-2.0 paper Table 1 | - | 0.890 | - | 0.980 | 0.980 | 0.860 | 0.880 | 0.750 |
| cdskit targeting5-v1, seed2 strict external holdout | 1,801 | 0.891 | 0.895 | 0.773 | 0.921 | 0.851 | 0.950 | 0.957 |
| cdskit targeting5-v1, fresh unused balanced holdout | 1,000 | 0.907 | 0.909 | 0.796 | 0.915 | 0.861 | 0.976 | 0.990 |

These numbers do not establish equivalent accuracy to TargetP-2.0. That claim
would require both predictors to be evaluated on the same independently held-out
data with a compatible protocol. This model predicts targeting-peptide classes,
not general DeepLoc localization labels; its original `p_peroxisome` head is a
constant zero, not a trained peroxisome detector.

## Sources

- [TargetP-2.0 data page](https://services.healthtech.dtu.dk/services/TargetP-2.0/3-Data.php)
- [TargetP-2.0 source repository](https://github.com/JJAlmagro/TargetP-2.0)
- [cdskit targeting5-v1 release](https://github.com/kfuku52/cdskit/releases/tag/localize-targeting5-v1)

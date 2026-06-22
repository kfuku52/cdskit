# cdskit localize peroxisome head

`p_peroxisome` is available in the experimental prerelease model
`cdskit-localize-targeting5-perox-deeploc21-et-v1.pt`, registered under the
alias `targeting5-perox-deeploc21-et-v1`. When that alias is used with
`cdskit localize --model`, CDSKIT downloads the model from GitHub Releases into
the local model cache if it is not already present and verifies its SHA-256
checksum.

```bash
cdskit localize \
  --seqfile proteins.faa \
  --seqtype protein \
  --model targeting5-perox-deeploc21-et-v1 \
  --organism_group non_plant \
  --report localize.tsv
```

The peroxisome head is a CPU-runtime scikit-learn ExtraTrees classifier trained
on DeepLoc21 Swiss-Prot train/validation rows with sequence-level features,
including C-terminal PTS-like features. Interpret it as a peroxisome
sequence-label probability that is strongest for PTS-like targeting signals,
not as a general peroxisome-associated localization detector.

## Release asset

- Release: https://github.com/kfuku52/cdskit/releases/tag/localize-targeting5-perox-deeploc21-et-v1
- Asset: `cdskit-localize-targeting5-perox-deeploc21-et-v1.pt`
- SHA-256: `d0998df8819d975b4392342ab78dccc0dd95cf301e4d2df8f38c73d0b5aab445`

## Fairness checks

- External accession and exact-sequence overlap are excluded or reported.
- A constant-zero baseline and a regex PTS baseline are reported.
- MMseqs homology subsets and cluster out-of-fold evaluation are used to expose
  similarity-driven optimism.
- HPA is kept as a difficult broad-localization stress test.

## Candidate comparison

The current candidate uses ExtraTrees because it improved the independent
external and cluster-OOF checks compared with the earlier HGB candidate, while
keeping inference CPU-only.

| Candidate | UniProt external AUPRC | UniProt external F1 | Cluster OOF AUPRC | Cluster OOF F1 | HPA stress AUPRC | HPA stress F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HGB, `perox_sequence_v1` | 0.239 | 0.293 | 0.191 | 0.212 | 0.007 | 0.000 |
| HGB, `broad_localize_v1` | 0.209 | 0.302 | 0.125 | 0.142 | - | - |
| ExtraTrees, `perox_sequence_v1` | 0.256 | 0.370 | 0.246 | 0.277 | 0.149 | 0.250 |

## Evaluation snapshot

| Model / evaluation | Rows | Positives | AUPRC | AUROC | F1 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DeepLoc21 validation, perox head | 5,462 | 53 | 0.482 | 0.895 | 0.583 | threshold tuned on DeepLoc21 partition 4 |
| DeepLoc21 validation, regex PTS | 5,462 | 53 | 0.080 | 0.765 | 0.217 | simple PTS1/PTS2 signal baseline |
| UniProt experimental CC external, perox head | 11,395 | 138 | 0.256 | 0.847 | 0.370 | no accession/exact-sequence overlap with DeepLoc21 training |
| UniProt experimental CC external, regex PTS | 11,395 | 138 | 0.041 | 0.642 | 0.156 | same external rows |
| UniProt experimental CC cluster OOF, perox head | 11,395 | 138 | 0.246 | 0.854 | 0.277 | MMseqs clusters at 30% identity / 80% coverage |
| HPA external stress test, DeepLoc21-trained perox head | 1,717 | 7 | 0.149 | 0.665 | 0.250 | broad peroxisome-associated task; no positive homology hits |

These results support replacing the previous constant-zero `p_peroxisome`
placeholder for signal-like use cases, but they do not yet support advertising
the head as a broad peroxisome-localization model.

## Reproduce

Main candidate model and UniProt experimental CC external benchmark:

```bash
python -m cdskit.perox_benchmark \
  --train_tsv data/localize_bench/deeploc21/deeploc21_localization_train_validation.tsv \
  --external_test_tsv data/localize_bench/eukaryota_full_with_lineage.tsv \
  --external_format uniprot_exp_cc \
  --feature_profile perox_sequence_v1 \
  --model_kind extra_trees \
  --base_model targeting5 \
  --model_out data/localize_bench/perox_deeploc21_et_v1/cdskit-localize-targeting5-perox-deeploc21-et-v1.pt \
  --report_json data/localize_bench/perox_deeploc21_et_v1/perox_benchmark_uniprot_exp_external.json \
  --report_md data/localize_bench/perox_deeploc21_et_v1/perox_benchmark_uniprot_exp_external.md \
  --predictions_prefix data/localize_bench/perox_deeploc21_et_v1/perox_predictions_uniprot_exp \
  --homology_check yes \
  --homology_threads 4 \
  --cluster_oof yes \
  --cluster_oof_source external \
  --cluster_oof_folds 5 \
  --cluster_oof_method mmseqs
```

HPA stress test for the same DeepLoc21-trained candidate:

```bash
python -m cdskit.perox_benchmark \
  --train_tsv data/localize_bench/deeploc21/deeploc21_localization_train_validation.tsv \
  --external_test_tsv data/localize_bench/deeploc21/deeploc21_hpa_test.tsv \
  --external_format prepared \
  --feature_profile perox_sequence_v1 \
  --model_kind extra_trees \
  --homology_check yes \
  --homology_threads 4 \
  --report_json data/localize_bench/perox_deeploc21_et_v1/perox_benchmark_hpa_external.json \
  --report_md data/localize_bench/perox_deeploc21_et_v1/perox_benchmark_hpa_external.md \
  --predictions_prefix data/localize_bench/perox_deeploc21_et_v1/perox_predictions_hpa
```

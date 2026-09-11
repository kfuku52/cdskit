# cdskit localize peroxisome head

The CDSKIT-trained weights are MIT-licensed; the model release includes LICENSE
and SHA256SUMS. The license update leaves the checkpoint bytes and legacy runtime
requirements unchanged. Source datasets retain their original terms.

`p_peroxisome` is available in the experimental prerelease model
`cdskit-localize-targeting5-perox-deeploc21-et-v1.pt`, registered under the
alias `targeting5-perox-deeploc21-et-v1`. When that alias is used with
`cdskit localize --model`, CDSKIT downloads the model from GitHub Releases into
the local model cache if it is not already present and verifies its SHA-256
checksum.

```bash
cdskit localize \
  --seq_file proteins.faa \
  --seq_type protein \
  --model targeting5-perox-deeploc21-et-v1 \
  --organism_group non_plant \
  --report localize.tsv
```

The peroxisome head is a CPU-runtime scikit-learn ExtraTrees classifier trained
on DeepLoc21 Swiss-Prot train/validation rows with sequence-level features,
including C-terminal PTS-like features. Its `p_peroxisome` output is a learned
sequence-label score, not an independently calibrated biological probability.
The historical evaluation below does not establish broad peroxisome-associated
localization accuracy, and a PTS motif match alone does not prove targeting.

## Release asset

Use the [pretrained runtime setup](https://github.com/kfuku52/cdskit/wiki/Installation-and-dependencies#pretrained-targeting5-runtime).
This artifact contains legacy scikit-learn 1.5.2 estimators. Current CDSKIT
supports their known loss layout on scikit-learn 1.9.0 and has been compared
against the 1.5.2 reference on a retained synthetic fixture.

- [GitHub model release](https://github.com/kfuku52/cdskit/releases/tag/localize-targeting5-perox-deeploc21-et-v1)
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

These are historical development results, not metrics rerun for the current
CDSKIT version. The validation threshold was selected on that validation set;
use the independent external and cluster evaluations to assess generalization.

| Model / evaluation | Rows | Positives | AUPRC | AUROC | F1 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DeepLoc21 validation, perox head | 5,462 | 53 | 0.482 | 0.895 | 0.583 | threshold tuned on DeepLoc21 `fold_id=4` |
| DeepLoc21 validation, regex PTS | 5,462 | 53 | 0.080 | 0.765 | 0.217 | simple PTS1/PTS2 signal baseline |
| UniProt experimental CC external, perox head | 11,395 | 138 | 0.256 | 0.847 | 0.370 | no accession/exact-sequence overlap with DeepLoc21 training |
| UniProt experimental CC external, regex PTS | 11,395 | 138 | 0.041 | 0.642 | 0.156 | same external rows |
| UniProt experimental CC cluster OOF, perox head | 11,395 | 138 | 0.246 | 0.854 | 0.277 | MMseqs clusters at 30% identity / 80% coverage |
| HPA external stress test, DeepLoc21-trained perox head | 1,717 | 7 | 0.149 | 0.665 | 0.250 | broad peroxisome-associated task; no positive homology hits |

These results support replacing the previous constant-zero `p_peroxisome`
placeholder for signal-like use cases, but they do not yet support advertising
the head as a broad peroxisome-localization model.

## Reproduce

These are research commands that need prepared inputs, optional ML dependencies,
and an `mmseqs` executable on `PATH`. Data files below are not included in the
package. Download/prepare the DeepLoc tables with
`python -m cdskit.deeploc_benchmark --download yes --prepare yes --benchmark no`
and supply the indicated UniProt snapshot separately. A new live UniProt query
does not reproduce a frozen historical dataset exactly.

Main candidate model and UniProt experimental CC external benchmark:

```bash
python -m cdskit.perox_benchmark \
  --training_tsv data/localize_bench/deeploc21/deeploc21_localization_train_validation.tsv \
  --external_test_tsv data/localize_bench/eukaryota_full_with_lineage.tsv \
  --external_format uniprot_exp_cc \
  --feature_profile perox_sequence_v1 \
  --model_kind extra_trees \
  --base_model targeting5 \
  --model_out data/localize_bench/perox_deeploc21_et_v1/cdskit-localize-targeting5-perox-deeploc21-et-v1.pt \
  --out_json data/localize_bench/perox_deeploc21_et_v1/perox_benchmark_uniprot_exp_external.json \
  --out_md data/localize_bench/perox_deeploc21_et_v1/perox_benchmark_uniprot_exp_external.md \
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
  --training_tsv data/localize_bench/deeploc21/deeploc21_localization_train_validation.tsv \
  --external_test_tsv data/localize_bench/deeploc21/deeploc21_hpa_test.tsv \
  --external_format prepared \
  --feature_profile perox_sequence_v1 \
  --model_kind extra_trees \
  --homology_check yes \
  --homology_threads 4 \
  --out_json data/localize_bench/perox_deeploc21_et_v1/perox_benchmark_hpa_external.json \
  --out_md data/localize_bench/perox_deeploc21_et_v1/perox_benchmark_hpa_external.md \
  --predictions_prefix data/localize_bench/perox_deeploc21_et_v1/perox_predictions_hpa
```

## Feature and decision versions

See the [localization feature and decision contract](https://github.com/kfuku52/cdskit/blob/master/docs/localize-scientific-contract.md)
for the corrected nine-residue PTS2 definition, legacy model compatibility,
`--decision_policy safe-v1`, v2 output and explicit taxonomy constraints.
Published artifacts retain their historical defaults; new staged pipeline models
default to safe inference with `ensure_one_label=False`. Scores remain unverified
as calibrated biological probabilities.

## References

Cite the method or resource actually used, in addition to the CDSKIT version and
command. Papers below do not validate newly trained CDSKIT models or new options.

- Thumuluri V et al. (2022). DeepLoc 2.0: multi-label subcellular localization prediction using protein language models. *Nucleic Acids Research* 50:W228–W234. [Paper](https://doi.org/10.1093/nar/gkac278) — Localization labels, sorting-signal data and protein-language-model methodology where used.
- Ødum M et al. (2024). DeepLoc 2.1: multi-label membrane protein type prediction using protein language models. *Nucleic Acids Research* 52:W215–W220. [Paper](https://doi.org/10.1093/nar/gkae237) — DeepLoc 2.1 data/partition provenance; citing the dataset does not mean CDSKIT executes the DeepLoc predictor.
- Geurts P, Ernst D, Wehenkel L (2006). Extremely randomized trees. *Machine Learning* 63:3–42. [Paper](https://doi.org/10.1007/s10994-006-6226-1) — ExtraTrees method used for the peroxisome/specialist head.
- The UniProt Consortium (2025). UniProt: the Universal Protein Knowledgebase in 2025. *Nucleic Acids Research* 53:D609–D617. [Paper](https://doi.org/10.1093/nar/gkae1010) — Protein sequences and annotations; also report the downloaded snapshot/query and evidence filters.
- Thul PJ et al. (2017). A subcellular map of the human proteome. *Science* 356:eaal3321. [Paper](https://doi.org/10.1126/science.aal3321) — Human Protein Atlas Cell Atlas annotation resource for the HPA evaluations; report the actual snapshot and label mapping.
- Steinegger M, Söding J (2017). MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. *Nature Biotechnology* 35:1026–1028. [Paper](https://doi.org/10.1038/nbt.3988) — Cite when MMseqs2 is used for homology filtering or clustering in the research workflow.
- Flynn CR, Mullen RT, Trelease RN (1998). Mutational analyses of a type 2 peroxisomal targeting signal that is capable of directing oligomeric protein import into tobacco BY-2 glyoxysomes. *The Plant Journal* 16:709–720. [Paper](https://doi.org/10.1046/j.1365-313x.1998.00344.x) — Experimental PTS2 nonapeptide evidence; the CDSKIT motif/window detector is a heuristic.
- Gonzalez NH et al. (2011). A Single Peroxisomal Targeting Signal Mediates Matrix Protein Import in Diatoms. *PLOS ONE* 6:e25316. [Paper](https://doi.org/10.1371/journal.pone.0025316) — Evidence that PTS2-mediated import is not universal across taxa; motif matches alone do not prove localization.

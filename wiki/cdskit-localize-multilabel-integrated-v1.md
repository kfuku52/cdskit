# Integrated ten-label localization model v1

This published CDSKIT model combines a CNN with ten sequence-feature specialists.
It predicts nucleus, cytoplasm, extracellular, mitochondrion, cell_membrane,
endoplasmic_reticulum, chloroplast, golgi_apparatus, lysosome_vacuole and peroxisome.
Multiple labels can be predicted for one protein.

## Install and download

Install CDSKIT >=0.29.0 and PyTorch using the
[integrated runtime setup](https://github.com/kfuku52/cdskit/wiki/Installation-and-dependencies#integrated-ten-label-model-runtime).
Inference runs on CPU without scikit-learn, Transformers or an ESM download.

Download the `.pt` checkpoint from the
[model release](https://github.com/kfuku52/cdskit/releases/tag/localize-multilabel-integrated-v1).
The release also includes MODEL_CARD.md, evaluation.json, LICENSE and SHA256SUMS.
For example, in a POSIX shell:

```bash
curl --fail --location --output cdskit-localize-multilabel-integrated-v1.pt \
  https://github.com/kfuku52/cdskit/releases/download/localize-multilabel-integrated-v1/cdskit-localize-multilabel-integrated-v1.pt
shasum -a 256 cdskit-localize-multilabel-integrated-v1.pt
```

The checkpoint SHA-256 must be:

```text
e9b35ead3eca4dcdf833e469f18f1d67e05a3de274b00861cbf08bf138d01621
```

## Predict

```bash
cdskit localize --seq_file proteins.faa --seq_type protein \
  --model cdskit-localize-multilabel-integrated-v1.pt \
  --threads 1 --report localization.tsv
```

For CDS input, use `--seq_type dna` (the default). There is no short download
alias for this model; use the downloaded file path. Keep the default safe loading
setting. After installation and download, inference can run offline.

Output contains `seq_id`, semicolon-separated `predicted_labels`, ten `p_LABEL`
columns and `perox_signal_type`. The model uses a separate threshold for each
label and always returns at least one label. A `.json` report contains the same
row objects; `predicted_labels` remains a string.

## Evaluation and limits

Five-fold development evaluation across three seeds gave mean macro/micro F1
0.5040/0.5965, compared with 0.4028/0.5357 for the baseline. On the previously
inspected 1,717-protein HPA set, macro F1 was effectively unchanged
(0.2195 to 0.2203), while micro F1 improved (0.4974 to 0.5327).
Peroxisome F1 was zero on HPA's seven positives. On 115 additional experimental
UniProt positives, recall was 25/115 versus baseline 31/115; that positive-only
panel cannot estimate precision or F1.

These results do not establish universal improvement or calibrated biological
confidence. The model has no abstention or taxonomy masking: `--organism_group`
does not suppress chloroplast predictions in this ten-label model. See the
[full experiment and audit](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-full-experiment)
and the release model card for split design, homology checks and limitations.

## License

The CDSKIT-trained weights are MIT-licensed. The release's model card preserves
UniProt, DeepLoc and HPA attribution; source datasets retain their own terms.
The MIT license update changed no checkpoint bytes.

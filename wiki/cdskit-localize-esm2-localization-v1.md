# ESM2 localization baseline v1

`esm2-localization-v1` is the default ten-label localization model in cdskit
0.30.3. It uses a frozen ESM2 650M encoder and a Light Attention classifier
trained with ordinary binary cross-entropy (BCE). Validation BCE selects the
training epoch, and validation data select independent label thresholds.
Weighted BCE, ASL, label attention, terminal attention, and macro-AP epoch
selection are not used in this checkpoint.

## Use and distribution

Install cdskit with its `ml` extra, or `ml-cpu` for CPU-only use, following the
[installation guide](https://github.com/kfuku52/cdskit/wiki/Installation-and-dependencies). Then run:

```sh
cdskit localize --seq_file proteins.faa --seq_type protein --threads 8 --report predictions.tsv
```

The model alias downloads the approximately 12 MB checkpoint from the
[model release](https://github.com/kfuku52/cdskit/releases/tag/localize-esm2-localization-v1)
and verifies its SHA-256:

```text
534e0e096c95bede49be56e3bbde5a070c41684c86db6f993bdecbfc1c93131d
```

The ESM2 backbone is downloaded separately from
[`facebook/esm2_t33_650M_UR50D`](https://huggingface.co/facebook/esm2_t33_650M_UR50D/tree/08e4846e537177426273712802403f7ba8261b6c),
revision `08e4846e537177426273712802403f7ba8261b6c`. Allow several GB of disk space
for backbone weights and several GB of RAM. CPU inference is supported; CUDA
is not required. The checkpoint loads with the normal safe reader, without
`--allow_unsafe_model`.

For offline use, first run an online prediction to populate both the cdskit
checkpoint cache and the Hugging Face backbone cache. Then set
`CDSKIT_OFFLINE=1`. `--model_download no` alone only disables the cdskit
checkpoint download; it does not by itself disable backbone downloads.
`CDSKIT_MODEL_DIR` changes the cdskit checkpoint cache root; Hugging Face's cache
settings control the backbone cache. This exported checkpoint contains no
training-machine residue-cache path and does not force local-only downloads.

## Training and evaluation

Training source: cdskit 0.30.2, commit
`a27f868e814351747b7dcb44bc74300051924b96`. The corpus is the DeepLoc-derived
27,763-protein, ten-label dataset after removing 540 proteins in similarity
clusters crossing the original folds. The dataset's closed-world label
convention is retained; unlisted labels are not experimental proof of absence.

The released checkpoint was trained on 22,412 proteins, with a separate 5,351
proteins for epoch and threshold selection. Seed 1, AdamW learning rate 0.001,
weight decay 0.0001, batch size 8, at most 12 epochs, patience 3; epoch 3 was
selected. ESM2 residue windows are 1,000 residues with 128-residue overlap.
Overlapping embeddings are averaged. The head uses width-9 feature and
attention convolutions, 128 hidden channels, dropout 0.25, attention-weighted
and max pooling, and ten sigmoid outputs.

Five-fold evaluation of the **baseline recipe**, repeated with three seeds,
gave pooled-test macro-F1 0.65397, micro-F1 0.72813, and macro-AP 0.67110
(means across seeds). These scores come from separately trained fold models;
they are not a test of the released checkpoint on its own training data.
Previously examined outer folds make this a development comparison rather than
an untouched confirmation study.

The released checkpoint's independent HPA evaluation used 537 proteins,
excluding previously used sequences and homologs, with six common labels.
Observed-annotation agreement was macro-F1 0.6251, micro-F1 0.7095, and macro-AP
0.6304. HPA annotations are incomplete: these values do not establish that
unlisted localizations are absent. Minority-label support is limited.

On the validation server, full CPU inference for 32 proteins using 8 CPU threads
had a median wall time of 69.71 seconds and peak RSS of 3.44 GiB. Weights were
already downloaded; one warmup pair was excluded and three fresh CLI runs were
measured. These timings depend on sequence lengths and hardware.

## Scope and limitations

The outputs are nucleus, cytoplasm, extracellular, mitochondrion, cell membrane,
endoplasmic reticulum, chloroplast, Golgi apparatus, lysosome/vacuole, and
peroxisome. Multiple locations or no accepted location are possible. The
`safe-v1` policy preserves abstentions; it does not force one label per protein.
Scores are model outputs, not independently calibrated probabilities.
Localization predictions do not establish targeting-peptide presence or
cleavage sites.

Experimental improvement candidates were evaluated but not promoted: their
macro-F1 gain did not meet the predeclared target and macro-AP declined in the
full retraining comparison. The default intentionally remains the baseline.

To re-export a frozen-PLM checkpoint with a pinned remote encoder:

```sh
python -m scripts.export_localize_plm trained-model.pt portable-model.pt
```

Run this from the source checkout with ML dependencies installed. The exporter
preserves weights, thresholds and encoder identity, clears only the residue
cache path and local-only flag, and rejects local or changed encoder identities.

## References

Cite the method or resource actually used, in addition to the CDSKIT version and
command. Papers below do not validate newly trained CDSKIT models or new options.

- Lin Z et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379:1123–1130. [Paper](https://doi.org/10.1126/science.ade2574) — ESM2 encoder used by frozen-encoder models; CDSKIT does not run the paper’s structure predictor.
- Stärk H, Dallago C, Heinzinger M, Rost B (2021). Light attention predicts protein location from the language of life. *Bioinformatics Advances* 1:vbab035. [Paper](https://doi.org/10.1093/bioadv/vbab035) — Light Attention architecture adapted for the CDSKIT multilabel head.
- Thumuluri V et al. (2022). DeepLoc 2.0: multi-label subcellular localization prediction using protein language models. *Nucleic Acids Research* 50:W228–W234. [Paper](https://doi.org/10.1093/nar/gkac278) — Localization labels, sorting-signal data and protein-language-model methodology where used.
- Ødum M et al. (2024). DeepLoc 2.1: multi-label membrane protein type prediction using protein language models. *Nucleic Acids Research* 52:W215–W220. [Paper](https://doi.org/10.1093/nar/gkae237) — DeepLoc 2.1 data/partition provenance; citing the dataset does not mean CDSKIT executes the DeepLoc predictor.
- The UniProt Consortium (2025). UniProt: the Universal Protein Knowledgebase in 2025. *Nucleic Acids Research* 53:D609–D617. [Paper](https://doi.org/10.1093/nar/gkae1010) — Protein sequences and annotations; also report the downloaded snapshot/query and evidence filters.
- Thul PJ et al. (2017). A subcellular map of the human proteome. *Science* 356:eaal3321. [Paper](https://doi.org/10.1126/science.aal3321) — Human Protein Atlas Cell Atlas annotation resource for the HPA evaluations; report the actual snapshot and label mapping.

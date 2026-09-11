# cdskit localize

`cdskit localize` predicts protein targeting or localization labels from CDS or
protein FASTA input. It can use a local model file or a published pretrained
model alias.

The default is `esm2-localization-v1`: a frozen ESM2 650M encoder plus a trained
ten-label localization head. The checksum-verified checkpoint downloads on first
use; ESM2 backbone weights are fetched separately at an immutable revision.
An explicit model path or published alias overrides the default. Install the
`ml` or `ml-cpu` extra for PyTorch and Transformers.

The default checkpoint uses the baseline recipe: Light Attention, ordinary BCE
loss, and epoch selection by validation BCE. Experimental improvement candidates
are not adopted as the default; alternative losses, attention heads, and macro-AP
epoch selection remain opt-in training options. See the
[model card](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-esm2-localization-v1) for evaluation and offline use.

## Choose a model

| Model | Prediction task | Runtime |
| --- | --- | --- |
| `esm2-localization-v1` (default) | Ten subcellular locations, multiple labels per protein | ESM2 650M + baseline Light Attention head; PyTorch and Transformers; automatic download |
| [Integrated v1](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-multilabel-integrated-v1) | Ten subcellular locations, multiple labels per protein | CDSKIT >=0.29.0 and PyTorch; downloaded file path |
| `targeting5` | Five targeting-peptide classes, one class per protein | PyTorch and scikit-learn; verified with 1.5.2 and 1.9.0; registered alias |
| `targeting5-perox-deeploc21-et-v1` | Five targeting-peptide classes plus a peroxisome score | Same legacy runtime; experimental alias |

The three published CDSKIT-trained model releases are MIT-licensed. Source datasets retain
their own licenses and attribution requirements. The ten-label model's peroxisome
recall is not consistently better than its baseline; choose based on the prediction
task and inspect the model-specific evaluation.

## Examples

### Published ten-label localization model

Follow the [download and runtime instructions](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-multilabel-integrated-v1), then run:

```bash
cdskit localize --seq_file proteins.faa --seq_type protein \
  --model cdskit-localize-multilabel-integrated-v1.pt \
  --threads 1 --report localization.tsv
```

This model uses a local path, not a registered download alias. It loads safely
without scikit-learn, Transformers, or `--allow_unsafe_model yes`.

### Pretrained TargetP-compatible model

The pretrained `targeting5` model predicts `noTP`, `SP`, `mTP`, `cTP`, and
`lTP`. It runs on CPU. First follow the
[pretrained runtime setup](https://github.com/kfuku52/cdskit/wiki/Installation-and-dependencies#pretrained-targeting5-runtime):
the targeting5 release artifacts need scikit-learn as well as torch. Current
CDSKIT restores their legacy loss objects on scikit-learn 1.9.0, with prediction
checks against 1.5.2. This does not establish compatibility with every version.

```bash
cdskit localize \
  --seq_file proteins.faa \
  --seq_type protein \
  --model targeting5 \
  --report localize.tsv
```

### Experimental peroxisome-head model

The prerelease alias `targeting5-perox-deeploc21-et-v1` adds a CPU-runtime
ExtraTrees peroxisome sequence-label head to the `targeting5` model. It is
strongest for PTS-like peroxisomal targeting signals and should not yet be
treated as a broad peroxisome-associated localization detector.

```bash
cdskit localize \
  --seq_file proteins.faa \
  --seq_type protein \
  --model targeting5-perox-deeploc21-et-v1 \
  --organism_group non_plant \
  --report localize.tsv
```

CDS input is translated in frame before prediction:

```bash
cdskit localize \
  --seq_file cds.fasta \
  --model targeting5 \
  --report localize.tsv
```

### Local model file

```bash
cdskit localize \
  --seq_file proteins.faa \
  --seq_type protein \
  --model localize_model.json \
  --report localize.tsv
```

### Example input and output

Example protein FASTA input:

```fasta
>seq_sp
MKKLLLLLLLLLLAVAVAASAASA
>seq_mtp
MRRKRRAARAKRRNQAAARRRAA
```

Example command:

```bash
cdskit localize --seq_file proteins.faa --seq_type protein --model targeting5 --report localize.tsv
```

Example TSV output, with probabilities shortened for readability:

```tsv
seq_id	predicted_class	p_noTP	p_SP	p_mTP	p_cTP	p_lTP	p_peroxisome	perox_signal_type
seq_sp	SP	0.0004	0.9979	0.0009	0.0004	0.0004	0.0	none
seq_mtp	mTP	0.0742	0.0099	0.9122	0.0007	0.0029	0.0	none
```

## Input requirements

- CDS input is the default.
- CDS sequences must be DNA, in frame, and have no internal stop codons.
- Protein input can be used with `--seq_type protein`.
- Use `--codon_table INT` when translating CDS with a non-standard genetic code.

## Key options

- `--seq_file PATH`: Input FASTA. Use `-` for standard input.
- `--seq_type dna|protein`: Input sequence type. The default is `dna`.
- `--model PATH|ALIAS`: Model file path or pretrained alias. Defaults to `esm2-localization-v1`; downloads the baseline checkpoint on first use.
- `--report PATH`: Output report. Use `-` for standard output. `.json` writes JSON; other suffixes write TSV.
- `--organism_group unknown|plant|non_plant`: Optional organism group used to constrain targeting-model cTP/lTP predictions. The integrated ten-label model does not mask chloroplast predictions by taxonomy.
- `--include_features yes|no`: Include internal feature values in the output report.
- `--model_download yes|no`: Allow checksum-verified downloads for pretrained aliases. Downloads are enabled unless `CDSKIT_OFFLINE` is set; use `no` for offline-only operation.
- `--threads INT`: Requested CPU workers/ML threads. `0` detects CPUs available
  to the process (respecting CPU affinity on Linux), up to
  the safety limit (64 by default); small workloads may run serially.
- `--allow_unsafe_model yes|no`: Permit pickle loading for a trusted local
  legacy model. The default is `no`; see the safety notes below.

## Output

For the pretrained `targeting5` model and other single-label targeting-peptide
models, the TSV report includes:

- `seq_id`
- `predicted_class`
- `p_noTP`, `p_SP`, `p_mTP`, `p_cTP`, `p_lTP`
- `p_peroxisome`
- `perox_signal_type`

Compatible multi-label models instead write `predicted_labels` and one
probability column per model label. `predicted_labels` is a semicolon-separated
string in both TSV and JSON rows, not a JSON array.

TSV output is UTF-8, tab-delimited, rectangular, and LF-terminated. JSON output
contains the same row objects. In targeting models, `p_peroxisome` is a separate binary-head score;
it does not replace `predicted_class`. `perox_signal_type` describes the
detected PTS-like signal category when the loaded model provides that feature.
In the integrated model, `p_peroxisome` is one of the ten localization probabilities
and its threshold determines whether `peroxisome` appears in `predicted_labels`.
The original `targeting5` artifact has a constant-zero peroxisome head; a zero
there is not evidence that a protein is absent from peroxisomes. Use a model
with a trained peroxisome head when that score is needed. Scores and thresholded
classes should not be interpreted as calibrated biological confidence without
an independent evaluation.

## Model safety and offline use

Local model files first use JSON or PyTorch's restricted `weights_only` loader.
Some legacy and sklearn-containing `.pt` files need pickle deserialization,
which can execute code. Use `--allow_unsafe_model yes` only for a file you trust;
do not enable it merely to silence an error from an unknown model.

Registered pretrained aliases are treated as trusted artifacts only after
their SHA-256 is verified. They enable legacy loading internally, so you do
not need to add the unsafe flag when using those aliases. An explicit local
path to the same legacy file does not receive the alias's automatic trust.
Checksums verify the registered bytes; they do not make arbitrary pickle files
safe. See [runtime compatibility](https://github.com/kfuku52/cdskit/wiki/Installation-and-dependencies#pretrained-targeting5-runtime)
for the legacy targeting5 release models. The integrated v1 checkpoint uses
restricted loading and does not need the legacy runtime.

The cache root is `$CDSKIT_MODEL_DIR` when set, then
`$XDG_CACHE_HOME/cdskit/models`, otherwise `~/.cache/cdskit/models`.
Artifacts live below `localize/MODEL_NAME/v1/FILENAME` within that root. These
paths follow the same rule on macOS and Windows, not the OS-specific native
cache directory.

Concurrent jobs sharing this cache use a per-model file lock. One process
downloads and verifies the model; waiting processes recheck and reuse the
completed file. Downloads are published by atomic replacement only after the
SHA-256 check succeeds. A terminated downloader does not leave an OS lock held;
an interrupted temporary file is never treated as a completed model. Do not
delete lock files while jobs are running.

ESM weights use the standard Hugging Face Hub cache and its per-blob download
locks, via Transformers. Concurrent jobs may make separate metadata requests,
but reuse the same downloaded weights. This is a separate cache from
`CDSKIT_MODEL_DIR`: configure `HF_HUB_CACHE` (or `HF_HOME`) consistently across
jobs as well. For Slurm jobs on multiple nodes, both cache locations must be on
a shared filesystem that supports file locking across those nodes. Separate
node-local caches will each download their own copy. See the
[Hugging Face cache documentation](https://huggingface.co/docs/hub/en/local-cache).

`--model_download no` disables alias downloads and downloads by nested ESM
predictors. `CDSKIT_OFFLINE=1` also disables those downloads. Copy or populate
the cache before going offline; ESM encoders need their own Hugging Face cache
or a local directory recorded with `--esm_model_local_dir` during training.
Offline flags do not bypass checksum validation or make missing assets optional.

## Model aliases

| Alias | Labels | Notes |
| --- | --- | --- |
| `targeting5` | `noTP`, `SP`, `mTP`, `cTP`, `lTP` | TargetP-compatible pretrained model; downloaded and checksum-verified on first use |
| `esm2-localization-v1`, `esm2-localization` | Ten subcellular localization labels | Default ESM2 650M + baseline Light Attention head; checksum-verified download |
| `targeting5-perox-deeploc21-et-v1` | `noTP`, `SP`, `mTP`, `cTP`, `lTP`, `p_peroxisome` | Experimental prerelease with a DeepLoc21-trained ExtraTrees peroxisome sequence-label head; downloaded and checksum-verified on first use |

## Training custom models

`cdskit localize-learn` can train lightweight custom models from a TSV table or
from UniProt entries downloaded by query.

```bash
cdskit localize-learn \
  --training_tsv train.tsv \
  --seq_col sequence \
  --label_mode explicit \
  --localization_col localization \
  --perox_col peroxisome \
  --model_out localize_model.json
```

See [cdskit localize-learn](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-learn)
for input table format, UniProt download mode, model choices, and fair
evaluation options.

## Related pages

- [targeting5-v1 model details](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-targeting5-v1)
- [training custom localize models](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-learn)
- [experimental peroxisome head](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-peroxisome-head)
- [localize benchmarks and development notes](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-benchmarks-and-notes)

## Multi-label evaluation and research

For development-fitted decision thresholds, CNN terminal/window comparisons, frozen
ESM heads and audited specialist integration, see
[localization improvements](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-improvements)
and the [full-data experiment](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-full-experiment).

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

- Lin Z et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379:1123–1130. [Paper](https://doi.org/10.1126/science.ade2574) — ESM2 encoder used by frozen-encoder models; CDSKIT does not run the paper’s structure predictor.
- Stärk H, Dallago C, Heinzinger M, Rost B (2021). Light attention predicts protein location from the language of life. *Bioinformatics Advances* 1:vbab035. [Paper](https://doi.org/10.1093/bioadv/vbab035) — Light Attention architecture adapted for the CDSKIT multilabel head.
- Thumuluri V et al. (2022). DeepLoc 2.0: multi-label subcellular localization prediction using protein language models. *Nucleic Acids Research* 50:W228–W234. [Paper](https://doi.org/10.1093/nar/gkac278) — Localization labels, sorting-signal data and protein-language-model methodology where used.
- Ødum M et al. (2024). DeepLoc 2.1: multi-label membrane protein type prediction using protein language models. *Nucleic Acids Research* 52:W215–W220. [Paper](https://doi.org/10.1093/nar/gkae237) — DeepLoc 2.1 data/partition provenance; citing the dataset does not mean CDSKIT executes the DeepLoc predictor.
- Almagro Armenteros JJ et al. (2019). Detecting sequence signals in targeting peptides using deep learning. *Life Science Alliance* 2:e201900429. [Paper](https://doi.org/10.26508/lsa.201900429) — TargetP 2.0 methodology and targeting-peptide dataset; CDSKIT targeting5 weights are separately trained, not the official TargetP predictor.
- Flynn CR, Mullen RT, Trelease RN (1998). Mutational analyses of a type 2 peroxisomal targeting signal that is capable of directing oligomeric protein import into tobacco BY-2 glyoxysomes. *The Plant Journal* 16:709–720. [Paper](https://doi.org/10.1046/j.1365-313x.1998.00344.x) — Experimental PTS2 nonapeptide evidence; the CDSKIT motif/window detector is a heuristic.
- Gonzalez NH et al. (2011). A Single Peroxisomal Targeting Signal Mediates Matrix Protein Import in Diatoms. *PLOS ONE* 6:e25316. [Paper](https://doi.org/10.1371/journal.pone.0025316) — Evidence that PTS2-mediated import is not universal across taxa; motif matches alone do not prove localization.

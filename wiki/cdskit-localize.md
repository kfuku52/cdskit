# cdskit localize

`cdskit localize` predicts protein targeting or localization labels from CDS or
protein FASTA input. It can use a local model file or a published pretrained
model alias.

## Examples

### Pretrained TargetP-compatible model

The pretrained `targeting5` model predicts `noTP`, `SP`, `mTP`, `cTP`, and
`lTP`. It runs on CPU. First follow the
[pretrained runtime setup](https://github.com/kfuku52/cdskit/wiki/Installation-and-dependencies#pretrained-targeting5-runtime):
the existing release artifacts need scikit-learn 1.5.2 as well as torch, and
do not load with every newer scikit-learn version.

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
- `--model PATH|ALIAS`: Model file path or pretrained alias such as `targeting5`.
- `--report PATH`: Output report. Use `-` for standard output. `.json` writes JSON; other suffixes write TSV.
- `--organism_group unknown|plant|non_plant`: Optional organism group used to constrain plant-only cTP/lTP predictions.
- `--include_features yes|no`: Include internal feature values in the output report.
- `--model_download yes|no`: Allow checksum-verified downloads for pretrained aliases. The default is `yes`; use `no` for offline-only operation.
- `--threads INT`: Requested CPU workers/ML threads. `0` detects CPUs up to
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
contains the same row objects. `p_peroxisome` is a separate binary-head score;
it does not replace `predicted_class`. `perox_signal_type` describes the
detected PTS-like signal category when the loaded model provides that feature.
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
for the existing release models.

The cache root is `$CDSKIT_MODEL_DIR` when set, then
`$XDG_CACHE_HOME/cdskit/models`, otherwise `~/.cache/cdskit/models`.
Artifacts live below `localize/MODEL_NAME/v1/FILENAME` within that root. These
paths follow the same rule on macOS and Windows, not the OS-specific native
cache directory.

`--model_download no` disables alias downloads and downloads by nested ESM
predictors. `CDSKIT_OFFLINE=1` also disables those downloads. Copy or populate
the cache before going offline; ESM encoders need their own Hugging Face cache
or a local directory recorded with `--esm_model_local_dir` during training.
Offline flags do not bypass checksum validation or make missing assets optional.

## Model aliases

| Alias | Labels | Notes |
| --- | --- | --- |
| `targeting5` | `noTP`, `SP`, `mTP`, `cTP`, `lTP` | TargetP-compatible pretrained model; downloaded and checksum-verified on first use |
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

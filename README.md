![](logo/logo_cdskit_large.png)

[![Run Tests](https://github.com/kfuku52/cdskit/actions/workflows/test.yml/badge.svg)](https://github.com/kfuku52/cdskit/actions/workflows/test.yml)
[![GitHub release](https://img.shields.io/github/v/tag/kfuku52/cdskit?label=release)](https://github.com/kfuku52/cdskit/releases)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/cdskit.svg)](https://anaconda.org/bioconda/cdskit)
[![Python](https://img.shields.io/badge/python-3.10--3.14-blue)](https://github.com/kfuku52/cdskit)
[![Platforms](https://img.shields.io/conda/pn/bioconda/cdskit.svg)](https://anaconda.org/bioconda/cdskit)
[![Downloads](https://img.shields.io/conda/dn/bioconda/cdskit.svg)](https://anaconda.org/bioconda/cdskit)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Overview
**CDSKIT** ([/sidieskit/](http://ipa-reader.xyz/?text=sidieskit&voice=Joanna)) is a Python program that processes DNA sequences, especially protein-coding sequences. Many functions of this program are designed to handle DNA sequences using codons (sets of three nucleotides) as the unit, and therefore, edits the coding sequences without causing a frameshift. Input and output format names follow [Biopython SeqIO](https://biopython.org/wiki/SeqIO), but transformed records can only be written to formats whose required annotations remain meaningful. In particular, translations and length-changing operations should normally use FASTA rather than FASTQ.

## Installation
Published stable versions of CDSKIT are available from
[Bioconda](https://anaconda.org/bioconda/cdskit). The `master` branch can be
newer than the latest Bioconda package. For users requiring a `conda`
installation, please refer to [Miniforge](https://github.com/conda-forge/miniforge)
for a lightweight conda environment.

#### Install from Bioconda
```
conda install bioconda::cdskit
```

#### Verify the installation by displaying the available options
```
cdskit -h 
```

#### (For advanced users) Install the development version from GitHub
```
pip install git+https://github.com/kfuku52/cdskit
```

Pretrained `cdskit localize` TargetP-style `.pt` models run on CPU, but they
need the optional machine-learning dependencies (`torch` and `scikit-learn`).
CDSKIT is not published on PyPI. To install the GitHub development version
with the `ml` extra, include the repository URL explicitly:

```
pip install 'cdskit[ml] @ git+https://github.com/kfuku52/cdskit.git'
```

## Subcommands
See [Wiki](https://github.com/kfuku52/cdskit/wiki) for detailed descriptions.

- [`accession2fasta`](https://github.com/kfuku52/cdskit/wiki/cdskit-accession2fasta): Retrieving fasta sequences from a list of GenBank accessions

- [`aggregate`](https://github.com/kfuku52/cdskit/wiki/cdskit-aggregate): Extracting the longest sequences combined with a sequence name regex

- [`backalign`](https://github.com/kfuku52/cdskit/wiki/cdskit-backalign): Back-aligning CDS from unaligned CDS + aligned proteins

- [`backtrim`](https://github.com/kfuku52/cdskit/wiki/cdskit-backtrim): Back-translating a trimmed protein alignment

- [`codonstats`](https://github.com/kfuku52/cdskit/wiki/cdskit-codonstats): Printing codon-aware per-sequence and aggregate codon-usage statistics

- [`degeneracy`](https://github.com/kfuku52/cdskit/wiki/cdskit-degeneracy): Extracting aligned 0/2/3/4-fold degenerate nucleotide positions

- [`filter`](https://github.com/kfuku52/cdskit/wiki/cdskit-filter): Filtering CDS by sequence-level quality rules

- [`gapjust`](https://github.com/kfuku52/cdskit/wiki/cdskit-gapjust): Adjusting consecutive Ns to the fixed length

- [`hammer`](https://github.com/kfuku52/cdskit/wiki/cdskit-hammer): Removing less-occupied codon columns from a gappy alignment

- [`intersection`](https://github.com/kfuku52/cdskit/wiki/cdskit-intersection): Dropping non-overlapping sequence labels between two sequences files or between a sequence file and a gff file

- [`label`](https://github.com/kfuku52/cdskit/wiki/cdskit-label): Modifying sequence labels

- [`longestorf`](https://github.com/kfuku52/cdskit/wiki/cdskit-longestorf): Finding the longest ORF by six-frame translation (+/- strands, 3 frames each)

- [`localize`](https://github.com/kfuku52/cdskit/wiki/cdskit-localize): Predicting targeting peptide classes (`noTP`, `SP`, `mTP`, `cTP`, `lTP`) or compatible multi-label localization models from CDS or protein input

- `localize-learn`: Training a `localize` model from tab-separated data, or auto-downloaded UniProt entries (`explicit` labels or `uniprot_cc` text inference mode)

- [`mask`](https://github.com/kfuku52/cdskit/wiki/cdskit-mask): Masking ambiguous and/or stop codons

- [`maxalign`](https://github.com/kfuku52/cdskit/wiki/cdskit-maxalign): Removing sequences to maximize codon-based alignment area ([MaxAlign](https://link.springer.com/article/10.1186/1471-2105-8-312))

- [`pad`](https://github.com/kfuku52/cdskit/wiki/cdskit-pad): Making nucleotide sequences in-frame by head and tail paddings

- [`parsegb`](https://github.com/kfuku52/cdskit/wiki/cdskit-parsegb): Converting the GenBank format

- [`plot`](https://github.com/kfuku52/cdskit/wiki/cdskit-plot): Plotting aligned CDS summaries, codon-state maps, or nucleotide alignment views with consensus codon/AA and AA frequency logos using matplotlib (`--mode summary|map|msa`; default output is PDF, override with `--format`)

- [`printseq`](https://github.com/kfuku52/cdskit/wiki/cdskit-printseq): Print a subset of sequences with a regex

- [`rmseq`](https://github.com/kfuku52/cdskit/wiki/cdskit-rmseq): Removing a subset of sequences by using a sequence name regex and by detecting problematic sequence characters

- [`split`](https://github.com/kfuku52/cdskit/wiki/cdskit-split): Splitting 1st, 2nd, and 3rd codon positions

- [`stats`](https://github.com/kfuku52/cdskit/wiki/cdskit-stats): Printing sequence statistics

- [`translate`](https://github.com/kfuku52/cdskit/wiki/cdskit-translate): Translating CDS nucleotide sequences to amino acids

- [`trimcodon`](https://github.com/kfuku52/cdskit/wiki/cdskit-trimcodon): Trimming aligned CDS codon columns by clean-codon fraction

- [`validate`](https://github.com/kfuku52/cdskit/wiki/cdskit-validate): Validating aligned CDS quality and reporting issues

## Streamlined analysis
CDSKIT is designed for data flow through [standard input and output](https://en.wikipedia.org/wiki/Standard_streams). Streamlined processing may be combined with other sequence processing tools, such as [SeqKit](https://bioinf.shenwei.me/seqkit/), with pipes (`|`).

```
# Example 
seqkit seq input.fasta.gz | cdskit pad | cdskit mask | seqkit translate | cdskit aggregate -x ":.*"  > output.fasta
```

## Parallel execution
Commands that have independent record- or search-level work expose
`--threads INT`. Small metadata-only commands intentionally run serially.

- `--threads 1`: single-threaded (default)
- `--threads 2` or larger: multi-threaded
- `--threads 0`: auto-detect available CPU count

For resource safety, cdskit caps worker counts at 64 by default. Set
`CDSKIT_MAX_THREADS` to change that ceiling.

CPU-bound record commands select process workers from total sequence workload
rather than record count. The default crossover is 16,000,000 residues and can
be tuned with `CDSKIT_PROCESS_PARALLEL_MIN_RESIDUES`. Vectorized translation is
processed serially because worker startup costs more than it saves.

## CLI and TSV conventions

Long options use snake_case consistently, for example `--seq_file`,
`--out_file`, `--codon_table`, and `--seq_name_regex`. Older compact names
remain accepted for backward compatibility, but print a deprecation warning
that names the replacement. Boolean values consistently accept
`yes/no`, `true/false`, `on/off`, and `1/0`.

cdskit TSV files are UTF-8, tab-delimited, rectangular, and written with LF
line endings. Input tables must have a unique, non-empty header and all
columns required by the selected operation. Shared biological columns use
`accession`, `sequence`, `organism_group`, `localization`, `peroxisome`, and
`fold_id`. Peroxisome values are written as `yes` or `no`. Multi-part TSV
reports use one header, a `schema_version` column, and a `section` column
instead of concatenating tables with different widths. Versioned reports
currently use schema version `2` on every row. See [the 0.24 migration
guide](MIGRATION.md) for the old-to-new CLI and TSV mappings and
[the changelog](CHANGELOG.md) for release details.

## Localization prediction

Lightweight JSON models only need the base installation. Pretrained TargetP-
style `.pt` models, including experimental peroxisome-head candidates, run on
CPU but require the optional machine-learning dependencies described above.

Train with a local TSV:

```
cdskit localize-learn \
  --training_tsv train.tsv \
  --seq_col sequence \
  --label_mode explicit \
  --localization_col localization \
  --perox_col peroxisome \
  --model_out localize_model.json
```

Train by auto-downloading UniProt data:

```
cdskit localize-learn \
  --uniprot_preset viridiplantae \
  --uniprot_query "keyword:Transit peptide" \
  --label_mode uniprot_cc \
  --seq_col sequence \
  --localization_col cc_subcellular_location \
  --uniprot_fields accession,sequence,cc_subcellular_location \
  --uniprot_exclude_fragments yes \
  --uniprot_out_tsv uniprot_download.tsv \
  --model_out localize_model.json
```

Train a BiLSTM+attention model (PyTorch required):

```
cdskit localize-learn \
  --training_tsv uniprot_download.tsv \
  --seq_col sequence \
  --seq_type protein \
  --label_mode uniprot_cc \
  --localization_col cc_subcellular_location \
  --model_arch bilstm_attention \
  --dl_seq_len 120 \
  --dl_embed_dim 32 \
  --dl_hidden_dim 64 \
  --dl_epochs 10 \
  --cv_folds 3 \
  --model_out localize_bilstm.pt
```

Notes:

- `--uniprot_preset` can be used alone or combined with `--uniprot_query`.
- If `--label_mode explicit` is used with UniProt source, `--uniprot_fields` must include both `--localization_col` and `--perox_col`.
- Reports include overall `class_train_accuracy` plus per-class values (`class_train_accuracy_noTP`, etc.). With `--cv_folds`, per-class CV accuracies are also reported (`cv_class_accuracy_noTP`, etc.).

Predict from in-frame CDS:

```
cdskit localize \
  --seq_file cds.fasta \
  --model localize_model.json \
  --report localize.tsv
```

Predict from protein sequences:

```
cdskit localize \
  --seq_file proteins.faa \
  --seq_type protein \
  --model localize_model.json \
  --report localize.tsv
```

Compatible multi-label models can also be trained from the public DeepLoc 2.1
benchmark data with `python -m cdskit.deeploc_benchmark`. TargetP 2.0 comparison
helpers are available through `python -m cdskit.targetp_benchmark`. See the
[`localize` wiki page](https://github.com/kfuku52/cdskit/wiki/cdskit-localize)
for supported localization, membrane, and sorting-signal labels plus benchmark
commands.

## Performance benchmarks

Run the repeatable CPU hot-path suite before and after performance changes:

```bash
python -m cdskit.benchmark_hotpaths
python -m cdskit.benchmark_hotpaths --full
```

## Development and tests

The dependency-light development loop runs unit and integration tests without
loading optional ML frameworks:

```bash
python scripts/check.py quick
```

The committed `uv.lock` is the reproducible development and CI resolution.
Package consumers can continue to install cdskit with pip; the lock is not an
upper-bound policy for the library's declared compatibility ranges.

See [TESTING.md](TESTING.md) for the ML, coverage, quality, package, and focused
rerun commands, plus the test layout and fixture conventions.

## Citation
There is no published paper on CDSKIT itself, but we used and cited CDSKIT in several papers including [Fukushima & Pollock (2023, Nat Ecol Evol 7: 155-170)](https://www.nature.com/articles/s41559-022-01932-7).


## Licensing
This program is BSD-licensed (3 clause). See [LICENSE](LICENSE) for details.

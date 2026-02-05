![](logo/logo_cdskit_large.png)

[![Run Tests](https://github.com/kfuku52/cdskit/actions/workflows/test.yml/badge.svg)](https://github.com/kfuku52/cdskit/actions/workflows/test.yml)
[![GitHub release](https://img.shields.io/github/v/tag/kfuku52/cdskit?label=release)](https://github.com/kfuku52/cdskit/releases)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/cdskit.svg)](https://anaconda.org/bioconda/cdskit)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/kfuku52/cdskit)
[![Platforms](https://img.shields.io/conda/pn/bioconda/cdskit.svg)](https://anaconda.org/bioconda/cdskit)
[![Downloads](https://img.shields.io/conda/dn/bioconda/cdskit.svg)](https://anaconda.org/bioconda/cdskit)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Overview
**CDSKIT** ([/sidieskit/](http://ipa-reader.xyz/?text=sidieskit&voice=Joanna)) is a Python program that processes DNA sequences, especially protein-coding sequences. Many functions of this program are designed to handle DNA sequences using codons (sets of three nucleotides) as the unit, and therefore, edits the coding sequences without causing a frameshift. [All sequence formats supported by Biopython](https://biopython.org/wiki/SeqIO) are available in this tool for both inputs and outputs.

## Installation
The latest version of CDSKIT is available from [Bioconda](https://anaconda.org/bioconda/cdskit). For users requiring a `conda` installation, please refer to [Miniforge](https://github.com/conda-forge/miniforge) for a lightweight conda environment.

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

## Subcommands
See [Wiki](https://github.com/kfuku52/cdskit/wiki) for detailed descriptions.

- [`accession2fasta`](https://github.com/kfuku52/cdskit/wiki/cdskit-accession2fasta): Retrieving fasta sequences from a list of GenBank accessions

- [`aggregate`](https://github.com/kfuku52/cdskit/wiki/cdskit-aggregate): Extracting the longest sequences combined with a sequence name regex

- [`backtrim`](https://github.com/kfuku52/cdskit/wiki/cdskit-backtrim): Back-translating a trimmed protein alignment

- [`gapjust`](https://github.com/kfuku52/cdskit/wiki/cdskit-gapjust): Adjusting consecutive Ns to the fixed length

- [`hammer`](https://github.com/kfuku52/cdskit/wiki/cdskit-hammer): Removing less-occupied codon columns from a gappy alignment

- [`intersection`](https://github.com/kfuku52/cdskit/wiki/cdskit-intersection): Dropping non-overlapping sequence labels between two sequences files or between a sequence file and a gff file

- [`label`](https://github.com/kfuku52/cdskit/wiki/cdskit-label): Modifying sequence labels

- [`mask`](https://github.com/kfuku52/cdskit/wiki/cdskit-mask): Masking ambiguous and/or stop codons

- [`pad`](https://github.com/kfuku52/cdskit/wiki/cdskit-pad): Making nucleotide sequences in-frame by head and tail paddings

- [`parsegb`](https://github.com/kfuku52/cdskit/wiki/cdskit-parsegb): Converting the GenBank format

- [`printseq`](https://github.com/kfuku52/cdskit/wiki/cdskit-printseq): Print a subset of sequences with a regex

- [`rmseq`](https://github.com/kfuku52/cdskit/wiki/cdskit-rmseq): Removing a subset of sequences by using a sequence name regex and by detecting problematic sequence characters

- [`split`](https://github.com/kfuku52/cdskit/wiki/cdskit-split): Splitting 1st, 2nd, and 3rd codon positions

- [`stats`](https://github.com/kfuku52/cdskit/wiki/cdskit-stats): Printing sequence statistics

## Streamlined analysis
CDSKIT is designed for data flow through [standard input and output](https://en.wikipedia.org/wiki/Standard_streams). Streamlined processing may be combined with other sequence processing tools, such as [SeqKit](https://bioinf.shenwei.me/seqkit/), with pipes (`|`).

```
# Example 
seqkit seq input.fasta.gz | cdskit pad | cdskit mask | seqkit translate | cdskit aggregate -x ":.*"  > output.fasta
```

## Citation
There is no published paper on CDSKIT itself, but we used and cited CDSKIT in several papers including [Fukushima & Pollock (2023, Nat Ecol Evol 7: 155-170)](https://www.nature.com/articles/s41559-022-01932-7).


## Licensing
This program is BSD-licensed (3 clause). See [LICENSE](LICENSE) for details.


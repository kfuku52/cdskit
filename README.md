![](logo/logo_cdskit_large.png)

## Overview
**CDSKIT** is a Python program that manipulates protein-coding nucleotide sequences. This program is designed to handle DNA sequences using codons (sets of three nucleotides) as the unit, and therefore, edits the coding sequences without causing a frameshift. [All sequence formats supported by Biopython](https://biopython.org/wiki/SeqIO) are available in this tool for both inputs and outputs.

## Installation
```
# Installation with pip
pip install git+https://github.com/kfuku52/cdskit

# This should show complete options if installation is successful
cdskit -h 
```

## Functions
See [Wiki](https://github.com/kfuku52/cdskit/wiki) for detailed descriptions.

- [`accession2fasta`](https://github.com/kfuku52/cdskit/wiki/cdskit-accession2fasta): Retrieving fasta sequences from a list of GenBank accessions

- [`aggregate`](https://github.com/kfuku52/cdskit/wiki/cdskit-aggregate): Extracting the longest sequences combined with a sequence name regex

- [`backtrim`](https://github.com/kfuku52/cdskit/wiki/cdskit-backtrim): Back-translating a trimmed protein alignment

- [`hammer`](https://github.com/kfuku52/cdskit/wiki/cdskit-hammer): Removing less-occupied codon columns from a gappy alignment

- [`mask`](https://github.com/kfuku52/cdskit/wiki/cdskit-mask): Masking ambiguous and/or stop codons

- [`pad`](https://github.com/kfuku52/cdskit/wiki/cdskit-pad): Making nucleotide sequences in-frame by head and tail paddings

- [`parsegb`](https://github.com/kfuku52/cdskit/wiki/cdskit-parsegb): Converting the GenBank format

- [`printseq`](https://github.com/kfuku52/cdskit/wiki/cdskit-printseq): Print a subset of sequences with a regex

- [`rmseq`](https://github.com/kfuku52/cdskit/wiki/cdskit-rmseq): Removing a subset of sequences by using a sequence name regex and by detecting problematic sequence characters.

- [`stats`](https://github.com/kfuku52/cdskit/wiki/cdskit-stats): Printing sequence statistics.

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


![](logo/logo_cdskit_large.svg)

## Overview
**CDSKIT** is a Python program that manipulates protein-coding nucleotide sequences. This program is designed to handle DNA sequences using codons (sets of three nucleotides) as the unit, and therefore, edits the coding sequences without causing a frameshift. [All sequence formats supported by Biopython](https://biopython.org/wiki/SeqIO) are available in this tool for both inputs and outputs.


## Installation
```
# Installation with pip
pip install git+https://github.com/kfuku52/cdskit

# This should show complete options if installation is successful
cdskit -h 
```

## Commands
See [wiki](https://github.com/kfuku52/cdskit/wiki) for detailed descriptions.

`accession2fasta`: Retrieving fasta sequences from a list of GenBank accessions

`aggregate`: Extracting the longest sequences combined with a sequence name regex

`backtrim`: Back-translating a trimmed protein alignment

`hammer`: Removing less-occupied codon columns from a gappy alignment

`mask`: Masking ambiguous and/or stop codons

`pad`: Making nucleotide sequences in-frame by head and tail paddings

`parsegb`: Converting the GenBank format

`printseq`: Print a subset of sequences with a regex

`rmseq`: Removing a subset of sequences by using a sequence name regex and by detecting problematic sequence characters.

`stats`: Printing sequence statistics.


## Pipe for streamlined analysis
CDSKIT is designed with data flow through standard input. Streamlined processing may be combined with other sequence processing tools such as [SeqKit](https://bioinf.shenwei.me/seqkit/).
```
seqkit seq input.fasta | cdskit pad | cdskit mask | cdskit aggregate > output.fasta
```

# Licensing
This program is BSD-licensed (3 clause). See [LICENSE](LICENSE) for details.


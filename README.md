## Overview

**cdskit** is a pre- and post-processing tool for protein-coding nucleotide sequences. Many functions manipulate sequences without causing frameshift. [All sequence formats supported by biopython](https://biopython.org/wiki/SeqIO) are available in this tool for both inputs and outputs.

## Dependency
* [python 3](https://www.python.org/)
* [biopython](https://biopython.org/)
* [numpy](http://www.numpy.org/)

## Installation
```
# Installation with pip
pip install git+https://github.com/kfuku52/cdskit

# This should show complete options if installation is successful
cdskit -h 
```

## Subcommands
See [wiki](https://github.com/kfuku52/cdskit/wiki) for the complete description.

`accession2fasta`: Retrieve fasta sequences from a list of GenBank accessions

`aggregate`: Extracting the longest sequences combined with regex in sequence names

`backtrim`: Back-translating trimmed protein alignment

`hammer`: "Hammer down" long alignments

`mask`: Masking ambiguous and/or stop codons

`pad`: Making nucleotide sequences in-frame by N-padding

`parsegb`: Parsing the GenBank format

`printseq`: Print a subset of sequences with regex

## Pipe for streamlined analysis
The streamlined processing may be combined with other sequence processing tools such as [SeqKit](https://bioinf.shenwei.me/seqkit/).
```
cat input.fasta \
| cdskit pad \
| cdskit mask \
| cdskit aggregate \
> output.fasta
```

# Licensing
This program is BSD-licensed (3 clause). See [LICENSE](LICENSE) for details.


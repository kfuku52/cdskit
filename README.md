## Overview

**CDSKIT** is a Python program that manipulates protein-coding nucleotide sequences. A unique feature of this program is to handle sequences using codons (sets of three nucleotides) as the unit, and therefore, edits the coding sequences without causing a frameshift. [All sequence formats supported by biopython](https://biopython.org/wiki/SeqIO) are available in this tool for both inputs and outputs.


## Installation
```
# Installation with pip
pip install git+https://github.com/kfuku52/cdskit

# This should show complete options if installation is successful
cdskit -h 
```

## Subcommands
See [wiki](https://github.com/kfuku52/cdskit/wiki) for the complete description.

`accession2fasta`: Retrieving fasta sequences from a list of GenBank accessions

`aggregate`: Extracting the longest sequences combined with a sequence name regex

`backtrim`: Back-translating a trimmed protein alignment

`hammer`: "Hammer down" long alignments

`mask`: Masking ambiguous and/or stop codons

`pad`: Making nucleotide sequences in-frame by head and tail paddings

`parsegb`: Converting the GenBank format

`printseq`: Print a subset of sequences with a regex

`rmseq`: Removing a subset of sequences by using a sequence name regex and by detecting problematic sequence characters.

`stats`: Printing sequence statistics.


## Pipe for streamlined analysis
Streamlined processing may be combined with other sequence processing tools such as [SeqKit](https://bioinf.shenwei.me/seqkit/).
```
seqkit seq input.fasta | cdskit pad | cdskit mask | cdskit aggregate > output.fasta
```

# Licensing
This program is BSD-licensed (3 clause). See [LICENSE](LICENSE) for details.


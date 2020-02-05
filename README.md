## Overview

**cdskit** is a pre- and post-processing tool for protein-coding sequences. 
[All sequence formats supported by biopython](https://biopython.org/wiki/SeqIO) are available in this tool for both input and output.

## Dependency
* [python 3](https://www.python.org/)
* [biopython](https://biopython.org/)
* [numpy](http://www.numpy.org/)

## Installation
```
# Installation with pip
pip install git+https://github.com/kfuku52/cdskit

# This should show complete options
cdskit -h 
```

## Example

### Padding truncated coding sequences to make them in-frame

`cdskit pad --seqfile input.fasta --outfile output.fasta`

```
# input.fasta
>miss_1nt_5prime
TGCTAAGCGGTAATCTAAGCGGTAATTGA
>miss_2nt_5prime
GCTAAGCGGTAATCTAAGCGGTAATTGA
>miss_1nt_3prime
ATGCTAAGCGGTAATCTAAGCGGTAATTG
>miss_2nt_3prime
ATGCTAAGCGGTAATCTAAGCGGTAATT
>miss_1nt_both
TGCTAAGCGGTAATCTAAGCGGTAATTG
>miss_2nt_both
GCTAAGCGGTAATCTAAGCGGTAATT
>complete
ATGCTAAGCGGTAATCTAAGCGGTAATTGA
```

```
# output.fasta
>miss_1nt_5prime
NTGCTAAGCGGTAATCTAAGCGGTAATTGA
>miss_2nt_5prime
NNGCTAAGCGGTAATCTAAGCGGTAATTGA
>miss_1nt_3prime
ATGCTAAGCGGTAATCTAAGCGGTAATTGN
>miss_2nt_3prime
ATGCTAAGCGGTAATCTAAGCGGTAATTNN
>miss_1nt_both
NTGCTAAGCGGTAATCTAAGCGGTAATTGN
>miss_2nt_both
NNGCTAAGCGGTAATCTAAGCGGTAATTNN
>complete
ATGCTAAGCGGTAATCTAAGCGGTAATTGA
```

### Extracting longest sequences among isoforms using regular expression

`cdskit aggregate --seqfile input.fasta --outfile output.fasta --expression ":.*" "\|.*"`

```
# input.fasta
>seq1:1.length=30nt
ATGCTAAGCGGTAATCTAAGCGGTAATTGA
>seq1:2.length=27nt
ATGCTAAGCGGTAATCTAAGCGGTTGA
>seq1:3.length=33nt
ATGCAACTAAGCGGTAATCTAAGCGGTAATTGA
>seq2|1.length=45nt
ATGTCGGAATTGCGAGTAAGCACCAGCTTCTCAAAACCAAAATAA
>seq2|2.length=54nt
ATGTCGAGATCCCGAGAATTGCGAGTAAGCACCAGCTTCTCAAAACCAAAATAA
```

```
# output.fasta
>seq1:3.length=33nt
ATGCAACTAAGCGGTAATCTAAGCGGTAATTGA
>seq2|2.length=54nt
ATGTCGAGATCCCGAGAATTGCGAGTAAGCACCAGCTTCTCAAAACCAAAATAA
```
### Mask ambiguous and/or stop codons
`cdskit mask --seqfile input.fasta --outfile output.fasta`

```
# input.fasta
>stop
---ATGTAAATTATGTTGAAG---
>ambiguous1
---ATGTNAATTATGTTGAAG---
>ambiguous2
---ATGT-AATTATGTTGAAG---
>all
---ATGTAAATT--GTTGANG---
```

```
# output.fasta
>stop
---ATGNNNATTATGTTGAAG---
>ambiguous1
---ATGNNNATTATGTTGAAG---
>ambiguous2
---ATGNNNATTATGTTGAAG---
>all
---ATGNNNATTNNNTTGNNN---
```


### Pipe for streamlined analysis

```
cat input.fasta \
| cdskit pad \
| cdskit mask \
| cdskit aggregate \
> output.fasta
```

# Licensing
This program is BSD-licensed (3 clause). See [LICENSE](LICENSE) for details.


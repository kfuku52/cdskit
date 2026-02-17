# cdskit translate

`cdskit translate` translates CDS nucleotide sequences to amino acid sequences.

### Example

#### Command

    cdskit translate --seqfile cds.fasta --outfile protein.fasta --codontable 1

#### cds.fasta

    >seq1
    ATGAAATGA
    >seq2
    ATGCCCTAA

#### protein.fasta

    >seq1
    MK*
    >seq2
    MP*

### Key options

- `--codontable INT`: NCBI codon table ID (default: `1`).
- `--to_stop yes|no`: Stop translation at the first in-frame stop codon (default: `no`).

### Notes

- Input sequence lengths should be multiples of three.
- Sequence names and order are preserved in the output.

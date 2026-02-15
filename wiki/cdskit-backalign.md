# cdskit backalign

`cdskit backalign` back-aligns CDS sequences based on an amino acid alignment.

### Example

In this example, CDSKIT generates a codon alignment from unaligned CDS sequences and an aligned amino acid sequence file.

#### Command

    cdskit backalign --seqfile unaligned_cds.fasta --aa_aln aligned_aa.fasta --outfile aligned_cds.fasta

#### unaligned_cds.fasta

    >seq1
    ATGAAACCC
    >seq2
    ATGAAAGGG

#### aligned_aa.fasta

    >seq1
    MK-P
    >seq2
    MKG-

#### aligned_cds.fasta

    >seq1
    ATGAAA---CCC
    >seq2
    ATGAAAGGG---

### Notes

- Sequence IDs in `--seqfile` and `--aa_aln` should match.
- `--aa_aln` should be aligned (all sequences should have the same length).
- CDS lengths in `--seqfile` should be multiples of three after removing gap characters.
- A terminal stop codon in CDS can be omitted in `--aa_aln`.

# cdskit backalign

`cdskit backalign` back-aligns CDS sequences based on an amino acid alignment.

## Example

In this example, CDSKIT generates a codon alignment from unaligned CDS sequences and an aligned amino acid sequence file.

### Command

```bash
cdskit backalign --seq_file unaligned_cds.fasta --aa_aln aligned_aa.fasta --out_file aligned_cds.fasta
```

### unaligned_cds.fasta

```fasta
>seq1
ATGAAACCC
>seq2
ATGAAAGGG
```

### aligned_aa.fasta

```fasta
>seq1
MK-P
>seq2
MKG-
```

### aligned_cds.fasta

```fasta
>seq1
ATGAAA---CCC
>seq2
ATGAAAGGG---
```

## Notes

- Sequence IDs in `--seq_file` and `--aa_aln` should match.
- `--aa_aln` should be aligned (all sequences should have the same length).
- CDS lengths in `--seq_file` should be multiples of three after removing gap characters.
- A terminal stop codon in CDS can be omitted in `--aa_aln`.

For trimmed protein alignments, use [backtrim](https://github.com/kfuku52/cdskit/wiki/cdskit-backtrim) with the trimmer's retained column positions. Backalign requires the complete corresponding protein sequence (apart from the documented terminal stop exception); it cannot infer deleted codon positions from repeated amino acids.

## References

Cite the method or resource actually used and report the CDSKIT version and
options.

- Cock PJA et al. (2009). Biopython: freely available Python tools for computational molecular biology and bioinformatics. *Bioinformatics* 25:1422–1423. [Paper](https://doi.org/10.1093/bioinformatics/btp163) — Sequence parsing, feature extraction and genetic-code infrastructure used by CDSKIT.

# cdskit translate

`cdskit translate` converts CDS nucleotide sequences to amino-acid sequences.

## Example

```bash
cdskit translate --seq_file cds.fasta --out_file proteins.fasta --codon_table 1
```

## Stop-aware translation

```bash
cdskit translate --seq_file cds.fasta --out_file proteins.fasta --to_stop yes
```

With `--to_stop yes`, translation stops at the first in-frame stop codon.

## Key options

- `--codon_table INT`: NCBI codon table ID.
- `--to_stop yes|no`: Stop translation at the first stop codon instead of emitting `*`.

## Notes

- Input sequences must be DNA and their lengths must be multiples of three.
- Gap-only codons translate to `-`.
- Partial-gap codons and unresolved ambiguous codons translate to `X`.
  Ambiguities with one translation remain resolvable, for example `GCN` is
  `A` and `TAR` is `*` under the standard code.
- For genetic codes with context-dependent stops (27, 28, and 31), translation
  uses the amino-acid assignment for dual-use codons because the sequence alone
  does not identify termination context. `--to_stop yes` therefore does not
  stop at a dual-use codon. Backalignment follows the same rule.

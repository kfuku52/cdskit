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

## Ordinary translation versus an explicitly complete CDS

Ordinary translation uses the forward amino acid for dual-coding codons in
codes 27/28/31. `--to_stop yes` stops only at a definite translated `*`, including
standard-code TAR; it does not infer context-dependent termination. This is an
intentional CDSKIT contract; Biopython's direct `to_stop=True` rejects these
code tables instead.

`--complete_cds yes` explicitly asserts complete CDS boundaries. It requires a
valid start codon, length divisible by three, a terminal-compatible stop, and no
definite internal stops or missing/invalid codons. The initiator becomes M and
the terminator is omitted, regardless of `--to_stop`. The ordinary default is
unchanged. Possible ambiguous internal stops remain X; acceptance is not proof
of biological completeness or function.

```bash
cdskit translate --seq_file complete_cds.fasta --out_file proteins.fasta \
  --codon_table 27 --complete_cds yes
```

See [codon semantics](https://github.com/kfuku52/cdskit/wiki/codon-semantics) for uncertainty and compatibility.

`X` in DNA is treated as any base, like `N`, consistently in ordinary,
partial-tail and complete-CDS translation. Invalid alphabet characters are
rejected even when the same codon also contains a missing character.

## References

Cite the method or resource actually used and report the CDSKIT version and
options.

- Cock PJA et al. (2009). Biopython: freely available Python tools for computational molecular biology and bioinformatics. *Bioinformatics* 25:1422–1423. [Paper](https://doi.org/10.1093/bioinformatics/btp163) — Sequence parsing, feature extraction and genetic-code infrastructure used by CDSKIT.
- Swart EC, Serra V, Petroni G, Nowacki M (2016). Genetic Codes with No Dedicated Stop Codon: Context-Dependent Translation Termination. *Cell* 166:691–702. [Paper](https://doi.org/10.1016/j.cell.2016.06.020) — Biological background for context-dependent termination in the discussed ciliate codes; CDSKIT does not infer termination context.

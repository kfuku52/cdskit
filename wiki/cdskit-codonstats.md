# cdskit codonstats

`cdskit codonstats` prints codon-aware sequence statistics and aggregate codon-usage tables.

## Examples

### Per-sequence summary

```bash
cdskit codonstats --seq_file cds.fasta --mode summary
```

This prints one row per sequence with nucleotide length, codon counts, stop-codon count, and GC / GC1 / GC2 / GC3 percentages.

### Aggregate codon usage

```bash
cdskit codonstats --seq_file cds.fasta --mode usage --codon_table 1
```

This prints a `codon / aa / count / fraction` table aggregated across the input set.

### Both tables

```bash
cdskit codonstats --seq_file cds.fasta --mode both
```

`both` prints two separately headed tables with a blank line between them.
It is intended for inspection, not parsing as one rectangular TSV. For a
machine-readable pipeline, run `summary` and `usage` separately and save each
standard-output stream to its own file. These tables do not use the versioned
`schema_version`/`section` format of the QC report commands.

## Key options

- `--mode summary|usage|both`: Print the per-sequence summary table, the aggregate usage table, or both.
- `--codon_table INT`: NCBI codon table ID used to interpret stop codons and amino-acid assignments.

## Notes

- Input sequences must be DNA and their lengths must be multiples of three.
- Missing codons are counted separately from ambiguous codons.
- GC statistics use only unambiguous A/C/G/T bases in the denominator, for
  all positions combined and for each codon position separately. In contrast,
  `cdskit stats` includes ambiguous bases and gaps in its GC denominator.

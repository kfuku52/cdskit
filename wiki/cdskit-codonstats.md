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
- Missing codons are counted separately from ambiguous codons. Ambiguous and
  definite-stop counts can overlap (for example, standard-code `TAR`).
- Summary rows also include `codons_possible_stop`, `codons_context_dependent`
  and `codon_semantics_version`; read fields by column name.
- GC statistics use only unambiguous A/C/G/T bases in the denominator, for
  all positions combined and for each codon position separately. In contrast,
  `cdskit stats --mode sequence` includes ambiguous bases and gaps in its GC denominator.

## Codon semantics

See [codon meaning and uncertainty](https://github.com/kfuku52/cdskit/wiki/codon-semantics) for definite versus
possible stops, dual-coding tables 27/28/31, overlapping ambiguity/stop counts,
and report compatibility. These rules describe sequence QC, not gene function.

## Citation

No separate method paper is designated for this utility. Report the CDSKIT
version, command and options; see [citing CDSKIT](https://github.com/kfuku52/cdskit/wiki#citing-cdskit).

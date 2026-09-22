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
Only observed unambiguous A/C/G/T codons appear, including stop codons.
`fraction` is the count divided by the total of those usage counts (0–1),
not a within-amino-acid frequency or RSCU. Ambiguous codons such as `GCN`
are excluded even when their amino acid is resolvable.

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

- `--mode summary|usage|both` (default `summary`): Print the per-sequence summary table, the aggregate usage table, or both.
- `--codon_table INT`: NCBI codon table ID used to interpret stop codons and amino-acid assignments.

## Notes

- Input sequences must be DNA and their lengths must be multiples of three.
- Missing codons are counted separately from ambiguous codons. Ambiguous and
  definite-stop counts can overlap (for example, standard-code `TAR`).
- `codons_complete` counts triplets without missing characters, including
  ambiguous and stop codons; it does not mean clean codons or complete CDSs.
  `codons_stop` includes terminal stops.
- Summary rows also include `codons_possible_stop`, `codons_context_dependent`
  and `codon_semantics_version`; read fields by column name.
- GC statistics use only unambiguous A/C/G/T bases in the denominator, for
  all positions combined and for each codon position separately. In contrast,
  `cdskit stats --mode sequence` includes ambiguous bases and gaps in its GC denominator.
  The output fields `gc_all`, `gc1`, `gc2`, and `gc3` are percentages (0–100),
  printed to six decimals. A zero denominator produces `0.000000`, not a
  measured zero GC fraction. FASTA `seq_id` is the first whitespace-delimited
  header token; summary mode retains one row per input record, including duplicate IDs.

## Codon semantics

See [codon meaning and uncertainty](https://github.com/kfuku52/cdskit/wiki/codon-semantics) for definite versus
possible stops, dual-coding tables 27/28/31, overlapping ambiguity/stop counts,
and report compatibility. These rules describe sequence QC, not gene function.

## Citation

No separate method paper is designated for this utility. Report the CDSKIT
version, command and options; see [citing CDSKIT](https://github.com/kfuku52/cdskit/wiki#citing-cdskit).

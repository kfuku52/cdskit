# cdskit validate

`cdskit validate` runs basic quality checks for CDS alignments and prints a summary.

## Example

### Command

```bash
cdskit validate --seq_file alignment.fasta --codon_table 1 --report validate.json
```

## What is checked

- Whether all sequence lengths are identical (`aligned`).
- Sequence lengths that are not multiples of three.
- Duplicate sequence IDs.
- Gap-only sequences.
- Sequences containing internal stop codons.
- Ambiguous codons and ambiguous-codon rate.

## Key options

- `--codon_table INT`: NCBI codon table ID used for internal stop checks.
- `--report PATH`: Optional output report path (`.json` or tab-separated text).
  No report file is produced unless this option is specified. Use `--report -`
  to write only the TSV report to standard output (CDSKIT >=0.31.7).

## TSV report format

TSV reports use schema version 2 and begin with `schema_version` and `section`.
`summary` rows contain counts and rates; `id_set` rows contain compact JSON
arrays for each category of affected sequence IDs.

## Notes

- Without `--report -`, the human-readable summary is printed to standard
  output, including when saving a report to a named file. Successful reporting exits
  with status 0 even when QC issues are found; inspect the report fields to
  decide whether an alignment meets your criteria.
- `--report -` suppresses the human-readable summary so stdout contains one
  rectangular TSV. Runtime logs remain on stderr. In versions through 0.31.6,
  this option mixed the summary and TSV; use a named report file on those versions.
- `ambiguous_codon_rate` is a 0–1 ratio: ambiguous complete codons divided by
  evaluable complete codons (excluding codons with `-`, `?`, or `.`). A zero
  denominator reports 0. `gap_only_ids` also includes all-N/all-X sequences,
  not just literal dashes.
- This command does not modify input sequences.

## Codon semantics

See [codon meaning and uncertainty](https://github.com/kfuku52/cdskit/wiki/codon-semantics) for definite versus
possible stops, dual-coding tables 27/28/31, overlapping ambiguity/stop counts,
and report compatibility. These rules describe sequence QC, not gene function.

## Citation

No separate method paper is designated for this utility. Report the CDSKIT
version, command and options; see [citing CDSKIT](https://github.com/kfuku52/cdskit/wiki#citing-cdskit).

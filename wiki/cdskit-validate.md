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
  No report file is produced unless this option is specified.

## TSV report format

TSV reports use schema version 2 and begin with `schema_version` and `section`.
`summary` rows contain counts and rates; `id_set` rows contain compact JSON
arrays for each category of affected sequence IDs.

## Notes

- Validation output is printed to standard output.
- This command does not modify input sequences.

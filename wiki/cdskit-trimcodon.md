# cdskit trimcodon

`cdskit trimcodon` removes aligned codon columns using a clean-codon fraction threshold.

## Example

```bash
cdskit trimcodon --seq_file alignment.fasta --out_file trimmed.fasta --min_clean_fraction 0.8
```

## How sites are evaluated

For each codon column, `cdskit trimcodon` records:

- clean fraction: fraction of sequences with a clean codon
- clean, missing, ambiguous, and stop-codon counts

Sites are retained when their clean fraction is at least `--min_clean_fraction`.

A clean codon contains no missing character (`-`, `?`, or `.`), no ambiguous nucleotide (`N`, `R`, `Y`, etc.), and no unambiguous stop codon.

## Key options

- `--min_clean_fraction FLOAT`: Minimum fraction of sequences with a clean codon required to retain a site.
- `--report PATH`: Optional JSON or TSV site-level report.

## TSV report format

TSV reports use schema version 2 and begin with `schema_version` and `section`.
`summary` rows contain run-level counts and the threshold; `site` rows contain
the codon position, category counts, clean fraction, and `keep=yes|no` result.

## Notes

- Input sequences must be DNA, aligned, and multiples of three in length.
- Sequence order is preserved in the trimmed output.

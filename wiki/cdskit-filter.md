# cdskit filter

`cdskit filter` removes CDS records that fail sequence-level quality rules.

## Example

```bash
cdskit filter --seq_file input.fasta --out_file filtered.fasta --drop_internal_stop yes --min_clean_codon_fraction 0.8 --dedup keep-longest
```

## What can be filtered

- Non-triplet sequence lengths.
- Sequences with internal in-frame stop codons.
- Sequences whose clean-codon fraction is below a threshold.
- Duplicate IDs, resolved by `keep-first` or `keep-longest`.

## Key options

- `--drop_non_triplet yes|no`: Drop sequences whose length is not divisible by three.
- `--drop_internal_stop yes|no`: Drop sequences with internal stop codons.
- `--min_clean_codon_fraction FLOAT`: Minimum fraction of clean codons required to retain a sequence.
- `--dedup no|keep-first|keep-longest`: Duplicate-ID handling after filtering.
- `--report PATH`: Optional JSON or TSV report listing kept and dropped IDs.

A clean codon contains no missing character (`-`, `?`, or `.`), no ambiguous nucleotide (`N`, `R`, `Y`, etc.), and no unambiguous stop codon.

## TSV report format

TSV reports use schema version 2. Every row starts with `schema_version` and
`section`. Use `summary` for run settings/counts, `id_set` for kept or dropped
ID lists, `drop_reason` for per-reason counts, and `sequence` for one row per
input record. List-valued cells such as `ids` and `drop_reasons` contain compact
JSON arrays.

## Notes

- Input sequences must be DNA.
- The command filters whole sequences, not alignment columns.

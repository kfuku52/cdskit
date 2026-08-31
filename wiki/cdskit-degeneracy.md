# cdskit degeneracy

`cdskit degeneracy` extracts aligned nucleotide positions by codon degeneracy class.

## Example

```bash
cdskit degeneracy --seq_file alignment.fasta --prefix genes --fold 4 2 --codon_table 1
```

This writes separate files such as `genes_4fold_positions.fasta` and `genes_2fold_positions.fasta`.

## What it does

- Examines each aligned codon column.
- Classifies each nucleotide position as `0`, `2`, `3`, or `4` fold degenerate when the assignment is consistent across usable sequences.
- Skips positions that are unassigned or conflicting across sequences.

## Key options

- `--prefix PATH`: Output filename prefix.
- `--fold 0 2 3 4`: Degeneracy class or classes to export.
- `--codon_table INT`: NCBI codon table ID used for degeneracy classification.
- `--report PATH`: Optional JSON or TSV summary report.

## Notes

- Input sequences must be DNA, aligned, and multiples of three in length.
- Output alignments preserve sequence order while retaining only the selected nucleotide positions.

# cdskit maxalign

`cdskit maxalign` removes sequences to maximize codon-based alignment area, inspired by [MaxAlign (Gouveia-Oliveira et al. 2007)](https://link.springer.com/article/10.1186/1471-2105-8-312).

CDSKIT computes alignment area in **codon units (3 nt)**:

`alignment area = (number of retained sequences) x (number of complete codon sites)`

A complete codon site is a codon column where every retained sequence has no missing character in that codon.

## Example

In this example, CDSKIT removes one gap-heavy sequence and keeps codon sites that are complete in the retained set.

### Command

```bash
cdskit maxalign --seq_file input.fasta --out_file output.fasta --mode exact
```

### input.fasta

```fasta
>seq1
ATGAAACCCGGG
>seq2
ATGAAACCCGGG
>seq3
ATGAAACCCGGG
>seq4
---AAA---GGG
```

### output.fasta

```fasta
>seq1
ATGAAACCCGGG
>seq2
ATGAAACCCGGG
>seq3
ATGAAACCCGGG
```

## Key options

- `--mode auto|exact|greedy|branch-and-bound` selects the solver. `auto` uses exact subset search for small inputs and greedy search for larger inputs.
- `--max_exact_sequences INT` sets the sequence limit for the `exact` solver. The default is `16` and the accepted range is `1`–`24`.
- `--missing_chars STR`: Characters treated as missing within a codon (default: `-?.`).
- `--keep_seq_name_regex REGEX [REGEX ...]`: Never remove matching sequence IDs.
- `--max_removed INT`: Maximum number of sequences that may be removed.
- `--report PATH`: Optional JSON or TSV optimization report.
  Empty input writes an empty sequence output and a zero-count report.

## TSV report format

TSV reports use schema version 2 and begin with `schema_version` and `section`.
`summary` rows contain run-level metrics; `step` rows describe the initial state
and each removal step. List-valued `removed_ids` cells are JSON arrays.

## Notes

- Input sequences should already be aligned (equal sequence lengths).
- Input sequence lengths should be multiples of three.
- Output keeps only retained sequences and codon sites that are complete in the retained set.
- `--mode auto` switches to `greedy` when the number of input sequences exceeds `--max_exact_sequences`.
- Exact search scales exponentially in the number of unprotected sequences. For 24 unprotected sequences and up to 65,535 codon columns, the subset count table uses at most 32 MiB. Input records and other working memory are additional.

### Exact search with a removal limit

`--mode branch-and-bound` proves the optimum by searching weighted gap
patterns. It can handle more than 24 sequences when few removals or gap
patterns are feasible. Use `--max_removed` to restrict the search. For the
example above, replacing `--mode exact` with `--mode branch-and-bound
--max_removed 1` produces the same `output.fasta`.

The solver considers retaining or removing a row at each branch. It prunes
branches whose best possible area falls short of the best solution already
found. It applies the same protected-row and tie rules as `exact`. The
search runs in one process and stops with an error if it exceeds its fixed
work limit, without publishing a result. [Original MaxAlign algorithm](https://services.healthtech.dtu.dk/services/MaxAlign-1.1/4-Algorithm.php)

## References

Cite the method or resource actually used and report the CDSKIT version and
options.

- Gouveia-Oliveira R, Sackett PW, Pedersen AG (2007). MaxAlign: maximizing usable data in an alignment. *BMC Bioinformatics* 8:312. [Paper](https://doi.org/10.1186/1471-2105-8-312) — Original alignment-area objective; CDSKIT adapts the objective to codons and supplies its own solvers.

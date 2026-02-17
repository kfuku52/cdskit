# cdskit maxalign

`cdskit maxalign` removes sequences to maximize codon-based alignment area, inspired by [MaxAlign (Gouveia-Oliveira et al. 2007)](https://link.springer.com/article/10.1186/1471-2105-8-312).

CDSKIT computes alignment area in **codon units (3 nt)**:

`alignment area = (number of retained sequences) x (number of complete codon sites)`

A complete codon site is a codon column where every retained sequence has no missing character in that codon.

### Example

In this example, CDSKIT removes one gap-heavy sequence and keeps codon sites that are complete in the retained set.

#### Command

    cdskit maxalign --seqfile input.fasta --outfile output.fasta --mode exact

#### input.fasta

    >seq1
    ATGAAACCCGGG
    >seq2
    ATGAAACCCGGG
    >seq3
    ATGAAACCCGGG
    >seq4
    ---AAA---GGG

#### output.fasta

    >seq1
    ATGAAACCCGGG
    >seq2
    ATGAAACCCGGG
    >seq3
    ATGAAACCCGGG

### Key options

- `--mode auto|exact|greedy`: Solver mode. `auto` uses exact search for small inputs and greedy search for larger inputs.
- `--max_exact_sequences INT`: Maximum number of sequences allowed in exact mode (default: `16`).
- `--missing_char STR`: Characters treated as missing within a codon (default: `-?.`).
- `--keep REGEX1,REGEX2,...`: Comma-separated regex patterns of sequence names that should not be dropped.
- `--max_removed INT`: Maximum total number of sequences that can be removed.
- `--report PATH`: Optional report path. If `PATH` ends with `.json`, JSON is written; otherwise TSV is written. No report is produced unless this is specified.

### Notes

- Input sequences should already be aligned (equal sequence lengths).
- Input sequence lengths should be multiples of three.
- Output keeps only retained sequences and codon sites that are complete in the retained set.
- `--mode auto` switches to `greedy` when the number of input sequences exceeds `--max_exact_sequences`.

# cdskit aggregate

`cdskit aggregate` groups sequence IDs after removing one or more regular
expression matches, then retains the longest sequence in each group. The
original ID of the retained record is preserved.

## Example

### Command
```
cdskit aggregate --seq_file input.fasta --out_file output.fasta --expression ":.*" "\|.*"
```

### input.fasta
```
>seq1:1.length=30nt
ATGCTAAGCGGTAATCTAAGCGGTAATTGA
>seq1:2.length=27nt
ATGCTAAGCGGTAATCTAAGCGGTTGA
>seq1:3.length=33nt
ATGCAACTAAGCGGTAATCTAAGCGGTAATTGA
>seq2|1.length=45nt
ATGTCGGAATTGCGAGTAAGCACCAGCTTCTCAAAACCAAAATAA
>seq2|2.length=54nt
ATGTCGAGATCCCGAGAATTGCGAGTAAGCACCAGCTTCTCAAAACCAAAATAA
```

### output.fasta
```
>seq1:3.length=33nt
ATGCAACTAAGCGGTAATCTAAGCGGTAATTGA
>seq2|2.length=54nt
ATGTCGAGATCCCGAGAATTGCGAGTAAGCACCAGCTTCTCAAAACCAAAATAA
```

## Notes

- Values passed to `--expression` are applied sequentially. In this example,
  `:.*` and `\|.*` reduce the IDs to the group keys `seq1` and `seq2`.
- `--mode longest` is currently the only selection mode. Ties retain the first
  record encountered in the input.
- If `--expression` is omitted, records are copied without aggregation.

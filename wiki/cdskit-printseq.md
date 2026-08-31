# cdskit printseq

`cdskit printseq` writes records whose sequence IDs fully match a regular
expression to standard output.

## Example

### Command
```
cdskit printseq --seq_file input.fasta --seq_name_regex 'seq_[AG]' > output.fasta
```

### input.fasta
```
>seq_A
AAAAAAAAAAAA
>seq_T
TTTTTTTTTTTT
>seq_G
GGGGGGGGGGGG
>seq_C
CCCCCCCCCCCC
```

### output.fasta
```
>seq_A
AAAAAAAAAAAA
>seq_G
GGGGGGGGGGGG
```

## Notes

- Matching uses the complete record ID. For substring matching, include `.*`
  explicitly, for example `--seq_name_regex '.*kinase.*'`.
- `--show_seq_name no` writes sequence strings without FASTA header lines.
- This command has no `--out_file` option; redirect standard output as shown
  above when a file is required.

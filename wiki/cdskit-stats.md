# cdskit stats

`cdskit stats` prints aggregate nucleotide-sequence statistics to standard
output.

## Examples

### Command
```
cdskit stats --seq_file input.fasta
```

### input.fasta
```
>seq1
ACGTacgtNN--
>seq2
GGCC
```

### stdout
```
Number of sequences: 2
Total length: 16
Total softmasked length: 4
Total N length: 2
Total gap (-) length: 2
GC content: 50.0%
```

Lowercase characters are counted as soft-masked. `N` and `-` counts are
case-insensitive for `N`; GC percentage uses total sequence length as its
denominator, including ambiguous bases and gaps. Redirect standard output if
the summary should be saved to a file.

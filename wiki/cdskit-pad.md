# cdskit pad

`cdskit pad` adds characters at the 5' and/or 3' end of nucleotide sequences
to restore a length divisible by three. When more than one placement is
possible, it chooses the placement with the fewest internal stop codons under
the selected genetic code.

## Example

### Command
```
cdskit pad --seq_file input.fasta --out_file output.fasta
```

### input.fasta
```
>miss_1nt_5prime
TGCTAAGCGGTAATCTAAGCGGTAATTGA
>miss_2nt_5prime
GCTAAGCGGTAATCTAAGCGGTAATTGA
>miss_1nt_3prime
ATGCTAAGCGGTAATCTAAGCGGTAATTG
>miss_2nt_3prime
ATGCTAAGCGGTAATCTAAGCGGTAATT
>miss_1nt_both
TGCTAAGCGGTAATCTAAGCGGTAATTG
>miss_2nt_both
GCTAAGCGGTAATCTAAGCGGTAATT
>complete
ATGCTAAGCGGTAATCTAAGCGGTAATTGA
```

### output.fasta
```
>miss_1nt_5prime
NTGCTAAGCGGTAATCTAAGCGGTAATTGA
>miss_2nt_5prime
NNGCTAAGCGGTAATCTAAGCGGTAATTGA
>miss_1nt_3prime
ATGCTAAGCGGTAATCTAAGCGGTAATTGN
>miss_2nt_3prime
ATGCTAAGCGGTAATCTAAGCGGTAATTNN
>miss_1nt_both
NTGCTAAGCGGTAATCTAAGCGGTAATTGN
>miss_2nt_both
NNGCTAAGCGGTAATCTAAGCGGTAATTNN
>complete
ATGCTAAGCGGTAATCTAAGCGGTAATTGA
```

## Key options

- `--pad_char CHAR` sets the padding character (`N` by default).
- `--codon_table INT` selects the genetic code used to count internal stops.
- `--drop_pseudo yes` omits sequences that still contain internal stop codons
  after the best padding is chosen. The default is `no`.

The command writes the selected head/tail padding and stop counts to standard
error for records that require evaluation.

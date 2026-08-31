# cdskit mask

`cdskit mask` replaces ambiguous codons, stop codons, and partial-gap codons
with a repeated mask character while preserving the reading frame. Complete
gap codons (`---`) remain gaps.

## Example

### Command
```
cdskit mask --seq_file input.fasta --out_file output.fasta
```

### input.fasta
```
>stop
---ATGTAAATTATGTTGAAG---
>ambiguous1
---ATGTNAATTATGTTGAAG---
>ambiguous2
---ATGT-AATTATGTTGAAG---
>all
---ATGTAAATT--GTTGANG---
```

### output.fasta
```
>stop
---ATGNNNATTATGTTGAAG---
>ambiguous1
---ATGNNNATTATGTTGAAG---
>ambiguous2
---ATGNNNATTATGTTGAAG---
>all
---ATGNNNATTNNNTTGNNN---
```

## Key options

- `--mask_char CHAR` sets the replacement character (`N` by default); it is
  repeated three times for each masked codon.
- `--ambiguous_codon yes|no` controls masking of codons that cannot be
  resolved under the selected genetic code (`yes` by default).
- `--stop_codon yes|no` controls stop-codon masking (`yes` by default).
- `--codon_table INT` selects the genetic code.

Input must be nucleotide data with every sequence length divisible by three.
Partial-gap codons such as `AT-` are masked even when both optional masking
switches are set to `no`.

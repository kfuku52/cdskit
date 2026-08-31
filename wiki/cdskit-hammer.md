# cdskit hammer

`cdskit hammer` removes sparsely occupied codon columns from an aligned CDS.
With an integer `--nail N`, a codon column is retained when at least `N`
sequences translate to a non-missing amino acid there. Translated `-`, `?`,
`X`, and `*` do not count as occupied. Use `--nail all` to require occupancy
in every sequence. This translation-based rule differs from the strict
ACGT-only clean-codon rule used by `trimcodon`.

## Example

### Command
```
cdskit hammer --seq_file input.fasta --out_file output.fasta --nail 4
```

### input.fasta
```
>seq1
---ATGTAAATTATGTTGAAG---TGATGA---------
>seq2
---ATGTNAATTATGTTGAAG---TATTGA---------
>seq3
---ATGTGAATTATGTTGAAG---TATTGA---------
>seq4
---ATGTAAATT---TTGANG---TATTGATTTTCATCA
>seq5
---ATGTAAATTATGTTGANG---TATTGATTTTCATCA
>seq6
---ATGTAAATT---TTGANG---TATTGATTTTCATCA
```

### output.fasta
```
>seq1
ATGATTATGTTGTGA
>seq2
ATGATTATGTTGTAT
>seq3
ATGATTATGTTGTAT
>seq4
ATGATT---TTGTAT
>seq5
ATGATTATGTTGTAT
>seq6
ATGATT---TTGTAT
```

## Key options

- `--nail INT/all` controls the occupancy threshold (`4` by default).
- `--prevent_gap_only yes` may lower the threshold when the requested value
  would leave a sequence containing gaps only. Set it to `no` for strict
  threshold application.
- `--codon_table INT` selects the genetic code used by codon-aware checks.

Input must be an aligned nucleotide sequence set whose lengths are multiples
of three.

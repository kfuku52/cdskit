# cdskit longestorf

`cdskit longestorf` searches six translated frames (3 on `+` strand and 3 on `-` strand) and returns the longest CDS candidate for each input sequence.

## Example

### Command

```bash
cdskit longestorf --seq_file unaligned_nt.fasta --out_file longest_orf.fasta --codon_table 1 --annotate_seq_name yes
```

### unaligned_nt.fasta

```fasta
>seq1
AAAATGAAACCCTAGGGGATGAAAAAAACCCCTGAATGATGCCCTAA
```

### longest_orf.fasta

```fasta
>seq1 strand=+ frame=1 start=19 end=39 nt_len=21 aa_len=7 category=complete
ATGAAAAAAACCCCTGAATGA
```

## Multiple ORFs in one sequence

For `seq1`, multiple complete ORFs exist in the same frame (`+`, frame 1).
`cdskit longestorf` selects the longest one.

```text
Sequence (+, frame 1 codons):
AAA | ATG AAA CCC TAG | GGG | ATG AAA AAA ACC CCT GAA TGA | TGC CCT ...
      [ORF1: 12 nt]              [ORF2: 21 nt, selected]
```

## Candidate priority

Candidates are prioritized in the following order:

1. `complete`: start codon + in-frame stop codon
2. `partial`: start codon to frame end (no in-frame stop)
3. `no_start`: longest stop-free segment (when no start-based CDS is found)

## Notes

- `longestcds` is a deprecated alias with the same behavior; use `longestorf`
  in new commands.
- Input sequences do not need to be aligned.
- Output sequence orientation follows the predicted coding strand.
- Coordinates in output description (`start`, `end`) are reported on the original input strand coordinates (1-based), with strand indicated by `strand=+/-`.
- Header annotation (`strand=... frame=... start=...`) is optional via `--annotate_seq_name yes|no` (default: `no`).
- `aa_len` is `nt_len / 3`; for a complete candidate it includes the terminal
  stop position. It is not the length of the protein after removing `*`.

# cdskit longestorf

`cdskit longestorf` searches six translated frames (3 on `+` strand and 3 on `-` strand) and returns the longest ORF candidate for each input sequence.

This is the canonical command name. `cdskit longestcds` remains available as a deprecated alias.

### Example

#### Command

    cdskit longestorf --seqfile unaligned_nt.fasta --outfile longest_orf.fasta --codontable 1 --annotate_seqname yes

#### unaligned_nt.fasta

    >seq1
    AAAATGAAACCCTAGGGGATGAAAAAAACCCCTGAATGATGCCCTAA

#### longest_orf.fasta

    >seq1 strand=+ frame=1 start=19 end=39 nt_len=21 aa_len=7 category=complete
    ATGAAAAAAACCCCTGAATGA

### Multiple ORFs in one sequence

For `seq1`, multiple complete ORFs exist in the same frame (`+`, frame 1).  
`cdskit longestorf` selects the longest one.

    Sequence (+, frame 1 codons):
    AAA | ATG AAA CCC TAG | GGG | ATG AAA AAA ACC CCT GAA TGA | TGC CCT ...
          [ORF1: 12 nt]              [ORF2: 21 nt, selected]

### Candidate priority

Candidates are prioritized in the following order:

1. `complete`: start codon + in-frame stop codon
2. `partial`: start codon to frame end (no in-frame stop)
3. `no_start`: longest stop-free segment (when no start-based ORF is found)

### Notes

- Input sequences do not need to be aligned.
- Output sequence orientation follows the predicted coding strand.
- Coordinates in output description (`start`, `end`) are reported on the original input strand coordinates (1-based), with strand indicated by `strand=+/-`.
- Header annotation (`strand=... frame=... start=...`) is optional via `--annotate_seqname yes|no` (default: `no`).

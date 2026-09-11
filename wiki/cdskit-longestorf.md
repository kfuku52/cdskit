# cdskit longestorf

`cdskit longestorf` searches six translated frames (3 on `+` strand and 3 on `-` strand) and returns one ranked ORF candidate for each input sequence. The default prioritizes complete candidates before length.

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

## Explicit selection and candidate reports

The default is `--selection complete-first`: category priority above, then
length. `--selection longest` instead compares length first across **all**
start-based candidates and maximal stop-free segments in all six frames, then
category. A `no_start` candidate may win even when start-based candidates exist.
The search is not limited to one longest complete candidate per strand.

Both modes resolve remaining ties deterministically: plus strand, then the
existing stop/start flags, smaller frame number, and smaller original start
coordinate. Length is nucleotide length, including the terminating codon for a
complete candidate. This convention is shared by both modes.

```bash
cdskit longestorf --seq_file transcripts.fasta --out_file candidates.fasta \
  --selection longest --report candidates.json --annotate_seq_name yes
```

`--report` records all start-to-first-definite-stop (or frame-end) candidates
and maximal stop-free segments, including unselected ones, with ranks, ties
before deterministic ordering, coordinates, stop evidence and uncertainty.
Nested starts are included; arbitrary subsegments are not enumerated. Only the
selected candidate contains `output_seq`; unselected sequences are reconstructed
from coordinates and strand. Counts are computed from per-frame prefix sums,
so nested candidates do not repeatedly scan or copy the same long sequence.
Reports still contain one row per candidate. `sort_key` records the actual
selection policy, and missing/ambiguous counts supplement stop uncertainty.
Coordinates refer to the original input; output sequence is in coding orientation.

Codes 27/28/31 do not terminate a candidate at a dual-coding codon based solely
on table membership. Such boundaries are unconfirmed, and the candidate may
extend to frame end. An explicit annotated CDS can instead be checked with
`translate --complete_cds yes`. Standard-code TAR is a definite boundary;
TAN is not. See [codon semantics](https://github.com/kfuku52/cdskit/wiki/codon-semantics).

Neither selection mode identifies a functional gene. For example, the default
selects the 6 nt complete `ATGTAA` even if a much longer partial region follows.
Longer candidates are not inherently more accurate gene annotations; compare
coordinates to independent annotation, homology and other biological evidence.

## Citation

No separate method paper is designated for this utility. Report the CDSKIT
version, command and options; see [citing CDSKIT](https://github.com/kfuku52/cdskit/wiki#citing-cdskit).

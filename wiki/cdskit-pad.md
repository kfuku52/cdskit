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

## Frame preservation and provenance

`--mode min-stop` (default) retains the historical candidate order and picks the
first candidate with the fewest **definite internal stops**. Alternative frames
are evaluated only when the initially tail-padded sequence has such a stop.
Otherwise, the original sequence or its tail-padded form is retained. Equal
scores do not establish a unique biological frame.

Use `--mode preserve-frame` for a supplied/annotated reading frame. It only adds
0–2 characters at the tail, preserving the original bases (including X and
letter case); internal stops remain visible. `--drop_internal_stop
yes` drops records with definite internal stops after the selected padding;
`--drop_pseudo yes` is a compatibility alias. Neither option diagnoses a
pseudogene, and retained records can still contain uncertain stop possibilities.

```bash
cdskit pad --seq_file cds.fasta --out_file padded.fasta \
  --mode preserve-frame --drop_internal_stop yes --report padding.json
```

The optional report includes original and initially tail-padded counts, every
evaluated placement, head/tail padding, frame offset in the original sequence,
selected and tied candidate indexes, and the keep/drop reason. The stderr field
`tail_padded_num_stop` replaces the misleading `original_num_stop` label.
An original frame offset is zero-based; it is `(-head_padding) mod 3` after
padding. Artificial padding is not recovered sequence evidence.

For example, default padding can turn `ATGTAAGGG` into `NATGTAAGGGNN` with zero
internal stops. This does not demonstrate that the new frame is correct.
Internal frameshifts cannot in general be repaired by changing only the ends.
See [codon meaning and uncertainty](codon-semantics.md) for terminal rules,
report formats and interpretation.

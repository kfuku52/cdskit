# cdskit stats

`cdskit stats` prints aggregate nucleotide-sequence statistics to standard
output by default (`--mode sequence`). Use `--mode alignment` for AMAS-compatible
DNA/protein alignment TSV summaries. `--seq_type` defaults to `dna`; `aa` is
accepted only in alignment mode. `--out_file PATH` saves either output format
atomically. `--threads` controls the existing sequence-mode parallel work;
alignment summaries run serially.

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
denominator, including ambiguous bases and gaps. Use `--out_file` or redirect
standard output to save the summary.

## Alignment mode

Available in CDSKIT 0.31.0 and newer.

Summarize one DNA or protein alignment as a rectangular TSV, without an AMAS
runtime dependency. This mode replaces the alignment-summary operation, not
AMAS concatenation, conversion, trimming, or per-taxon reports.

```bash
cdskit stats --mode alignment --seq_file alignment.fasta --seq_type dna --out_file summary.tsv
cdskit stats --mode alignment --seq_file proteins.fasta --seq_type aa > summary.tsv
gzip -cd alignment.fa.gz | cdskit stats --mode alignment --seq_type dna > summary.tsv
```

`--seq_file` and `--out_file` default to standard input/output (`-`). Input format
is FASTA by default; `--in_seq_format` uses Biopython SeqIO format names. Decompress
compressed files before passing them to this command. Logs go to standard error.
Output is written atomically to named files. Input/output path collisions fail.

### Output and AMAS compatibility

DNA output has the AMAS `summary` column names, order, character counts, and
numeric rounding. Protein output additionally includes `AT_content` and
`GC_content` as `NA`, so consumers selecting a common schema can read both modes.
The one-table output does not add `schema_version` or `section` columns.

The summary columns are `Alignment_name`, `No_of_taxa`, `Alignment_length`,
`Total_matrix_cells`, `Undetermined_characters`, `Missing_percent`,
`No_variable_sites`, `Proportion_variable_sites`, `Parsimony_informative_sites`,
`Proportion_parsimony_informative`, `AT_content`, and `GC_content`, followed by
per-character counts. Alignment name is the input basename (`-` for stdin).

- Length and site counts are nucleotide or amino-acid positions, not codons.
  DNA length need not be a multiple of three. Matrix cells = taxa × length.
- DNA missing characters are `X N O - ?`; protein missing characters are
  `X . * - ?`. Missing percentage uses all matrix cells and is rounded to
  three decimals. Other ambiguity codes are not counted as missing.
- Variable sites have at least two distinct unambiguous states. Informative
  sites have at least two states each observed in at least two taxa. Only
  `A C G T` (DNA) or the standard 20 amino acids count as states; ambiguity
  and missing characters are excluded, without removing the whole column.
  Site proportions use the full alignment length and three-decimal rounding.
- **GC is an AMAS compatibility statistic on a 0–1 scale, not a pooled GC
  percentage.** Per taxon, AT = `(A+T+W)/(A+T+W+G+C+S)`, rounded to three
  decimals. These AT values are averaged and rounded again; GC = `1−AT`,
  rounded to three decimals. Taxa without any ATW/GCS contribute AT=0.
  Consequently an entirely missing DNA alignment reports AT=0 and GC=1:
  this inherited convention must not be interpreted as measured 100% GC.
  Use `codonstats` for per-CDS GC/GC1/GC2/GC3 with its separately documented
  denominator, rather than interpreting this compatibility field as that metric.

Lowercase is normalized to uppercase. Accepted DNA alphabet is
`ACGTKMRYSWBVHDXNO-?`; accepted protein alphabet is
`ACDEFGHIKLMNPQRSTVWYBJZX.*-?`. Empty alignments, zero-length sequences,
unequal lengths, empty or duplicate record IDs, and other characters are rejected.
These checks deliberately avoid AMAS behavior that can silently overwrite
duplicate taxa or proceed with invalid alignments. FASTA IDs follow SeqIO's
first-whitespace-delimited token convention.

Definitions were checked against the upstream
[AMAS implementation](https://github.com/marekborowiec/AMAS/blob/9ffc9d688bdb701847be931c2e107d9831caa129/amas/AMAS.py).
Repository-owned DNA/protein fixtures include saved AMAS reference outputs;
tests do not install or run AMAS. CDSKIT implements the statistics independently.

## References

Cite the method or resource actually used and report the CDSKIT version and
options.

- Borowiec ML (2016). AMAS: a fast tool for alignment manipulation and computing of summary statistics. *PeerJ* 4:e1660. [Paper](https://doi.org/10.7717/peerj.1660) — Cite when using AMAS-compatible alignment statistics; CDSKIT computes these independently without running AMAS.

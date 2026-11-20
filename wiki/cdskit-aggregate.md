# cdskit aggregate

`cdskit aggregate` groups sequence IDs after removing one or more regular
expression matches, then retains the longest sequence in each group. The
original ID of the retained record is preserved.

## Example

### Command
```
cdskit aggregate --seq_file input.fasta --out_file output.fasta --expression ":.*" "\|.*"
```

### input.fasta
```
>seq1:1.length=30nt
ATGCTAAGCGGTAATCTAAGCGGTAATTGA
>seq1:2.length=27nt
ATGCTAAGCGGTAATCTAAGCGGTTGA
>seq1:3.length=33nt
ATGCAACTAAGCGGTAATCTAAGCGGTAATTGA
>seq2|1.length=45nt
ATGTCGGAATTGCGAGTAAGCACCAGCTTCTCAAAACCAAAATAA
>seq2|2.length=54nt
ATGTCGAGATCCCGAGAATTGCGAGTAAGCACCAGCTTCTCAAAACCAAAATAA
```

### output.fasta
```
>seq1:3.length=33nt
ATGCAACTAAGCGGTAATCTAAGCGGTAATTGA
>seq2|2.length=54nt
ATGTCGAGATCCCGAGAATTGCGAGTAAGCACCAGCTTCTCAAAACCAAAATAA
```

## Notes

- Values passed to `--expression` are applied sequentially. In this example,
  `:.*` and `\|.*` reduce the IDs to the group keys `seq1` and `seq2`.
- `--mode longest` is currently the only selection mode. Ties retain the first
  record encountered in the input.
- If `--expression` is omitted, records are copied without aggregation.

## Protein-supported representatives

Use `--group_table groups.tsv` to supply explicit groups with the columns
`transcript_id` and `group_id`. The representative FASTA retains the original
sequence ID. When combined with `--expression`, the expressions are applied to
sequence IDs before looking up their group. For example, this lets alternative
ORFs with a `.cdskit-orf-N` suffix share the source transcript's group.

Pass `--hit_table hits.tsv --selection_report choices.tsv` with an aggregation
expression to use protein homology. The TSV requires this header, with protein
coordinates in 1-based inclusive units:

```text
query_id	target_id	evalue	bitscore	query_start	query_end	query_length	target_start	target_end	target_length
```

These columns correspond to DIAMOND `blastp --outfmt 6 qseqid sseqid evalue bitscore
qstart qend qlen sstart send slen`. The query ID matches the FASTA ID. Compare
hits from the same search settings and database. A qualifying HSP has
E-value at most 1e-5 and covers at least 60% of both sequences.

The longest candidate remains the baseline. A qualifying alternative can
replace an unsupported baseline if all supported candidates share a near-best
target. For a supported baseline, an alternative must hit the baseline's best
target within a 1.10-fold margin of its own best bitscore, gain at least 10% in bitscore
on that target, and cover at least as much of the target. The report records
the baseline, selected sequence and evidence for every group.

## Citation

No separate method paper is designated for this utility. Report the CDSKIT
version, command and options; see [citing CDSKIT](https://github.com/kfuku52/cdskit/wiki#citing-cdskit).

# Concatenate codon alignments

```sh
cdskit concat --taxa_tsv taxa.tsv --genes_tsv genes.tsv --out_dir matrix
```

Use one `taxon` column listing unique FASTA IDs in the desired matrix order,
including separate IDs for populations. In the gene manifest, supply unique
`gene` names and an `alignment` path relative to that manifest, in locus order.
Each aligned FASTA must have equal, nonempty, in-frame rows. Record IDs are
the first whitespace-delimited header field. Gaps must occupy entire codons.
The input alphabet is IUPAC DNA, `?` and `-`. An optional `backtrim_json`
column points to a successful, complete CDSKIT backtrim mapping report.

A new output directory contains `alignment.fa`, IQ-TREE DNA gene partitions
in `partitions.txt`, inclusive nucleotide and codon ranges in `partitions.tsv`,
and a one-based `sites.tsv` mapping every matrix codon to its input alignment
column and its pre-trim column when supplied. Without a trimming report, source
coordinates refer to the supplied alignment. In `coverage.tsv`, absent records
have `present=0` and are filled with gaps. Supplied all-missing sequences retain
`present=1`; `nonmissing_nt` excludes N, ? and gaps. Unknown or repeated IDs
and partial codon gaps fail the operation before publication. The matrix is
transposed in bounded batches using temporary space in the output filesystem.

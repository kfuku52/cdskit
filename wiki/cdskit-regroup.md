# cdskit regroup

`regroup` audits rnaSPAdes transcript groups and can subdivide them using
external evidence of distinct loci.

```bash
cdskit regroup --seq_file transcripts.fasta --paths transcripts.paths \
  --fastg assembly_graph.fastg --kmer_size 49 \
  --out_file groups.tsv --report group-audit.tsv
```

Use the final assembly graph's k-mer size. The audit finds shared cut edges
and cut transcript paths within each original multi-transcript group. It uses
the rnaSPAdes edge criterion of `FASTG edge length - k > 300`. A cut indicates
that removing the edge or transcript disconnects at least two transcript
paths. The report lists such connections for review and retains all original
assignments by default. The graph and paths must come from the same assembly.

Add `--locus_table loci.tsv` to supply `transcript_id` and `locus_id` columns.
Use original FASTA IDs and one row per uniquely assigned transcript. Omit
ambiguous or unmapped transcripts. Assign locus IDs from the same
species-matched reference and annotation. A group is split only when every
member has a locus assignment and the assignments cover more than one locus.
Published child group IDs use `gN.split1`, `gN.split2`, and so on in lexical
locus order. A locus table can also be used on its own.

The optional `--junction_table` adds externally measured paired-read counts to
the graph report. Its columns are `transcript_id`, `from_edge`, `to_edge`, and
`read_pairs`. Use the signed edges in the forward transcript path, such as
`7+` and `9-`. Counts should represent pairs spanning that junction. The JSON
column `bridge_junctions` distinguishes measured zero from a missing count.
Junction support is reported as evidence for review.

Use `cdskit aggregate --seq_file cds.fasta --group_table groups.tsv` with
CDS FASTA IDs that match the source transcript IDs.

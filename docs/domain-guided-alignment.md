# Domain search and alignment

Install MAFFT and NCBI BLAST+ on `PATH`. Specify the input sequence type as
`dna`, `codon`, or `aa`. FASTA inputs and named FASTA outputs can use gzip.
With `codon`, CDSKIT translates in the specified genetic code, aligns in amino
acid space, and maps the original codons back onto that alignment. CDS lengths
must be multiples of three. An optional `--aa_file` supplies existing
translations with matching IDs and residues, allowing `X` at masked sites.

```bash
cdskit align --seq_type dna --seq_file loci.fna --out_file loci.aln.fna
cdskit align --seq_type aa --seq_file family.faa --out_file family.aln.faa.gz
cdskit align --seq_type codon --seq_file family.fna --codon_table 1 \
  --out_file family.codon.aln.fna.gz --threads 8
```

## Search for conserved domains

`cdskit rpsblast` writes a headered BLAST TSV with one row per hit and an
explicit no-hit row for each remaining sequence. Coordinates are amino acid
positions for `aa` and `codon`. The `dna` mode uses `rpstblastn` with all six
frames and reports nucleotide coordinates and `qframe`. `--rps_mt_mode queries`
can improve throughput for batches with enough query sequences.

```bash
cdskit rpsblast --seq_type codon --seq_file family.fna --threads 8 \
  --out_file family.domains.tsv --manifest family.domains.json
```

Use `--db_prefix` to select an existing BLAST profile database. By default
CDSKIT can use a bundled Pfam database or download and validate NCBI's
`Pfam_LE` distribution into `${XDG_CACHE_HOME:-$HOME/.cache}/cdskit/rpsblast`.
Set `--db_cache` to share a cache across jobs. Downloaders take a lock using
exclusive file creation, validate the database, and publish it with a rename
on the cache filesystem. Other workers reuse the published copy. Use
`--no_download` on nodes that cannot access the download server, or provide a
local archive as a `file://` URL through `--db_url`.

## Select and extend a backbone

```bash
cdskit align --strategy query_backbone --seq_type codon \
  --seq_file family.fna --query_type aa --query_file references.faa \
  --rpsblast_tsv family.domains.tsv --domain_manifest family.domains.json \
  --search_missing_domains --backbone_size 100 --threads 8 \
  --backbone_report backbone.tsv --column_map columns.tsv \
  --out_file family.aln.fna.gz
```

References already present with matching sequences enter the backbone first.
Other candidates are ranked by ordered domain-profile coverage. An optional
`--priority_tsv` with `gene_id` and integer `priority` takes precedence, with
lower values sorted first. An optional `--species_tsv` with `gene_id` and
`species` prioritizes representing distinct species. If the references have
no domain hits, selection uses these inputs and sequence IDs.

Complete precomputed results skip domain searches. A separate reference or an
incomplete table requires `--search_missing_domains` to search the missing
sequences. Use `--domain_manifest` to verify input and table hashes, coordinate
units and the E-value threshold. Set `--search_translated` to derive domain
evidence automatically for DNA. To skip selection and reuse a pre-aligned
backbone, supply `--backbone_file` in the alignment alphabet (AA for codon).

MAFFT adds remaining sequences while retaining insertion columns. Output IDs,
residues and codons are checked. In `--column_map`, an empty backbone column
marks an insertion. The map uses amino acid columns for codon alignments.

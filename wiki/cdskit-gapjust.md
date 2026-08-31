# cdskit gapjust

`cdskit gapjust` normalizes runs of `N` to a fixed length. When a matching GFF
file is supplied, feature coordinates after each edited gap are shifted to stay
consistent with the output sequence.

## Example

```bash
cdskit gapjust \
  --seq_file input.fasta \
  --out_file output.fasta \
  --gap_len 100 \
  --gap_just_min 20 \
  --gap_just_max 1000
```

This changes `N` runs of 20–1000 bases to exactly 100 bases. Shorter or longer
runs are left unchanged. Without `--gap_just_min` or `--gap_just_max`, every
`N` run whose length differs from `--gap_len` is adjusted.

## GFF-aware use

```bash
cdskit gapjust \
  --seq_file assembly.fasta \
  --in_gff annotations.gff \
  --out_file assembly.gapjust.fasta \
  --out_gff annotations.gapjust.gff \
  --gap_len 100
```

FASTA record IDs and GFF `seqid` values must match. Coordinates after an edited
gap shift by the inserted or removed length. Coordinates in a retained gap
segment keep their relative position; those removed by shortening are clamped
to the retained gap's end (or the preceding coordinate, at least 1, when the
gap is deleted). Feature phase is not recalculated.

## Key options

- `--gap_len INT`: Target length for adjusted `N` runs; default `100`.
- `--gap_just_min INT`: Do not extend gaps shorter than this value.
- `--gap_just_max INT`: Do not shorten gaps longer than this value.
- `--threads INT`: Requested workers; `0` detects CPUs up to the safety limit
  (64 by default). Small workloads can run serially.

The input must be DNA. A summary of the number and original sizes of edited
gaps is written to standard error.

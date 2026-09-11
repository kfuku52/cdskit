# cdskit gapjust

Normalize selected runs of `N` bases in a DNA assembly to a fixed length.
This is scaffold gap formatting, not CDS repair or inference of missing sequence.

```bash
cdskit gapjust \
  --seq_file input.fasta --out_file output.fasta \
  --gap_len 100 --gap_just_min 20 --gap_just_max 1000
```

This changes `N` runs of 20–1000 bases to exactly 100 bases. Shorter or longer
runs are left unchanged. Without the thresholds, every `N` run whose length
differs from `--gap_len` is adjusted. Lowercase `n` becomes uppercase `N`.

## GFF-aware use

```bash
cdskit gapjust \
  --seq_file assembly.fasta --in_gff annotations.gff \
  --out_file assembly.gapjust.fasta --out_gff annotations.gapjust.gff \
  --gap_len 100 --edit_report gapjust.edits.json
```

By default, a length-changing N run overlapping any CDS is rejected, even if
its length difference is divisible by three. The check covers the entire
original N interval, including boundaries and runs spanning multiple CDSs.
Both `CDS` and `SO:0000316` are recognized. Missing Parent, missing phase, and
unknown strand do not disable overlap protection. A run already at the target
length is not an edit and is allowed inside CDS.

Use `--cds_overlap skip` to leave each CDS-overlapping N run at its original
length while editing other runs. A run shared by several transcripts is skipped
in its entirety if it overlaps any CDS; it is never split at CDS boundaries.
The result can therefore contain different gap lengths.

Edits confined to introns or intergenic regions preserve CDS bases and phase
while moving coordinates. This does not establish unchanged biological splicing
or recover the true gap length. Changing N counts within CDS can alter translation;
recalculating phase cannot restore the original biological reading frame.

Shortening retains the leftmost bases of a run in genomic coordinates. If any
feature start or end would be deleted, the operation fails rather than clamping
it onto a neighboring base. This applies to non-CDS features too, and is not
overridden by `skip`. Automatic feature deletion or reannotation is not provided.
Coordinates in retained parts of gaps preserve their relative positions.

FASTA IDs must be unique when GFF is supplied, and every GFF seqid must have a
matching FASTA record. Feature bounds and `##sequence-region` bounds are checked
against FASTA lengths. Sequence-region directives are updated, including terminal
gap deletions; edits crossing a directive boundary or deleting its entire region
are rejected. Other headers and feature attributes are preserved.

Input Parent cycles, unresolved references, unknown CDS strand/phase, and ordinary
spliced-CDS phase inconsistencies are reported without repairing them. Phase
continuity uses each Parent, transcription order, and the original initial phase;
partial CDSs are not forced to phase zero. These diagnostics are not a complete
GFF3 validator and cannot interpret programmed frameshifts or other translation
exceptions. GFF with embedded FASTA is not supported by the existing reader.

All edits are checked before output, including standard output. FASTA and GFF
cannot both use standard output. A rejected edit
leaves existing output files untouched. File outputs (FASTA, GFF, and optional
report) are staged together. Streaming output cannot be rolled back after an I/O
failure. No-GFF mode retains sequence-only normalization, reports that CDS overlap
was not checked, and must not be treated as CDS-safe processing.

## Edit report

`--edit_report PATH` writes a separate JSON file on success (not to stdout).
Schema version 1 records `coordinate_system: "1-based-inclusive"`, `cds_overlap`,
`cds_checked`, input diagnostics, and an `edits` list. Each entry includes:

- `record_index` (1-based input FASTA order) and `seqid`;
- `original_start`, `original_end`, `target_length`, and `length_delta`;
- `action` (`apply` or `skip`) and `reason` (`non_cds_edit`, `cds_overlap`, or
  `cds_not_checked`);
- overlapping feature types, original bounds, strand, decoded IDs and Parents.

Unchanged runs and runs excluded by length thresholds are not report entries.
For repeated FASTA IDs in no-GFF mode, `record_index` identifies the input record.
Rejected runs produce an error diagnostic and no new report or sequence output.
Keep original input files with the report when reproducible regeneration is needed.

## Python API and migration

`normalize_record_gap_lengths(..., gff=gff, cds_overlap="error")` optionally
protects a single sequence with an original-coordinate GFF. It returns the
accepted edits and existing count/min/max summary tuple. It changes only the
sequence, not the supplied GFF.

For paired sequence/GFF transformations, prefer `gapjust_main`, or call
`plan_record_gap_lengths` for each record, then
`gapjust_gff.select_gap_edits(gff, plans, cds_overlap="skip")` once and apply
**the returned accepted plans** to both `apply_record_gap_edits` and
`apply_gap_justifications_to_gff`. Sequence application also validates the original N intervals and rejects stale
or malformed plans before changing the record. Embedded SeqRecord features and
per-base quality values are cleared after any length-changing edit, including
changes whose total length differences cancel; their coordinates are not lifted.
The GFF application rejects CDS intersections itself;
it does not independently skip edits, which would desynchronize FASTA and GFF.

GFF application requires interval dictionaries with `original_gap_start`
(0-based), `original_gap_length`, and `target_gap_length`. Legacy point shifts
or tuples lack deletion/overlap information and are rejected by the GFF API.
The low-level `vectorized_coordinate_update` still accepts legacy point shifts
for compatibility, but provides no CDS safety guarantee. Range-aware coordinate
mapping now raises for deleted endpoints instead of clamping them.

Existing CDS-overlapping or endpoint-deleting GFF workflows now stop deliberately.
Use `skip` for protected CDS gaps, adjust gap thresholds to exclude problematic
runs, or obtain separately reannotated data. There is no unsafe allow mode.
Safe scaffold edits retain their previous sequence and feature outputs, except
for updated sequence-region metadata and added diagnostics.

## Key options

- `--gap_len INT`: Target length; default `100`, zero deletes selected runs.
- `--gap_just_min INT`: Do not extend gaps shorter than this value.
- `--gap_just_max INT`: Do not shorten gaps longer than this value.
- `--cds_overlap {error,skip}`: CDS protection policy; default `error`.
- `--edit_report PATH`: Optional JSON audit file.
- `--threads INT`: Requested workers; `0` detects CPUs up to the safety limit.

No model format, feature definition, or trained weight changes are involved.

See the [validation record](https://github.com/kfuku52/cdskit/blob/master/docs/gapjust-safety-validation.md) for the regression
checks, independent CDS extraction, and validation limitations. Coordinate-valued
custom attributes are preserved verbatim, not lifted; check those separately when
using annotation dialects that embed additional genomic coordinates in attributes.

## Citation

No separate method paper is designated for this utility. Report the CDSKIT
version, command and options; see [citing CDSKIT](https://github.com/kfuku52/cdskit/wiki#citing-cdskit).

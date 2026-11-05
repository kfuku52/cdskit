# Shared workflow APIs

`cdskit validate --profile dna|protein|codon` applies strict streaming FASTA
artifact contracts. The default `quality` profile retains historical quality
reporting. Codon artifacts require equal aligned lengths and both aligned and
ungapped lengths divisible by three. `fasta_contract` exposes the same checks
for already-open gzip/archive streams.

`label --mapping-out PATH` exports the old_name/new_name TSV assigned to the
sequence output; `--mapping-in PATH` applies a saved map. `label_mapping` also
offers deterministic bounded shortening that reserves existing short names.
Its `iter_fasta_headers` and `map_fasta_headers` helpers stream complete headers
without materializing sequence text, preserving descriptions and line endings.
The same TSV can be passed to NWKIT rename. A caller publishing paired FASTA
and tree outputs must validate their ID sets and publish both together.
Mapped IDs must remain unique, including IDs left untouched by a partial map.
Sequence IDs cannot contain whitespace that would change their FASTA meaning.
When clipping creates collisions, use `--unique` before exporting the map.

`sequence_selection` provides bounded record iteration, exact-ID selection and
the streaming FASTA tuple protocol. Ordering, duplicate rejection and missing-ID
checks are explicit; this does not replace a workflow's persistent sequence index.

GFF3/GTF attribute syntax, coding streams, ordered feature geometry, phase carry,
strand-aware indexed reconstruction and transcript/UTR coordinates are in
`gff_attributes`, `gff_attribute_syntax`, `coding_gff_stream`,
`gff_feature_structure`, `cds_features` and `transcript_coordinates`.
Embedded `##FASTA` ends feature parsing. Source-model admission and correction
decisions remain the caller's responsibility.

`coordinates.trimmed_mapping` reuses the complete subsequence mapping kernel,
reports ambiguity, and accepts leftmost/unresolved/error policies. Its module
entrypoint processes batches for R clients. Packaged R coordinate functions
preserve intron phase and unresolved positional evidence. `pairwise_identity`
uses only amino-acid sites observed in both members of each pair.
`protein_similarity` exposes coverage, identity and phase-aware splice-junction
scores without selecting representative isoforms.
Identical-sequence shortcuts check the caller's actual alignment scores;
custom matrices, wildcards and gap rewards still use the supplied aligner.

`alignment_metrics` supplies basic alignment lengths and F3X4 frequencies with
explicit policies: U becomes T, an ambiguous triplet is omitted as a whole, and
unambiguous stops remain in the denominator. These are separate from codon-usage
statistics that omit stops.
Length summaries use constant auxiliary memory regardless of the record count.

`translate.coding_protein` preserves ordinary codon translation; it does not
infer complete-CDS initiation or termination. Use the explicit complete-CDS
translation API only with known CDS boundaries, especially for codes 27/28/31.

`localize_models.resolve_localize_model_path("latest")` resolves a published
checkpoint once per cache and records the selected release and checksum.
`latest_model.resolve` exposes the selection manifest and a configurable cache
namespace. Named model aliases keep their existing semantics. Offline reuse
verifies the selected artifact instead of silently selecting a different model.

For explicit model-family/feature-schema selection, pass `model_family` and `feature_schema` to `latest_model.latest_model` or `resolve`. Selector evidence is a JSON GitHub asset label containing these fields; absent evidence is never guessed from a filename. A frozen cache that differs from the requested selectors is rejected; use a distinct namespace for a distinct model contract.

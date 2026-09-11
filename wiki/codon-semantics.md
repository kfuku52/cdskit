# Codon meaning, uncertainty, and sequence selection

CDSKIT codon semantics version **2** separates base ambiguity from termination.
`TAR` in code 1 is both ambiguous and a definite stop. `TAN` is ambiguous with
possible termination, not a definite stop. Codes 27, 28 and 31 contain codons
that can encode an amino acid or terminate depending on context. Their presence
alone does not establish an internal stop or a complete ORF.

The source definitions are the [NCBI genetic codes](https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi?chapter=cgencodes).
Ordinary translation uses the forward-table amino acid for dual-coding codons,
consistent with [Biopython ordinary translation](https://biopython.org/docs/latest/api/Bio.Seq.html).
CDSKIT does not infer termination from poly(A) distance, taxon-specific signals,
selenocysteine/pyrrolysine annotations, or programmed frameshifts.

## Shared attributes

`cdskit.codonutil.analyze_codon(codon, codontable, context="unknown")` returns
independent immutable attributes:

| Attribute | Meaning |
| --- | --- |
| `ambiguous` | Valid IUPAC uncertainty, including X as A/C/G/T; excludes missing codons |
| `missing` | Contains `-`, `.`, or `?` |
| `invalid`, `partial` | Invalid alphabet or a length other than three; never clean or a definite stop |
| `amino_acids` | Amino acids represented by the expansions under the forward table |
| `definite_stop` | Every expansion is an unconditional stop in the supplied context |
| `possible_stop` | At least one expansion permits termination, but not a definite stop |
| `context_dependent` | At least one expansion has both sense and stop assignments |
| `terminal_stop_compatible` | All expansions can terminate if explicitly used as a CDS terminator |
| `clean` | Complete A/C/G/T codon, no missing/invalid bases and not a definite stop |

`possible_stop` includes dual coding; it does not include definite stops.
`context="ordinary"` and `"unknown"` both preserve dual coding as uncertainty
while allowing forward translation. Only `"complete_terminal"`, explicitly
asserted by the caller, accepts all terminal-compatible expansions as stops.
`clean` describes sequence usability for ordinary translation, not biological
proof. A dual-coding codon can be clean and context-dependent simultaneously.

The old integer `classify_codon` is a lossy compatibility projection, prioritizing
missing, then definite stop, then ambiguous/invalid/partial, then clean. Use the
new attributes for independent counts. `get_stop_codons` in codonutil still
exposes the raw table for callers that need terminal compatibility; do not use
raw membership as an internal-stop test.

## Position and termination

`filter` and `validate` use `last_evaluable`: the last complete codon without
missing characters may be exempt from the **internal** stop count. Trailing
alignment gaps do not shift this boundary; trailing N codons do. An incomplete
tail does not establish a terminal complete codon. Stop totals still include a
terminal stop, and `trimcodon` does not count terminal stops as clean.

`pad` uses `physical`: only the final complete physical codon may be exempt.
Artificial `-` padding must not make a preceding stop disappear as an apparent
terminal codon. Both policies are positional conventions; neither certifies a
complete CDS. Reports record the policy where it is relevant.

## Reports and compatibility

- `filter` and `trimcodon` add possible-stop and context-dependent counts;
  `validate` adds counts and sequence ID lists. These uncertainties do not by
  themselves trigger the definite-internal-stop exclusion rule. Ambiguous
  codons can still fail an independently selected clean-fraction threshold.
- `codonstats` appends `codons_possible_stop`, `codons_context_dependent` and
  `codon_semantics_version` to summary TSV output. `codons_ambiguous` and
  `codons_stop` may overlap. Ambiguous codons are excluded from concrete codon
  usage, even when they are definite stops. GC denominators are unchanged.
- Sectioned reports retain the sectioned TSV format version 2 and separately
  identify codon semantics version 2. Consumers should read column names, not
  fixed positions. JSON gains fields without dropping existing fields.
- `pad --report` and `longestorf --report` accept JSON or sectioned TSV. JSON
  contains metadata and `sequences`; TSV contains `metadata` and `sequence`
  rows with a JSON `data` cell. Duplicate IDs remain distinguishable by
  `input_order` (one-based). Candidate indexes are zero-based; genomic
  coordinates are one-based inclusive. Empty padding inputs have null original
  span coordinates. Only the selected ORF candidate includes its sequence; all
  other candidates are represented by strand and coordinates. Sequence output
  and a report may not both use standard output.
- Existing CLI defaults and FASTA IDs remain unchanged. Definite-stop corrections
  intentionally change filtering, trimming and candidate choices for affected
  inputs. Preserve old output provenance rather than mixing regenerated data.
- Low-level validate worker helpers accept `codontable=` and return uncertainty
  counts as well. Their legacy `stop_codons=` form retains the six-field return
  value, but requires an explicitly unconditional set and cannot recover
  dual-coding context. Call `summarize_records` for a named report.

No model weights or localization feature definitions change. If regenerated
translations or QC outputs feed a model, record the new input version and
assess its effects separately; this change does not automatically retrain it.

## Scope of validation

Tests independently enumerate all IUPAC triplets for each supported code,
check source-backed dual-coding examples, compare six-frame candidate generation
and ranking to a simple reference algorithm, and exercise CLI reports and process
workers. These establish software behavior, not gene functionality.

A fixed small RefSeq panel and synthetic perturbations are documented in the
[2026-09-10 evaluation](https://github.com/kfuku52/cdskit/blob/master/docs/codon-semantics-evaluation-2026-09-10.md).

## References

- Swart EC, Serra V, Petroni G, Nowacki M (2016). Genetic Codes with No Dedicated Stop Codon: Context-Dependent Translation Termination. *Cell* 166:691–702. [Paper](https://doi.org/10.1016/j.cell.2016.06.020) — Biological background for context-dependent termination in the discussed ciliate codes; CDSKIT does not infer termination context.
- Cock PJA et al. (2009). Biopython: freely available Python tools for computational molecular biology and bioinformatics. *Bioinformatics* 25:1422–1423. [Paper](https://doi.org/10.1093/bioinformatics/btp163) — Sequence parsing, feature extraction and genetic-code infrastructure used by CDSKIT.

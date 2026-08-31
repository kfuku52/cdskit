# cdskit wiki

cdskit is a command-line toolkit for frame-aware processing, validation,
visualization, and localization analysis of protein-coding sequences.

## Setting up

- [Installation and dependencies](https://github.com/kfuku52/cdskit/wiki/Installation-and-dependencies)

After installation, run `cdskit --help` for the command list and
`cdskit COMMAND --help` for current options and defaults.

Long options use `snake_case`. Older compact spellings are accepted for
backward compatibility but print a deprecation warning. Structured TSV reports are UTF-8,
tab-delimited, rectangular, and LF-terminated. Multi-section reports contain
`schema_version` and `section` columns. See the
[0.24 migration guide](https://github.com/kfuku52/cdskit/blob/master/MIGRATION.md)
for old-to-new mappings.

`codonstats --mode both` is an exception: it prints two separately headed
tables. Use `--mode summary` or `--mode usage` for a single TSV table.
`longestcds` is a deprecated alias of `longestorf`.

This wiki describes the current `master` source; packaged releases can lag
behind it. Check `cdskit --version` when comparing installed behavior. Page
sources and images are also maintained in the repository's
[`wiki/` directory](https://github.com/kfuku52/cdskit/tree/master/wiki);
see [documentation maintenance](https://github.com/kfuku52/cdskit/blob/master/docs/documentation.md).

## CDSKIT commands

- [`accession2fasta`](https://github.com/kfuku52/cdskit/wiki/cdskit-accession2fasta): Retrieving fasta sequences from a list of GenBank accessions

- [`aggregate`](https://github.com/kfuku52/cdskit/wiki/cdskit-aggregate): Extracting the longest sequences combined with a sequence name regex

- [`backalign`](https://github.com/kfuku52/cdskit/wiki/cdskit-backalign): Back-aligning CDS from unaligned CDS plus aligned proteins

- [`backtrim`](https://github.com/kfuku52/cdskit/wiki/cdskit-backtrim): Back-translating a trimmed protein alignment

- [`codonstats`](https://github.com/kfuku52/cdskit/wiki/cdskit-codonstats): Printing codon-aware per-sequence and aggregate codon-usage statistics

- [`degeneracy`](https://github.com/kfuku52/cdskit/wiki/cdskit-degeneracy): Extracting aligned 0/2/3/4-fold degenerate nucleotide positions

- [`filter`](https://github.com/kfuku52/cdskit/wiki/cdskit-filter): Filtering CDS by sequence-level clean-codon fraction and quality rules

- [`gapjust`](https://github.com/kfuku52/cdskit/wiki/cdskit-gapjust): Adjusting consecutive Ns to the fixed length

- [`hammer`](https://github.com/kfuku52/cdskit/wiki/cdskit-hammer): Removing less-occupied codon columns from a gappy alignment

- [`intersection`](https://github.com/kfuku52/cdskit/wiki/cdskit-intersection): Dropping non-overlapping sequence labels between two sequences files or between a sequence file and a gff file

- [`label`](https://github.com/kfuku52/cdskit/wiki/cdskit-label): Modifying sequence labels

- [`longestorf`](https://github.com/kfuku52/cdskit/wiki/cdskit-longestorf): Finding the longest ORF by six-frame translation (+/- strands, 3 frames each)

- [`localize`](https://github.com/kfuku52/cdskit/wiki/cdskit-localize): Predicting targeting peptide classes (`noTP`, `SP`, `mTP`, `cTP`, `lTP`) or compatible multi-label localization models from CDS or protein input

- [`localize-learn`](https://github.com/kfuku52/cdskit/wiki/cdskit-localize-learn): Training custom `cdskit localize` models from TSV or UniProt-derived labels

- [`mask`](https://github.com/kfuku52/cdskit/wiki/cdskit-mask): Masking ambiguous and/or stop codons

- [`maxalign`](https://github.com/kfuku52/cdskit/wiki/cdskit-maxalign): Removing sequences to maximize codon-based alignment area ([MaxAlign](https://link.springer.com/article/10.1186/1471-2105-8-312))

- [`pad`](https://github.com/kfuku52/cdskit/wiki/cdskit-pad): Making nucleotide sequences in-frame by head and tail paddings

- [`parsegb`](https://github.com/kfuku52/cdskit/wiki/cdskit-parsegb): Converting the GenBank format

- [`plot`](https://github.com/kfuku52/cdskit/wiki/cdskit-plot): Plotting aligned CDS clean-fraction summaries, codon-state maps, and nucleotide alignment views with consensus codon/AA and AA frequency logos

- [`printseq`](https://github.com/kfuku52/cdskit/wiki/cdskit-printseq): Print a subset of sequences with a regex

- [`rmseq`](https://github.com/kfuku52/cdskit/wiki/cdskit-rmseq): Removing a subset of sequences by using a sequence name regex and by detecting problematic sequence characters

- [`split`](https://github.com/kfuku52/cdskit/wiki/cdskit-split): Splitting 1st, 2nd, and 3rd codon positions

- [`stats`](https://github.com/kfuku52/cdskit/wiki/cdskit-stats): Printing sequence statistics

- [`translate`](https://github.com/kfuku52/cdskit/wiki/cdskit-translate): Translating CDS nucleotide sequences to amino acids

- [`trimcodon`](https://github.com/kfuku52/cdskit/wiki/cdskit-trimcodon): Trimming aligned CDS codon columns by clean-codon fraction

- [`validate`](https://github.com/kfuku52/cdskit/wiki/cdskit-validate): Validating aligned CDS quality and reporting issues

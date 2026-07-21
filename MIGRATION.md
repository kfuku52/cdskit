# Migrating to cdskit 0.24

cdskit 0.24 standardizes CLI option names and TSV contracts. Input-side
renames are backward compatible. The four multi-part TSV reports listed below
use a new rectangular schema; their JSON equivalents are unchanged.

## CLI options

Use `snake_case` for long options. For example:

| Command or scope | Deprecated spelling | Canonical spelling |
| --- | --- | --- |
| shared input | `--seqfile` | `--seq_file` |
| shared input | `--inseqformat` | `--in_seq_format` |
| shared output | `--outfile` | `--out_file` |
| shared output | `--outseqformat` | `--out_seq_format` |
| shared sequence type | `--seqtype` | `--seq_type` |
| shared codon table | `--codontable` | `--codon_table` |
| sequence selection | `--seqname` | `--seq_name_regex` |
| sequence selection | `--seq_name` | `--seq_name_regex` |
| `aggregate` | `--seqnamefmt` | `--seq_name_format` |
| `backalign` | `--stopcodon` | `--stop_codon` |
| `filter` | `--ambiguouscodon` | `--ambiguous_codon` |
| `intersection` | `--seqfile2` | `--seq_file_2` |
| `intersection` | `--inseqformat2` | `--in_seq_format_2` |
| `intersection` | `--outfile2` | `--out_file_2` |
| `intersection` | `--outseqformat2` | `--out_seq_format_2` |
| `intersection` | `--ingff` | `--in_gff` |
| `intersection` | `--outgff` | `--out_gff` |
| `intersection` | `--fix_outrange_gff_records` | `--fix_out_of_range_gff_records` |
| `label` | `--list_seqname_keys` | `--list_seq_name_keys` |
| `localize` | `--no_model_download BOOL` | `--model_download BOOL` with the value inverted |
| `mask` | `--maskchar` | `--mask_char` |
| `maxalign` | `--keep` | `--keep_seq_name_regex` |
| `maxalign` | `--missing_char` | `--missing_chars` |
| `pad` | `--padchar` | `--pad_char` |
| `pad` | `--nopseudo` | `--drop_pseudo yes` |
| `parsegb` | `--annotate_seqname` | `--annotate_seq_name` |
| `plot` | `--plotformat` | `--format` |
| `printseq` | `--show_seqname` | `--show_seq_name` |
| `rmseq` | `--problematic_char` | `--problematic_chars` |
| TargetP benchmark | `--prepared_tsv` | `--out_tsv` |
| TargetP external augmentation | `--external_tsv_out` | `--external_out_tsv` |
| TargetP external Torch | `--external_tsv_out` | `--external_out_tsv` |
| TargetP external evaluation | `--targetp_tsv` | `--targetp_reference_tsv` |
| TargetP Torch evaluation | `--reuse_cache` | `--reuse_oof_cache` |
| DeepLoc benchmark | `--report_json` | `--prepare_report_json` |
| peroxisome benchmark | `--train_tsv` | `--training_tsv` |
| peroxisome benchmark | `--report_json` | `--out_json` |
| peroxisome benchmark | `--report_md` | `--out_md` |
| peroxisome benchmark | `--validation_partition` | `--validation_fold_id` |
| UniProt preset split | `--input_tsv` | `--eukaryota_tsv` |
| UniProt preset split | `--report_json` | `--out_json` |

Every renamed option remains accepted during the 0.24 release series. Using a
deprecated spelling writes a warning such as the following to standard error:

```text
Warning: --seqfile is deprecated; use --seq_file instead.
```

Boolean arguments accept `yes/no`, `true/false`, `on/off`, and `1/0`.
`--threads 0` selects all detected CPUs; positive values select an explicit
worker count.

Run the relevant command with `--help` to see its complete canonical option
list. The warning is authoritative for less common renamed options.

## Shared biological TSV columns

Newly written tables use these shared names:

| Legacy column | Canonical column | Canonical values |
| --- | --- | --- |
| `kingdom` | `organism_group` | `plant`, `non_plant`, or `unknown` |
| `partition` | `fold_id` | Dataset-defined fold identifier |
| localization-specific variants | `localization` | Dataset-defined localization class |
| numeric peroxisome flags | `peroxisome` | `yes` or `no` |

Legacy `kingdom` and `partition` inputs remain accepted by compatibility-aware
DeepLoc and peroxisome dataset loaders. Producers write only the canonical
names. New training tables should contain `accession`, `sequence`,
`organism_group`, `localization`, `peroxisome`, and `fold_id` as required by
the selected operation.

## Versioned TSV reports

Report schema version 2 is UTF-8, tab-delimited, rectangular, and LF-terminated.
Every row contains `schema_version=2`. Use `section` to interpret the remaining
cells; cells outside that section are empty. Lists and dictionaries in cells
are compact JSON rather than comma-joined text.

### `filter --report report.tsv`

The old report concatenated summary, ID, reason-count, and per-sequence tables
with incompatible headers. Schema 2 uses these sections:

- `summary`: `metric`, `value`
- `id_set`: `metric`, `ids`
- `drop_reason`: `drop_reason`, `count`
- `sequence`: `input_order`, `id`, `kept`, `drop_reasons`, `length_nt`,
  `tail_nt`, `non_triplet`, `internal_stop`, `total_codons`, `clean_codons`,
  `unclean_codons`, `missing_codons`, `ambiguous_codons`, `stop_codons`, and
  `clean_codon_fraction`

### `trimcodon --report report.tsv`

The old report concatenated a summary table and a codon-site table. Schema 2
uses:

- `summary`: `metric`, `value`
- `site`: `codon_site_1based`, `clean_fraction`, `clean_codons`,
  `unclean_codons`, `missing_codons`, `ambiguous_codons`, `stop_codons`, and
  `keep`

### `validate --report report.tsv`

The old report concatenated summary metrics and ID sets. Schema 2 uses:

- `summary`: `metric`, `value`
- `id_set`: `metric`, `ids`

### `maxalign --report report.tsv`

The old report contained summary key/value rows without an explicit schema.
Schema 2 uses:

- `summary`: `metric`, `value`
- `step`: `label`, `num_kept`, `complete_codon_sites`, `area`, `removed_id`,
  and `removed_ids`

## Consumer example

```python
import csv

with open('report.tsv', encoding='utf-8', newline='') as handle:
    rows = list(csv.DictReader(handle, delimiter='\t'))

if rows and rows[0]['schema_version'] != '2':
    raise ValueError('Unsupported cdskit report schema')

summary = {
    row['metric']: row['value']
    for row in rows
    if row['section'] == 'summary'
}
```

If an unchanged legacy TSV report is temporarily required, request the JSON
report instead and transform it in the downstream workflow. cdskit does not
emit the ambiguous concatenated-table TSV format after 0.23.

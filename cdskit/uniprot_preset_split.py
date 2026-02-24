import argparse
import csv
import json
import os
import re

from cdskit.localize_learn import (
    fetch_uniprot_training_rows,
    write_uniprot_rows_tsv,
)

LINEAGE_COL_CANDIDATES = (
    'lineage_ids',
    'Taxonomic lineage (Ids)',
    'Taxonomic lineage IDs',
    'taxonomic_lineage_ids',
)

TAXON_EUKARYOTA = '2759'
TAXON_VIRIDIPLANTAE = '33090'
TAXON_METAZOA = '33208'
TAXON_FUNGI = '4751'

UNIPROT_EUKARYOTA_FIELDS = (
    'accession',
    'sequence',
    'cc_subcellular_location',
    'organism_id',
    'lineage_ids',
)


def parse_yes_no(value, label):
    txt = str(value).strip().lower()
    if txt in ['yes', 'y', 'true', '1']:
        return True
    if txt in ['no', 'n', 'false', '0']:
        return False
    raise ValueError('{} should be yes or no.'.format(label))


def parse_taxon_ids(lineage_text):
    txt = str(lineage_text or '').strip()
    if txt == '':
        return set()
    return set(re.findall(r'\b\d+\b', txt))


def resolve_lineage_col(fieldnames, lineage_col=''):
    if fieldnames is None:
        fieldnames = []
    fieldnames = list(fieldnames)
    requested = str(lineage_col or '').strip()
    if requested != '':
        if requested not in fieldnames:
            txt = 'Lineage column "{}" was not found in input TSV.'
            raise ValueError(txt.format(requested))
        return requested
    for col_name in LINEAGE_COL_CANDIDATES:
        if col_name in fieldnames:
            return col_name
    txt = (
        'No lineage column was found. Expected one of: {}. '
        'Download eukaryota with lineage_ids included.'
    )
    raise ValueError(txt.format(', '.join(LINEAGE_COL_CANDIDATES)))


def classify_lineage_ids(taxon_ids):
    ids = set(taxon_ids)
    is_euk = TAXON_EUKARYOTA in ids
    is_viridiplantae = TAXON_VIRIDIPLANTAE in ids
    is_metazoa = TAXON_METAZOA in ids
    is_fungi = TAXON_FUNGI in ids
    return {
        'eukaryota': is_euk,
        'viridiplantae': is_viridiplantae,
        'metazoa': is_metazoa,
        'fungi': is_fungi,
        'non_viridiplantae_euk': is_euk and (not is_viridiplantae),
        'protist_core': is_euk and (not is_viridiplantae) and (not is_metazoa) and (not is_fungi),
    }


def split_rows_by_eukaryota_presets(rows, lineage_col):
    out = {
        'viridiplantae': [],
        'metazoa': [],
        'fungi': [],
        'non_viridiplantae_euk': [],
        'protist_core': [],
    }
    count_missing_lineage = 0
    count_non_eukaryota = 0
    for row in rows:
        taxon_ids = parse_taxon_ids(row.get(lineage_col, ''))
        if len(taxon_ids) == 0:
            count_missing_lineage += 1
            continue
        flags = classify_lineage_ids(taxon_ids=taxon_ids)
        if not flags['eukaryota']:
            count_non_eukaryota += 1
            continue
        for preset_name in out.keys():
            if flags[preset_name]:
                out[preset_name].append(row)
    return out, count_missing_lineage, count_non_eukaryota


def read_rows_tsv(path):
    with open(path, 'r', encoding='utf-8', newline='') as inp:
        reader = csv.DictReader(inp, delimiter='\t')
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_rows_tsv(rows, fieldnames, path):
    with open(path, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(
            out,
            fieldnames=fieldnames,
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def split_uniprot_eukaryota_tsv(input_tsv, out_dir, out_prefix='', lineage_col='', report_json=''):
    rows, fieldnames = read_rows_tsv(path=input_tsv)
    resolved_lineage_col = resolve_lineage_col(
        fieldnames=fieldnames,
        lineage_col=lineage_col,
    )
    split_rows, count_missing_lineage, count_non_eukaryota = split_rows_by_eukaryota_presets(
        rows=rows,
        lineage_col=resolved_lineage_col,
    )

    if out_prefix == '':
        basename = os.path.basename(input_tsv)
        out_prefix = os.path.splitext(basename)[0]
    os.makedirs(out_dir, exist_ok=True)

    output_paths = dict()
    counts_by_dataset = dict()
    for preset_name, subset_rows in split_rows.items():
        out_name = '{}_{}.tsv'.format(out_prefix, preset_name)
        out_path = os.path.join(out_dir, out_name)
        write_rows_tsv(
            rows=subset_rows,
            fieldnames=fieldnames,
            path=out_path,
        )
        output_paths[preset_name] = out_path
        counts_by_dataset[preset_name] = int(len(subset_rows))

    report = {
        'input_tsv': input_tsv,
        'out_dir': out_dir,
        'out_prefix': out_prefix,
        'lineage_col': resolved_lineage_col,
        'total_rows': int(len(rows)),
        'rows_missing_lineage': int(count_missing_lineage),
        'rows_non_eukaryota': int(count_non_eukaryota),
        'counts_by_dataset': counts_by_dataset,
        'output_paths': output_paths,
    }
    if report_json != '':
        with open(report_json, 'w', encoding='utf-8') as out:
            json.dump(report, out, indent=2)
    return report


def build_parser():
    parser = argparse.ArgumentParser(
        description='Download eukaryota once and split it into derived preset TSVs.',
    )
    parser.add_argument(
        '--input_tsv',
        required=True,
        type=str,
        help='Input eukaryota TSV path. If --download_eukaryota yes, this path is overwritten.',
    )
    parser.add_argument(
        '--out_dir',
        default='.',
        type=str,
        help='Output directory for derived preset TSVs.',
    )
    parser.add_argument(
        '--out_prefix',
        default='',
        type=str,
        help='Output file prefix. Default is input TSV basename without extension.',
    )
    parser.add_argument(
        '--lineage_col',
        default='',
        type=str,
        help='Optional lineage column name override.',
    )
    parser.add_argument(
        '--report_json',
        default='',
        type=str,
        help='Optional JSON report path.',
    )
    parser.add_argument(
        '--download_eukaryota',
        default='no',
        type=str,
        help='yes|no. If yes, download eukaryota to --input_tsv first.',
    )
    parser.add_argument(
        '--uniprot_reviewed',
        default='yes',
        type=str,
        help='yes|no. Download only reviewed entries.',
    )
    parser.add_argument(
        '--uniprot_exclude_fragments',
        default='yes',
        type=str,
        help='yes|no. Exclude entries annotated as fragment:true.',
    )
    parser.add_argument(
        '--uniprot_page_size',
        default=500,
        type=int,
        help='Rows per request page (max 500).',
    )
    parser.add_argument(
        '--uniprot_max_rows',
        default=0,
        type=int,
        help='Maximum downloaded rows. 0 means no hard limit.',
    )
    parser.add_argument(
        '--uniprot_sampling',
        default='head',
        type=str,
        choices=['head', 'random'],
        help='Sampling mode when --uniprot_max_rows > 0.',
    )
    parser.add_argument(
        '--uniprot_sampling_seed',
        default=1,
        type=int,
        help='Random seed for random sampling.',
    )
    parser.add_argument(
        '--uniprot_timeout_sec',
        default=60,
        type=int,
        help='Timeout seconds per HTTP request.',
    )
    parser.add_argument(
        '--uniprot_retries',
        default=2,
        type=int,
        help='Number of HTTP retries.',
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    download_euk = parse_yes_no(args.download_eukaryota, '--download_eukaryota')
    reviewed = parse_yes_no(args.uniprot_reviewed, '--uniprot_reviewed')
    exclude_fragments = parse_yes_no(args.uniprot_exclude_fragments, '--uniprot_exclude_fragments')

    if download_euk:
        rows = fetch_uniprot_training_rows(
            query='taxonomy_id:2759',
            fields=list(UNIPROT_EUKARYOTA_FIELDS),
            reviewed=reviewed,
            exclude_fragments=exclude_fragments,
            page_size=int(args.uniprot_page_size),
            max_rows=int(args.uniprot_max_rows),
            timeout_sec=int(args.uniprot_timeout_sec),
            retries=int(args.uniprot_retries),
            sampling_mode=str(args.uniprot_sampling),
            sampling_seed=int(args.uniprot_sampling_seed),
        )
        if len(rows) == 0:
            raise ValueError('No row was downloaded for eukaryota.')
        input_dir = os.path.dirname(args.input_tsv)
        if input_dir != '':
            os.makedirs(input_dir, exist_ok=True)
        write_uniprot_rows_tsv(
            rows=rows,
            fields=list(UNIPROT_EUKARYOTA_FIELDS),
            out_path=args.input_tsv,
        )

    report = split_uniprot_eukaryota_tsv(
        input_tsv=args.input_tsv,
        out_dir=args.out_dir,
        out_prefix=args.out_prefix,
        lineage_col=args.lineage_col,
        report_json=args.report_json,
    )
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

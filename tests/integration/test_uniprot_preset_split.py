import csv

from cdskit.uniprot_preset_split import (
    classify_lineage_ids,
    parse_taxon_ids,
    split_uniprot_eukaryota_tsv,
)


def test_parse_taxon_ids_and_classify_lineage_ids():
    ids = parse_taxon_ids('131567 (no rank), 2759 (domain), 33090 (clade), 4751 (kingdom)')
    assert '2759' in ids
    assert '33090' in ids
    assert '4751' in ids

    flags = classify_lineage_ids(ids)
    assert flags['eukaryota'] is True
    assert flags['viridiplantae'] is True
    assert flags['non_viridiplantae_euk'] is False
    assert flags['protist_core'] is False


def test_split_uniprot_eukaryota_tsv_outputs_expected_counts(temp_dir):
    input_tsv = temp_dir / 'eukaryota_with_lineage.tsv'
    out_dir = temp_dir / 'split'
    report_json = temp_dir / 'report.json'

    rows = [
        {
            'accession': 'VIR1',
            'sequence': 'MAAA',
            'cc_subcellular_location': 'Chloroplast',
            'lineage_ids': '2759,33090',
        },
        {
            'accession': 'MET1',
            'sequence': 'MBBB',
            'cc_subcellular_location': 'Membrane',
            'lineage_ids': '2759,33208',
        },
        {
            'accession': 'FUN1',
            'sequence': 'MCCC',
            'cc_subcellular_location': 'Cytoplasm',
            'lineage_ids': '2759,4751',
        },
        {
            'accession': 'PRO1',
            'sequence': 'MDDD',
            'cc_subcellular_location': 'Nucleus',
            'lineage_ids': '2759',
        },
        {
            'accession': 'BAC1',
            'sequence': 'MEEE',
            'cc_subcellular_location': 'Cytoplasm',
            'lineage_ids': '2',
        },
        {
            'accession': 'MISS1',
            'sequence': 'MFFF',
            'cc_subcellular_location': 'Unknown',
            'lineage_ids': '',
        },
    ]

    with open(input_tsv, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(
            out,
            fieldnames=['accession', 'sequence', 'cc_subcellular_location', 'lineage_ids'],
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    report = split_uniprot_eukaryota_tsv(
        input_tsv=str(input_tsv),
        out_dir=str(out_dir),
        out_prefix='demo',
        lineage_col='lineage_ids',
        report_json=str(report_json),
    )

    assert report['total_rows'] == 6
    assert report['rows_missing_lineage'] == 1
    assert report['rows_non_eukaryota'] == 1
    assert report['counts_by_dataset']['viridiplantae'] == 1
    assert report['counts_by_dataset']['metazoa'] == 1
    assert report['counts_by_dataset']['fungi'] == 1
    assert report['counts_by_dataset']['non_viridiplantae_euk'] == 3
    assert report['counts_by_dataset']['protist_core'] == 1

    for preset_name in [
        'viridiplantae',
        'metazoa',
        'fungi',
        'non_viridiplantae_euk',
        'protist_core',
    ]:
        out_path = report['output_paths'][preset_name]
        with open(out_path, 'r', encoding='utf-8') as inp:
            subset = list(csv.DictReader(inp, delimiter='\t'))
        assert len(subset) == report['counts_by_dataset'][preset_name]

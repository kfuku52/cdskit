import csv

import pytest

from cdskit.targetp_external_eval import (
    build_deeploc_hpa_broad_rows,
    build_deeploc_sorting_rows,
    build_uniprot_holdout_rows,
    compute_single_label_metrics,
    filter_rows_by_mmseqs_similarity,
    load_targetp_exclusion_keys,
    stratified_sample_rows,
)


def _write_tsv(path, fieldnames, rows):
    with open(path, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, delimiter='\t', fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_deeploc_sorting_rows_remove_targetp_exact_overlaps(temp_dir):
    targetp = temp_dir / 'targetp.tsv'
    sorting = temp_dir / 'sorting.tsv'
    _write_tsv(
        targetp,
        ['accession', 'sequence'],
        [{'accession': 'P1', 'sequence': 'MATS'}],
    )
    _write_tsv(
        sorting,
        ['source', 'accession', 'kingdom', 'sequence', 'sorting_signal_labels'],
        [
            {
                'source': 'deeploc',
                'accession': 'P1.1',
                'kingdom': 'Metazoa',
                'sequence': 'MATS',
                'sorting_signal_labels': 'SP',
            },
            {
                'source': 'deeploc',
                'accession': 'P2',
                'kingdom': 'Viridiplantae',
                'sequence': 'MAAA',
                'sorting_signal_labels': 'TH',
            },
            {
                'source': 'deeploc',
                'accession': 'P3',
                'kingdom': 'Metazoa',
                'sequence': 'MBBB',
                'sorting_signal_labels': 'GPI',
            },
        ],
    )

    rows, skipped = build_deeploc_sorting_rows(
        path=str(sorting),
        targetp_keys=load_targetp_exclusion_keys(str(targetp)),
    )

    assert [row['true_class'] for row in rows] == ['lTP']
    assert rows[0]['organism_group'] == 'plant'
    assert skipped['targetp_exact_overlap'] == 1
    assert skipped['no_targetp_equivalent_label'] == 1


def test_deeploc_hpa_broad_maps_mature_locations_to_targetp_proxy(temp_dir):
    targetp = temp_dir / 'targetp.tsv'
    hpa = temp_dir / 'hpa.tsv'
    _write_tsv(targetp, ['accession', 'sequence'], [])
    _write_tsv(
        hpa,
        ['source', 'accession', 'kingdom', 'sequence', 'localization_labels'],
        [
            {
                'source': 'hpa',
                'accession': 'H1',
                'kingdom': 'Metazoa',
                'sequence': 'MAAA',
                'localization_labels': 'mitochondrion;nucleus',
            },
            {
                'source': 'hpa',
                'accession': 'H2',
                'kingdom': 'Metazoa',
                'sequence': 'MBBB',
                'localization_labels': 'nucleus;cytoplasm',
            },
        ],
    )

    rows, skipped = build_deeploc_hpa_broad_rows(
        path=str(hpa),
        targetp_keys=load_targetp_exclusion_keys(str(targetp)),
    )

    assert [row['true_class'] for row in rows] == ['mTP', 'noTP']
    assert skipped == {}


def test_uniprot_holdout_uses_cdskit_cc_label_rules_and_skips_ambiguous(temp_dir):
    targetp = temp_dir / 'targetp.tsv'
    uniprot = temp_dir / 'uniprot.tsv'
    _write_tsv(
        targetp,
        ['accession', 'sequence'],
        [{'accession': 'P1', 'sequence': 'MATS'}],
    )
    _write_tsv(
        uniprot,
        ['accession', 'sequence', 'cc_subcellular_location', 'lineage_ids'],
        [
            {
                'accession': 'P1',
                'sequence': 'MATS',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Secreted.',
                'lineage_ids': '2759, 33208',
            },
            {
                'accession': 'P2',
                'sequence': 'MAAA',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Chloroplast.',
                'lineage_ids': '2759, 33090',
            },
            {
                'accession': 'P3',
                'sequence': 'MBBB',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Secreted. Mitochondrion.',
                'lineage_ids': '2759',
            },
        ],
    )

    rows, skipped = build_uniprot_holdout_rows(
        path=str(uniprot),
        targetp_keys=load_targetp_exclusion_keys(str(targetp)),
    )

    assert [row['true_class'] for row in rows] == ['cTP']
    assert rows[0]['organism_group'] == 'plant'
    assert skipped['targetp_exact_overlap'] == 1
    assert skipped['ambiguous_uniprot_cc'] == 1


def test_external_eval_metrics_and_stratified_sampling_are_deterministic():
    rows = [
        {'true_class': 'SP', 'predicted_class': 'SP'},
        {'true_class': 'SP', 'predicted_class': 'noTP'},
        {'true_class': 'noTP', 'predicted_class': 'noTP'},
    ]

    metrics = compute_single_label_metrics(rows)

    assert metrics['n_rows'] == 3
    assert metrics['accuracy'] == pytest.approx(2.0 / 3.0)
    assert metrics['by_class']['SP']['recall'] == pytest.approx(0.5)
    assert metrics['by_class']['noTP']['precision'] == pytest.approx(0.5)

    sample = stratified_sample_rows(
        rows=[
            {'true_class': 'SP', 'accession': 'S1'},
            {'true_class': 'SP', 'accession': 'S2'},
            {'true_class': 'noTP', 'accession': 'N1'},
            {'true_class': 'noTP', 'accession': 'N2'},
        ],
        max_per_class=1,
        seed=7,
    )

    assert [row['true_class'] for row in sample] == ['noTP', 'SP']
    assert len(sample) == 2


def test_mmseqs_similarity_filter_removes_hit_queries(temp_dir, monkeypatch):
    bin_dir = temp_dir / 'bin'
    bin_dir.mkdir()
    mmseqs = bin_dir / 'mmseqs'
    mmseqs.write_text(
        '#!/bin/sh\n'
        'printf "q0\\tt0\\t100\\t4\\t4\\t4\\t0\\t50\\n" > "$4"\n',
        encoding='utf-8',
    )
    mmseqs.chmod(0o755)
    monkeypatch.setenv('PATH', str(bin_dir))

    kept, report = filter_rows_by_mmseqs_similarity(
        rows=[
            {'accession': 'Q0', 'sequence': 'MATS'},
            {'accession': 'Q1', 'sequence': 'MAAA'},
        ],
        targetp_rows=[
            {'accession': 'T0', 'sequence': 'MATS'},
        ],
        min_seq_id=0.30,
        min_coverage=0.80,
        threads=2,
        enabled=True,
    )

    assert [row['accession'] for row in kept] == ['Q1']
    assert report['available'] is True
    assert report['removed'] == 1
    assert report['kept'] == 1
    assert report['status'] == 'ok'

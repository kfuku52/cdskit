import csv

import numpy as np
import pytest

from cdskit.targetp_external_aug import (
    build_external_augmented_training_rows,
    run_external_augmented_feature_oof,
    strict_uniprot_targetp_label,
)


pytest.importorskip('sklearn')


def _write_tsv(path, fieldnames, rows):
    with open(path, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, delimiter='\t', fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _targetp_rows():
    rows = list()
    seq_by_class = {
        'noTP': 'MGGGGGGGGGGGGGGGGGGG',
        'SP': 'MKKLLLLLLLLAAAAAGGGGG',
        'mTP': 'MARRRRAAASSSLLLGGGGG',
        'cTP': 'MASTSTSTSTSSRRRGGGGG',
        'lTP': 'MRRSTSTSTSTSSGGGGGGG',
    }
    for fold_id in ['fold1', 'fold2']:
        for class_name, seq in seq_by_class.items():
            rows.append({
                'accession': '{}_{}'.format(fold_id, class_name),
                'sequence': seq + fold_id[-1],
                'localization': class_name,
                'peroxisome': 'no',
                'organism_group': 'plant' if class_name in ['cTP', 'lTP'] else 'non_plant',
                'fold_id': fold_id,
            })
    return rows


def test_strict_uniprot_targetp_label_skips_lumen_noise():
    assert strict_uniprot_targetp_label(
        'SUBCELLULAR LOCATION: Endoplasmic reticulum lumen.',
        organism_group='non_plant',
    ) == ('noTP', '')
    assert strict_uniprot_targetp_label(
        'SUBCELLULAR LOCATION: Plastid, chloroplast thylakoid membrane.',
        organism_group='plant',
    ) == (None, 'thylakoid_not_lumen')
    assert strict_uniprot_targetp_label(
        'SUBCELLULAR LOCATION: Plastid, chloroplast thylakoid lumen.',
        organism_group='plant',
    ) == ('lTP', '')


def test_build_external_augmented_rows_filters_overlaps_and_conflicts(temp_dir):
    targetp = temp_dir / 'targetp.tsv'
    uniprot = temp_dir / 'uniprot.tsv'
    extra_uniprot = temp_dir / 'extra_uniprot.tsv'
    _write_tsv(
        targetp,
        ['accession', 'sequence', 'localization', 'peroxisome', 'organism_group', 'fold_id'],
        _targetp_rows(),
    )
    _write_tsv(
        uniprot,
        ['accession', 'sequence', 'cc_subcellular_location', 'lineage_ids'],
        [
            {
                'accession': 'fold1_noTP',
                'sequence': 'MOVERLAP',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Secreted.',
                'lineage_ids': '2759',
            },
            {
                'accession': 'U1',
                'sequence': 'MSTRICTCTP',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Chloroplast.',
                'lineage_ids': '2759,33090',
            },
            {
                'accession': 'U2',
                'sequence': 'MSTRICTLTP',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Chloroplast thylakoid lumen.',
                'lineage_ids': '2759,33090',
            },
            {
                'accession': 'U3',
                'sequence': 'MCONFLICT',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Secreted.',
                'lineage_ids': '2759',
            },
            {
                'accession': 'U4',
                'sequence': 'MCONFLICT',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Mitochondrion.',
                'lineage_ids': '2759',
            },
        ],
    )
    _write_tsv(
        extra_uniprot,
        ['accession', 'sequence', 'cc_subcellular_location', 'lineage_ids'],
        [
            {
                'accession': 'U5',
                'sequence': 'MEXTRALTP',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Plastid, chloroplast thylakoid lumen.',
                'lineage_ids': '2759,33090',
            },
        ],
    )

    rows, report = build_external_augmented_training_rows(
        targetp_tsv=str(targetp),
        uniprot_tsv=str(uniprot),
        extra_uniprot_tsvs=[str(extra_uniprot)],
        include_deeploc=False,
        max_per_class=10,
        seed=1,
    )

    assert sorted(row['localization'] for row in rows) == ['cTP', 'lTP', 'lTP']
    assert report['skipped']['uniprot_targetp_exact_overlap'] == 1
    assert 'extra_uniprot1_missing_location_text' not in report['skipped']
    assert report['skipped']['conflicting_duplicate_sequence'] == 1
    assert report['sampled_counts'] == {'cTP': 1, 'lTP': 2}


def test_external_augmented_feature_oof_is_foldwise(temp_dir):
    targetp = temp_dir / 'targetp.tsv'
    uniprot = temp_dir / 'uniprot.tsv'
    _write_tsv(
        targetp,
        ['accession', 'sequence', 'localization', 'peroxisome', 'organism_group', 'fold_id'],
        _targetp_rows(),
    )
    _write_tsv(
        uniprot,
        ['accession', 'sequence', 'cc_subcellular_location', 'lineage_ids'],
        [
            {
                'accession': 'U1',
                'sequence': 'MKKLLLLLLLLAAAAAX',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Secreted.',
                'lineage_ids': '2759',
            },
            {
                'accession': 'U2',
                'sequence': 'MARRRRAAASSSLLLQ',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Mitochondrion.',
                'lineage_ids': '2759',
            },
            {
                'accession': 'U3',
                'sequence': 'MASTSTSTSTSSRRRQ',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Chloroplast.',
                'lineage_ids': '2759,33090',
            },
            {
                'accession': 'U4',
                'sequence': 'MRRSTSTSTSTSSQ',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Chloroplast thylakoid lumen.',
                'lineage_ids': '2759,33090',
            },
        ],
    )

    result = run_external_augmented_feature_oof(
        training_tsv=str(targetp),
        uniprot_tsv=str(uniprot),
        include_deeploc=False,
        max_external_per_class=10,
        external_weight=0.25,
        n_estimators=5,
        random_state=3,
        threshold_grid=[0.5, 1.0, 2.0],
    )

    assert result['prob_matrix'].shape == (10, 5)
    np.testing.assert_allclose(result['prob_matrix'].sum(axis=1), np.ones((10,)))
    assert [fold['fold_id'] for fold in result['folds']] == ['fold1', 'fold2']
    assert all(fold['n_external_train'] == 4 for fold in result['folds'])
    assert result['foldwise_threshold']['metrics']['macro_f1'] >= 0.0

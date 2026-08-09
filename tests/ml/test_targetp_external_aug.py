import csv

import numpy as np
import pytest

from cdskit.targetp_external_aug import (
    build_external_augmented_training_rows,
    fit_external_augmented_feature_runtime_model,
    _parse_class_thresholds,
    _parse_external_class_weights,
    run_external_augmented_feature_oof,
    split_external_train_calibration_rows,
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
    assert report['mmseqs_similarity_filter']['status'] == 'disabled'


def test_build_external_augmented_rows_can_filter_mmseqs_similarity(temp_dir, monkeypatch):
    bin_dir = temp_dir / 'bin'
    bin_dir.mkdir()
    mmseqs = bin_dir / 'mmseqs'
    mmseqs.write_text(
        '#!/bin/sh\n'
        'printf "q0\\tt0\\t100\\t20\\t20\\t20\\t0\\t80\\n" > "$4"\n',
        encoding='utf-8',
    )
    mmseqs.chmod(0o755)
    monkeypatch.setenv('PATH', str(bin_dir))

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
        ],
    )

    rows, report = build_external_augmented_training_rows(
        targetp_tsv=str(targetp),
        uniprot_tsv=str(uniprot),
        include_deeploc=False,
        max_per_class=10,
        seed=1,
        use_mmseqs=True,
        threads=2,
    )

    assert [row['accession'] for row in rows] == ['U2']
    assert report['mmseqs_similarity_filter']['status'] == 'ok'
    assert report['mmseqs_similarity_filter']['removed'] == 1
    assert report['filtered_rows'] == 1


def test_build_external_augmented_rows_can_exclude_holdout_rows(temp_dir):
    targetp = temp_dir / 'targetp.tsv'
    uniprot = temp_dir / 'uniprot.tsv'
    holdout = temp_dir / 'holdout.tsv'
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
                'sequence': 'MSTRICTSP',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Secreted.',
                'lineage_ids': '2759',
            },
            {
                'accession': 'U2',
                'sequence': 'MSTRICTMTP',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Mitochondrion.',
                'lineage_ids': '2759',
            },
        ],
    )
    _write_tsv(
        holdout,
        ['accession', 'sequence'],
        [{'accession': 'U2', 'sequence': 'MOTHER'}],
    )

    rows, report = build_external_augmented_training_rows(
        targetp_tsv=str(targetp),
        uniprot_tsv=str(uniprot),
        exclusion_tsvs=[str(holdout)],
        include_deeploc=False,
        max_per_class=10,
        seed=1,
    )

    assert [row['accession'] for row in rows] == ['U1']
    assert report['skipped']['uniprot_external_exclusion_overlap'] == 1
    assert report['external_exclusion_accessions'] == 1


def test_build_external_augmented_rows_filters_holdout_similarity(temp_dir, monkeypatch):
    calls = list()

    def fake_filter_rows_by_mmseqs_similarity(
        rows,
        targetp_rows,
        min_seq_id=0.30,
        min_coverage=0.80,
        threads=1,
        enabled=True,
    ):
        calls.append([row.get('accession', '') for row in targetp_rows])
        remove_u2 = any(row.get('accession', '') == 'HOLDOUT' for row in targetp_rows)
        kept = [
            row for row in rows
            if not (remove_u2 and row.get('accession', '') == 'U2')
        ]
        return kept, {
            'requested': bool(enabled),
            'available': True,
            'min_seq_id': float(min_seq_id),
            'min_coverage': float(min_coverage),
            'removed': int(len(rows) - len(kept)),
            'kept': int(len(kept)),
            'status': 'ok',
        }

    monkeypatch.setattr(
        'cdskit.targetp_external_aug.filter_rows_by_mmseqs_similarity',
        fake_filter_rows_by_mmseqs_similarity,
    )
    targetp = temp_dir / 'targetp.tsv'
    uniprot = temp_dir / 'uniprot.tsv'
    holdout = temp_dir / 'holdout.tsv'
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
                'sequence': 'MSTRICTSP',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Secreted.',
                'lineage_ids': '2759',
            },
            {
                'accession': 'U2',
                'sequence': 'MSTRICTMTP',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Mitochondrion.',
                'lineage_ids': '2759',
            },
        ],
    )
    _write_tsv(
        holdout,
        ['accession', 'sequence'],
        [{'accession': 'HOLDOUT', 'sequence': 'MNEARHOLDOUT'}],
    )

    rows, report = build_external_augmented_training_rows(
        targetp_tsv=str(targetp),
        uniprot_tsv=str(uniprot),
        exclusion_tsvs=[str(holdout)],
        include_deeploc=False,
        max_per_class=10,
        seed=1,
        use_mmseqs=False,
        exclusion_mmseqs=True,
    )

    assert [row['accession'] for row in rows] == ['U1']
    assert len(calls) == 1
    assert calls[0] == ['HOLDOUT']
    assert report['external_exclusion_similarity_filter']['status'] == 'ok'
    assert report['external_exclusion_similarity_filter']['removed'] == 1


def test_split_external_train_calibration_rows_is_stratified():
    rows = [
        {'accession': '{}{}'.format(class_name, row_i), 'localization': class_name}
        for class_name in ['noTP', 'SP', 'mTP', 'cTP', 'lTP']
        for row_i in range(4)
    ]

    train_rows, calibration_rows, report = split_external_train_calibration_rows(
        rows=rows,
        calibration_fraction=0.25,
        seed=9,
    )

    assert len(train_rows) == 15
    assert len(calibration_rows) == 5
    assert report['enabled'] is True
    assert report['calibration_counts'] == {
        'noTP': 1,
        'SP': 1,
        'mTP': 1,
        'cTP': 1,
        'lTP': 1,
    }


def test_parse_class_thresholds_accepts_partial_overrides():
    assert _parse_class_thresholds('') is None

    thresholds = _parse_class_thresholds('noTP=0.6,SP=0.65,lTP=0.1')

    assert thresholds['noTP'] == pytest.approx(0.6)
    assert thresholds['SP'] == pytest.approx(0.65)
    assert thresholds['lTP'] == pytest.approx(0.1)
    assert thresholds['mTP'] == pytest.approx(1.0)
    assert thresholds['cTP'] == pytest.approx(1.0)


def test_parse_external_class_weights_accepts_partial_overrides():
    assert _parse_external_class_weights('') is None

    weights = _parse_external_class_weights('cTP=2.5,lTP=0.75')

    assert set(weights.keys()) == {'cTP', 'lTP'}
    assert weights['cTP'] == pytest.approx(2.5)
    assert weights['lTP'] == pytest.approx(0.75)


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


def test_fit_external_augmented_feature_runtime_model_records_external_training(temp_dir):
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
            {
                'accession': 'U5',
                'sequence': 'MGGGGGGGGGGGGGX',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Cytoplasm.',
                'lineage_ids': '2759',
            },
            {
                'accession': 'U6',
                'sequence': 'MKKLLLLLLLLAAAAB',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Secreted.',
                'lineage_ids': '2759',
            },
            {
                'accession': 'U7',
                'sequence': 'MARRRRAAASSSLLLB',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Mitochondrion.',
                'lineage_ids': '2759',
            },
            {
                'accession': 'U8',
                'sequence': 'MASTSTSTSTSSRRRB',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Chloroplast.',
                'lineage_ids': '2759,33090',
            },
            {
                'accession': 'U9',
                'sequence': 'MRRSTSTSTSTSSB',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Chloroplast thylakoid lumen.',
                'lineage_ids': '2759,33090',
            },
            {
                'accession': 'U10',
                'sequence': 'MGGGGGGGGGGGGGB',
                'cc_subcellular_location': 'SUBCELLULAR LOCATION: Cytoplasm.',
                'lineage_ids': '2759',
            },
        ],
    )

    model = fit_external_augmented_feature_runtime_model(
        training_tsv=str(targetp),
        uniprot_tsv=str(uniprot),
        include_deeploc=False,
        max_external_per_class=10,
        external_weight=0.25,
        calibration_fraction=0.5,
        n_estimators=5,
        random_state=3,
    )

    assert model['model_type'] == 'targetp_feature_ensemble_v1'
    assert model['metadata']['num_target_rows'] == 10
    assert model['metadata']['num_external_rows'] == 10
    assert model['metadata']['num_external_train_rows'] == 5
    assert model['metadata']['num_external_calibration_rows'] == 5
    assert model['metadata']['external_calibration']['threshold_tuning']['enabled'] is True
    assert model['localization_model']['classifier_profile']['external_augmented'] is True
    assert set(model['localization_model']['class_thresholds'].keys()) == {
        'noTP',
        'SP',
        'mTP',
        'cTP',
        'lTP',
    }

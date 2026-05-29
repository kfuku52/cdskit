import csv

import numpy as np
import pytest

from cdskit.localize_learn import LOCALIZATION_CLASSES
from cdskit.targetp_stack import (
    build_ltp_ctp_specialist_feature_matrix,
    evaluate_foldwise_classwise_blend_ltp_ctp_override,
    evaluate_foldwise_ltp_ctp_override,
    evaluate_foldwise_notp_ctp_ltp_override,
    run_targetp_stack_oof,
    stack_feature_matrix,
)


pytest.importorskip('sklearn')


def _write_targetp_fixture(path):
    seq_by_class = {
        'noTP': 'MGGGGGGGGGGGGGGGGGGG',
        'SP': 'MKKLLLLLLLLAAAAAGGGGG',
        'mTP': 'MARRRRAAASSSLLLGGGGG',
        'cTP': 'MASTSTSTSTSSRRRGGGGG',
        'lTP': 'MRRSTSTSTSTSSGGGGGGG',
    }
    rows = list()
    for fold_id in ['fold1', 'fold2']:
        for class_name in LOCALIZATION_CLASSES:
            rows.append({
                'accession': '{}_{}'.format(fold_id, class_name),
                'sequence': seq_by_class[class_name],
                'localization': class_name,
                'peroxisome': 'no',
                'organism_group': 'plant' if class_name in ['cTP', 'lTP'] else 'non_plant',
                'fold_id': fold_id,
            })
    with open(path, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                'accession',
                'sequence',
                'localization',
                'peroxisome',
                'organism_group',
                'fold_id',
            ],
            delimiter='\t',
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def _write_base_oof_npz(path, true_idx, confidence=0.8):
    n_classes = len(LOCALIZATION_CLASSES)
    prob = np.full((len(true_idx), n_classes), (1.0 - confidence) / (n_classes - 1), dtype=np.float64)
    for row_i, class_i in enumerate(true_idx):
        prob[row_i, int(class_i)] = confidence
    np.savez_compressed(
        str(path),
        prob_matrix=prob,
        true_idx=np.asarray(true_idx, dtype=np.int64),
        class_names=np.asarray(list(LOCALIZATION_CLASSES)),
    )
    return prob


def test_stack_feature_matrix_combines_base_probabilities_and_sequence_features(temp_dir):
    training_tsv = temp_dir / 'targetp.tsv'
    rows = _write_targetp_fixture(training_tsv)
    true_idx = np.asarray([i for _ in ['fold1', 'fold2'] for i in range(len(LOCALIZATION_CLASSES))])
    base_prob = _write_base_oof_npz(temp_dir / 'base.npz', true_idx)

    with_sequence = stack_feature_matrix(
        rows=rows,
        base_prob_matrices=[base_prob],
        include_sequence_features=True,
    )
    without_sequence = stack_feature_matrix(
        rows=rows,
        base_prob_matrices=[base_prob],
        include_sequence_features=False,
    )

    assert with_sequence.shape[0] == len(rows)
    assert without_sequence.shape == (len(rows), len(LOCALIZATION_CLASSES) + 3)
    assert with_sequence.shape[1] > without_sequence.shape[1]


def test_ltp_ctp_specialist_features_extend_base_features(temp_dir):
    training_tsv = temp_dir / 'targetp.tsv'
    rows = _write_targetp_fixture(training_tsv)

    specialist_features = build_ltp_ctp_specialist_feature_matrix(rows=rows)

    assert specialist_features.shape[0] == len(rows)
    assert specialist_features.shape[1] > 0
    assert np.all(np.isfinite(specialist_features))


def test_targetp_stack_oof_is_foldwise_and_normalized(temp_dir):
    training_tsv = temp_dir / 'targetp.tsv'
    _write_targetp_fixture(training_tsv)
    true_idx = np.asarray([i for _ in ['fold1', 'fold2'] for i in range(len(LOCALIZATION_CLASSES))])
    base_path = temp_dir / 'base.npz'
    _write_base_oof_npz(base_path, true_idx)

    oof = run_targetp_stack_oof(
        training_tsv=str(training_tsv),
        base_oof_npzs=[str(base_path)],
        n_estimators=5,
        random_state=3,
        include_sequence_features=False,
    )

    assert oof['prob_matrix'].shape == (10, len(LOCALIZATION_CLASSES))
    np.testing.assert_allclose(oof['prob_matrix'].sum(axis=1), np.ones((10,)))
    np.testing.assert_allclose(oof['true_idx'], true_idx)
    assert [fold['fold_id'] for fold in oof['folds']] == ['fold1', 'fold2']
    assert oof['feature_dim'] == len(LOCALIZATION_CLASSES) + 3


def test_targetp_stack_can_train_organism_specialized_models(temp_dir):
    training_tsv = temp_dir / 'targetp.tsv'
    _write_targetp_fixture(training_tsv)
    true_idx = np.asarray([i for _ in ['fold1', 'fold2'] for i in range(len(LOCALIZATION_CLASSES))])
    base_path = temp_dir / 'base.npz'
    _write_base_oof_npz(base_path, true_idx)

    oof = run_targetp_stack_oof(
        training_tsv=str(training_tsv),
        base_oof_npzs=[str(base_path)],
        n_estimators=5,
        random_state=3,
        include_sequence_features=False,
        organism_specialized_stack=True,
    )

    assert oof['prob_matrix'].shape == (10, len(LOCALIZATION_CLASSES))
    np.testing.assert_allclose(oof['prob_matrix'].sum(axis=1), np.ones((10,)))
    assert oof['profile']['organism_specialized_stack'] is True
    assert [fold['fold_id'] for fold in oof['folds']] == ['fold1', 'fold2']
    assert all(len(fold['organism_groups']) == 2 for fold in oof['folds'])
    assert all(
        group['used_global_fallback'] is False
        for fold in oof['folds']
        for group in fold['organism_groups']
    )


def test_ltp_ctp_override_is_foldwise(temp_dir):
    training_tsv = temp_dir / 'targetp.tsv'
    rows = _write_targetp_fixture(training_tsv)
    true_idx = np.asarray([i for _ in ['fold1', 'fold2'] for i in range(len(LOCALIZATION_CLASSES))])
    base_prob = _write_base_oof_npz(temp_dir / 'base.npz', true_idx)
    fold_ids = np.asarray([row['fold_id'] for row in rows])

    result = evaluate_foldwise_ltp_ctp_override(
        prob_matrix=base_prob,
        true_idx=true_idx,
        fold_ids=fold_ids,
        rows=rows,
        class_names=list(LOCALIZATION_CLASSES),
        threshold_grid=[0.5, 1.0, 2.0],
        score_grid=[0.1, 0.5, 0.9],
        n_estimators=5,
        random_state=3,
    )

    assert result['metrics']['macro_f1'] >= 0.0
    assert [fold['fold_id'] for fold in result['folds']] == ['fold1', 'fold2']
    assert all(fold['n_specialist_train'] == 2 for fold in result['folds'])


def test_notp_ctp_ltp_override_is_foldwise(temp_dir):
    training_tsv = temp_dir / 'targetp.tsv'
    rows = _write_targetp_fixture(training_tsv)
    true_idx = np.asarray([i for _ in ['fold1', 'fold2'] for i in range(len(LOCALIZATION_CLASSES))])
    base_prob = _write_base_oof_npz(temp_dir / 'base.npz', true_idx)
    fold_ids = np.asarray([row['fold_id'] for row in rows])

    result = evaluate_foldwise_notp_ctp_ltp_override(
        prob_matrix=base_prob,
        true_idx=true_idx,
        fold_ids=fold_ids,
        rows=rows,
        class_names=list(LOCALIZATION_CLASSES),
        threshold_grid=[0.5, 1.0, 2.0],
        score_grid=[0.1, 0.5, 0.9],
        notp_ctp_n_estimators=5,
        ltp_ctp_n_estimators=5,
        notp_ctp_random_state=3,
        ltp_ctp_random_state=4,
        notp_ctp_min_samples_leaf=2,
        ltp_ctp_class_weight='balanced_subsample',
        ltp_ctp_min_samples_leaf=2,
    )

    assert result['metrics']['macro_f1'] >= 0.0
    assert [fold['fold_id'] for fold in result['folds']] == ['fold1', 'fold2']
    assert all(fold['n_ltp_ctp_specialist_train'] == 2 for fold in result['folds'])
    assert result['profile']['notp_ctp_min_samples_leaf'] == 2
    assert result['profile']['ltp_ctp_class_weight'] == 'balanced_subsample'
    assert result['profile']['ltp_ctp_min_samples_leaf'] == 2


def test_foldwise_classwise_blend_ltp_ctp_override_is_foldwise(temp_dir):
    training_tsv = temp_dir / 'targetp.tsv'
    rows = _write_targetp_fixture(training_tsv)
    true_idx = np.asarray([i for _ in ['fold1', 'fold2'] for i in range(len(LOCALIZATION_CLASSES))])
    base_prob = _write_base_oof_npz(temp_dir / 'base.npz', true_idx, confidence=0.70)
    blend_prob = _write_base_oof_npz(temp_dir / 'blend.npz', true_idx, confidence=0.85)
    fold_ids = np.asarray([row['fold_id'] for row in rows])

    result = evaluate_foldwise_classwise_blend_ltp_ctp_override(
        prob_a=base_prob,
        prob_b=blend_prob,
        true_idx=true_idx,
        fold_ids=fold_ids,
        rows=rows,
        class_names=list(LOCALIZATION_CLASSES),
        alpha_grid=[0.0, 0.5, 1.0],
        threshold_grid=[0.5, 1.0, 2.0],
        score_grid=[0.1, 0.5, 0.9],
        n_estimators=5,
        random_state=3,
        ltp_ctp_min_samples_leaf=2,
    )

    assert result['metrics']['macro_f1'] >= 0.0
    assert [fold['fold_id'] for fold in result['folds']] == ['fold1', 'fold2']
    assert all(fold['n_ltp_ctp_specialist_train'] == 2 for fold in result['folds'])
    assert 'alpha_by_class' in result['folds'][0]
    assert result['profile']['min_samples_leaf'] == 2

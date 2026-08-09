import csv

import numpy as np
import pytest

from cdskit.localize_learn import LOCALIZATION_CLASSES
from cdskit.targetp_stack import (
    TARGETP_STACK_MTP_SPECIALIST_DEFAULTS,
    build_ltp_ctp_specialist_feature_matrix,
    build_parser,
    evaluate_foldwise_classwise_multi_blend,
    evaluate_foldwise_classwise_multi_blend_ltp_ctp_override,
    evaluate_foldwise_classwise_multi_blend_sp_override,
    evaluate_foldwise_classwise_blend_ltp_ctp_override,
    evaluate_foldwise_ltp_ctp_override,
    evaluate_foldwise_notp_ctp_ltp_override,
    run_targetp_stack_oof,
    stack_feature_matrix,
)


pytest.importorskip('sklearn')


def test_targetp_stack_parser_exposes_mtp_threshold_grid_defaults():
    parser = build_parser()
    args = parser.parse_args(['--base_oof_npzs', 'base.npz'])

    assert args.post_blend_sp_max_iter == 350
    assert args.post_blend_sp_random_states == '2,13,31'
    assert args.post_blend_mtp_model_kind == 'extra_trees'
    assert args.post_blend_mtp_n_estimators == 300
    assert args.post_blend_mtp_score_min == pytest.approx(
        TARGETP_STACK_MTP_SPECIALIST_DEFAULTS['mtp_score_min']
    )
    assert args.post_blend_mtp_score_max == pytest.approx(
        TARGETP_STACK_MTP_SPECIALIST_DEFAULTS['mtp_score_max']
    )
    assert args.post_blend_mtp_score_steps == TARGETP_STACK_MTP_SPECIALIST_DEFAULTS['mtp_score_steps']


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
    assert result['profile']['ltp_source_classes'] == ['cTP']


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


def test_foldwise_classwise_multi_blend_is_foldwise(temp_dir):
    training_tsv = temp_dir / 'targetp.tsv'
    rows = _write_targetp_fixture(training_tsv)
    true_idx = np.asarray([i for _ in ['fold1', 'fold2'] for i in range(len(LOCALIZATION_CLASSES))])
    prob_a = _write_base_oof_npz(temp_dir / 'a.npz', true_idx, confidence=0.65)
    prob_b = _write_base_oof_npz(temp_dir / 'b.npz', true_idx, confidence=0.75)
    prob_c = _write_base_oof_npz(temp_dir / 'c.npz', true_idx, confidence=0.85)
    fold_ids = np.asarray([row['fold_id'] for row in rows])

    result = evaluate_foldwise_classwise_multi_blend(
        prob_matrices=[prob_a, prob_b, prob_c],
        true_idx=true_idx,
        fold_ids=fold_ids,
        class_names=list(LOCALIZATION_CLASSES),
        source_labels=['a', 'b', 'c'],
        weight_grid=[
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
        ],
        threshold_grid=[0.5, 1.0, 2.0],
    )

    assert result['metrics']['macro_f1'] >= 0.0
    assert [fold['fold_id'] for fold in result['folds']] == ['fold1', 'fold2']
    assert result['profile']['n_sources'] == 3
    weights = result['folds'][0]['weights_by_class']['lTP']
    assert sorted(weights.keys()) == ['a', 'b', 'c']
    assert sum(weights.values()) == pytest.approx(1.0)


def test_foldwise_classwise_multi_blend_ltp_override_is_foldwise(temp_dir):
    training_tsv = temp_dir / 'targetp.tsv'
    rows = _write_targetp_fixture(training_tsv)
    true_idx = np.asarray([i for _ in ['fold1', 'fold2'] for i in range(len(LOCALIZATION_CLASSES))])
    prob_a = _write_base_oof_npz(temp_dir / 'a.npz', true_idx, confidence=0.65)
    prob_b = _write_base_oof_npz(temp_dir / 'b.npz', true_idx, confidence=0.75)
    prob_c = _write_base_oof_npz(temp_dir / 'c.npz', true_idx, confidence=0.85)
    fold_ids = np.asarray([row['fold_id'] for row in rows])
    weight_grid = [
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
    ]
    fixed_blend = evaluate_foldwise_classwise_multi_blend(
        prob_matrices=[prob_a, prob_b, prob_c],
        true_idx=true_idx,
        fold_ids=fold_ids,
        class_names=list(LOCALIZATION_CLASSES),
        source_labels=['a', 'b', 'c'],
        weight_grid=weight_grid,
        threshold_grid=[0.5, 1.0, 2.0],
    )

    result = evaluate_foldwise_classwise_multi_blend_ltp_ctp_override(
        prob_matrices=[prob_a, prob_b, prob_c],
        true_idx=true_idx,
        fold_ids=fold_ids,
        rows=rows,
        class_names=list(LOCALIZATION_CLASSES),
        source_labels=['a', 'b', 'c'],
        weight_grid=weight_grid,
        threshold_grid=[0.5, 1.0, 2.0],
        score_grid=[0.1, 0.5, 0.9],
        n_estimators=5,
        random_state=3,
        ltp_ctp_min_samples_leaf=2,
        ltp_source_classes=['cTP', 'noTP'],
        fixed_fold_rows=fixed_blend['folds'],
    )

    assert result['metrics']['macro_f1'] >= 0.0
    assert [fold['fold_id'] for fold in result['folds']] == ['fold1', 'fold2']
    assert all(fold['n_ltp_ctp_specialist_train'] == 2 for fold in result['folds'])
    assert result['profile']['n_sources'] == 3
    assert result['profile']['ltp_source_classes'] == ['cTP', 'noTP']
    assert result['profile']['fixed_fold_rows'] is True
    weights = result['folds'][0]['weights_by_class']['lTP']
    assert sorted(weights.keys()) == ['a', 'b', 'c']


def test_foldwise_classwise_multi_blend_sp_override_is_foldwise(temp_dir):
    training_tsv = temp_dir / 'targetp.tsv'
    rows = _write_targetp_fixture(training_tsv)
    true_idx = np.asarray([i for _ in ['fold1', 'fold2'] for i in range(len(LOCALIZATION_CLASSES))])
    prob_a = _write_base_oof_npz(temp_dir / 'a.npz', true_idx, confidence=0.65)
    prob_b = _write_base_oof_npz(temp_dir / 'b.npz', true_idx, confidence=0.75)
    prob_c = _write_base_oof_npz(temp_dir / 'c.npz', true_idx, confidence=0.85)
    fold_ids = np.asarray([row['fold_id'] for row in rows])
    weight_grid = [
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
    ]
    fixed_blend = evaluate_foldwise_classwise_multi_blend(
        prob_matrices=[prob_a, prob_b, prob_c],
        true_idx=true_idx,
        fold_ids=fold_ids,
        class_names=list(LOCALIZATION_CLASSES),
        source_labels=['a', 'b', 'c'],
        weight_grid=weight_grid,
        threshold_grid=[0.5, 1.0, 2.0],
    )

    result = evaluate_foldwise_classwise_multi_blend_sp_override(
        prob_matrices=[prob_a, prob_b, prob_c],
        true_idx=true_idx,
        fold_ids=fold_ids,
        rows=rows,
        class_names=list(LOCALIZATION_CLASSES),
        source_labels=['a', 'b', 'c'],
        weight_grid=weight_grid,
        threshold_grid=[0.5, 1.0, 2.0],
        fixed_fold_rows=fixed_blend['folds'],
        sp_random_states=[3],
        sp_weights=[1.0],
        sp_max_iter=5,
    )

    assert result['metrics']['macro_f1'] >= 0.0
    assert [fold['fold_id'] for fold in result['folds']] == ['fold1', 'fold2']
    assert result['profile']['fixed_fold_rows'] is True
    assert result['profile']['sp_random_states'] == [3]
    assert 'sp_score_threshold' in result['folds'][0]


def test_foldwise_classwise_multi_blend_sp_mtp_override_is_foldwise(temp_dir):
    training_tsv = temp_dir / 'targetp.tsv'
    rows = _write_targetp_fixture(training_tsv)
    true_idx = np.asarray([i for _ in ['fold1', 'fold2'] for i in range(len(LOCALIZATION_CLASSES))])
    prob_a = _write_base_oof_npz(temp_dir / 'a.npz', true_idx, confidence=0.65)
    prob_b = _write_base_oof_npz(temp_dir / 'b.npz', true_idx, confidence=0.75)
    prob_c = _write_base_oof_npz(temp_dir / 'c.npz', true_idx, confidence=0.85)
    fold_ids = np.asarray([row['fold_id'] for row in rows])
    weight_grid = [
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
    ]
    fixed_blend = evaluate_foldwise_classwise_multi_blend(
        prob_matrices=[prob_a, prob_b, prob_c],
        true_idx=true_idx,
        fold_ids=fold_ids,
        class_names=list(LOCALIZATION_CLASSES),
        source_labels=['a', 'b', 'c'],
        weight_grid=weight_grid,
        threshold_grid=[0.5, 1.0, 2.0],
    )

    result = evaluate_foldwise_classwise_multi_blend_sp_override(
        prob_matrices=[prob_a, prob_b, prob_c],
        true_idx=true_idx,
        fold_ids=fold_ids,
        rows=rows,
        class_names=list(LOCALIZATION_CLASSES),
        source_labels=['a', 'b', 'c'],
        weight_grid=weight_grid,
        threshold_grid=[0.5, 1.0, 2.0],
        fixed_fold_rows=fixed_blend['folds'],
        sp_random_states=[3],
        sp_weights=[1.0],
        sp_max_iter=5,
        mtp_override=True,
        mtp_n_estimators=5,
        mtp_random_state=7,
        mtp_threshold_grid=[0.2, 0.3],
    )

    assert result['metrics']['macro_f1'] >= 0.0
    assert [fold['fold_id'] for fold in result['folds']] == ['fold1', 'fold2']
    assert result['profile']['mtp_override'] is True
    assert result['profile']['mtp_feature_profile'] == 'targetp_sp_signal_plus_sources_v1'
    assert result['profile']['mtp_n_estimators'] == 5
    assert 'mtp_score_threshold' in result['folds'][0]


def test_foldwise_classwise_multi_blend_sp_mtp_ltp_after_override_is_foldwise(temp_dir):
    training_tsv = temp_dir / 'targetp.tsv'
    rows = _write_targetp_fixture(training_tsv)
    true_idx = np.asarray([i for _ in ['fold1', 'fold2'] for i in range(len(LOCALIZATION_CLASSES))])
    prob_a = _write_base_oof_npz(temp_dir / 'a.npz', true_idx, confidence=0.65)
    prob_b = _write_base_oof_npz(temp_dir / 'b.npz', true_idx, confidence=0.75)
    prob_c = _write_base_oof_npz(temp_dir / 'c.npz', true_idx, confidence=0.85)
    fold_ids = np.asarray([row['fold_id'] for row in rows])
    weight_grid = [
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
    ]
    fixed_blend = evaluate_foldwise_classwise_multi_blend(
        prob_matrices=[prob_a, prob_b, prob_c],
        true_idx=true_idx,
        fold_ids=fold_ids,
        class_names=list(LOCALIZATION_CLASSES),
        source_labels=['a', 'b', 'c'],
        weight_grid=weight_grid,
        threshold_grid=[0.5, 1.0, 2.0],
    )

    result = evaluate_foldwise_classwise_multi_blend_sp_override(
        prob_matrices=[prob_a, prob_b, prob_c],
        true_idx=true_idx,
        fold_ids=fold_ids,
        rows=rows,
        class_names=list(LOCALIZATION_CLASSES),
        source_labels=['a', 'b', 'c'],
        weight_grid=weight_grid,
        threshold_grid=[0.5, 1.0, 2.0],
        fixed_fold_rows=fixed_blend['folds'],
        sp_random_states=[3],
        sp_weights=[1.0],
        sp_max_iter=5,
        mtp_override=True,
        mtp_n_estimators=5,
        mtp_random_state=7,
        mtp_threshold_grid=[0.2, 0.3],
        ltp_after_override=True,
        ltp_after_n_estimators=5,
        ltp_after_random_state=11,
        ltp_after_threshold_grid=[0.2, 0.3],
        ltp_after_source_classes=['cTP'],
        ltp_after_negative_classes=['cTP'],
    )

    assert result['metrics']['macro_f1'] >= 0.0
    assert [fold['fold_id'] for fold in result['folds']] == ['fold1', 'fold2']
    assert result['profile']['ltp_after_override'] is True
    assert result['profile']['ltp_after_source_classes'] == ['cTP']
    assert all(fold['n_ltp_after_specialist_train'] == 2 for fold in result['folds'])
    assert 'ltp_after_score_threshold' in result['folds'][0]

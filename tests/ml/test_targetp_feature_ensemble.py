import csv

import numpy as np
import pytest

from cdskit.localize_model import (
    FEATURE_NAMES,
    extract_targetp_feature_ensemble_features,
    predict_localization_and_peroxisome,
)
from cdskit.localize_learn import LOCALIZATION_CLASSES
from cdskit.targetp_feature_ensemble import (
    build_targetp_feature_matrix,
    build_targetp_feature_blend_runtime_model,
    evaluate_foldwise_thresholds,
    fit_targetp_feature_runtime_model,
    run_targetp_feature_ensemble_oof,
)


pytest.importorskip('sklearn')


class _FixedClassifier:
    def __init__(self, classes, probabilities):
        self.classes_ = np.asarray(classes)
        self.probabilities = np.asarray(probabilities, dtype=np.float64)

    def predict_proba(self, features):
        return np.tile(self.probabilities.reshape((1, -1)), (features.shape[0], 1))


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


def test_targetp_feature_matrix_has_stable_row_count(temp_dir):
    training_tsv = temp_dir / 'targetp.tsv'
    rows = _write_targetp_fixture(training_tsv)

    features = build_targetp_feature_matrix(rows=rows)

    assert features.shape[0] == len(rows)
    assert features.shape[1] > 1000


def test_targetp_feature_ensemble_oof_and_foldwise_threshold_eval(temp_dir):
    training_tsv = temp_dir / 'targetp.tsv'
    _write_targetp_fixture(training_tsv)

    oof = run_targetp_feature_ensemble_oof(
        training_tsv=str(training_tsv),
        n_estimators=5,
        random_state=3,
    )
    report = evaluate_foldwise_thresholds(
        prob_matrix=oof['prob_matrix'],
        true_idx=oof['true_idx'],
        fold_ids=oof['fold_ids'],
        class_names=list(LOCALIZATION_CLASSES),
        threshold_grid=[0.5, 1.0],
    )

    assert oof['prob_matrix'].shape == (10, len(LOCALIZATION_CLASSES))
    np.testing.assert_allclose(oof['prob_matrix'].sum(axis=1), np.ones((10,)))
    assert report['metrics']['macro_f1'] >= 0.0
    assert len(report['folds']) == 2


def test_targetp_feature_binary_ovr_oof(temp_dir):
    training_tsv = temp_dir / 'targetp.tsv'
    _write_targetp_fixture(training_tsv)

    oof = run_targetp_feature_ensemble_oof(
        training_tsv=str(training_tsv),
        model_kind='binary_extra_trees',
        n_estimators=5,
        random_state=3,
    )

    assert oof['prob_matrix'].shape == (10, len(LOCALIZATION_CLASSES))
    np.testing.assert_allclose(oof['prob_matrix'].sum(axis=1), np.ones((10,)))
    assert oof['profile']['model_kind'] == 'binary_extra_trees'


def test_fit_targetp_feature_runtime_model_uses_localize_perox_features(temp_dir):
    training_tsv = temp_dir / 'targetp.tsv'
    _write_targetp_fixture(training_tsv)

    model = fit_targetp_feature_runtime_model(
        training_tsv=str(training_tsv),
        n_estimators=5,
        random_state=3,
    )

    assert model['model_type'] == 'targetp_feature_ensemble_v1'
    assert model['localization_model']['feature_dim'] > 1000
    assert model['perox_model']['mode'] in ['constant', 'centroid']


def test_fit_targetp_binary_feature_runtime_model(temp_dir):
    training_tsv = temp_dir / 'targetp.tsv'
    _write_targetp_fixture(training_tsv)

    model = fit_targetp_feature_runtime_model(
        training_tsv=str(training_tsv),
        model_kind='binary_extra_trees',
        n_estimators=5,
        random_state=3,
    )

    assert model['model_type'] == 'targetp_feature_ensemble_v1'
    assert model['localization_model']['classifier'] is None
    assert len(model['localization_model']['binary_classifiers']) == len(LOCALIZATION_CLASSES)


def test_targetp_binary_feature_runtime_predicts_with_binary_classifiers(temp_dir):
    training_tsv = temp_dir / 'targetp.tsv'
    _write_targetp_fixture(training_tsv)
    model = fit_targetp_feature_runtime_model(
        training_tsv=str(training_tsv),
        model_kind='binary_extra_trees',
        n_estimators=5,
        random_state=3,
    )

    result = predict_localization_and_peroxisome(
        aa_seq='MKKLLLLLLLLAAAAAGGGGG',
        model=model,
        organism_group='non_plant',
    )

    assert result['predicted_class'] in LOCALIZATION_CLASSES
    np.testing.assert_allclose(
        sum(result['class_probabilities'].values()),
        1.0,
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_targetp_feature_ltp_specialist_rescues_plant_ctp_prediction():
    seq = 'MRRSTSTSTSTSSGGGGGGG'
    feature_dim = int(extract_targetp_feature_ensemble_features(seq, 'plant').shape[0])
    model = {
        'model_type': 'targetp_feature_ensemble_v1',
        'feature_names': list(FEATURE_NAMES),
        'localization_model': {
            'mode': 'targetp_feature_ensemble',
            'class_order': list(LOCALIZATION_CLASSES),
            'classifier': _FixedClassifier(
                classes=[0, 1, 2, 3, 4],
                probabilities=[0.02, 0.03, 0.03, 0.86, 0.06],
            ),
            'binary_classifiers': None,
            'class_thresholds': {class_name: 1.0 for class_name in LOCALIZATION_CLASSES},
            'feature_dim': feature_dim,
            'targetp_feature_ltp_specialist': {
                'enabled': True,
                'models': [_FixedClassifier(classes=[0, 1], probabilities=[0.05, 0.95])],
                'weights': [1.0],
                'threshold': 0.5,
                'source_classes': ['cTP'],
                'mass_threshold': 0.0,
            },
        },
        'perox_model': {'mode': 'constant', 'yes_probability': 0.0},
    }

    result = predict_localization_and_peroxisome(
        aa_seq=seq,
        model=model,
        organism_group='plant',
    )

    assert result['predicted_class'] == 'lTP'


def test_build_targetp_feature_blend_runtime_model_extracts_source_base():
    feature_model = {
        'model_type': 'targetp_feature_ensemble_v1',
        'localization_model': {'class_order': list(LOCALIZATION_CLASSES)},
        'perox_model': {'mode': 'constant', 'yes_probability': 0.0},
    }
    source = {
        'model_type': 'targetp_blend_v1',
        'localization_model': {
            'base_models': [
                {'model_type': 'nearest_centroid_v1', 'localization_model': {'id': 'a'}},
                {'model_type': 'esm_head_v1', 'localization_model': {'id': 'b'}},
            ],
        },
    }

    model = build_targetp_feature_blend_runtime_model(
        feature_model=feature_model,
        blend_source_model=source,
        alpha_by_class={class_name: 0.5 for class_name in LOCALIZATION_CLASSES},
        class_thresholds={class_name: 1.0 for class_name in LOCALIZATION_CLASSES},
        blend_base_index=1,
    )

    assert model['model_type'] == 'targetp_blend_v1'
    assert model['localization_model']['base_models'][0]['model_type'] == 'targetp_feature_ensemble_v1'
    assert model['localization_model']['base_models'][1]['model_type'] == 'esm_head_v1'

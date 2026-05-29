import csv

import numpy as np
import pytest

from cdskit.localize_learn import LOCALIZATION_CLASSES
from cdskit.targetp_stack import (
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

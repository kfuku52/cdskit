import json
import sys

import numpy as np
import pytest

from cdskit.targetp_blend import (
    LOCALIZATION_CLASSES,
    _load_oof_npz,
    _metrics_from_prob_matrix,
    _oof_rows_to_prob_and_true,
    _optimize_classwise_alpha,
    _optimize_global_alpha,
    _save_oof_npz,
    main,
)


def test_oof_rows_to_prob_and_true_sorts_and_normalizes():
    class_names = ['noTP', 'SP']
    oof_rows = [
        {
            'index': '1',
            'true_class': 'SP',
            'class_probabilities': {'noTP': 2.0, 'SP': 2.0},
        },
        {
            'index': '0',
            'true_class': 'noTP',
            'class_probabilities': {'noTP': -1.0, 'SP': 0.0},
        },
    ]

    prob, true_idx = _oof_rows_to_prob_and_true(
        oof_rows=oof_rows,
        class_names=class_names,
    )

    assert true_idx.tolist() == [0, 1]
    np.testing.assert_allclose(prob[0, :], np.asarray([1.0, 0.0]))
    np.testing.assert_allclose(prob[1, :], np.asarray([0.5, 0.5]))


def test_oof_cache_save_and_load_roundtrip(temp_dir):
    class_names = list(LOCALIZATION_CLASSES)
    prob_matrix = np.asarray(
        [
            [0.70, 0.20, 0.10, 0.00, 0.00],
            [0.10, 0.70, 0.10, 0.10, 0.00],
            [0.10, 0.10, 0.70, 0.10, 0.00],
        ],
        dtype=np.float64,
    )
    true_idx = np.asarray([0, 1, 2], dtype=np.int64)
    npz_path = temp_dir / 'oof.npz'

    _save_oof_npz(
        path=str(npz_path),
        prob_matrix=prob_matrix,
        true_idx=true_idx,
        class_names=class_names,
    )
    loaded_prob, loaded_true, loaded_names = _load_oof_npz(path=str(npz_path))

    np.testing.assert_allclose(loaded_prob, prob_matrix)
    np.testing.assert_allclose(loaded_true, true_idx)
    assert loaded_names == class_names


def test_classwise_blend_beats_global_when_models_specialize():
    class_names = list(LOCALIZATION_CLASSES)
    true_idx = np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int64)
    n_rows = int(true_idx.shape[0])
    n_class = len(class_names)

    bilstm_prob = np.zeros((n_rows, n_class), dtype=np.float64)
    esm_prob = np.zeros((n_rows, n_class), dtype=np.float64)
    for i, true_class_idx in enumerate(true_idx.tolist()):
        if true_class_idx in [0, 1]:
            bilstm_prob[i, true_class_idx] = 1.0
            esm_prob[i, 2] = 1.0
        else:
            bilstm_prob[i, 0] = 1.0
            esm_prob[i, true_class_idx] = 1.0

    best_global_alpha, global_metrics = _optimize_global_alpha(
        prob_a=bilstm_prob,
        prob_b=esm_prob,
        true_idx=true_idx,
        class_names=class_names,
        grid=[0.0, 1.0],
    )
    alpha_by_class, classwise_metrics = _optimize_classwise_alpha(
        prob_a=bilstm_prob,
        prob_b=esm_prob,
        true_idx=true_idx,
        class_names=class_names,
        grid=[0.0, 1.0],
        init_alpha=best_global_alpha,
    )

    assert classwise_metrics['macro_f1'] > global_metrics['macro_f1']
    assert classwise_metrics['macro_f1'] == pytest.approx(0.7333333333333333)
    assert alpha_by_class.tolist() == pytest.approx([0.0, 1.0, 0.0, 0.0, 0.0])


def test_main_runs_with_cached_oof_only(temp_dir, monkeypatch):
    class_names = list(LOCALIZATION_CLASSES)
    true_idx = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
    bilstm_prob = np.asarray(
        [
            [0.95, 0.03, 0.01, 0.01, 0.00],
            [0.60, 0.35, 0.02, 0.02, 0.01],
            [0.30, 0.20, 0.30, 0.10, 0.10],
            [0.30, 0.20, 0.10, 0.30, 0.10],
            [0.30, 0.20, 0.10, 0.10, 0.30],
        ],
        dtype=np.float64,
    )
    esm_prob = np.asarray(
        [
            [0.70, 0.20, 0.05, 0.03, 0.02],
            [0.10, 0.80, 0.04, 0.03, 0.03],
            [0.10, 0.10, 0.75, 0.03, 0.02],
            [0.10, 0.10, 0.03, 0.72, 0.05],
            [0.10, 0.10, 0.03, 0.05, 0.72],
        ],
        dtype=np.float64,
    )
    bilstm_npz = temp_dir / 'bilstm_oof.npz'
    esm_npz = temp_dir / 'esm_oof.npz'
    out_json = temp_dir / 'blend_out.json'
    out_md = temp_dir / 'blend_out.md'
    dummy_tsv = temp_dir / 'dummy.tsv'

    _save_oof_npz(
        path=str(bilstm_npz),
        prob_matrix=bilstm_prob,
        true_idx=true_idx,
        class_names=class_names,
    )
    _save_oof_npz(
        path=str(esm_npz),
        prob_matrix=esm_prob,
        true_idx=true_idx,
        class_names=class_names,
    )
    dummy_tsv.write_text(
        'sequence\tlocalization\tperoxisome\tfold_id\n'
        'MAAA\tnoTP\tno\tfold1\n',
        encoding='utf-8',
    )

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'targetp_blend.py',
            '--training_tsv',
            str(dummy_tsv),
            '--reuse_oof_cache',
            'yes',
            '--bilstm_oof_npz',
            str(bilstm_npz),
            '--esm_oof_npz',
            str(esm_npz),
            '--blend_grid_step',
            '0.5',
            '--out_json',
            str(out_json),
            '--out_md',
            str(out_md),
        ],
    )
    main()

    with open(out_json, 'r', encoding='utf-8') as inp:
        result = json.load(inp)

    assert result['bilstm']['used_cache'] is True
    assert result['esm']['used_cache'] is True
    assert 'blend_global' in result
    assert 'blend_classwise' in result
    md = out_md.read_text(encoding='utf-8')
    assert '| Metric | TargetP | bilstm | esm | blend(global) | blend(classwise) |' in md


def test_metrics_from_prob_matrix_computes_expected_accuracy():
    class_names = ['noTP', 'SP']
    prob_matrix = np.asarray(
        [
            [0.80, 0.20],
            [0.40, 0.60],
            [0.60, 0.40],
            [0.10, 0.90],
        ],
        dtype=np.float64,
    )
    true_idx = np.asarray([0, 1, 1, 1], dtype=np.int64)
    metrics = _metrics_from_prob_matrix(
        prob_matrix=prob_matrix,
        true_idx=true_idx,
        class_names=class_names,
    )
    assert metrics['overall_accuracy'] == pytest.approx(0.75)
    assert metrics['macro_f1'] == pytest.approx(0.7333333333333334)

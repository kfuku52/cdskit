import csv
import json
import sys

import numpy as np
import pytest

import cdskit.targetp_blend as targetp_blend_module
from cdskit.targetp_blend import (
    LOCALIZATION_CLASSES,
    _aggregate_score_columns,
    _build_targetp_blend_runtime_model,
    _evaluate_foldwise_classwise_blend,
    _export_targetp_blend_runtime_model,
    _load_oof_npz,
    _apply_organism_gate,
    _apply_specialist_postprocess_predictions,
    _metrics_from_prob_matrix,
    _oof_rows_to_prob_and_true,
    _optimize_classwise_alpha,
    _optimize_global_alpha,
    _save_oof_npz,
    _targetp_margin_summary,
    main,
)
from cdskit.localize_model import predict_localization_and_peroxisome


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


def test_oof_cache_load_can_use_fallback_true_idx(temp_dir):
    class_names = list(LOCALIZATION_CLASSES)
    prob_matrix = np.asarray(
        [
            [0.70, 0.20, 0.10, 0.00, 0.00],
            [0.10, 0.70, 0.10, 0.10, 0.00],
        ],
        dtype=np.float64,
    )
    true_idx = np.asarray([0, 1], dtype=np.int64)
    npz_path = temp_dir / 'oof_without_true.npz'
    np.savez_compressed(
        str(npz_path),
        prob_matrix=prob_matrix,
        class_names=np.asarray(class_names),
    )

    loaded_prob, loaded_true, loaded_names = _load_oof_npz(
        path=str(npz_path),
        fallback_true_idx=true_idx,
    )

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


def test_foldwise_classwise_blend_uses_held_out_folds():
    class_names = list(LOCALIZATION_CLASSES)
    true_idx = np.asarray([0, 1, 0, 1], dtype=np.int64)
    fold_ids = np.asarray(['fold1', 'fold1', 'fold2', 'fold2'])
    prob_a = np.asarray(
        [
            [0.90, 0.10, 0.00, 0.00, 0.00],
            [0.80, 0.20, 0.00, 0.00, 0.00],
            [0.90, 0.10, 0.00, 0.00, 0.00],
            [0.80, 0.20, 0.00, 0.00, 0.00],
        ],
        dtype=np.float64,
    )
    prob_b = np.asarray(
        [
            [0.20, 0.80, 0.00, 0.00, 0.00],
            [0.10, 0.90, 0.00, 0.00, 0.00],
            [0.20, 0.80, 0.00, 0.00, 0.00],
            [0.10, 0.90, 0.00, 0.00, 0.00],
        ],
        dtype=np.float64,
    )

    metrics, fold_rows = _evaluate_foldwise_classwise_blend(
        prob_a=prob_a,
        prob_b=prob_b,
        true_idx=true_idx,
        fold_ids=fold_ids,
        class_names=class_names,
        alpha_grid=[0.0, 1.0],
        threshold_grid=[1.0],
    )

    assert len(fold_rows) == 2
    assert metrics['macro_f1'] > 0.0
    assert {row['fold_id'] for row in fold_rows} == {'fold1', 'fold2'}


def test_apply_organism_gate_removes_non_plant_ctp_ltp_mass():
    class_names = list(LOCALIZATION_CLASSES)
    prob = np.asarray(
        [
            [0.10, 0.10, 0.10, 0.35, 0.35],
            [0.10, 0.10, 0.10, 0.35, 0.35],
        ],
        dtype=np.float64,
    )

    gated = _apply_organism_gate(
        prob_matrix=prob,
        plant_mask=np.asarray([True, False], dtype=bool),
        class_names=class_names,
    )

    np.testing.assert_allclose(gated[0, :], prob[0, :])
    assert gated[1, class_names.index('cTP')] == pytest.approx(0.0)
    assert gated[1, class_names.index('lTP')] == pytest.approx(0.0)
    assert float(gated[1, :].sum()) == pytest.approx(1.0)


def test_aggregate_score_columns_supports_mean_and_weights():
    scores = [
        np.asarray([0.0, 0.4, 1.0], dtype=np.float64),
        np.asarray([0.2, 0.6, 0.8], dtype=np.float64),
        np.asarray([0.4, 0.8, 0.6], dtype=np.float64),
    ]

    np.testing.assert_allclose(
        _aggregate_score_columns(scores),
        np.asarray([0.2, 0.6, 0.8], dtype=np.float64),
    )
    np.testing.assert_allclose(
        _aggregate_score_columns(scores, weights=[0.2, 0.3, 0.5]),
        np.asarray([0.26, 0.66, 0.74], dtype=np.float64),
    )


def test_build_targetp_blend_runtime_model_predicts_with_wrapped_base_models():
    class_names = list(LOCALIZATION_CLASSES)
    model = _build_targetp_blend_runtime_model(
        base_model_a={
            'model_type': 'nearest_centroid_v1',
            'localization_model': {
                'mode': 'constant',
                'class_label': 'noTP',
                'class_order': class_names,
            },
        },
        base_model_b={
            'model_type': 'nearest_centroid_v1',
            'localization_model': {
                'mode': 'constant',
                'class_label': 'lTP',
                'class_order': class_names,
            },
        },
        perox_model={'mode': 'constant', 'yes_probability': 0.0},
        alpha_by_class={
            'noTP': 1.0,
            'SP': 1.0,
            'mTP': 1.0,
            'cTP': 1.0,
            'lTP': 0.0,
        },
        class_thresholds={
            'noTP': 1.0,
            'SP': 1.0,
            'mTP': 1.0,
            'cTP': 1.0,
            'lTP': 0.4,
        },
        metadata={'source': 'unit-test'},
    )

    pred = predict_localization_and_peroxisome(
        aa_seq='MARRVAAARRLLLLLVVVVVAAST',
        model=model,
        organism_group='plant',
    )

    assert model['model_type'] == 'targetp_blend_v1'
    assert model['metadata']['source'] == 'unit-test'
    assert pred['predicted_class'] == 'lTP'


def test_export_targetp_blend_runtime_model_uses_full_training_table(temp_dir, monkeypatch):
    class_names = list(LOCALIZATION_CLASSES)
    training_tsv = temp_dir / 'targetp_export.tsv'
    with open(training_tsv, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(
            out,
            fieldnames=['sequence', 'localization', 'peroxisome'],
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writeheader()
        for class_name, seq in [
            ('noTP', 'MGPVNQDEGPVNQDEGPVNQDE'),
            ('SP', 'MKKLLLLLLLLLLAVAVAASAASA'),
            ('mTP', 'MRRKRRAARAKRRNQAAARRRAA'),
            ('cTP', 'MSTSTSTTSTASSSAATSTASSTT'),
            ('lTP', 'MARRVAAARRLLLLLVVVVVAAST'),
        ]:
            writer.writerow({
                'sequence': seq,
                'localization': class_name,
                'peroxisome': 'no',
            })

    def fake_fit_localization_model(**kwargs):
        model_arch = kwargs['model_arch']
        return {
            'mode': 'constant',
            'class_label': 'noTP' if model_arch == 'bilstm_attention' else 'lTP',
            'class_order': class_names,
        }

    saved = {}

    def fake_save_localize_model(model, path):
        saved['model'] = model
        saved['path'] = path

    monkeypatch.setattr(
        targetp_blend_module,
        'fit_localization_model',
        fake_fit_localization_model,
    )
    monkeypatch.setattr(
        targetp_blend_module,
        'save_localize_model',
        fake_save_localize_model,
    )
    args = targetp_blend_module.build_parser().parse_args([
        '--training_tsv',
        str(training_tsv),
        '--model_out',
        str(temp_dir / 'targetp_blend.pt'),
        '--model_out_specialist_postprocess',
        'no',
    ])
    prob_a = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64), (5, 1))
    prob_b = np.tile(np.asarray([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64), (5, 1))
    true_idx = np.arange(5, dtype=np.int64)
    info = _export_targetp_blend_runtime_model(
        args=args,
        prob_a=prob_a,
        prob_b=prob_b,
        base_prob=(prob_a + prob_b) / 2.0,
        true_idx=true_idx,
        alpha_by_class=np.asarray([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64),
        class_thresholds=np.ones((5,), dtype=np.float64),
        benchmark_out={
            'targetp_macro_f1': 0.89,
            'blend_threshold': {'metrics': {'macro_f1': 0.90}},
        },
    )

    assert info['model_type'] == 'targetp_blend_v1'
    assert info['specialist_postprocess'] is False
    assert saved['path'] == str(temp_dir / 'targetp_blend.pt')
    assert saved['model']['model_type'] == 'targetp_blend_v1'
    assert saved['model']['metadata']['num_used_rows'] == 5
    assert [
        row['model_type']
        for row in saved['model']['localization_model']['base_models']
    ] == ['bilstm_attention_v1', 'esm_head_v1']


def test_targetp_margin_summary_flags_all_class_pass():
    class_names = ['noTP', 'SP']
    targetp_ref = {
        'noTP': {'f1': 0.80},
        'SP': {'f1': 0.70},
    }
    metrics = {
        'by_class': {
            'noTP': {'f1': 0.82},
            'SP': {'f1': 0.71},
        },
    }

    summary = _targetp_margin_summary(
        metrics=metrics,
        targetp_ref=targetp_ref,
        class_names=class_names,
    )

    assert summary['beats_targetp_all_classes'] is True
    assert summary['min_class_f1_margin'] == pytest.approx(0.01)
    assert summary['class_beats_targetp'] == {'noTP': True, 'SP': True}


def test_specialist_postprocess_applies_sp_gate_and_ltp_rerank():
    class_names = list(LOCALIZATION_CLASSES)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    prob = np.asarray(
        [
            [0.30, 0.60, 0.05, 0.03, 0.02],
            [0.30, 0.20, 0.05, 0.35, 0.10],
            [0.10, 0.10, 0.05, 0.25, 0.50],
        ],
        dtype=np.float64,
    )

    pred = _apply_specialist_postprocess_predictions(
        base_prob=prob,
        class_thresholds=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        sp_scores=np.asarray([0.20, 0.95, 0.10], dtype=np.float64),
        sp_threshold=0.90,
        ltp_scores=np.asarray([0.10, 0.80, 0.30], dtype=np.float64),
        ltp_threshold=0.50,
        plant_mask=np.asarray([False, True, True], dtype=bool),
        class_names=class_names,
        ltp_mass_threshold=0.20,
    )

    assert pred.tolist() == [
        class_to_idx['noTP'],
        class_to_idx['SP'],
        class_to_idx['cTP'],
    ]


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
    assert 'targetp_margin' in result['blend_threshold']
    md = out_md.read_text(encoding='utf-8')
    assert '| Metric | TargetP | bilstm | esm | blend(global) | blend(classwise) |' in md
    assert '| All classes > TargetP |' in md


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

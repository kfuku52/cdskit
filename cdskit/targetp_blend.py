import argparse
import csv
import json
import os

import numpy as np

from cdskit.localize_learn import (
    LOCALIZATION_CLASSES,
    build_training_matrix,
    evaluate_cross_validation,
)
from cdskit.targetp_benchmark import (
    TARGETP_TABLE1_REFERENCE,
    compute_prf_by_class,
)


def _read_training_rows(path):
    with open(path, 'r', encoding='utf-8', newline='') as inp:
        return list(csv.DictReader(inp, delimiter='\t'))


def _oof_rows_to_prob_and_true(oof_rows, class_names):
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    rows = sorted(oof_rows, key=lambda r: int(r['index']))
    n = len(rows)
    k = len(class_names)
    prob = np.zeros((n, k), dtype=np.float64)
    true_idx = np.zeros((n,), dtype=np.int64)
    for i, row in enumerate(rows):
        probs = row.get('class_probabilities', {})
        total = 0.0
        for j, class_name in enumerate(class_names):
            p = float(probs.get(class_name, 0.0))
            if p < 0.0:
                p = 0.0
            prob[i, j] = p
            total += p
        if total <= 0.0:
            prob[i, 0] = 1.0
            total = 1.0
        if total != 1.0:
            prob[i, :] = prob[i, :] / total
        true_name = str(row['true_class'])
        true_idx[i] = int(class_to_idx[true_name])
    return prob, true_idx


def _metrics_from_prob_matrix(prob_matrix, true_idx, class_names):
    pred_idx = np.argmax(prob_matrix, axis=1).astype(np.int64)
    return _metrics_from_prediction_indices(
        pred_idx=pred_idx,
        true_idx=true_idx,
        class_names=class_names,
    )


def _metrics_from_prediction_indices(pred_idx, true_idx, class_names):
    true_names = [class_names[int(i)] for i in true_idx.tolist()]
    pred_names = [class_names[int(i)] for i in pred_idx.tolist()]
    by_class = compute_prf_by_class(
        true_classes=true_names,
        pred_classes=pred_names,
        class_names=class_names,
    )
    overall = float(np.mean(pred_idx == true_idx))
    macro_f1 = float(np.mean(np.asarray(
        [by_class[name]['f1'] for name in class_names],
        dtype=np.float64,
    )))
    return {
        'overall_accuracy': overall,
        'macro_f1': macro_f1,
        'by_class': by_class,
    }


def _prediction_indices_with_thresholds(prob_matrix, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float64).reshape((1, -1))
    thresholds[thresholds <= 0.0] = 1.0
    scores = np.asarray(prob_matrix, dtype=np.float64) / thresholds
    return np.argmax(scores, axis=1).astype(np.int64)


def _blend_global(prob_a, prob_b, alpha):
    alpha = float(alpha)
    out = (alpha * prob_a) + ((1.0 - alpha) * prob_b)
    out = out / np.clip(out.sum(axis=1, keepdims=True), a_min=1.0e-12, a_max=None)
    return out


def _blend_classwise(prob_a, prob_b, alpha_by_class):
    alpha_vec = np.asarray(alpha_by_class, dtype=np.float64).reshape((1, -1))
    out = (alpha_vec * prob_a) + ((1.0 - alpha_vec) * prob_b)
    out = out / np.clip(out.sum(axis=1, keepdims=True), a_min=1.0e-12, a_max=None)
    return out


def _optimize_global_alpha(prob_a, prob_b, true_idx, class_names, grid):
    best_alpha = float(grid[0])
    best_metrics = None
    for alpha in grid:
        blend = _blend_global(prob_a=prob_a, prob_b=prob_b, alpha=alpha)
        metrics = _metrics_from_prob_matrix(
            prob_matrix=blend,
            true_idx=true_idx,
            class_names=class_names,
        )
        if (best_metrics is None) or (metrics['macro_f1'] > best_metrics['macro_f1']):
            best_metrics = metrics
            best_alpha = float(alpha)
    return best_alpha, best_metrics


def _optimize_classwise_alpha(prob_a, prob_b, true_idx, class_names, grid, init_alpha):
    n_class = len(class_names)
    alpha = np.full((n_class,), float(init_alpha), dtype=np.float64)
    best = _metrics_from_prob_matrix(
        prob_matrix=_blend_classwise(prob_a, prob_b, alpha),
        true_idx=true_idx,
        class_names=class_names,
    )
    improved = True
    while improved:
        improved = False
        for c in range(n_class):
            best_local_alpha = float(alpha[c])
            best_local_metrics = best
            for trial in grid:
                tmp = alpha.copy()
                tmp[c] = float(trial)
                metrics = _metrics_from_prob_matrix(
                    prob_matrix=_blend_classwise(prob_a, prob_b, tmp),
                    true_idx=true_idx,
                    class_names=class_names,
                )
                if metrics['macro_f1'] > best_local_metrics['macro_f1']:
                    best_local_alpha = float(trial)
                    best_local_metrics = metrics
            if best_local_alpha != float(alpha[c]):
                alpha[c] = best_local_alpha
                best = best_local_metrics
                improved = True
    return alpha, best


def _optimize_class_thresholds(prob_matrix, true_idx, class_names, grid):
    thresholds = np.ones((len(class_names),), dtype=np.float64)
    pred_idx = _prediction_indices_with_thresholds(
        prob_matrix=prob_matrix,
        thresholds=thresholds,
    )
    best_metrics = _metrics_from_prediction_indices(
        pred_idx=pred_idx,
        true_idx=true_idx,
        class_names=class_names,
    )
    improved = True
    while improved:
        improved = False
        for class_i in range(len(class_names)):
            best_local = float(thresholds[class_i])
            best_local_metrics = best_metrics
            for trial in grid:
                tmp = thresholds.copy()
                tmp[class_i] = float(trial)
                pred_idx = _prediction_indices_with_thresholds(
                    prob_matrix=prob_matrix,
                    thresholds=tmp,
                )
                metrics = _metrics_from_prediction_indices(
                    pred_idx=pred_idx,
                    true_idx=true_idx,
                    class_names=class_names,
                )
                if metrics['macro_f1'] > best_local_metrics['macro_f1']:
                    best_local = float(trial)
                    best_local_metrics = metrics
            if best_local != float(thresholds[class_i]):
                thresholds[class_i] = best_local
                best_metrics = best_local_metrics
                improved = True
    return thresholds, best_metrics


def _save_oof_npz(path, prob_matrix, true_idx, class_names):
    out_dir = os.path.dirname(path)
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        path,
        prob_matrix=np.asarray(prob_matrix, dtype=np.float64),
        true_idx=np.asarray(true_idx, dtype=np.int64),
        class_names=np.asarray(class_names),
    )


def _load_oof_npz(path, fallback_true_idx=None):
    data = np.load(path, allow_pickle=True)
    prob_matrix = np.asarray(data['prob_matrix'], dtype=np.float64)
    if 'true_idx' in data.files:
        true_idx = np.asarray(data['true_idx'], dtype=np.int64)
    elif fallback_true_idx is not None:
        true_idx = np.asarray(fallback_true_idx, dtype=np.int64)
    else:
        raise KeyError(
            'true_idx is not a file in the archive and no fallback_true_idx was provided.'
        )
    class_names = [str(v) for v in data['class_names'].tolist()]
    return prob_matrix, true_idx, class_names


def _read_true_idx_from_training_tsv(training_tsv, class_names):
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    rows = _read_training_rows(path=training_tsv)
    out = list()
    for row in rows:
        class_name = str(row.get('localization', '')).strip()
        if class_name not in class_to_idx:
            raise ValueError('Unsupported localization label in training_tsv: {}'.format(class_name))
        out.append(class_to_idx[class_name])
    return np.asarray(out, dtype=np.int64)


def _read_fold_ids_from_training_tsv(training_tsv):
    rows = _read_training_rows(path=training_tsv)
    fold_ids = list()
    for row_i, row in enumerate(rows):
        fold_id = str(row.get('fold_id', '')).strip()
        if fold_id == '':
            fold_id = str(row.get('targetp_fold', '')).strip()
        if fold_id == '':
            fold_id = 'row{}'.format(row_i)
        fold_ids.append(fold_id)
    return np.asarray(fold_ids)


def _read_organism_group_mask(training_tsv):
    rows = _read_training_rows(path=training_tsv)
    return np.asarray([
        str(row.get('organism_group', '')).strip().lower() == 'plant'
        for row in rows
    ], dtype=bool)


def _apply_organism_gate(prob_matrix, plant_mask, class_names):
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64).copy()
    plant_mask = np.asarray(plant_mask, dtype=bool)
    if prob_matrix.shape[0] != plant_mask.shape[0]:
        raise ValueError('organism_group row count does not match OOF probability rows.')
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    for class_name in ['cTP', 'lTP']:
        if class_name in class_to_idx:
            prob_matrix[~plant_mask, class_to_idx[class_name]] = 0.0
    row_sum = prob_matrix.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    return prob_matrix / row_sum


def _run_model_oof(
    training_tsv,
    model_arch,
    localize_strategy,
    dl_train_params,
    cv_seed,
):
    rows = _read_training_rows(path=training_tsv)
    x, aa_sequences, class_labels, perox_labels, skipped, fold_ids = build_training_matrix(
        rows=rows,
        seq_col='sequence',
        seqtype='protein',
        codontable=1,
        label_mode='explicit',
        localization_col='localization',
        perox_col='peroxisome',
        skip_ambiguous=True,
        cv_fold_col='fold_id',
    )
    cv = evaluate_cross_validation(
        x=x,
        aa_sequences=aa_sequences,
        class_labels=class_labels,
        perox_labels=perox_labels,
        n_folds=5,
        seed=int(cv_seed),
        model_arch=model_arch,
        dl_train_params=dl_train_params,
        dl_device=str(dl_train_params.get('device', 'cpu')),
        localize_strategy=localize_strategy,
        fold_ids=fold_ids,
    )
    prob_matrix, true_idx = _oof_rows_to_prob_and_true(
        oof_rows=cv['oof_rows'],
        class_names=list(LOCALIZATION_CLASSES),
    )
    metrics = _metrics_from_prob_matrix(
        prob_matrix=prob_matrix,
        true_idx=true_idx,
        class_names=list(LOCALIZATION_CLASSES),
    )
    return {
        'prob_matrix': prob_matrix,
        'true_idx': true_idx,
        'metrics': metrics,
        'n_rows_total': int(len(rows)),
        'n_rows_used': int(prob_matrix.shape[0]),
        'n_rows_skipped': int(skipped),
    }


def _evaluate_foldwise_classwise_blend(
    prob_a,
    prob_b,
    true_idx,
    fold_ids,
    class_names,
    alpha_grid,
    threshold_grid,
):
    prob_a = np.asarray(prob_a, dtype=np.float64)
    prob_b = np.asarray(prob_b, dtype=np.float64)
    true_idx = np.asarray(true_idx, dtype=np.int64)
    fold_ids = np.asarray(fold_ids)
    if prob_a.shape != prob_b.shape:
        raise ValueError('Shape mismatch between foldwise blend probability matrices.')
    if prob_a.shape[0] != true_idx.shape[0]:
        raise ValueError('true_idx row count does not match foldwise blend probabilities.')
    if prob_a.shape[0] != fold_ids.shape[0]:
        raise ValueError('fold_id row count does not match foldwise blend probabilities.')

    pred_idx = np.zeros((prob_a.shape[0],), dtype=np.int64)
    fold_rows = list()
    for fold_id in sorted(set([str(v) for v in fold_ids.tolist()])):
        valid_mask = np.asarray([str(v) == fold_id for v in fold_ids.tolist()], dtype=bool)
        train_mask = ~valid_mask
        if int(np.sum(valid_mask)) == 0 or int(np.sum(train_mask)) == 0:
            continue
        best_alpha, _ = _optimize_global_alpha(
            prob_a=prob_a[train_mask],
            prob_b=prob_b[train_mask],
            true_idx=true_idx[train_mask],
            class_names=class_names,
            grid=alpha_grid,
        )
        alpha_by_class, _ = _optimize_classwise_alpha(
            prob_a=prob_a[train_mask],
            prob_b=prob_b[train_mask],
            true_idx=true_idx[train_mask],
            class_names=class_names,
            grid=alpha_grid,
            init_alpha=best_alpha,
        )
        train_blend = _blend_classwise(
            prob_a=prob_a[train_mask],
            prob_b=prob_b[train_mask],
            alpha_by_class=alpha_by_class,
        )
        threshold_by_class, _ = _optimize_class_thresholds(
            prob_matrix=train_blend,
            true_idx=true_idx[train_mask],
            class_names=class_names,
            grid=threshold_grid,
        )
        valid_blend = _blend_classwise(
            prob_a=prob_a[valid_mask],
            prob_b=prob_b[valid_mask],
            alpha_by_class=alpha_by_class,
        )
        pred_idx[valid_mask] = _prediction_indices_with_thresholds(
            prob_matrix=valid_blend,
            thresholds=threshold_by_class,
        )
        fold_rows.append({
            'fold_id': fold_id,
            'n_train': int(np.sum(train_mask)),
            'n_valid': int(np.sum(valid_mask)),
            'global_alpha': float(best_alpha),
            'alpha_by_class': {class_names[i]: float(alpha_by_class[i]) for i in range(len(class_names))},
            'class_thresholds': {class_names[i]: float(threshold_by_class[i]) for i in range(len(class_names))},
        })

    metrics = _metrics_from_prediction_indices(
        pred_idx=pred_idx,
        true_idx=true_idx,
        class_names=class_names,
    )
    return metrics, fold_rows


def _build_summary_rows(
    targetp_ref,
    bilstm_metrics,
    esm_metrics,
    blend_global_metrics,
    blend_class_metrics,
    blend_threshold_metrics,
    blend_foldwise_metrics=None,
):
    rows = list()
    for class_name in LOCALIZATION_CLASSES:
        row = {
            'class': class_name,
            'targetp_f1': float(targetp_ref[class_name]['f1']),
            'bilstm_f1': float(bilstm_metrics['by_class'][class_name]['f1']),
            'esm_f1': float(esm_metrics['by_class'][class_name]['f1']),
            'blend_global_f1': float(blend_global_metrics['by_class'][class_name]['f1']),
            'blend_classwise_f1': float(blend_class_metrics['by_class'][class_name]['f1']),
            'blend_threshold_f1': float(blend_threshold_metrics['by_class'][class_name]['f1']),
        }
        if blend_foldwise_metrics is not None:
            row['blend_foldwise_f1'] = float(blend_foldwise_metrics['by_class'][class_name]['f1'])
        rows.append(row)
    return rows


def _render_markdown(out):
    rows = out['class_rows']
    md = list()
    has_foldwise = 'blend_foldwise' in out
    if has_foldwise:
        md.append('| Class | TargetP F1 | bilstm F1 | esm F1 | blend(global) F1 | blend(classwise) F1 | blend(threshold) F1 | blend(foldwise) F1 |')
        md.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    else:
        md.append('| Class | TargetP F1 | bilstm F1 | esm F1 | blend(global) F1 | blend(classwise) F1 | blend(threshold) F1 |')
        md.append('|---|---:|---:|---:|---:|---:|---:|')
    for row in rows:
        if has_foldwise:
            md.append(
                '| {c} | {t:.3f} | {b:.3f} | {e:.3f} | {g:.3f} | {cw:.3f} | {th:.3f} | {fw:.3f} |'.format(
                    c=row['class'],
                    t=row['targetp_f1'],
                    b=row['bilstm_f1'],
                    e=row['esm_f1'],
                    g=row['blend_global_f1'],
                    cw=row['blend_classwise_f1'],
                    th=row['blend_threshold_f1'],
                    fw=row['blend_foldwise_f1'],
                )
            )
        else:
            md.append(
                '| {c} | {t:.3f} | {b:.3f} | {e:.3f} | {g:.3f} | {cw:.3f} | {th:.3f} |'.format(
                    c=row['class'],
                    t=row['targetp_f1'],
                    b=row['bilstm_f1'],
                    e=row['esm_f1'],
                    g=row['blend_global_f1'],
                    cw=row['blend_classwise_f1'],
                    th=row['blend_threshold_f1'],
                )
            )
    md.append('')
    if has_foldwise:
        md.append('| Metric | TargetP | bilstm | esm | blend(global) | blend(classwise) | blend(threshold) | blend(foldwise) |')
        md.append('|---|---:|---:|---:|---:|---:|---:|---:|')
        md.append(
            '| Macro F1 | {tp:.3f} | {b:.3f} | {e:.3f} | {g:.3f} | {cw:.3f} | {th:.3f} | {fw:.3f} |'.format(
                tp=out['targetp_macro_f1'],
                b=out['bilstm']['metrics']['macro_f1'],
                e=out['esm']['metrics']['macro_f1'],
                g=out['blend_global']['metrics']['macro_f1'],
                cw=out['blend_classwise']['metrics']['macro_f1'],
                th=out['blend_threshold']['metrics']['macro_f1'],
                fw=out['blend_foldwise']['metrics']['macro_f1'],
            )
        )
        md.append(
            '| Overall accuracy | - | {b:.3f} | {e:.3f} | {g:.3f} | {cw:.3f} | {th:.3f} | {fw:.3f} |'.format(
                b=out['bilstm']['metrics']['overall_accuracy'],
                e=out['esm']['metrics']['overall_accuracy'],
                g=out['blend_global']['metrics']['overall_accuracy'],
                cw=out['blend_classwise']['metrics']['overall_accuracy'],
                th=out['blend_threshold']['metrics']['overall_accuracy'],
                fw=out['blend_foldwise']['metrics']['overall_accuracy'],
            )
        )
    else:
        md.append('| Metric | TargetP | bilstm | esm | blend(global) | blend(classwise) | blend(threshold) |')
        md.append('|---|---:|---:|---:|---:|---:|---:|')
        md.append(
            '| Macro F1 | {tp:.3f} | {b:.3f} | {e:.3f} | {g:.3f} | {cw:.3f} | {th:.3f} |'.format(
                tp=out['targetp_macro_f1'],
                b=out['bilstm']['metrics']['macro_f1'],
                e=out['esm']['metrics']['macro_f1'],
                g=out['blend_global']['metrics']['macro_f1'],
                cw=out['blend_classwise']['metrics']['macro_f1'],
                th=out['blend_threshold']['metrics']['macro_f1'],
            )
        )
        md.append(
            '| Overall accuracy | - | {b:.3f} | {e:.3f} | {g:.3f} | {cw:.3f} | {th:.3f} |'.format(
                b=out['bilstm']['metrics']['overall_accuracy'],
                e=out['esm']['metrics']['overall_accuracy'],
                g=out['blend_global']['metrics']['overall_accuracy'],
                cw=out['blend_classwise']['metrics']['overall_accuracy'],
                th=out['blend_threshold']['metrics']['overall_accuracy'],
            )
        )
    md.append('')
    md.append('global alpha (bilstm weight): {:.3f}'.format(out['blend_global']['alpha']))
    md.append('classwise alpha (bilstm weight): {}'.format(out['blend_classwise']['alpha_by_class']))
    md.append('class thresholds: {}'.format(out['blend_threshold']['class_thresholds']))
    return '\n'.join(md)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Blend bilstm-cdskit and esm-cdskit on TargetP fold-fixed benchmark.',
    )
    parser.add_argument('--training_tsv', default='data/localize_bench/targetp2_benchmark.tsv', type=str)
    parser.add_argument('--reuse_oof_cache', default='yes', choices=['yes', 'no'], type=str)
    parser.add_argument('--organism_gate', default='no', choices=['yes', 'no'], type=str)
    parser.add_argument('--bilstm_oof_npz', default='data/localize_bench/targetp2_oof_bilstm.npz', type=str)
    parser.add_argument('--esm_oof_npz', default='data/localize_bench/targetp2_oof_esm.npz', type=str)
    parser.add_argument('--cv_seed', default=1, type=int)
    parser.add_argument('--localize_strategy', default='single_stage', choices=['single_stage', 'two_stage', 'two_stage_ctp_ltp'], type=str)
    parser.add_argument('--bilstm_dl_seq_len', default=200, type=int)
    parser.add_argument('--bilstm_dl_embed_dim', default=32, type=int)
    parser.add_argument('--bilstm_dl_hidden_dim', default=64, type=int)
    parser.add_argument('--bilstm_dl_num_layers', default=1, type=int)
    parser.add_argument('--bilstm_dl_dropout', default=0.2, type=float)
    parser.add_argument('--bilstm_dl_epochs', default=15, type=int)
    parser.add_argument('--bilstm_dl_batch_size', default=128, type=int)
    parser.add_argument('--bilstm_dl_lr', default=1.0e-3, type=float)
    parser.add_argument('--bilstm_dl_weight_decay', default=1.0e-4, type=float)
    parser.add_argument('--bilstm_dl_class_weight', default='yes', choices=['yes', 'no'], type=str)
    parser.add_argument('--bilstm_dl_loss', default='ce', choices=['ce', 'focal'], type=str)
    parser.add_argument('--bilstm_dl_balanced_batch', default='no', choices=['yes', 'no'], type=str)
    parser.add_argument('--bilstm_dl_feature_fusion', default='yes', choices=['yes', 'no'], type=str)
    parser.add_argument('--bilstm_dl_seed', default=1, type=int)
    parser.add_argument('--bilstm_dl_device', default='cpu', choices=['cpu', 'cuda', 'mps', 'auto'], type=str)
    parser.add_argument('--esm_model_name', default='facebook/esm2_t6_8M_UR50D', type=str)
    parser.add_argument('--esm_model_local_dir', default='', type=str)
    parser.add_argument('--esm_pooling', default='cls', choices=['cls', 'mean'], type=str)
    parser.add_argument('--esm_max_len', default=200, type=int)
    parser.add_argument('--esm_dl_epochs', default=1, type=int)
    parser.add_argument('--esm_dl_batch_size', default=32, type=int)
    parser.add_argument('--esm_dl_lr', default=1.0e-3, type=float)
    parser.add_argument('--esm_dl_weight_decay', default=1.0e-4, type=float)
    parser.add_argument('--esm_dl_class_weight', default='yes', choices=['yes', 'no'], type=str)
    parser.add_argument('--esm_dl_seed', default=1, type=int)
    parser.add_argument('--esm_dl_device', default='cpu', choices=['cpu', 'cuda', 'mps', 'auto'], type=str)
    parser.add_argument('--blend_grid_step', default=0.05, type=float)
    parser.add_argument('--threshold_grid', default='0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.5,0.65,0.8,1.0,1.25,1.5,2.0,3.0,5.0', type=str)
    parser.add_argument('--foldwise_blend_eval', default='no', choices=['yes', 'no'], type=str)
    parser.add_argument('--out_json', default='data/localize_bench/targetp2_bilstm_esm_blend.json', type=str)
    parser.add_argument('--out_md', default='data/localize_bench/targetp2_bilstm_esm_blend.md', type=str)
    return parser


def _to_bool_yes_no(value):
    return str(value).strip().lower() in ['yes', 'y', 'true', '1']


def main():
    args = build_parser().parse_args()
    class_names = list(LOCALIZATION_CLASSES)
    reuse_cache = _to_bool_yes_no(args.reuse_oof_cache)
    fallback_true_idx = None
    if os.path.exists(args.training_tsv):
        fallback_true_idx = _read_true_idx_from_training_tsv(
            training_tsv=args.training_tsv,
            class_names=class_names,
        )

    bilstm_prob = None
    bilstm_true = None
    if reuse_cache and os.path.exists(args.bilstm_oof_npz):
        bilstm_prob, bilstm_true, class_names_from_file = _load_oof_npz(
            args.bilstm_oof_npz,
            fallback_true_idx=fallback_true_idx,
        )
        if class_names_from_file != class_names:
            raise ValueError('Class names in bilstm_oof_npz do not match LOCALIZATION_CLASSES.')
        if bilstm_prob.shape[0] != bilstm_true.shape[0]:
            raise ValueError('Row count mismatch between bilstm_oof_npz probabilities and true labels.')
        bilstm_metrics = _metrics_from_prob_matrix(
            prob_matrix=bilstm_prob,
            true_idx=bilstm_true,
            class_names=class_names,
        )
        bilstm_info = {
            'metrics': bilstm_metrics,
            'oof_npz': args.bilstm_oof_npz,
            'used_cache': True,
        }
    else:
        bilstm_dl = {
            'seq_len': int(args.bilstm_dl_seq_len),
            'embed_dim': int(args.bilstm_dl_embed_dim),
            'hidden_dim': int(args.bilstm_dl_hidden_dim),
            'num_layers': int(args.bilstm_dl_num_layers),
            'dropout': float(args.bilstm_dl_dropout),
            'epochs': int(args.bilstm_dl_epochs),
            'batch_size': int(args.bilstm_dl_batch_size),
            'learning_rate': float(args.bilstm_dl_lr),
            'weight_decay': float(args.bilstm_dl_weight_decay),
            'use_class_weight': _to_bool_yes_no(args.bilstm_dl_class_weight),
            'loss_name': str(args.bilstm_dl_loss),
            'balanced_batch': _to_bool_yes_no(args.bilstm_dl_balanced_batch),
            'feature_fusion': _to_bool_yes_no(args.bilstm_dl_feature_fusion),
            'aux_tp_weight': 0.0,
            'aux_ctp_ltp_weight': 0.0,
            'seed': int(args.bilstm_dl_seed),
            'device': str(args.bilstm_dl_device),
            'esm_model_name': '',
            'esm_model_local_dir': '',
            'esm_pooling': 'cls',
            'esm_max_len': 0,
        }
        out = _run_model_oof(
            training_tsv=args.training_tsv,
            model_arch='bilstm_attention',
            localize_strategy=args.localize_strategy,
            dl_train_params=bilstm_dl,
            cv_seed=int(args.cv_seed),
        )
        bilstm_prob = out['prob_matrix']
        bilstm_true = out['true_idx']
        _save_oof_npz(
            path=args.bilstm_oof_npz,
            prob_matrix=bilstm_prob,
            true_idx=bilstm_true,
            class_names=class_names,
        )
        bilstm_info = {
            'metrics': out['metrics'],
            'oof_npz': args.bilstm_oof_npz,
            'used_cache': False,
            'n_rows_total': out['n_rows_total'],
            'n_rows_used': out['n_rows_used'],
            'n_rows_skipped': out['n_rows_skipped'],
        }

    esm_prob = None
    esm_true = None
    if reuse_cache and os.path.exists(args.esm_oof_npz):
        esm_prob, esm_true, class_names_from_file = _load_oof_npz(
            args.esm_oof_npz,
            fallback_true_idx=fallback_true_idx,
        )
        if class_names_from_file != class_names:
            raise ValueError('Class names in esm_oof_npz do not match LOCALIZATION_CLASSES.')
        if esm_prob.shape[0] != esm_true.shape[0]:
            raise ValueError('Row count mismatch between esm_oof_npz probabilities and true labels.')
        esm_metrics = _metrics_from_prob_matrix(
            prob_matrix=esm_prob,
            true_idx=esm_true,
            class_names=class_names,
        )
        esm_info = {
            'metrics': esm_metrics,
            'oof_npz': args.esm_oof_npz,
            'used_cache': True,
        }
    else:
        esm_dl = {
            'seq_len': 0,
            'embed_dim': 0,
            'hidden_dim': 0,
            'num_layers': 0,
            'dropout': 0.0,
            'epochs': int(args.esm_dl_epochs),
            'batch_size': int(args.esm_dl_batch_size),
            'learning_rate': float(args.esm_dl_lr),
            'weight_decay': float(args.esm_dl_weight_decay),
            'use_class_weight': _to_bool_yes_no(args.esm_dl_class_weight),
            'loss_name': 'ce',
            'balanced_batch': False,
            'seed': int(args.esm_dl_seed),
            'device': str(args.esm_dl_device),
            'esm_model_name': str(args.esm_model_name),
            'esm_model_local_dir': str(args.esm_model_local_dir),
            'esm_pooling': str(args.esm_pooling),
            'esm_max_len': int(args.esm_max_len),
        }
        out = _run_model_oof(
            training_tsv=args.training_tsv,
            model_arch='esm_head',
            localize_strategy=args.localize_strategy,
            dl_train_params=esm_dl,
            cv_seed=int(args.cv_seed),
        )
        esm_prob = out['prob_matrix']
        esm_true = out['true_idx']
        _save_oof_npz(
            path=args.esm_oof_npz,
            prob_matrix=esm_prob,
            true_idx=esm_true,
            class_names=class_names,
        )
        esm_info = {
            'metrics': out['metrics'],
            'oof_npz': args.esm_oof_npz,
            'used_cache': False,
            'n_rows_total': out['n_rows_total'],
            'n_rows_used': out['n_rows_used'],
            'n_rows_skipped': out['n_rows_skipped'],
        }

    if bilstm_prob.shape != esm_prob.shape:
        raise ValueError('Shape mismatch between bilstm and esm OOF probabilities.')
    if np.any(bilstm_true != esm_true):
        raise ValueError('True labels differ between bilstm and esm OOF caches.')
    organism_gate = _to_bool_yes_no(args.organism_gate)
    if organism_gate:
        plant_mask = _read_organism_group_mask(training_tsv=args.training_tsv)
        bilstm_prob = _apply_organism_gate(
            prob_matrix=bilstm_prob,
            plant_mask=plant_mask,
            class_names=class_names,
        )
        esm_prob = _apply_organism_gate(
            prob_matrix=esm_prob,
            plant_mask=plant_mask,
            class_names=class_names,
        )
        bilstm_info['metrics'] = _metrics_from_prob_matrix(
            prob_matrix=bilstm_prob,
            true_idx=bilstm_true,
            class_names=class_names,
        )
        esm_info['metrics'] = _metrics_from_prob_matrix(
            prob_matrix=esm_prob,
            true_idx=esm_true,
            class_names=class_names,
        )
    bilstm_info['organism_gate'] = bool(organism_gate)
    esm_info['organism_gate'] = bool(organism_gate)

    step = float(args.blend_grid_step)
    if (step <= 0.0) or (step > 1.0):
        raise ValueError('--blend_grid_step should be in (0, 1].')
    n_tick = int(np.floor(1.0 / step)) + 1
    grid = [float(i) * step for i in range(n_tick)]
    if grid[-1] != 1.0:
        grid.append(1.0)
    grid = sorted(set([round(v, 10) for v in grid]))

    best_alpha, blend_global_metrics = _optimize_global_alpha(
        prob_a=bilstm_prob,
        prob_b=esm_prob,
        true_idx=bilstm_true,
        class_names=class_names,
        grid=grid,
    )
    alpha_by_class, blend_class_metrics = _optimize_classwise_alpha(
        prob_a=bilstm_prob,
        prob_b=esm_prob,
        true_idx=bilstm_true,
        class_names=class_names,
        grid=grid,
        init_alpha=best_alpha,
    )
    blend_class_prob = _blend_classwise(
        prob_a=bilstm_prob,
        prob_b=esm_prob,
        alpha_by_class=alpha_by_class,
    )
    threshold_grid = [
        float(v.strip()) for v in str(args.threshold_grid).split(',')
        if str(v).strip() != ''
    ]
    if len(threshold_grid) == 0:
        raise ValueError('--threshold_grid should contain at least one value.')
    threshold_grid = sorted(set(threshold_grid))
    threshold_by_class, blend_threshold_metrics = _optimize_class_thresholds(
        prob_matrix=blend_class_prob,
        true_idx=bilstm_true,
        class_names=class_names,
        grid=threshold_grid,
    )
    blend_foldwise = None
    if _to_bool_yes_no(args.foldwise_blend_eval):
        fold_ids = _read_fold_ids_from_training_tsv(training_tsv=args.training_tsv)
        foldwise_metrics, foldwise_rows = _evaluate_foldwise_classwise_blend(
            prob_a=bilstm_prob,
            prob_b=esm_prob,
            true_idx=bilstm_true,
            fold_ids=fold_ids,
            class_names=class_names,
            alpha_grid=grid,
            threshold_grid=threshold_grid,
        )
        blend_foldwise = {
            'description': 'Each held-out fold is predicted using classwise alpha and class thresholds optimized on the other folds.',
            'metrics': foldwise_metrics,
            'folds': foldwise_rows,
        }

    targetp_macro_f1 = float(np.mean(np.asarray(
        [TARGETP_TABLE1_REFERENCE[c]['f1'] for c in class_names],
        dtype=np.float64,
    )))

    out = {
        'training_tsv': args.training_tsv,
        'class_names': class_names,
        'targetp_reference': TARGETP_TABLE1_REFERENCE,
        'targetp_macro_f1': targetp_macro_f1,
        'organism_gate': bool(organism_gate),
        'bilstm': bilstm_info,
        'esm': esm_info,
        'blend_global': {
            'alpha': float(best_alpha),
            'metrics': blend_global_metrics,
        },
        'blend_classwise': {
            'alpha_by_class': {class_names[i]: float(alpha_by_class[i]) for i in range(len(class_names))},
            'metrics': blend_class_metrics,
        },
        'blend_threshold': {
            'alpha_by_class': {class_names[i]: float(alpha_by_class[i]) for i in range(len(class_names))},
            'class_thresholds': {class_names[i]: float(threshold_by_class[i]) for i in range(len(class_names))},
            'metrics': blend_threshold_metrics,
        },
    }
    if blend_foldwise is not None:
        out['blend_foldwise'] = blend_foldwise
    out['class_rows'] = _build_summary_rows(
        targetp_ref=TARGETP_TABLE1_REFERENCE,
        bilstm_metrics=out['bilstm']['metrics'],
        esm_metrics=out['esm']['metrics'],
        blend_global_metrics=out['blend_global']['metrics'],
        blend_class_metrics=out['blend_classwise']['metrics'],
        blend_threshold_metrics=out['blend_threshold']['metrics'],
        blend_foldwise_metrics=(None if blend_foldwise is None else out['blend_foldwise']['metrics']),
    )
    out['markdown'] = _render_markdown(out=out)

    out_json_dir = os.path.dirname(args.out_json)
    if out_json_dir != '':
        os.makedirs(out_json_dir, exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as out_json:
        json.dump(out, out_json, indent=2)

    out_md_dir = os.path.dirname(args.out_md)
    if out_md_dir != '':
        os.makedirs(out_md_dir, exist_ok=True)
    with open(args.out_md, 'w', encoding='utf-8') as out_md:
        out_md.write(out['markdown'] + '\n')

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

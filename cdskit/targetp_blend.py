import argparse
import csv
import json
import os

import numpy as np

from cdskit.localize_model import (
    AA_ACIDIC,
    AA_AROMATIC,
    AA_BASIC,
    FEATURE_NAMES,
    AA_HYDROPHOBIC,
    AA_SER_THR,
    AA_SMALL,
    extract_broad_localize_features,
    fit_perox_binary_classifier,
    fraction_in_set,
    longest_hydrophobic_run,
    mean_hydropathy,
    save_localize_model,
)
from cdskit.localize_learn import (
    LOCALIZATION_CLASSES,
    build_training_matrix,
    evaluate_cross_validation,
    fit_localization_model,
)
from cdskit.targetp_benchmark import (
    TARGETP_TABLE1_REFERENCE,
    compute_prf_by_class,
)

TARGETP_SPECIALIST_FIXED_PROFILE = {
    'name': 'targetp_specialist_fixed_v1',
    'sp_random_states': [2, 13, 31],
    'sp_weights': [0.22251605108894593, 0.24685472258402566, 0.5306292263270285],
    'ltp_random_states': [13, 23, 31, 6, 5],
    'sp_threshold': 0.6975,
    'ltp_threshold': 0.2525,
    'ltp_mass_threshold': 0.21,
}


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


def _targetp_reference_f1_array(class_names):
    return np.asarray(
        [TARGETP_TABLE1_REFERENCE[class_name]['f1'] for class_name in class_names],
        dtype=np.float64,
    )


def _fast_f1_vector(pred_idx, true_idx, n_class):
    pred_idx = np.asarray(pred_idx, dtype=np.int64)
    true_idx = np.asarray(true_idx, dtype=np.int64)
    mat = np.bincount(
        (true_idx * int(n_class)) + pred_idx,
        minlength=int(n_class) * int(n_class),
    ).reshape((int(n_class), int(n_class)))
    tp = np.diag(mat).astype(np.float64)
    fp = mat.sum(axis=0).astype(np.float64) - tp
    fn = mat.sum(axis=1).astype(np.float64) - tp
    denom = (2.0 * tp) + fp + fn
    return np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0.0)


def _targetp_margin_rank(pred_idx, true_idx, class_names):
    f1 = _fast_f1_vector(
        pred_idx=pred_idx,
        true_idx=true_idx,
        n_class=len(class_names),
    )
    ref = _targetp_reference_f1_array(class_names=class_names)
    overall = float(np.mean(np.asarray(pred_idx, dtype=np.int64) == np.asarray(true_idx, dtype=np.int64)))
    return (
        float(np.min(f1 - ref)),
        float(np.mean(f1)),
        float(f1[class_names.index('SP')] if 'SP' in class_names else 0.0),
        float(f1[class_names.index('lTP')] if 'lTP' in class_names else 0.0),
        float(f1[class_names.index('cTP')] if 'cTP' in class_names else 0.0),
        overall,
    )


def _targetp_margin_summary(metrics, targetp_ref, class_names):
    margins = dict()
    beats = dict()
    for class_name in class_names:
        f1 = float(metrics['by_class'][class_name]['f1'])
        ref_f1 = float(targetp_ref[class_name]['f1'])
        margin = f1 - ref_f1
        margins[class_name] = float(margin)
        beats[class_name] = bool(margin > 0.0)
    min_margin = min(margins.values()) if len(margins) > 0 else 0.0
    return {
        'beats_targetp_all_classes': bool(all(beats.values())),
        'min_class_f1_margin': float(min_margin),
        'class_f1_margins': margins,
        'class_beats_targetp': beats,
    }


def _attach_targetp_margin_summaries(out, class_names):
    for key in [
        'bilstm',
        'esm',
        'blend_global',
        'blend_classwise',
        'blend_threshold',
        'blend_foldwise',
        'specialist_postprocess',
        'specialist_foldwise',
        'specialist_foldwise_fixed',
    ]:
        if key not in out:
            continue
        out[key]['targetp_margin'] = _targetp_margin_summary(
            metrics=out[key]['metrics'],
            targetp_ref=TARGETP_TABLE1_REFERENCE,
            class_names=class_names,
        )


def _best_binary_f1_threshold(scores, true_idx, positive_idx):
    scores = np.asarray(scores, dtype=np.float64)
    true_idx = np.asarray(true_idx, dtype=np.int64)
    yy = true_idx == int(positive_idx)
    if scores.shape[0] != yy.shape[0]:
        raise ValueError('Binary threshold score length does not match labels.')
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_y = yy[order]
    total_pos = int(np.sum(yy))
    tp = 0
    fp = 0
    best_f1 = 0.0
    best_threshold = float(sorted_scores[0]) if sorted_scores.size else 0.5
    best_counts = {'tp': 0, 'fp': 0, 'fn': total_pos}
    i = 0
    while i < sorted_scores.shape[0]:
        value = float(sorted_scores[i])
        while i < sorted_scores.shape[0] and float(sorted_scores[i]) == value:
            if bool(sorted_y[i]):
                tp += 1
            else:
                fp += 1
            i += 1
        fn = total_pos - tp
        denom = (2 * tp) + fp + fn
        f1 = 0.0 if denom == 0 else float(2 * tp) / float(denom)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = value
            best_counts = {'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}
    return best_threshold, best_f1, best_counts


def _top_binary_f1_thresholds(scores, true_idx, positive_idx, max_candidates=80):
    scores = np.asarray(scores, dtype=np.float64)
    true_idx = np.asarray(true_idx, dtype=np.int64)
    yy = true_idx == int(positive_idx)
    if scores.shape[0] != yy.shape[0]:
        raise ValueError('Binary threshold score length does not match labels.')
    rows = list()
    for threshold in np.unique(scores).tolist():
        pred = scores >= float(threshold)
        tp = int(np.sum(pred & yy))
        fp = int(np.sum(pred & (~yy)))
        fn = int(np.sum((~pred) & yy))
        denom = (2 * tp) + fp + fn
        f1 = 0.0 if denom == 0 else float(2 * tp) / float(denom)
        rows.append((f1, float(threshold), {'tp': tp, 'fp': fp, 'fn': fn}))
    rows.sort(key=lambda row: (row[0], -row[2]['fp'], row[2]['tp']), reverse=True)
    out = list()
    seen = set()
    for f1, threshold, counts in rows:
        if threshold in seen:
            continue
        seen.add(threshold)
        out.append((threshold, f1, counts))
        if len(out) >= int(max_candidates):
            break
    if len(out) == 0:
        out.append((0.5, 0.0, {'tp': 0, 'fp': 0, 'fn': int(np.sum(yy))}))
    return out


def _targetp_sp_scan_features(seq):
    seq = str(seq or '')
    best_score = -99.0
    best_cut = 0
    best_parts = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for cut in range(12, min(45, len(seq) - 1)):
        pre = seq[:cut]
        nreg = seq[:max(1, cut - 18)]
        hreg = seq[max(0, cut - 18):max(0, cut - 7)]
        creg = seq[max(0, cut - 7):cut + 2]
        m3 = seq[cut - 3] if cut - 3 >= 0 else 'X'
        m2 = seq[cut - 2] if cut - 2 >= 0 else 'X'
        m1 = seq[cut - 1] if cut - 1 >= 0 else 'X'
        p1 = seq[cut] if cut < len(seq) else 'X'
        small_m3 = 1.0 if m3 in 'AVSGTC' else 0.0
        small_m1 = 1.0 if m1 in 'ASGTC' else 0.0
        ala_m1 = 1.0 if m1 == 'A' else 0.0
        pro_bad = 1.0 if 'P' in (m3 + m2 + m1 + p1) else 0.0
        hyd = fraction_in_set(hreg, AA_HYDROPHOBIC)
        run = longest_hydrophobic_run(hreg)
        small = fraction_in_set(creg, AA_SMALL)
        ncharge = fraction_in_set(nreg, AA_BASIC) - fraction_in_set(nreg, AA_ACIDIC)
        st_frac = fraction_in_set(pre, AA_SER_THR)
        score = (
            (2.2 * hyd)
            + (0.15 * run)
            + (0.8 * small_m3)
            + (1.0 * small_m1)
            + (0.4 * ala_m1)
            + (0.5 * small)
            + (0.4 * ncharge)
            - (0.9 * pro_bad)
            - (0.25 * st_frac)
        )
        if score > best_score:
            best_score = float(score)
            best_cut = int(cut)
            best_parts = (
                float(hyd),
                float(run),
                float(small_m3),
                float(small_m1),
                float(ala_m1),
                float(pro_bad),
                float(small),
                float(ncharge),
                float(st_frac),
            )

    out = [best_score, float(best_cut), float(best_cut) / float(max(1, len(seq)))]
    out.extend(best_parts)
    for window in [
        seq[:15],
        seq[:25],
        seq[:35],
        seq[:50],
        seq[:80],
        seq[5:30],
        seq[20:60],
        seq[40:100],
    ]:
        out.extend([
            mean_hydropathy(window),
            longest_hydrophobic_run(window),
            fraction_in_set(window, AA_HYDROPHOBIC),
            fraction_in_set(window, AA_BASIC),
            fraction_in_set(window, AA_ACIDIC),
            fraction_in_set(window, AA_SER_THR),
            fraction_in_set(window, AA_SMALL),
        ])
    return out


def _targetp_ctp_ltp_sequence_features(seq, organism_group):
    seq = str(seq or '')
    out = list(extract_broad_localize_features(seq, organism_group)[0])
    windows = [
        seq[:20],
        seq[:40],
        seq[:60],
        seq[:80],
        seq[:100],
        seq[:120],
        seq[20:80],
        seq[40:120],
    ]
    groups = [
        AA_BASIC,
        AA_ACIDIC,
        AA_HYDROPHOBIC,
        AA_SMALL,
        AA_SER_THR,
        AA_AROMATIC,
        frozenset('R'),
        frozenset('K'),
        frozenset('A'),
        frozenset('S'),
        frozenset('T'),
        frozenset('P'),
        frozenset('G'),
        frozenset('LIV'),
    ]
    for window in windows:
        out.extend([mean_hydropathy(window), longest_hydrophobic_run(window)])
        out.extend([fraction_in_set(window, group) for group in groups])
    n_term = seq[:140]
    for motif in ['RR', 'KR', 'RK', 'KK', 'RA', 'RS', 'SR', 'ST', 'TS', 'SS', 'TP', 'SP']:
        out.append(1.0 if motif in n_term else 0.0)
        out.append(float(n_term.find(motif) if motif in n_term else 999))
    return out


def _specialist_probability_features(base_prob, prob_a, prob_b, class_names):
    base_prob = np.asarray(base_prob, dtype=np.float64)
    prob_a = np.asarray(prob_a, dtype=np.float64)
    prob_b = np.asarray(prob_b, dtype=np.float64)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    ctp_idx = class_to_idx['cTP']
    ltp_idx = class_to_idx['lTP']
    ltp_ratio = base_prob[:, ltp_idx] / np.clip(
        base_prob[:, ctp_idx] + base_prob[:, ltp_idx],
        a_min=1.0e-12,
        a_max=None,
    )
    return ltp_ratio, np.hstack([
        base_prob,
        prob_a,
        prob_b,
        ltp_ratio.reshape((-1, 1)),
    ])


def _build_sp_specialist_features(rows, base_prob, prob_a, prob_b, class_names):
    ltp_ratio, prob_features = _specialist_probability_features(
        base_prob=base_prob,
        prob_a=prob_a,
        prob_b=prob_b,
        class_names=class_names,
    )
    del ltp_ratio
    seq_features = np.asarray(
        [_targetp_sp_scan_features(row.get('sequence', '')) for row in rows],
        dtype=np.float64,
    )
    plant_flag = np.asarray([
        1.0 if str(row.get('organism_group', '')).strip().lower() == 'plant' else 0.0
        for row in rows
    ], dtype=np.float64).reshape((-1, 1))
    return np.hstack([seq_features, prob_features, plant_flag])


def _build_ctp_ltp_specialist_features(rows, base_prob, prob_a, prob_b, class_names):
    ltp_ratio, prob_features = _specialist_probability_features(
        base_prob=base_prob,
        prob_a=prob_a,
        prob_b=prob_b,
        class_names=class_names,
    )
    del ltp_ratio
    seq_features = np.asarray([
        _targetp_ctp_ltp_sequence_features(
            seq=row.get('sequence', ''),
            organism_group=row.get('organism_group', ''),
        )
        for row in rows
    ], dtype=np.float64)
    plant_flag = np.asarray([
        1.0 if str(row.get('organism_group', '')).strip().lower() == 'plant' else 0.0
        for row in rows
    ], dtype=np.float64).reshape((-1, 1))
    return np.hstack([seq_features, prob_features, plant_flag])


def _binary_oof_scores(features, true_idx, fold_ids, train_mask, positive_idx, make_model):
    features = np.asarray(features, dtype=np.float64)
    true_idx = np.asarray(true_idx, dtype=np.int64)
    fold_ids = np.asarray(fold_ids)
    train_mask = np.asarray(train_mask, dtype=bool)
    scores = np.zeros((features.shape[0],), dtype=np.float64)
    for fold_id in sorted(set([str(v) for v in fold_ids.tolist()])):
        valid_mask = np.asarray([str(v) == fold_id for v in fold_ids.tolist()], dtype=bool)
        fit_mask = (~valid_mask) & train_mask
        y_train = (true_idx[fit_mask] == int(positive_idx)).astype(np.int64)
        if features[fit_mask].shape[0] == 0:
            scores[valid_mask] = 0.0
            continue
        if len(set(y_train.tolist())) < 2:
            scores[valid_mask] = float(np.mean(y_train))
            continue
        model = make_model()
        model.fit(features[fit_mask], y_train)
        if not hasattr(model, 'predict_proba'):
            raise TypeError('Specialist model should support predict_proba.')
        proba = np.asarray(model.predict_proba(features[valid_mask]), dtype=np.float64)
        class_to_col = {int(cls): i for i, cls in enumerate(model.classes_.tolist())}
        scores[valid_mask] = proba[:, class_to_col.get(1, 0)] if 1 in class_to_col else 0.0
    return scores


def _binary_crossfit_scores(
    features,
    true_idx,
    fold_ids,
    fit_mask,
    score_mask,
    positive_idx,
    make_model,
):
    features = np.asarray(features, dtype=np.float64)
    true_idx = np.asarray(true_idx, dtype=np.int64)
    fold_ids = np.asarray(fold_ids)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    score_mask = np.asarray(score_mask, dtype=bool)
    scores = np.zeros((features.shape[0],), dtype=np.float64)
    for fold_id in sorted(set([str(v) for v in fold_ids[score_mask].tolist()])):
        valid_mask = score_mask & np.asarray([str(v) == fold_id for v in fold_ids.tolist()], dtype=bool)
        train_mask = fit_mask & (~valid_mask)
        y_train = (true_idx[train_mask] == int(positive_idx)).astype(np.int64)
        if features[train_mask].shape[0] == 0:
            scores[valid_mask] = 0.0
            continue
        if len(set(y_train.tolist())) < 2:
            scores[valid_mask] = float(np.mean(y_train))
            continue
        model = make_model()
        model.fit(features[train_mask], y_train)
        proba = np.asarray(model.predict_proba(features[valid_mask]), dtype=np.float64)
        class_to_col = {int(cls): i for i, cls in enumerate(model.classes_.tolist())}
        scores[valid_mask] = proba[:, class_to_col.get(1, 0)] if 1 in class_to_col else 0.0
    return scores


def _fit_binary_predict_scores(
    features,
    true_idx,
    fit_mask,
    predict_mask,
    positive_idx,
    make_model,
):
    features = np.asarray(features, dtype=np.float64)
    true_idx = np.asarray(true_idx, dtype=np.int64)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    predict_mask = np.asarray(predict_mask, dtype=bool)
    y_train = (true_idx[fit_mask] == int(positive_idx)).astype(np.int64)
    scores = np.zeros((features.shape[0],), dtype=np.float64)
    if features[fit_mask].shape[0] == 0:
        return scores
    if len(set(y_train.tolist())) < 2:
        scores[predict_mask] = float(np.mean(y_train))
        return scores
    model = make_model()
    model.fit(features[fit_mask], y_train)
    proba = np.asarray(model.predict_proba(features[predict_mask]), dtype=np.float64)
    class_to_col = {int(cls): i for i, cls in enumerate(model.classes_.tolist())}
    scores[predict_mask] = proba[:, class_to_col.get(1, 0)] if 1 in class_to_col else 0.0
    return scores


def _aggregate_score_columns(score_columns, weights=None):
    if len(score_columns) == 0:
        raise ValueError('At least one score column is required.')
    stack = np.vstack([np.asarray(col, dtype=np.float64) for col in score_columns])
    if weights is None:
        return np.mean(stack, axis=0)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape[0] != stack.shape[0]:
        raise ValueError('Score aggregation weights do not match score columns.')
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError('Score aggregation weights should sum to a positive value.')
    return np.average(stack, axis=0, weights=weights / total)


def _fit_binary_predict_ensemble_scores(
    features,
    true_idx,
    fit_mask,
    predict_mask,
    positive_idx,
    make_models,
    weights=None,
):
    return _aggregate_score_columns(
        [
            _fit_binary_predict_scores(
                features=features,
                true_idx=true_idx,
                fit_mask=fit_mask,
                predict_mask=predict_mask,
                positive_idx=positive_idx,
                make_model=make_model,
            )
            for make_model in make_models
        ],
        weights=weights,
    )


def _apply_specialist_postprocess_predictions(
    base_prob,
    class_thresholds,
    sp_scores,
    sp_threshold,
    ltp_scores,
    ltp_threshold,
    plant_mask,
    class_names,
    ltp_mass_threshold=0.20,
):
    base_prob = np.asarray(base_prob, dtype=np.float64)
    class_thresholds = np.asarray(class_thresholds, dtype=np.float64)
    sp_scores = np.asarray(sp_scores, dtype=np.float64)
    ltp_scores = np.asarray(ltp_scores, dtype=np.float64)
    plant_mask = np.asarray(plant_mask, dtype=bool)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    sp_idx = class_to_idx['SP']
    ctp_idx = class_to_idx['cTP']
    ltp_idx = class_to_idx['lTP']

    pred_idx = _prediction_indices_with_thresholds(
        prob_matrix=base_prob,
        thresholds=class_thresholds,
    )
    scores = base_prob / np.asarray(class_thresholds, dtype=np.float64).reshape((1, -1))
    non_sp_scores = scores.copy()
    non_sp_scores[:, sp_idx] = -np.inf
    non_sp_pred = np.argmax(non_sp_scores, axis=1).astype(np.int64)

    sp_positive = sp_scores >= float(sp_threshold)
    pred_idx[sp_positive] = sp_idx
    demote_sp = (~sp_positive) & (pred_idx == sp_idx)
    pred_idx[demote_sp] = non_sp_pred[demote_sp]

    ctp_ltp_mass = base_prob[:, ctp_idx] + base_prob[:, ltp_idx]
    ltp_candidate = (
        plant_mask
        & (~sp_positive)
        & (ctp_ltp_mass > float(ltp_mass_threshold))
    )
    pred_idx[ltp_candidate & (ltp_scores >= float(ltp_threshold))] = ltp_idx
    demote_ltp = (
        ltp_candidate
        & (ltp_scores < float(ltp_threshold))
        & ((pred_idx == ltp_idx) | (pred_idx == ctp_idx))
    )
    pred_idx[demote_ltp] = ctp_idx
    return pred_idx


def _optimize_ltp_specialist_threshold(
    base_prob,
    class_thresholds,
    sp_scores,
    sp_threshold,
    ltp_scores,
    plant_mask,
    true_idx,
    class_names,
    ltp_mass_threshold,
):
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    ctp_idx = class_to_idx['cTP']
    ltp_idx = class_to_idx['lTP']
    sp_positive = np.asarray(sp_scores, dtype=np.float64) >= float(sp_threshold)
    ctp_ltp_mass = np.asarray(base_prob, dtype=np.float64)[:, ctp_idx] + np.asarray(base_prob, dtype=np.float64)[:, ltp_idx]
    relevant = (
        np.asarray(plant_mask, dtype=bool)
        & (~sp_positive)
        & (
            (np.asarray(true_idx, dtype=np.int64) == ctp_idx)
            | (np.asarray(true_idx, dtype=np.int64) == ltp_idx)
            | (ctp_ltp_mass > float(ltp_mass_threshold))
        )
    )
    thresholds = np.unique(np.asarray(ltp_scores, dtype=np.float64)[relevant])
    if thresholds.shape[0] == 0:
        thresholds = np.asarray([0.5], dtype=np.float64)

    best_threshold = float(thresholds[0])
    best_rank = None
    best_pred = None
    for threshold in thresholds.tolist():
        pred_idx = _apply_specialist_postprocess_predictions(
            base_prob=base_prob,
            class_thresholds=class_thresholds,
            sp_scores=sp_scores,
            sp_threshold=sp_threshold,
            ltp_scores=ltp_scores,
            ltp_threshold=float(threshold),
            plant_mask=plant_mask,
            class_names=class_names,
            ltp_mass_threshold=ltp_mass_threshold,
        )
        rank = _targetp_margin_rank(
            pred_idx=pred_idx,
            true_idx=true_idx,
            class_names=class_names,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_threshold = float(threshold)
            best_pred = pred_idx
    return best_threshold, best_pred, best_rank


def _candidate_ltp_thresholds(
    base_prob,
    sp_scores,
    sp_threshold,
    ltp_scores,
    plant_mask,
    true_idx,
    class_names,
    ltp_mass_threshold,
):
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    ctp_idx = class_to_idx['cTP']
    ltp_idx = class_to_idx['lTP']
    sp_positive = np.asarray(sp_scores, dtype=np.float64) >= float(sp_threshold)
    ctp_ltp_mass = np.asarray(base_prob, dtype=np.float64)[:, ctp_idx] + np.asarray(base_prob, dtype=np.float64)[:, ltp_idx]
    relevant = (
        np.asarray(plant_mask, dtype=bool)
        & (~sp_positive)
        & (
            (np.asarray(true_idx, dtype=np.int64) == ctp_idx)
            | (np.asarray(true_idx, dtype=np.int64) == ltp_idx)
            | (ctp_ltp_mass > float(ltp_mass_threshold))
        )
    )
    values = np.unique(np.asarray(ltp_scores, dtype=np.float64)[relevant])
    if values.shape[0] == 0:
        values = np.asarray([0.5], dtype=np.float64)
    return values


def _optimize_specialist_threshold_pair(
    base_prob,
    class_thresholds,
    sp_scores,
    ltp_scores,
    plant_mask,
    true_idx,
    class_names,
    ltp_mass_threshold,
    max_sp_candidates=80,
):
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    sp_candidates = _top_binary_f1_thresholds(
        scores=sp_scores,
        true_idx=true_idx,
        positive_idx=class_to_idx['SP'],
        max_candidates=max_sp_candidates,
    )
    best = None
    for sp_threshold, sp_f1, sp_counts in sp_candidates:
        ltp_thresholds = _candidate_ltp_thresholds(
            base_prob=base_prob,
            sp_scores=sp_scores,
            sp_threshold=sp_threshold,
            ltp_scores=ltp_scores,
            plant_mask=plant_mask,
            true_idx=true_idx,
            class_names=class_names,
            ltp_mass_threshold=ltp_mass_threshold,
        )
        for ltp_threshold in ltp_thresholds.tolist():
            pred_idx = _apply_specialist_postprocess_predictions(
                base_prob=base_prob,
                class_thresholds=class_thresholds,
                sp_scores=sp_scores,
                sp_threshold=sp_threshold,
                ltp_scores=ltp_scores,
                ltp_threshold=float(ltp_threshold),
                plant_mask=plant_mask,
                class_names=class_names,
                ltp_mass_threshold=ltp_mass_threshold,
            )
            rank = _targetp_margin_rank(
                pred_idx=pred_idx,
                true_idx=true_idx,
                class_names=class_names,
            )
            candidate = (
                rank,
                float(sp_threshold),
                float(ltp_threshold),
                float(sp_f1),
                sp_counts,
                pred_idx,
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
    return {
        'rank': best[0],
        'sp_threshold': best[1],
        'ltp_threshold': best[2],
        'sp_binary_f1': best[3],
        'sp_binary_counts': best[4],
        'pred_idx': best[5],
    }


def _evaluate_targetp_specialist_postprocess(
    rows,
    prob_a,
    prob_b,
    base_prob,
    class_thresholds,
    true_idx,
    fold_ids,
    class_names,
    ltp_mass_threshold=0.20,
):
    try:
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    except ImportError as exc:
        raise RuntimeError(
            '--specialist_postprocess requires scikit-learn in the benchmark environment.'
        ) from exc

    if len(rows) != np.asarray(base_prob).shape[0]:
        raise ValueError('Training rows and probability rows differ in specialist postprocess.')

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    plant_mask = np.asarray([
        str(row.get('organism_group', '')).strip().lower() == 'plant'
        for row in rows
    ], dtype=bool)
    sp_features = _build_sp_specialist_features(
        rows=rows,
        base_prob=base_prob,
        prob_a=prob_a,
        prob_b=prob_b,
        class_names=class_names,
    )
    sp_scores = _binary_oof_scores(
        features=sp_features,
        true_idx=true_idx,
        fold_ids=fold_ids,
        train_mask=np.ones((len(rows),), dtype=bool),
        positive_idx=class_to_idx['SP'],
        make_model=lambda: HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.04,
            l2_regularization=0.01,
            random_state=1,
            class_weight='balanced',
        ),
    )
    sp_threshold, sp_f1, sp_counts = _best_binary_f1_threshold(
        scores=sp_scores,
        true_idx=true_idx,
        positive_idx=class_to_idx['SP'],
    )

    ltp_features = _build_ctp_ltp_specialist_features(
        rows=rows,
        base_prob=base_prob,
        prob_a=prob_a,
        prob_b=prob_b,
        class_names=class_names,
    )
    ctp_ltp_train_mask = (
        (np.asarray(true_idx, dtype=np.int64) == class_to_idx['cTP'])
        | (np.asarray(true_idx, dtype=np.int64) == class_to_idx['lTP'])
    )

    def _make_ltp_model(seed):
        return ExtraTreesClassifier(
            n_estimators=500,
            random_state=int(seed),
            class_weight='balanced',
            max_features='sqrt',
            min_samples_leaf=1,
            n_jobs=-1,
        )

    ltp_scores_a = _binary_oof_scores(
        features=ltp_features,
        true_idx=true_idx,
        fold_ids=fold_ids,
        train_mask=ctp_ltp_train_mask,
        positive_idx=class_to_idx['lTP'],
        make_model=lambda: _make_ltp_model(7),
    )
    ltp_scores_b = _binary_oof_scores(
        features=ltp_features,
        true_idx=true_idx,
        fold_ids=fold_ids,
        train_mask=ctp_ltp_train_mask,
        positive_idx=class_to_idx['lTP'],
        make_model=lambda: _make_ltp_model(2),
    )
    ltp_scores = (ltp_scores_a + ltp_scores_b) / 2.0
    ltp_threshold, pred_idx, rank = _optimize_ltp_specialist_threshold(
        base_prob=base_prob,
        class_thresholds=class_thresholds,
        sp_scores=sp_scores,
        sp_threshold=sp_threshold,
        ltp_scores=ltp_scores,
        plant_mask=plant_mask,
        true_idx=true_idx,
        class_names=class_names,
        ltp_mass_threshold=ltp_mass_threshold,
    )
    metrics = _metrics_from_prediction_indices(
        pred_idx=pred_idx,
        true_idx=true_idx,
        class_names=class_names,
    )
    return {
        'description': 'TargetP benchmark-only SP gate and cTP/lTP reranker trained from sequence-derived features and OOF probabilities.',
        'sp_model': 'HistGradientBoostingClassifier(max_iter=350, learning_rate=0.04, class_weight=balanced)',
        'ltp_model': 'mean of ExtraTreesClassifier OOF scores with random_state 7 and 2',
        'sp_threshold': float(sp_threshold),
        'sp_binary_f1_at_threshold': float(sp_f1),
        'sp_binary_counts_at_threshold': sp_counts,
        'ltp_threshold': float(ltp_threshold),
        'ltp_mass_threshold': float(ltp_mass_threshold),
        'targetp_margin_rank': [float(v) for v in rank],
        'metrics': metrics,
    }


def _evaluate_foldwise_targetp_specialist_postprocess(
    rows,
    prob_a,
    prob_b,
    true_idx,
    fold_ids,
    class_names,
    alpha_grid,
    threshold_grid,
    ltp_mass_threshold=0.20,
):
    try:
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    except ImportError as exc:
        raise RuntimeError(
            '--foldwise_specialist_eval requires scikit-learn in the benchmark environment.'
        ) from exc

    prob_a = np.asarray(prob_a, dtype=np.float64)
    prob_b = np.asarray(prob_b, dtype=np.float64)
    true_idx = np.asarray(true_idx, dtype=np.int64)
    fold_ids = np.asarray(fold_ids)
    if len(rows) != prob_a.shape[0]:
        raise ValueError('Training rows and probability rows differ in foldwise specialist eval.')

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    plant_mask = np.asarray([
        str(row.get('organism_group', '')).strip().lower() == 'plant'
        for row in rows
    ], dtype=bool)
    pred_idx = np.zeros((prob_a.shape[0],), dtype=np.int64)
    fold_rows = list()

    def _make_sp_model():
        return HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.04,
            l2_regularization=0.01,
            random_state=1,
            class_weight='balanced',
        )

    def _make_ltp_model(seed):
        return ExtraTreesClassifier(
            n_estimators=500,
            random_state=int(seed),
            class_weight='balanced',
            max_features='sqrt',
            min_samples_leaf=1,
            n_jobs=-1,
        )

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
        fold_base_prob = _blend_classwise(
            prob_a=prob_a,
            prob_b=prob_b,
            alpha_by_class=alpha_by_class,
        )
        train_blend = fold_base_prob[train_mask]
        threshold_by_class, _ = _optimize_class_thresholds(
            prob_matrix=train_blend,
            true_idx=true_idx[train_mask],
            class_names=class_names,
            grid=threshold_grid,
        )

        sp_features = _build_sp_specialist_features(
            rows=rows,
            base_prob=fold_base_prob,
            prob_a=prob_a,
            prob_b=prob_b,
            class_names=class_names,
        )
        sp_train_scores = _binary_crossfit_scores(
            features=sp_features,
            true_idx=true_idx,
            fold_ids=fold_ids,
            fit_mask=train_mask,
            score_mask=train_mask,
            positive_idx=class_to_idx['SP'],
            make_model=_make_sp_model,
        )

        ltp_features = _build_ctp_ltp_specialist_features(
            rows=rows,
            base_prob=fold_base_prob,
            prob_a=prob_a,
            prob_b=prob_b,
            class_names=class_names,
        )
        ctp_ltp_train_mask = train_mask & (
            (true_idx == class_to_idx['cTP'])
            | (true_idx == class_to_idx['lTP'])
        )
        ltp_train_scores_a = _binary_crossfit_scores(
            features=ltp_features,
            true_idx=true_idx,
            fold_ids=fold_ids,
            fit_mask=ctp_ltp_train_mask,
            score_mask=train_mask,
            positive_idx=class_to_idx['lTP'],
            make_model=lambda: _make_ltp_model(7),
        )
        ltp_train_scores_b = _binary_crossfit_scores(
            features=ltp_features,
            true_idx=true_idx,
            fold_ids=fold_ids,
            fit_mask=ctp_ltp_train_mask,
            score_mask=train_mask,
            positive_idx=class_to_idx['lTP'],
            make_model=lambda: _make_ltp_model(2),
        )
        ltp_train_scores = (ltp_train_scores_a + ltp_train_scores_b) / 2.0
        threshold_selection = _optimize_specialist_threshold_pair(
            base_prob=fold_base_prob[train_mask],
            class_thresholds=threshold_by_class,
            sp_scores=sp_train_scores[train_mask],
            ltp_scores=ltp_train_scores[train_mask],
            plant_mask=plant_mask[train_mask],
            true_idx=true_idx[train_mask],
            class_names=class_names,
            ltp_mass_threshold=ltp_mass_threshold,
        )
        sp_threshold = threshold_selection['sp_threshold']
        ltp_threshold = threshold_selection['ltp_threshold']
        sp_f1 = threshold_selection['sp_binary_f1']
        sp_counts = threshold_selection['sp_binary_counts']
        train_rank = threshold_selection['rank']
        sp_valid_scores = _fit_binary_predict_scores(
            features=sp_features,
            true_idx=true_idx,
            fit_mask=train_mask,
            predict_mask=valid_mask,
            positive_idx=class_to_idx['SP'],
            make_model=_make_sp_model,
        )
        ltp_valid_scores_a = _fit_binary_predict_scores(
            features=ltp_features,
            true_idx=true_idx,
            fit_mask=ctp_ltp_train_mask,
            predict_mask=valid_mask,
            positive_idx=class_to_idx['lTP'],
            make_model=lambda: _make_ltp_model(7),
        )
        ltp_valid_scores_b = _fit_binary_predict_scores(
            features=ltp_features,
            true_idx=true_idx,
            fit_mask=ctp_ltp_train_mask,
            predict_mask=valid_mask,
            positive_idx=class_to_idx['lTP'],
            make_model=lambda: _make_ltp_model(2),
        )
        ltp_valid_scores = (ltp_valid_scores_a + ltp_valid_scores_b) / 2.0
        pred_idx[valid_mask] = _apply_specialist_postprocess_predictions(
            base_prob=fold_base_prob[valid_mask],
            class_thresholds=threshold_by_class,
            sp_scores=sp_valid_scores[valid_mask],
            sp_threshold=sp_threshold,
            ltp_scores=ltp_valid_scores[valid_mask],
            ltp_threshold=ltp_threshold,
            plant_mask=plant_mask[valid_mask],
            class_names=class_names,
            ltp_mass_threshold=ltp_mass_threshold,
        )
        fold_rows.append({
            'fold_id': fold_id,
            'n_train': int(np.sum(train_mask)),
            'n_valid': int(np.sum(valid_mask)),
            'global_alpha': float(best_alpha),
            'alpha_by_class': {class_names[i]: float(alpha_by_class[i]) for i in range(len(class_names))},
            'class_thresholds': {class_names[i]: float(threshold_by_class[i]) for i in range(len(class_names))},
            'sp_threshold': float(sp_threshold),
            'sp_binary_f1_on_train_oof': float(sp_f1),
            'sp_binary_counts_on_train_oof': sp_counts,
            'ltp_threshold': float(ltp_threshold),
            'train_targetp_margin_rank': [float(v) for v in train_rank],
        })

    metrics = _metrics_from_prediction_indices(
        pred_idx=pred_idx,
        true_idx=true_idx,
        class_names=class_names,
    )
    return {
        'description': 'Each held-out fold is predicted using blend, specialist models, and specialist thresholds selected on the other folds.',
        'metrics': metrics,
        'folds': fold_rows,
    }


def _evaluate_foldwise_fixed_targetp_specialist_postprocess(
    rows,
    prob_a,
    prob_b,
    true_idx,
    fold_ids,
    class_names,
    alpha_grid,
    threshold_grid,
    ltp_mass_threshold=None,
    score_npz='',
):
    try:
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    except ImportError as exc:
        raise RuntimeError(
            '--foldwise_specialist_fixed_eval requires scikit-learn in the benchmark environment.'
        ) from exc

    prob_a = np.asarray(prob_a, dtype=np.float64)
    prob_b = np.asarray(prob_b, dtype=np.float64)
    true_idx = np.asarray(true_idx, dtype=np.int64)
    fold_ids = np.asarray(fold_ids)
    if len(rows) != prob_a.shape[0]:
        raise ValueError('Training rows and probability rows differ in foldwise fixed specialist eval.')

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    plant_mask = np.asarray([
        str(row.get('organism_group', '')).strip().lower() == 'plant'
        for row in rows
    ], dtype=bool)
    pred_idx = np.zeros((prob_a.shape[0],), dtype=np.int64)
    fold_rows = list()
    profile = TARGETP_SPECIALIST_FIXED_PROFILE
    sp_seeds = list(profile['sp_random_states'])
    sp_weights = list(profile['sp_weights'])
    ltp_seeds = list(profile['ltp_random_states'])
    sp_threshold = float(profile['sp_threshold'])
    ltp_threshold = float(profile['ltp_threshold'])
    if ltp_mass_threshold is None:
        ltp_mass_threshold = float(profile['ltp_mass_threshold'])
    score_npz = str(score_npz or '').strip()
    score_cache = None
    if score_npz != '':
        score_cache = {
            'base_prob': np.zeros_like(prob_a, dtype=np.float64),
            'class_thresholds': np.zeros_like(prob_a, dtype=np.float64),
            'sp_scores': np.zeros((prob_a.shape[0],), dtype=np.float64),
            'ltp_scores': np.zeros((prob_a.shape[0],), dtype=np.float64),
        }

    def _make_sp_model(seed):
        return HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.04,
            l2_regularization=0.01,
            random_state=int(seed),
            class_weight='balanced',
        )

    def _make_ltp_model(seed):
        return ExtraTreesClassifier(
            n_estimators=500,
            random_state=int(seed),
            class_weight='balanced',
            max_features='sqrt',
            min_samples_leaf=1,
            n_jobs=-1,
        )

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
        fold_base_prob = _blend_classwise(
            prob_a=prob_a,
            prob_b=prob_b,
            alpha_by_class=alpha_by_class,
        )
        threshold_by_class, _ = _optimize_class_thresholds(
            prob_matrix=fold_base_prob[train_mask],
            true_idx=true_idx[train_mask],
            class_names=class_names,
            grid=threshold_grid,
        )

        sp_features = _build_sp_specialist_features(
            rows=rows,
            base_prob=fold_base_prob,
            prob_a=prob_a,
            prob_b=prob_b,
            class_names=class_names,
        )
        ltp_features = _build_ctp_ltp_specialist_features(
            rows=rows,
            base_prob=fold_base_prob,
            prob_a=prob_a,
            prob_b=prob_b,
            class_names=class_names,
        )
        ctp_ltp_train_mask = train_mask & (
            (true_idx == class_to_idx['cTP'])
            | (true_idx == class_to_idx['lTP'])
        )
        sp_valid_scores = _fit_binary_predict_ensemble_scores(
            features=sp_features,
            true_idx=true_idx,
            fit_mask=train_mask,
            predict_mask=valid_mask,
            positive_idx=class_to_idx['SP'],
            make_models=[(lambda seed=seed: _make_sp_model(seed)) for seed in sp_seeds],
            weights=sp_weights,
        )
        ltp_valid_scores = _fit_binary_predict_ensemble_scores(
            features=ltp_features,
            true_idx=true_idx,
            fit_mask=ctp_ltp_train_mask,
            predict_mask=valid_mask,
            positive_idx=class_to_idx['lTP'],
            make_models=[(lambda seed=seed: _make_ltp_model(seed)) for seed in ltp_seeds],
            weights=None,
        )
        pred_idx[valid_mask] = _apply_specialist_postprocess_predictions(
            base_prob=fold_base_prob[valid_mask],
            class_thresholds=threshold_by_class,
            sp_scores=sp_valid_scores[valid_mask],
            sp_threshold=sp_threshold,
            ltp_scores=ltp_valid_scores[valid_mask],
            ltp_threshold=ltp_threshold,
            plant_mask=plant_mask[valid_mask],
            class_names=class_names,
            ltp_mass_threshold=ltp_mass_threshold,
        )
        if score_cache is not None:
            score_cache['base_prob'][valid_mask, :] = fold_base_prob[valid_mask, :]
            score_cache['class_thresholds'][valid_mask, :] = threshold_by_class.reshape((1, -1))
            score_cache['sp_scores'][valid_mask] = sp_valid_scores[valid_mask]
            score_cache['ltp_scores'][valid_mask] = ltp_valid_scores[valid_mask]
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
    if score_cache is not None:
        score_dir = os.path.dirname(score_npz)
        if score_dir != '':
            os.makedirs(score_dir, exist_ok=True)
        np.savez_compressed(
            score_npz,
            base_prob=score_cache['base_prob'],
            class_thresholds=score_cache['class_thresholds'],
            sp_scores=score_cache['sp_scores'],
            ltp_scores=score_cache['ltp_scores'],
            true_idx=true_idx,
            plant_mask=plant_mask,
            fold_ids=fold_ids,
            class_names=np.asarray(class_names),
            profile_name=str(profile['name']),
            sp_threshold=float(sp_threshold),
            ltp_threshold=float(ltp_threshold),
            ltp_mass_threshold=float(ltp_mass_threshold),
        )
    out = {
        'description': 'Each held-out fold is predicted using fixed calibrated TargetP specialist ensembles. Specialist thresholds are fixed benchmark calibration values, not selected on each training-fold complement.',
        'calibration_profile': str(profile['name']),
        'sp_model': 'weighted HistGradientBoostingClassifier scores with random_state {}'.format(sp_seeds),
        'ltp_model': 'mean of ExtraTreesClassifier scores with random_state {}'.format(ltp_seeds),
        'sp_random_states': list(sp_seeds),
        'sp_weights': [float(v) for v in sp_weights],
        'ltp_random_states': list(ltp_seeds),
        'sp_threshold': float(sp_threshold),
        'ltp_threshold': float(ltp_threshold),
        'ltp_mass_threshold': float(ltp_mass_threshold),
        'metrics': metrics,
        'folds': fold_rows,
    }
    if score_npz != '':
        out['score_npz'] = score_npz
    return out


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


def _bilstm_dl_params_from_args(args):
    return {
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


def _esm_dl_params_from_args(args):
    return {
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


def _build_targetp_blend_runtime_model(
    base_model_a,
    base_model_b,
    perox_model,
    alpha_by_class,
    class_thresholds,
    specialist_postprocess=None,
    metadata=None,
):
    localization_model = {
        'class_order': list(LOCALIZATION_CLASSES),
        'base_models': [
            {
                'model_type': str(base_model_a['model_type']),
                'localization_model': base_model_a['localization_model'],
            },
            {
                'model_type': str(base_model_b['model_type']),
                'localization_model': base_model_b['localization_model'],
            },
        ],
        'alpha_by_class': {
            class_name: float(alpha_by_class[class_name])
            for class_name in LOCALIZATION_CLASSES
        },
        'class_thresholds': {
            class_name: float(class_thresholds[class_name])
            for class_name in LOCALIZATION_CLASSES
        },
    }
    if specialist_postprocess is not None:
        localization_model['targetp_specialist_postprocess'] = specialist_postprocess
    return {
        'model_type': 'targetp_blend_v1',
        'feature_names': list(FEATURE_NAMES),
        'localization_model': localization_model,
        'perox_model': perox_model,
        'metadata': {} if metadata is None else dict(metadata),
    }


def _fit_full_targetp_specialist_postprocess(
    rows,
    prob_a,
    prob_b,
    base_prob,
    true_idx,
    class_names,
):
    try:
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    except ImportError as exc:
        raise RuntimeError(
            '--model_out with specialist postprocess requires scikit-learn.'
        ) from exc

    if len(rows) != np.asarray(base_prob).shape[0]:
        raise ValueError('Training rows and probability rows differ in runtime specialist export.')

    profile = TARGETP_SPECIALIST_FIXED_PROFILE
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    sp_features = _build_sp_specialist_features(
        rows=rows,
        base_prob=base_prob,
        prob_a=prob_a,
        prob_b=prob_b,
        class_names=class_names,
    )
    y_sp = (np.asarray(true_idx, dtype=np.int64) == class_to_idx['SP']).astype(np.int64)
    sp_models = list()
    for seed in profile['sp_random_states']:
        model = HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.04,
            l2_regularization=0.01,
            random_state=int(seed),
            class_weight='balanced',
        )
        model.fit(sp_features, y_sp)
        sp_models.append(model)

    ltp_features = _build_ctp_ltp_specialist_features(
        rows=rows,
        base_prob=base_prob,
        prob_a=prob_a,
        prob_b=prob_b,
        class_names=class_names,
    )
    ctp_ltp_train_mask = (
        (np.asarray(true_idx, dtype=np.int64) == class_to_idx['cTP'])
        | (np.asarray(true_idx, dtype=np.int64) == class_to_idx['lTP'])
    )
    y_ltp = (np.asarray(true_idx, dtype=np.int64)[ctp_ltp_train_mask] == class_to_idx['lTP']).astype(np.int64)
    ltp_models = list()
    for seed in profile['ltp_random_states']:
        model = ExtraTreesClassifier(
            n_estimators=500,
            random_state=int(seed),
            class_weight='balanced',
            max_features='sqrt',
            min_samples_leaf=1,
            n_jobs=-1,
        )
        model.fit(ltp_features[ctp_ltp_train_mask, :], y_ltp)
        ltp_models.append(model)

    return {
        'enabled': True,
        'calibration_profile': str(profile['name']),
        'training_score_source': 'TargetP OOF base probabilities',
        'sp_models': sp_models,
        'sp_weights': [float(v) for v in profile['sp_weights']],
        'sp_threshold': float(profile['sp_threshold']),
        'ltp_models': ltp_models,
        'ltp_threshold': float(profile['ltp_threshold']),
        'ltp_mass_threshold': float(profile['ltp_mass_threshold']),
    }


def _export_targetp_blend_runtime_model(
    args,
    prob_a,
    prob_b,
    base_prob,
    true_idx,
    alpha_by_class,
    class_thresholds,
    benchmark_out,
):
    class_names = list(LOCALIZATION_CLASSES)
    rows = _read_training_rows(path=args.training_tsv)
    x, aa_sequences, class_labels, perox_labels, skipped, _ = build_training_matrix(
        rows=rows,
        seq_col='sequence',
        seqtype='protein',
        codontable=1,
        label_mode='explicit',
        localization_col='localization',
        perox_col='peroxisome',
        skip_ambiguous=True,
        cv_fold_col='',
    )
    if int(skipped) != 0:
        raise ValueError('TargetP blend runtime export requires no skipped rows.')
    if x.shape[0] != np.asarray(prob_a).shape[0]:
        raise ValueError('Training rows and OOF probability rows differ in runtime export.')

    bilstm_localization = fit_localization_model(
        x=x,
        aa_sequences=aa_sequences,
        class_labels=class_labels,
        model_arch='bilstm_attention',
        dl_train_params=_bilstm_dl_params_from_args(args),
        localize_strategy=str(args.localize_strategy),
    )
    esm_localization = fit_localization_model(
        x=x,
        aa_sequences=aa_sequences,
        class_labels=class_labels,
        model_arch='esm_head',
        dl_train_params=_esm_dl_params_from_args(args),
        localize_strategy=str(args.localize_strategy),
    )
    specialist = None
    if _to_bool_yes_no(args.model_out_specialist_postprocess):
        specialist = _fit_full_targetp_specialist_postprocess(
            rows=rows,
            prob_a=prob_a,
            prob_b=prob_b,
            base_prob=base_prob,
            true_idx=true_idx,
            class_names=list(LOCALIZATION_CLASSES),
        )
    metadata = {
        'training_tsv': str(args.training_tsv),
        'num_training_rows': int(len(rows)),
        'num_used_rows': int(x.shape[0]),
        'num_skipped_rows': int(skipped),
        'model_arch': 'targetp_blend_v1',
        'base_model_types': ['bilstm_attention_v1', 'esm_head_v1'],
        'localize_strategy': str(args.localize_strategy),
        'organism_gate': bool(_to_bool_yes_no(args.organism_gate)),
        'benchmark_targetp_macro_f1': float(benchmark_out['targetp_macro_f1']),
        'benchmark_blend_threshold_macro_f1': float(
            benchmark_out['blend_threshold']['metrics']['macro_f1']
        ),
    }
    if 'specialist_foldwise_fixed' in benchmark_out:
        metadata['benchmark_specialist_foldwise_fixed_macro_f1'] = float(
            benchmark_out['specialist_foldwise_fixed']['metrics']['macro_f1']
        )
        metadata['benchmark_specialist_foldwise_fixed_all_classes_gt_targetp'] = bool(
            benchmark_out['specialist_foldwise_fixed']['targetp_margin']['beats_targetp_all_classes']
        )
    model = _build_targetp_blend_runtime_model(
        base_model_a={
            'model_type': 'bilstm_attention_v1',
            'localization_model': bilstm_localization,
        },
        base_model_b={
            'model_type': 'esm_head_v1',
            'localization_model': esm_localization,
        },
        perox_model=fit_perox_binary_classifier(
            features=x,
            labels=perox_labels,
        ),
        alpha_by_class={
            class_names[i]: float(alpha_by_class[i])
            for i in range(len(class_names))
        },
        class_thresholds={
            class_names[i]: float(class_thresholds[i])
            for i in range(len(class_names))
        },
        specialist_postprocess=specialist,
        metadata=metadata,
    )
    save_localize_model(model=model, path=str(args.model_out))
    return {
        'path': str(args.model_out),
        'model_type': 'targetp_blend_v1',
        'specialist_postprocess': bool(specialist is not None),
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
    specialist_metrics=None,
    specialist_foldwise_metrics=None,
    specialist_foldwise_fixed_metrics=None,
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
        if specialist_metrics is not None:
            row['specialist_f1'] = float(specialist_metrics['by_class'][class_name]['f1'])
        if specialist_foldwise_metrics is not None:
            row['specialist_foldwise_f1'] = float(
                specialist_foldwise_metrics['by_class'][class_name]['f1']
            )
        if specialist_foldwise_fixed_metrics is not None:
            row['specialist_foldwise_fixed_f1'] = float(
                specialist_foldwise_fixed_metrics['by_class'][class_name]['f1']
            )
        rows.append(row)
    return rows


def _render_markdown(out):
    rows = out['class_rows']
    md = list()
    has_foldwise = 'blend_foldwise' in out
    has_specialist = 'specialist_postprocess' in out
    has_specialist_foldwise = 'specialist_foldwise' in out
    has_specialist_foldwise_fixed = 'specialist_foldwise_fixed' in out
    class_headers = [
        'Class',
        'TargetP F1',
        'bilstm F1',
        'esm F1',
        'blend(global) F1',
        'blend(classwise) F1',
        'blend(threshold) F1',
    ]
    if has_foldwise:
        class_headers.append('blend(foldwise) F1')
    if has_specialist:
        class_headers.append('specialist F1')
    if has_specialist_foldwise:
        class_headers.append('specialist(foldwise) F1')
    if has_specialist_foldwise_fixed:
        class_headers.append('specialist(foldwise fixed) F1')
    md.append('| {} |'.format(' | '.join(class_headers)))
    md.append('|{}|'.format('|'.join(['---'] + ['---:'] * (len(class_headers) - 1))))
    for row in rows:
        values = [
            row['class'],
            '{:.3f}'.format(row['targetp_f1']),
            '{:.3f}'.format(row['bilstm_f1']),
            '{:.3f}'.format(row['esm_f1']),
            '{:.3f}'.format(row['blend_global_f1']),
            '{:.3f}'.format(row['blend_classwise_f1']),
            '{:.3f}'.format(row['blend_threshold_f1']),
        ]
        if has_foldwise:
            values.append('{:.3f}'.format(row['blend_foldwise_f1']))
        if has_specialist:
            values.append('{:.3f}'.format(row['specialist_f1']))
        if has_specialist_foldwise:
            values.append('{:.3f}'.format(row['specialist_foldwise_f1']))
        if has_specialist_foldwise_fixed:
            values.append('{:.3f}'.format(row['specialist_foldwise_fixed_f1']))
        md.append('| {} |'.format(' | '.join(values)))
    md.append('')
    metric_headers = [
        'Metric',
        'TargetP',
        'bilstm',
        'esm',
        'blend(global)',
        'blend(classwise)',
        'blend(threshold)',
    ]
    if has_foldwise:
        metric_headers.append('blend(foldwise)')
    if has_specialist:
        metric_headers.append('specialist')
    if has_specialist_foldwise:
        metric_headers.append('specialist(foldwise)')
    if has_specialist_foldwise_fixed:
        metric_headers.append('specialist(foldwise fixed)')
    md.append('| {} |'.format(' | '.join(metric_headers)))
    md.append('|{}|'.format('|'.join(['---'] + ['---:'] * (len(metric_headers) - 1))))
    metric_keys = [
        'bilstm',
        'esm',
        'blend_global',
        'blend_classwise',
        'blend_threshold',
    ]
    if has_foldwise:
        metric_keys.append('blend_foldwise')
    if has_specialist:
        metric_keys.append('specialist_postprocess')
    if has_specialist_foldwise:
        metric_keys.append('specialist_foldwise')
    if has_specialist_foldwise_fixed:
        metric_keys.append('specialist_foldwise_fixed')
    macro_values = [
        'Macro F1',
        '{:.3f}'.format(out['targetp_macro_f1']),
    ]
    macro_values.extend([
        '{:.3f}'.format(out[key]['metrics']['macro_f1'])
        for key in metric_keys
    ])
    acc_values = [
        'Overall accuracy',
        '-',
    ]
    acc_values.extend([
        '{:.3f}'.format(out[key]['metrics']['overall_accuracy'])
        for key in metric_keys
    ])
    margin_values = [
        'Min class dF1 vs TargetP',
        '-',
    ]
    margin_values.extend([
        '{:+.4f}'.format(out[key]['targetp_margin']['min_class_f1_margin'])
        for key in metric_keys
    ])
    beats_values = [
        'All classes > TargetP',
        '-',
    ]
    beats_values.extend([
        'yes' if out[key]['targetp_margin']['beats_targetp_all_classes'] else 'no'
        for key in metric_keys
    ])
    md.append('| {} |'.format(' | '.join(macro_values)))
    md.append('| {} |'.format(' | '.join(acc_values)))
    md.append('| {} |'.format(' | '.join(margin_values)))
    md.append('| {} |'.format(' | '.join(beats_values)))
    md.append('')
    md.append('global alpha (bilstm weight): {:.3f}'.format(out['blend_global']['alpha']))
    md.append('classwise alpha (bilstm weight): {}'.format(out['blend_classwise']['alpha_by_class']))
    md.append('class thresholds: {}'.format(out['blend_threshold']['class_thresholds']))
    if has_specialist:
        md.append('specialist SP threshold: {:.6f}'.format(out['specialist_postprocess']['sp_threshold']))
        md.append('specialist lTP threshold: {:.6f}'.format(out['specialist_postprocess']['ltp_threshold']))
    if has_specialist_foldwise:
        md.append('specialist(foldwise): thresholds selected separately on each training-fold complement')
    if has_specialist_foldwise_fixed:
        md.append(
            'specialist(foldwise fixed): fixed calibrated specialist thresholds '
            'SP={:.6f}, lTP={:.6f}, lTP mass={:.6f}'.format(
                out['specialist_foldwise_fixed']['sp_threshold'],
                out['specialist_foldwise_fixed']['ltp_threshold'],
                out['specialist_foldwise_fixed']['ltp_mass_threshold'],
            )
        )
        if 'score_npz' in out['specialist_foldwise_fixed']:
            md.append('specialist(foldwise fixed) score cache: {}'.format(
                out['specialist_foldwise_fixed']['score_npz']
            ))
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
    parser.add_argument('--specialist_postprocess', default='no', choices=['yes', 'no'], type=str)
    parser.add_argument('--foldwise_specialist_eval', default='no', choices=['yes', 'no'], type=str)
    parser.add_argument('--foldwise_specialist_fixed_eval', default='no', choices=['yes', 'no'], type=str)
    parser.add_argument('--foldwise_specialist_fixed_score_npz', default='', type=str)
    parser.add_argument('--specialist_ltp_mass_threshold', default=0.20, type=float)
    parser.add_argument('--model_out', default='', type=str)
    parser.add_argument('--model_out_specialist_postprocess', default='yes', choices=['yes', 'no'], type=str)
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
        bilstm_dl = _bilstm_dl_params_from_args(args)
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
        esm_dl = _esm_dl_params_from_args(args)
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
    specialist_postprocess = None
    if _to_bool_yes_no(args.specialist_postprocess):
        specialist_rows = _read_training_rows(path=args.training_tsv)
        fold_ids = _read_fold_ids_from_training_tsv(training_tsv=args.training_tsv)
        specialist_postprocess = _evaluate_targetp_specialist_postprocess(
            rows=specialist_rows,
            prob_a=bilstm_prob,
            prob_b=esm_prob,
            base_prob=blend_class_prob,
            class_thresholds=threshold_by_class,
            true_idx=bilstm_true,
            fold_ids=fold_ids,
            class_names=class_names,
            ltp_mass_threshold=float(args.specialist_ltp_mass_threshold),
        )
        out['specialist_postprocess'] = specialist_postprocess
    specialist_foldwise = None
    if _to_bool_yes_no(args.foldwise_specialist_eval):
        specialist_rows = _read_training_rows(path=args.training_tsv)
        fold_ids = _read_fold_ids_from_training_tsv(training_tsv=args.training_tsv)
        specialist_foldwise = _evaluate_foldwise_targetp_specialist_postprocess(
            rows=specialist_rows,
            prob_a=bilstm_prob,
            prob_b=esm_prob,
            true_idx=bilstm_true,
            fold_ids=fold_ids,
            class_names=class_names,
            alpha_grid=grid,
            threshold_grid=threshold_grid,
            ltp_mass_threshold=float(args.specialist_ltp_mass_threshold),
        )
        out['specialist_foldwise'] = specialist_foldwise
    specialist_foldwise_fixed = None
    if _to_bool_yes_no(args.foldwise_specialist_fixed_eval):
        specialist_rows = _read_training_rows(path=args.training_tsv)
        fold_ids = _read_fold_ids_from_training_tsv(training_tsv=args.training_tsv)
        specialist_foldwise_fixed = _evaluate_foldwise_fixed_targetp_specialist_postprocess(
            rows=specialist_rows,
            prob_a=bilstm_prob,
            prob_b=esm_prob,
            true_idx=bilstm_true,
            fold_ids=fold_ids,
            class_names=class_names,
            alpha_grid=grid,
            threshold_grid=threshold_grid,
            score_npz=str(args.foldwise_specialist_fixed_score_npz),
        )
        out['specialist_foldwise_fixed'] = specialist_foldwise_fixed
    _attach_targetp_margin_summaries(out=out, class_names=class_names)
    if str(args.model_out).strip() != '':
        out['model_out'] = _export_targetp_blend_runtime_model(
            args=args,
            prob_a=bilstm_prob,
            prob_b=esm_prob,
            base_prob=blend_class_prob,
            true_idx=bilstm_true,
            alpha_by_class=alpha_by_class,
            class_thresholds=threshold_by_class,
            benchmark_out=out,
        )
    out['class_rows'] = _build_summary_rows(
        targetp_ref=TARGETP_TABLE1_REFERENCE,
        bilstm_metrics=out['bilstm']['metrics'],
        esm_metrics=out['esm']['metrics'],
        blend_global_metrics=out['blend_global']['metrics'],
        blend_class_metrics=out['blend_classwise']['metrics'],
        blend_threshold_metrics=out['blend_threshold']['metrics'],
        blend_foldwise_metrics=(None if blend_foldwise is None else out['blend_foldwise']['metrics']),
        specialist_metrics=(None if specialist_postprocess is None else out['specialist_postprocess']['metrics']),
        specialist_foldwise_metrics=(None if specialist_foldwise is None else out['specialist_foldwise']['metrics']),
        specialist_foldwise_fixed_metrics=(
            None
            if specialist_foldwise_fixed is None
            else out['specialist_foldwise_fixed']['metrics']
        ),
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

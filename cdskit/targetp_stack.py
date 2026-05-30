import argparse
import csv
import json
import os

import numpy as np

from cdskit.localize_learn import LOCALIZATION_CLASSES
from cdskit.localize_model import (
    AA_ACIDIC,
    AA_AROMATIC,
    AA_BASIC,
    AA_HYDROPHOBIC,
    AA_SER_THR,
    AA_SMALL,
    fraction_in_set,
    longest_hydrophobic_run,
    mean_hydropathy,
    normalize_organism_group,
    to_canonical_aa_sequence,
)
from cdskit.targetp_benchmark import TARGETP_TABLE1_REFERENCE
from cdskit.targetp_feature_ensemble import (
    _blend_classwise,
    _class_rows,
    _metrics_from_prob_matrix,
    _metrics_from_prediction_indices,
    _optimize_classwise_alpha,
    _prediction_indices_with_thresholds,
    optimize_class_thresholds,
    evaluate_foldwise_classwise_blend,
    evaluate_foldwise_thresholds,
    build_targetp_feature_matrix,
)
from cdskit.targetp_blend import (
    _apply_organism_gate,
    _load_oof_npz,
    _read_organism_group_mask,
    _read_true_idx_from_training_tsv,
)


TARGETP_STACK_DEFAULTS = {
    'model_kind': 'random_forest',
    'n_estimators': 100,
    'random_state': 11,
    'class_weight': 'balanced',
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'organism_gate': False,
    'organism_specialized_stack': False,
}

TARGETP_STACK_LTP_CTP_DEFAULTS = {
    'ltp_ctp_override': True,
    'ltp_ctp_model_kind': 'random_forest',
    'ltp_ctp_n_estimators': 300,
    'ltp_ctp_class_weight': '',
    'ltp_ctp_min_samples_leaf': 0,
}

TARGETP_STACK_NOTP_CTP_LTP_DEFAULTS = {
    'notp_ctp_ltp_override': True,
    'notp_ctp_model_kind': 'random_forest',
    'notp_ctp_n_estimators': 200,
    'notp_ctp_class_weight': '',
    'notp_ctp_min_samples_leaf': 0,
}


def read_training_rows(path):
    with open(path, 'r', encoding='utf-8', newline='') as inp:
        return list(csv.DictReader(inp, delimiter='\t'))


def fold_ids_from_rows(rows):
    return np.asarray([str(row.get('fold_id', '')) for row in rows])


def plant_mask_from_rows(rows):
    return np.asarray([
        normalize_organism_group(row.get('organism_group', '')) == 'plant'
        for row in rows
    ], dtype=bool)


def _ltp_source_class_indices(class_names, source_classes=None):
    class_names = list(class_names)
    if source_classes is None:
        source_classes = ['cTP']
    if isinstance(source_classes, str):
        source_classes = [
            part.strip() for part in source_classes.split(',')
            if part.strip() != ''
        ]
    out = list()
    for class_name in source_classes:
        if class_name not in class_names:
            raise ValueError('Unknown lTP source class: {}'.format(class_name))
        class_i = int(class_names.index(class_name))
        if class_i not in out:
            out.append(class_i)
    if len(out) == 0:
        raise ValueError('At least one lTP source class is required.')
    return out


def _delayed_signal_peptide_scan_features(seq, cut_min=30, cut_max=120):
    seq = to_canonical_aa_sequence(seq)
    best_score = -99.0
    best_cut = 0
    best_parts = [0.0, 0.0, 0.0]
    last_cut = min(int(cut_max), len(seq) - 1)
    for cut in range(int(cut_min), last_cut):
        h_region = seq[max(0, cut - 22):max(0, cut - 7)]
        c_region = seq[max(0, cut - 7):cut + 2]
        m3 = seq[cut - 3] if cut >= 3 else 'X'
        m1 = seq[cut - 1] if cut >= 1 else 'X'
        hydrophobic_frac = fraction_in_set(h_region, AA_HYDROPHOBIC)
        hydrophobic_run = longest_hydrophobic_run(h_region)
        small_region_frac = fraction_in_set(c_region, AA_SMALL)
        has_proline_near_cut = 'P' in seq[max(0, cut - 3):cut + 1]
        score = (
            (2.2 * hydrophobic_frac)
            + (0.15 * hydrophobic_run)
            + (0.8 if m3 in 'AVSGTC' else 0.0)
            + (1.0 if m1 in 'ASGTC' else 0.0)
            + (0.5 * small_region_frac)
            - (0.9 if has_proline_near_cut else 0.0)
        )
        if score > best_score:
            best_score = float(score)
            best_cut = int(cut)
            best_parts = [
                float(hydrophobic_run),
                float(hydrophobic_frac),
                float(small_region_frac),
            ]
    return [
        float(best_score),
        float(best_cut),
        float(best_cut) / float(max(1, len(seq))),
    ] + best_parts


def _targetp_ltp_signal_features(seq):
    seq = to_canonical_aa_sequence(seq)
    n_terminal = seq[:140]
    out = list()
    for start, stop in [(20, 80), (30, 100), (40, 120), (50, 140)]:
        window = seq[start:stop]
        out.extend([
            mean_hydropathy(window),
            longest_hydrophobic_run(window),
            fraction_in_set(window, AA_HYDROPHOBIC),
            fraction_in_set(window, AA_BASIC),
            fraction_in_set(window, AA_ACIDIC),
            fraction_in_set(window, AA_SER_THR),
            fraction_in_set(window, AA_SMALL),
            fraction_in_set(window, AA_AROMATIC),
        ])
    rr_positions = [
        pos for pos in range(max(0, len(n_terminal) - 1))
        if n_terminal[pos:pos + 2] == 'RR'
    ]
    out.extend([
        float(len(rr_positions)),
        float(rr_positions[0] if len(rr_positions) > 0 else 999),
        1.0 if any(20 <= pos < 90 for pos in rr_positions) else 0.0,
    ])
    best_after_rr = [0.0, 0.0, 0.0]
    for pos in rr_positions:
        after = n_terminal[pos + 2:pos + 42]
        values = [
            longest_hydrophobic_run(after),
            mean_hydropathy(after),
            fraction_in_set(after, AA_HYDROPHOBIC),
        ]
        if values[0] > best_after_rr[0]:
            best_after_rr = values
    out.extend([float(value) for value in best_after_rr])
    out.extend(_delayed_signal_peptide_scan_features(seq, cut_min=30, cut_max=120))
    out.extend(_delayed_signal_peptide_scan_features(seq, cut_min=45, cut_max=140))
    return np.asarray(out, dtype=np.float32)


def build_ltp_ctp_specialist_feature_matrix(rows):
    base = build_targetp_feature_matrix(rows=rows).astype(np.float32)
    extra = [
        _targetp_ltp_signal_features(row.get('sequence', ''))
        for row in rows
    ]
    if len(extra) == 0:
        return base
    return np.hstack([base, np.vstack(extra).astype(np.float32)]).astype(np.float32)


def make_targetp_stack_classifier(
    model_kind='random_forest',
    n_estimators=100,
    random_state=11,
    class_weight='balanced',
    max_features='sqrt',
    min_samples_leaf=1,
):
    model_kind = str(model_kind).strip().lower()
    if model_kind == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=int(n_estimators),
            random_state=int(random_state),
            class_weight=None if str(class_weight).strip().lower() == 'none' else class_weight,
            max_features=max_features,
            min_samples_leaf=int(min_samples_leaf),
            n_jobs=1,
        )
    if model_kind == 'extra_trees':
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(
            n_estimators=int(n_estimators),
            random_state=int(random_state),
            class_weight=None if str(class_weight).strip().lower() == 'none' else class_weight,
            max_features=max_features,
            min_samples_leaf=int(min_samples_leaf),
            n_jobs=1,
        )
    if model_kind == 'hist_gradient_boosting':
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=int(n_estimators),
            learning_rate=0.04,
            l2_regularization=0.01,
            random_state=int(random_state),
            class_weight=None if str(class_weight).strip().lower() == 'none' else class_weight,
        )
    raise ValueError('Unsupported targetp stack model_kind: {}'.format(model_kind))


def stack_feature_matrix(rows, base_prob_matrices, include_sequence_features=True):
    pieces = list()
    if len(base_prob_matrices) == 0:
        raise ValueError('At least one base OOF matrix is required for TargetP stacking.')
    base_prob_matrices = [np.asarray(prob, dtype=np.float32) for prob in base_prob_matrices]
    n_rows = int(base_prob_matrices[0].shape[0])
    ctp_idx = list(LOCALIZATION_CLASSES).index('cTP')
    ltp_idx = list(LOCALIZATION_CLASSES).index('lTP')
    for prob in base_prob_matrices:
        if prob.ndim != 2:
            raise ValueError('Base probability matrix should be two-dimensional.')
        if int(prob.shape[0]) != n_rows:
            raise ValueError('Base probability matrices have different row counts.')
        if int(prob.shape[1]) != len(LOCALIZATION_CLASSES):
            raise ValueError('Base probability matrix class count does not match LOCALIZATION_CLASSES.')
    pieces.append(np.hstack(base_prob_matrices))
    for prob in base_prob_matrices:
        pieces.append(np.max(prob, axis=1, keepdims=True))
        ctp_ltp_mass = prob[:, ctp_idx:ctp_idx + 1] + prob[:, ltp_idx:ltp_idx + 1]
        pieces.append(prob[:, ltp_idx:ltp_idx + 1] / np.clip(ctp_ltp_mass, 1.0e-9, None))
        pieces.append(ctp_ltp_mass)
    if include_sequence_features:
        pieces.append(build_targetp_feature_matrix(rows=rows).astype(np.float32))
    return np.hstack(pieces).astype(np.float32)


def predict_stack_classifier_prob_matrix(classifier, feature_matrix, class_names):
    prob = np.asarray(classifier.predict_proba(feature_matrix), dtype=np.float64)
    out = np.zeros((feature_matrix.shape[0], len(class_names)), dtype=np.float64)
    class_to_col = {int(cls): i for i, cls in enumerate(list(classifier.classes_))}
    for class_i in range(len(class_names)):
        if class_i in class_to_col:
            out[:, class_i] = prob[:, class_to_col[class_i]]
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    return out / row_sum


def _convex_weight_grid(n_sources, step):
    n_sources = int(n_sources)
    step = float(step)
    if n_sources < 2:
        raise ValueError('At least two probability sources are required.')
    if step <= 0.0 or step > 1.0:
        raise ValueError('weight grid step should be in (0, 1].')
    n_level = int(round(1.0 / step))
    if n_level <= 0:
        raise ValueError('weight grid step is too large.')

    def gen(prefix, remaining, slots):
        if slots == 1:
            yield tuple(prefix + [remaining])
            return
        for value in range(remaining + 1):
            yield from gen(prefix + [value], remaining - value, slots - 1)

    return [
        np.asarray(values, dtype=np.float64) / float(n_level)
        for values in gen([], n_level, n_sources)
    ]


def _blend_classwise_multi(prob_matrices, weights_by_class):
    matrices = [np.asarray(prob, dtype=np.float64) for prob in prob_matrices]
    if len(matrices) < 2:
        raise ValueError('At least two probability matrices are required.')
    n_rows, n_classes = matrices[0].shape
    weights = np.asarray(weights_by_class, dtype=np.float64)
    if weights.shape != (n_classes, len(matrices)):
        raise ValueError('weights_by_class should have shape (n_classes, n_sources).')
    out = np.zeros((n_rows, n_classes), dtype=np.float64)
    for source_i, matrix in enumerate(matrices):
        if matrix.shape != (n_rows, n_classes):
            raise ValueError('All probability matrices should have the same shape.')
        out += matrix * weights[:, source_i].reshape((1, -1))
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    return out / row_sum


def _optimize_classwise_multi_weights(prob_matrices, true_idx, class_names, weight_grid):
    n_classes = len(class_names)
    best_weights = np.tile(weight_grid[0].reshape((1, -1)), (n_classes, 1))
    best_metrics = None
    for trial in weight_grid:
        weights = np.tile(np.asarray(trial, dtype=np.float64).reshape((1, -1)), (n_classes, 1))
        pred_idx = np.argmax(
            _blend_classwise_multi(prob_matrices=prob_matrices, weights_by_class=weights),
            axis=1,
        ).astype(np.int64)
        metrics = _metrics_from_prediction_indices(
            pred_idx=pred_idx,
            true_idx=true_idx,
            class_names=class_names,
        )
        if best_metrics is None or metrics['macro_f1'] > best_metrics['macro_f1']:
            best_metrics = metrics
            best_weights = weights

    improved = True
    while improved:
        improved = False
        for class_i in range(n_classes):
            best_local_weights = best_weights[class_i, :].copy()
            best_local_metrics = best_metrics
            for trial in weight_grid:
                weights = best_weights.copy()
                weights[class_i, :] = np.asarray(trial, dtype=np.float64)
                pred_idx = np.argmax(
                    _blend_classwise_multi(prob_matrices=prob_matrices, weights_by_class=weights),
                    axis=1,
                ).astype(np.int64)
                metrics = _metrics_from_prediction_indices(
                    pred_idx=pred_idx,
                    true_idx=true_idx,
                    class_names=class_names,
                )
                if metrics['macro_f1'] > best_local_metrics['macro_f1']:
                    best_local_weights = np.asarray(trial, dtype=np.float64)
                    best_local_metrics = metrics
            if not np.allclose(best_local_weights, best_weights[class_i, :]):
                best_weights[class_i, :] = best_local_weights
                best_metrics = best_local_metrics
                improved = True
    return best_weights, best_metrics


def evaluate_foldwise_classwise_multi_blend(
    prob_matrices,
    true_idx,
    fold_ids,
    class_names,
    source_labels,
    weight_grid,
    threshold_grid,
):
    prob_matrices = [np.asarray(prob, dtype=np.float64) for prob in prob_matrices]
    class_names = list(class_names)
    source_labels = list(source_labels)
    if len(source_labels) != len(prob_matrices):
        raise ValueError('source_labels should match prob_matrices.')
    pred_idx = np.zeros((np.asarray(true_idx).shape[0],), dtype=np.int64)
    fold_rows = list()
    fold_ids = np.asarray(fold_ids)
    for fold_id in sorted(set([str(value) for value in fold_ids.tolist()])):
        valid_mask = np.asarray([str(value) == fold_id for value in fold_ids.tolist()], dtype=bool)
        train_mask = ~valid_mask
        train_probs = [prob[train_mask, :] for prob in prob_matrices]
        valid_probs = [prob[valid_mask, :] for prob in prob_matrices]
        weights, weight_metrics = _optimize_classwise_multi_weights(
            prob_matrices=train_probs,
            true_idx=np.asarray(true_idx)[train_mask],
            class_names=class_names,
            weight_grid=weight_grid,
        )
        train_blend = _blend_classwise_multi(
            prob_matrices=train_probs,
            weights_by_class=weights,
        )
        thresholds, threshold_metrics = optimize_class_thresholds(
            prob_matrix=train_blend,
            true_idx=np.asarray(true_idx)[train_mask],
            class_names=class_names,
            grid=threshold_grid,
        )
        valid_blend = _blend_classwise_multi(
            prob_matrices=valid_probs,
            weights_by_class=weights,
        )
        pred_idx[valid_mask] = _prediction_indices_with_thresholds(
            prob_matrix=valid_blend,
            thresholds=thresholds,
        )
        fold_rows.append({
            'fold_id': str(fold_id),
            'weights_by_class': {
                class_names[class_i]: {
                    source_labels[source_i]: float(weights[class_i, source_i])
                    for source_i in range(len(source_labels))
                }
                for class_i in range(len(class_names))
            },
            'class_thresholds': {
                class_names[i]: float(thresholds[i]) for i in range(len(class_names))
            },
            'train_weight_macro_f1': float(weight_metrics['macro_f1']),
            'train_threshold_macro_f1': float(threshold_metrics['macro_f1']),
            'n_train': int(np.sum(train_mask)),
            'n_valid': int(np.sum(valid_mask)),
        })
    return {
        'description': 'Each held-out fold is predicted using classwise convex source weights and class thresholds optimized on the other folds.',
        'metrics': _metrics_from_prediction_indices(
            pred_idx=pred_idx,
            true_idx=np.asarray(true_idx, dtype=np.int64),
            class_names=class_names,
        ),
        'folds': fold_rows,
        'profile': {
            'source_labels': list(source_labels),
            'n_sources': int(len(source_labels)),
            'n_weight_grid': int(len(weight_grid)),
        },
    }


def evaluate_foldwise_classwise_multi_blend_ltp_ctp_override(
    prob_matrices,
    true_idx,
    fold_ids,
    rows,
    class_names,
    source_labels,
    weight_grid,
    threshold_grid,
    score_grid,
    model_kind='random_forest',
    n_estimators=100,
    random_state=101,
    class_weight='balanced',
    max_features='sqrt',
    min_samples_leaf=1,
    ltp_ctp_class_weight=None,
    ltp_ctp_min_samples_leaf=None,
    ltp_source_classes=None,
    fixed_fold_rows=None,
):
    prob_matrices = [np.asarray(prob, dtype=np.float64) for prob in prob_matrices]
    class_names = list(class_names)
    source_labels = list(source_labels)
    ctp_idx = int(class_names.index('cTP'))
    ltp_idx = int(class_names.index('lTP'))
    ltp_source_idx = _ltp_source_class_indices(
        class_names=class_names,
        source_classes=ltp_source_classes,
    )
    true_idx = np.asarray(true_idx, dtype=np.int64)
    fold_ids = np.asarray(fold_ids)
    plant_mask = plant_mask_from_rows(rows=rows)
    features = build_ltp_ctp_specialist_feature_matrix(rows=rows).astype(np.float32)
    ltp_ctp_class_weight = (
        class_weight
        if ltp_ctp_class_weight is None
        else ltp_ctp_class_weight
    )
    ltp_ctp_min_samples_leaf = (
        min_samples_leaf
        if ltp_ctp_min_samples_leaf is None
        else int(ltp_ctp_min_samples_leaf)
    )
    pred_idx = np.zeros((true_idx.shape[0],), dtype=np.int64)
    fold_rows = list()
    fixed_fold_by_id = None
    if fixed_fold_rows is not None:
        fixed_fold_by_id = {
            str(row['fold_id']): row for row in fixed_fold_rows
        }
    for fold_i, fold_id in enumerate(sorted(set([str(v) for v in fold_ids.tolist()]))):
        valid_mask = np.asarray([str(value) == fold_id for value in fold_ids.tolist()], dtype=bool)
        train_mask = ~valid_mask
        train_probs = [prob[train_mask, :] for prob in prob_matrices]
        valid_probs = [prob[valid_mask, :] for prob in prob_matrices]
        if fixed_fold_by_id is None:
            weights, weight_metrics = _optimize_classwise_multi_weights(
                prob_matrices=train_probs,
                true_idx=true_idx[train_mask],
                class_names=class_names,
                weight_grid=weight_grid,
            )
            thresholds, threshold_metrics = None, None
        else:
            fixed_fold = fixed_fold_by_id.get(str(fold_id))
            if fixed_fold is None:
                raise ValueError('Missing fixed fold row for fold {}'.format(fold_id))
            weights = np.asarray([
                [
                    fixed_fold['weights_by_class'][class_name][source_label]
                    for source_label in source_labels
                ]
                for class_name in class_names
            ], dtype=np.float64)
            weight_metrics = {
                'macro_f1': float(fixed_fold.get('train_weight_macro_f1', 0.0))
            }
            thresholds = np.asarray([
                fixed_fold['class_thresholds'][class_name]
                for class_name in class_names
            ], dtype=np.float64)
            threshold_metrics = {
                'macro_f1': float(fixed_fold.get('train_threshold_macro_f1', 0.0))
            }
        train_blend = _blend_classwise_multi(
            prob_matrices=train_probs,
            weights_by_class=weights,
        )
        valid_blend = _blend_classwise_multi(
            prob_matrices=valid_probs,
            weights_by_class=weights,
        )
        if thresholds is None:
            thresholds, threshold_metrics = optimize_class_thresholds(
                prob_matrix=train_blend,
                true_idx=true_idx[train_mask],
                class_names=class_names,
                grid=threshold_grid,
            )
        train_pred = _prediction_indices_with_thresholds(
            prob_matrix=train_blend,
            thresholds=thresholds,
        )
        valid_pred = _prediction_indices_with_thresholds(
            prob_matrix=valid_blend,
            thresholds=thresholds,
        )
        specialist_train = (
            train_mask
            & plant_mask
            & np.isin(true_idx, [ctp_idx, ltp_idx])
        )
        score_threshold = None
        specialist_train_macro = None
        train_override_count = 0
        valid_override_count = 0
        if len(set(true_idx[specialist_train].tolist())) >= 2:
            classifier = make_targetp_stack_classifier(
                model_kind=model_kind,
                n_estimators=n_estimators,
                random_state=int(random_state) + int(fold_i),
                class_weight=ltp_ctp_class_weight,
                max_features=max_features,
                min_samples_leaf=ltp_ctp_min_samples_leaf,
            )
            classifier.fit(
                features[specialist_train, :],
                (true_idx[specialist_train] == ltp_idx).astype(np.int64),
            )
            classes = [int(value) for value in list(classifier.classes_)]
            if 1 in classes:
                train_score = np.asarray(
                    classifier.predict_proba(features[train_mask, :]),
                    dtype=np.float64,
                )[:, classes.index(1)]
                valid_score = np.asarray(
                    classifier.predict_proba(features[valid_mask, :]),
                    dtype=np.float64,
                )[:, classes.index(1)]
                train_plant = plant_mask[train_mask]
                valid_plant = plant_mask[valid_mask]
                best_threshold = None
                best_metrics = None
                best_override_count = 0
                for trial in score_grid:
                    trial_pred = train_pred.copy()
                    override_mask = (
                        train_plant
                        & np.isin(train_pred, ltp_source_idx)
                        & (train_score >= float(trial))
                    )
                    trial_pred[override_mask] = ltp_idx
                    metrics = _metrics_from_prediction_indices(
                        pred_idx=trial_pred,
                        true_idx=true_idx[train_mask],
                        class_names=class_names,
                    )
                    if best_metrics is None or metrics['macro_f1'] > best_metrics['macro_f1']:
                        best_threshold = float(trial)
                        best_metrics = metrics
                        best_override_count = int(np.sum(override_mask))
                valid_override = (
                    valid_plant
                    & np.isin(valid_pred, ltp_source_idx)
                    & (valid_score >= float(best_threshold))
                )
                valid_pred[valid_override] = ltp_idx
                score_threshold = float(best_threshold)
                specialist_train_macro = float(best_metrics['macro_f1'])
                train_override_count = int(best_override_count)
                valid_override_count = int(np.sum(valid_override))
        pred_idx[valid_mask] = valid_pred
        fold_rows.append({
            'fold_id': str(fold_id),
            'weights_by_class': {
                class_names[class_i]: {
                    source_labels[source_i]: float(weights[class_i, source_i])
                    for source_i in range(len(source_labels))
                }
                for class_i in range(len(class_names))
            },
            'class_thresholds': {
                class_names[i]: float(thresholds[i]) for i in range(len(class_names))
            },
            'train_weight_macro_f1': float(weight_metrics['macro_f1']),
            'train_threshold_macro_f1': float(threshold_metrics['macro_f1']),
            'ltp_ctp_score_threshold': score_threshold,
            'ltp_ctp_train_macro_f1': specialist_train_macro,
            'ltp_ctp_train_override_count': int(train_override_count),
            'ltp_ctp_valid_override_count': int(valid_override_count),
            'n_train': int(np.sum(train_mask)),
            'n_valid': int(np.sum(valid_mask)),
            'n_ltp_ctp_specialist_train': int(np.sum(specialist_train)),
        })
    return {
        'description': (
            'Each held-out fold is predicted using classwise convex source '
            'weights and class thresholds optimized on the other folds, '
            'followed by a plant lTP specialist trained and thresholded only '
            'on the other folds.'
        ),
        'metrics': _metrics_from_prediction_indices(
            pred_idx=pred_idx,
            true_idx=true_idx,
            class_names=class_names,
        ),
        'folds': fold_rows,
        'profile': {
            'source_labels': list(source_labels),
            'n_sources': int(len(source_labels)),
            'n_weight_grid': int(len(weight_grid)),
            'ltp_ctp_feature_profile': 'targetp_ltp_signal_v1',
            'ltp_source_classes': [class_names[i] for i in ltp_source_idx],
            'fixed_fold_rows': fixed_fold_rows is not None,
            'model_kind': str(model_kind),
            'n_estimators': int(n_estimators),
            'random_state': int(random_state),
            'class_weight': str(ltp_ctp_class_weight),
            'max_features': str(max_features),
            'min_samples_leaf': int(ltp_ctp_min_samples_leaf),
        },
    }


def run_targetp_stack_oof(
    training_tsv,
    base_oof_npzs,
    model_kind='random_forest',
    n_estimators=100,
    random_state=11,
    class_weight='balanced',
    max_features='sqrt',
    min_samples_leaf=1,
    include_sequence_features=True,
    organism_gate=False,
    organism_specialized_stack=False,
):
    rows = read_training_rows(path=training_tsv)
    class_names = list(LOCALIZATION_CLASSES)
    true_idx = _read_true_idx_from_training_tsv(
        training_tsv=training_tsv,
        class_names=class_names,
    )
    fold_ids = fold_ids_from_rows(rows=rows)
    if np.any(fold_ids == ''):
        raise ValueError('TargetP stack OOF requires fold_id in every row.')
    base_prob_matrices = list()
    for path in base_oof_npzs:
        prob, labels, names = _load_oof_npz(path=path, fallback_true_idx=true_idx)
        if names != class_names:
            raise ValueError('Class names in {} do not match LOCALIZATION_CLASSES.'.format(path))
        if np.any(labels != true_idx):
            raise ValueError('True labels in {} do not match training_tsv.'.format(path))
        base_prob_matrices.append(prob)
    if organism_gate:
        plant_mask = _read_organism_group_mask(training_tsv=training_tsv)
        base_prob_matrices = [
            _apply_organism_gate(
                prob_matrix=prob,
                plant_mask=plant_mask,
                class_names=class_names,
            )
            for prob in base_prob_matrices
        ]
    features = stack_feature_matrix(
        rows=rows,
        base_prob_matrices=base_prob_matrices,
        include_sequence_features=include_sequence_features,
    )
    prob_matrix = np.zeros((features.shape[0], len(class_names)), dtype=np.float64)
    fold_rows = list()
    organism_specialized_stack = bool(organism_specialized_stack)
    plant_mask = plant_mask_from_rows(rows=rows) if organism_specialized_stack else None
    for fold_i, fold_id in enumerate(sorted(set(fold_ids.tolist()))):
        valid_mask = fold_ids == fold_id
        train_mask = ~valid_mask
        fold_row = {
            'fold_id': str(fold_id),
            'n_train': int(np.sum(train_mask)),
            'n_valid': int(np.sum(valid_mask)),
        }
        if organism_specialized_stack:
            group_rows = list()
            for group_name, group_mask in [
                ('plant', plant_mask),
                ('non_plant', ~plant_mask),
            ]:
                valid_group_mask = valid_mask & group_mask
                if int(np.sum(valid_group_mask)) == 0:
                    continue
                train_group_mask = train_mask & group_mask
                fit_mask = train_group_mask
                used_fallback = False
                if (
                    int(np.sum(fit_mask)) < 2
                    or len(set(true_idx[fit_mask].tolist())) < 2
                ):
                    fit_mask = train_mask
                    used_fallback = True
                classifier = make_targetp_stack_classifier(
                    model_kind=model_kind,
                    n_estimators=n_estimators,
                    random_state=int(random_state) + int(fold_i),
                    class_weight=class_weight,
                    max_features=max_features,
                    min_samples_leaf=min_samples_leaf,
                )
                classifier.fit(features[fit_mask, :], true_idx[fit_mask])
                prob_matrix[valid_group_mask, :] = predict_stack_classifier_prob_matrix(
                    classifier=classifier,
                    feature_matrix=features[valid_group_mask, :],
                    class_names=class_names,
                )
                group_rows.append({
                    'organism_group': group_name,
                    'n_train': int(np.sum(train_group_mask)),
                    'n_fit': int(np.sum(fit_mask)),
                    'n_valid': int(np.sum(valid_group_mask)),
                    'used_global_fallback': bool(used_fallback),
                })
            fold_row['organism_groups'] = group_rows
        else:
            classifier = make_targetp_stack_classifier(
                model_kind=model_kind,
                n_estimators=n_estimators,
                random_state=random_state,
                class_weight=class_weight,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
            )
            classifier.fit(features[train_mask, :], true_idx[train_mask])
            prob_matrix[valid_mask, :] = predict_stack_classifier_prob_matrix(
                classifier=classifier,
                feature_matrix=features[valid_mask, :],
                class_names=class_names,
            )
        fold_rows.append(fold_row)
    return {
        'prob_matrix': prob_matrix,
        'true_idx': true_idx,
        'class_names': class_names,
        'fold_ids': fold_ids,
        'folds': fold_rows,
        'feature_dim': int(features.shape[1]),
        'profile': dict(TARGETP_STACK_DEFAULTS, **{
            'model_kind': str(model_kind),
            'n_estimators': int(n_estimators),
            'random_state': int(random_state),
            'class_weight': str(class_weight),
            'max_features': str(max_features),
            'min_samples_leaf': int(min_samples_leaf),
            'include_sequence_features': bool(include_sequence_features),
            'organism_gate': bool(organism_gate),
            'organism_specialized_stack': bool(organism_specialized_stack),
            'base_oof_npzs': [str(path) for path in base_oof_npzs],
        }),
    }


def evaluate_foldwise_ltp_ctp_override(
    prob_matrix,
    true_idx,
    fold_ids,
    rows,
    class_names,
    threshold_grid,
    score_grid,
    model_kind='random_forest',
    n_estimators=300,
    random_state=101,
    class_weight='balanced',
    max_features='sqrt',
    min_samples_leaf=1,
    ltp_ctp_class_weight=None,
    ltp_ctp_min_samples_leaf=None,
    ltp_source_classes=None,
):
    class_names = list(class_names)
    ctp_idx = int(class_names.index('cTP'))
    ltp_idx = int(class_names.index('lTP'))
    ltp_source_idx = _ltp_source_class_indices(
        class_names=class_names,
        source_classes=ltp_source_classes,
    )
    true_idx = np.asarray(true_idx, dtype=np.int64)
    fold_ids = np.asarray(fold_ids)
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    plant_mask = plant_mask_from_rows(rows=rows)
    features = build_ltp_ctp_specialist_feature_matrix(rows=rows).astype(np.float32)
    ltp_ctp_class_weight = (
        class_weight
        if ltp_ctp_class_weight is None
        else ltp_ctp_class_weight
    )
    ltp_ctp_min_samples_leaf = (
        min_samples_leaf
        if ltp_ctp_min_samples_leaf is None
        else int(ltp_ctp_min_samples_leaf)
    )
    pred_idx = np.zeros((true_idx.shape[0],), dtype=np.int64)
    fold_rows = list()
    for fold_id in sorted(set([str(v) for v in fold_ids.tolist()])):
        valid_mask = np.asarray([str(v) == fold_id for v in fold_ids.tolist()], dtype=bool)
        train_mask = ~valid_mask
        base_thresholds, _ = optimize_class_thresholds(
            prob_matrix=prob_matrix[train_mask, :],
            true_idx=true_idx[train_mask],
            class_names=class_names,
            grid=threshold_grid,
        )
        train_base = _prediction_indices_with_thresholds(
            prob_matrix=prob_matrix[train_mask, :],
            thresholds=base_thresholds,
        )
        valid_base = _prediction_indices_with_thresholds(
            prob_matrix=prob_matrix[valid_mask, :],
            thresholds=base_thresholds,
        )
        specialist_train = (
            train_mask
            & plant_mask
            & np.isin(true_idx, [ctp_idx, ltp_idx])
        )
        if len(set(true_idx[specialist_train].tolist())) < 2:
            pred_idx[valid_mask] = valid_base
            fold_rows.append({
                'fold_id': str(fold_id),
                'class_thresholds': {
                    class_names[i]: float(base_thresholds[i]) for i in range(len(class_names))
                },
                'ltp_score_threshold': None,
                'train_macro_f1': None,
                'train_override_count': 0,
                'valid_override_count': 0,
                'n_train': int(np.sum(train_mask)),
                'n_valid': int(np.sum(valid_mask)),
                'n_specialist_train': int(np.sum(specialist_train)),
            })
            continue
        classifier = make_targetp_stack_classifier(
            model_kind=model_kind,
            n_estimators=n_estimators,
            random_state=int(random_state) + len(fold_rows),
            class_weight=ltp_ctp_class_weight,
            max_features=max_features,
            min_samples_leaf=ltp_ctp_min_samples_leaf,
        )
        classifier.fit(
            features[specialist_train, :],
            (true_idx[specialist_train] == ltp_idx).astype(np.int64),
        )
        train_score = np.asarray(
            classifier.predict_proba(features[train_mask, :]),
            dtype=np.float64,
        )
        valid_score = np.asarray(
            classifier.predict_proba(features[valid_mask, :]),
            dtype=np.float64,
        )
        classes = [int(value) for value in list(classifier.classes_)]
        if 1 not in classes:
            pred_idx[valid_mask] = valid_base
            continue
        train_score = train_score[:, classes.index(1)]
        valid_score = valid_score[:, classes.index(1)]
        train_plant = plant_mask[train_mask]
        valid_plant = plant_mask[valid_mask]
        best_threshold = None
        best_metrics = None
        best_override_count = 0
        for trial in score_grid:
            trial_pred = train_base.copy()
            override_mask = (
                train_plant
                & np.isin(train_base, ltp_source_idx)
                & (train_score >= float(trial))
            )
            trial_pred[override_mask] = ltp_idx
            metrics = _metrics_from_prediction_indices(
                pred_idx=trial_pred,
                true_idx=true_idx[train_mask],
                class_names=class_names,
            )
            if best_metrics is None or metrics['macro_f1'] > best_metrics['macro_f1']:
                best_threshold = float(trial)
                best_metrics = metrics
                best_override_count = int(np.sum(override_mask))
        valid_pred = valid_base.copy()
        valid_override = (
            valid_plant
            & np.isin(valid_base, ltp_source_idx)
            & (valid_score >= float(best_threshold))
        )
        valid_pred[valid_override] = ltp_idx
        pred_idx[valid_mask] = valid_pred
        fold_rows.append({
            'fold_id': str(fold_id),
            'class_thresholds': {
                class_names[i]: float(base_thresholds[i]) for i in range(len(class_names))
            },
            'ltp_score_threshold': float(best_threshold),
            'train_macro_f1': float(best_metrics['macro_f1']),
            'train_override_count': int(best_override_count),
            'valid_override_count': int(np.sum(valid_override)),
            'n_train': int(np.sum(train_mask)),
            'n_valid': int(np.sum(valid_mask)),
            'n_specialist_train': int(np.sum(specialist_train)),
        })
    return {
        'description': (
            'Each held-out fold uses class thresholds and a plant cTP-vs-lTP '
            'specialist trained only on the other folds; cTP predictions above '
            'the nested specialist threshold are overridden to lTP.'
        ),
        'metrics': _metrics_from_prediction_indices(
            pred_idx=pred_idx,
            true_idx=true_idx,
            class_names=class_names,
        ),
        'folds': fold_rows,
        'profile': {
            'ltp_ctp_feature_profile': 'targetp_ltp_signal_v1',
            'model_kind': str(model_kind),
            'n_estimators': int(n_estimators),
            'random_state': int(random_state),
            'class_weight': str(ltp_ctp_class_weight),
            'max_features': str(max_features),
            'min_samples_leaf': int(ltp_ctp_min_samples_leaf),
            'ltp_source_classes': [class_names[i] for i in ltp_source_idx],
        },
    }


def evaluate_foldwise_notp_ctp_ltp_override(
    prob_matrix,
    true_idx,
    fold_ids,
    rows,
    class_names,
    threshold_grid,
    score_grid,
    notp_ctp_model_kind='random_forest',
    notp_ctp_n_estimators=200,
    notp_ctp_random_state=400,
    ltp_ctp_model_kind='random_forest',
    ltp_ctp_n_estimators=300,
    ltp_ctp_random_state=101,
    class_weight='balanced',
    max_features='sqrt',
    min_samples_leaf=1,
    notp_ctp_class_weight=None,
    notp_ctp_min_samples_leaf=None,
    ltp_ctp_class_weight=None,
    ltp_ctp_min_samples_leaf=None,
):
    class_names = list(class_names)
    notp_idx = int(class_names.index('noTP'))
    ctp_idx = int(class_names.index('cTP'))
    ltp_idx = int(class_names.index('lTP'))
    true_idx = np.asarray(true_idx, dtype=np.int64)
    fold_ids = np.asarray(fold_ids)
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    plant_mask = plant_mask_from_rows(rows=rows)
    features = build_targetp_feature_matrix(rows=rows).astype(np.float32)
    ltp_features = build_ltp_ctp_specialist_feature_matrix(rows=rows).astype(np.float32)
    notp_ctp_class_weight = (
        class_weight
        if notp_ctp_class_weight is None
        else notp_ctp_class_weight
    )
    ltp_ctp_class_weight = (
        class_weight
        if ltp_ctp_class_weight is None
        else ltp_ctp_class_weight
    )
    notp_ctp_min_samples_leaf = (
        min_samples_leaf
        if notp_ctp_min_samples_leaf is None
        else int(notp_ctp_min_samples_leaf)
    )
    ltp_ctp_min_samples_leaf = (
        min_samples_leaf
        if ltp_ctp_min_samples_leaf is None
        else int(ltp_ctp_min_samples_leaf)
    )
    pred_idx = np.zeros((true_idx.shape[0],), dtype=np.int64)
    fold_rows = list()
    for fold_i, fold_id in enumerate(sorted(set([str(v) for v in fold_ids.tolist()]))):
        valid_mask = np.asarray([str(v) == fold_id for v in fold_ids.tolist()], dtype=bool)
        train_mask = ~valid_mask
        base_thresholds, _ = optimize_class_thresholds(
            prob_matrix=prob_matrix[train_mask, :],
            true_idx=true_idx[train_mask],
            class_names=class_names,
            grid=threshold_grid,
        )
        train_pred = _prediction_indices_with_thresholds(
            prob_matrix=prob_matrix[train_mask, :],
            thresholds=base_thresholds,
        )
        valid_pred = _prediction_indices_with_thresholds(
            prob_matrix=prob_matrix[valid_mask, :],
            thresholds=base_thresholds,
        )

        notp_train_rows = train_pred == notp_idx
        notp_threshold = None
        notp_train_override_count = 0
        notp_valid_override_count = 0
        notp_train_macro = None
        if (
            int(np.sum(notp_train_rows)) >= 10
            and int(np.sum(true_idx[train_mask][notp_train_rows] == ctp_idx)) >= 2
        ):
            notp_classifier = make_targetp_stack_classifier(
                model_kind=notp_ctp_model_kind,
                n_estimators=notp_ctp_n_estimators,
                random_state=int(notp_ctp_random_state) + int(fold_i),
                class_weight=notp_ctp_class_weight,
                max_features=max_features,
                min_samples_leaf=notp_ctp_min_samples_leaf,
            )
            notp_classifier.fit(
                features[train_mask, :][notp_train_rows, :],
                (true_idx[train_mask][notp_train_rows] == ctp_idx).astype(np.int64),
            )
            classes = [int(value) for value in list(notp_classifier.classes_)]
            if 1 in classes:
                train_score = np.asarray(
                    notp_classifier.predict_proba(features[train_mask, :]),
                    dtype=np.float64,
                )[:, classes.index(1)]
                valid_score = np.asarray(
                    notp_classifier.predict_proba(features[valid_mask, :]),
                    dtype=np.float64,
                )[:, classes.index(1)]
                best_threshold = None
                best_metrics = None
                best_override_count = 0
                for trial in score_grid:
                    trial_pred = train_pred.copy()
                    override_mask = (train_pred == notp_idx) & (train_score >= float(trial))
                    trial_pred[override_mask] = ctp_idx
                    metrics = _metrics_from_prediction_indices(
                        pred_idx=trial_pred,
                        true_idx=true_idx[train_mask],
                        class_names=class_names,
                    )
                    if best_metrics is None or metrics['macro_f1'] > best_metrics['macro_f1']:
                        best_threshold = float(trial)
                        best_metrics = metrics
                        best_override_count = int(np.sum(override_mask))
                train_override = (train_pred == notp_idx) & (train_score >= float(best_threshold))
                valid_override = (valid_pred == notp_idx) & (valid_score >= float(best_threshold))
                train_pred[train_override] = ctp_idx
                valid_pred[valid_override] = ctp_idx
                notp_threshold = float(best_threshold)
                notp_train_override_count = int(best_override_count)
                notp_valid_override_count = int(np.sum(valid_override))
                notp_train_macro = float(best_metrics['macro_f1'])

        specialist_train = (
            train_mask
            & plant_mask
            & np.isin(true_idx, [ctp_idx, ltp_idx])
        )
        ltp_threshold = None
        ltp_train_override_count = 0
        ltp_valid_override_count = 0
        ltp_train_macro = None
        if len(set(true_idx[specialist_train].tolist())) >= 2:
            ltp_classifier = make_targetp_stack_classifier(
                model_kind=ltp_ctp_model_kind,
                n_estimators=ltp_ctp_n_estimators,
                random_state=int(ltp_ctp_random_state) + int(fold_i),
                class_weight=ltp_ctp_class_weight,
                max_features=max_features,
                min_samples_leaf=ltp_ctp_min_samples_leaf,
            )
            ltp_classifier.fit(
                ltp_features[specialist_train, :],
                (true_idx[specialist_train] == ltp_idx).astype(np.int64),
            )
            classes = [int(value) for value in list(ltp_classifier.classes_)]
            if 1 in classes:
                train_score = np.asarray(
                    ltp_classifier.predict_proba(ltp_features[train_mask, :]),
                    dtype=np.float64,
                )[:, classes.index(1)]
                valid_score = np.asarray(
                    ltp_classifier.predict_proba(ltp_features[valid_mask, :]),
                    dtype=np.float64,
                )[:, classes.index(1)]
                train_plant = plant_mask[train_mask]
                valid_plant = plant_mask[valid_mask]
                best_threshold = None
                best_metrics = None
                best_override_count = 0
                for trial in score_grid:
                    trial_pred = train_pred.copy()
                    override_mask = (
                        train_plant
                        & (train_pred == ctp_idx)
                        & (train_score >= float(trial))
                    )
                    trial_pred[override_mask] = ltp_idx
                    metrics = _metrics_from_prediction_indices(
                        pred_idx=trial_pred,
                        true_idx=true_idx[train_mask],
                        class_names=class_names,
                    )
                    if best_metrics is None or metrics['macro_f1'] > best_metrics['macro_f1']:
                        best_threshold = float(trial)
                        best_metrics = metrics
                        best_override_count = int(np.sum(override_mask))
                valid_override = (
                    valid_plant
                    & (valid_pred == ctp_idx)
                    & (valid_score >= float(best_threshold))
                )
                valid_pred[valid_override] = ltp_idx
                ltp_threshold = float(best_threshold)
                ltp_train_override_count = int(best_override_count)
                ltp_valid_override_count = int(np.sum(valid_override))
                ltp_train_macro = float(best_metrics['macro_f1'])

        pred_idx[valid_mask] = valid_pred
        fold_rows.append({
            'fold_id': str(fold_id),
            'class_thresholds': {
                class_names[i]: float(base_thresholds[i]) for i in range(len(class_names))
            },
            'notp_ctp_score_threshold': notp_threshold,
            'notp_ctp_train_macro_f1': notp_train_macro,
            'notp_ctp_train_override_count': int(notp_train_override_count),
            'notp_ctp_valid_override_count': int(notp_valid_override_count),
            'ltp_ctp_score_threshold': ltp_threshold,
            'ltp_ctp_train_macro_f1': ltp_train_macro,
            'ltp_ctp_train_override_count': int(ltp_train_override_count),
            'ltp_ctp_valid_override_count': int(ltp_valid_override_count),
            'n_train': int(np.sum(train_mask)),
            'n_valid': int(np.sum(valid_mask)),
            'n_ltp_ctp_specialist_train': int(np.sum(specialist_train)),
        })
    return {
        'description': (
            'Each held-out fold first applies a noTP-to-cTP specialist and then '
            'a plant cTP-to-lTP specialist; both specialists and both score '
            'thresholds are fitted only on the other folds.'
        ),
        'metrics': _metrics_from_prediction_indices(
            pred_idx=pred_idx,
            true_idx=true_idx,
            class_names=class_names,
        ),
        'folds': fold_rows,
        'profile': {
            'ltp_ctp_feature_profile': 'targetp_ltp_signal_v1',
            'notp_ctp_model_kind': str(notp_ctp_model_kind),
            'notp_ctp_n_estimators': int(notp_ctp_n_estimators),
            'notp_ctp_random_state': int(notp_ctp_random_state),
            'notp_ctp_class_weight': str(notp_ctp_class_weight),
            'notp_ctp_min_samples_leaf': int(notp_ctp_min_samples_leaf),
            'ltp_ctp_model_kind': str(ltp_ctp_model_kind),
            'ltp_ctp_n_estimators': int(ltp_ctp_n_estimators),
            'ltp_ctp_random_state': int(ltp_ctp_random_state),
            'ltp_ctp_class_weight': str(ltp_ctp_class_weight),
            'ltp_ctp_min_samples_leaf': int(ltp_ctp_min_samples_leaf),
            'class_weight': str(class_weight),
            'max_features': str(max_features),
            'min_samples_leaf': int(min_samples_leaf),
        },
    }


def evaluate_foldwise_classwise_blend_ltp_ctp_override(
    prob_a,
    prob_b,
    true_idx,
    fold_ids,
    rows,
    class_names,
    alpha_grid,
    threshold_grid,
    score_grid,
    model_kind='random_forest',
    n_estimators=100,
    random_state=101,
    class_weight='balanced',
    max_features='sqrt',
    min_samples_leaf=1,
    ltp_ctp_class_weight=None,
    ltp_ctp_min_samples_leaf=None,
    ltp_source_classes=None,
):
    class_names = list(class_names)
    ctp_idx = int(class_names.index('cTP'))
    ltp_idx = int(class_names.index('lTP'))
    ltp_source_idx = _ltp_source_class_indices(
        class_names=class_names,
        source_classes=ltp_source_classes,
    )
    true_idx = np.asarray(true_idx, dtype=np.int64)
    fold_ids = np.asarray(fold_ids)
    prob_a = np.asarray(prob_a, dtype=np.float64)
    prob_b = np.asarray(prob_b, dtype=np.float64)
    plant_mask = plant_mask_from_rows(rows=rows)
    features = build_ltp_ctp_specialist_feature_matrix(rows=rows).astype(np.float32)
    ltp_ctp_class_weight = (
        class_weight
        if ltp_ctp_class_weight is None
        else ltp_ctp_class_weight
    )
    ltp_ctp_min_samples_leaf = (
        min_samples_leaf
        if ltp_ctp_min_samples_leaf is None
        else int(ltp_ctp_min_samples_leaf)
    )
    pred_idx = np.zeros((true_idx.shape[0],), dtype=np.int64)
    fold_rows = list()
    for fold_i, fold_id in enumerate(sorted(set([str(v) for v in fold_ids.tolist()]))):
        valid_mask = np.asarray([str(v) == fold_id for v in fold_ids.tolist()], dtype=bool)
        train_mask = ~valid_mask
        alpha, alpha_metrics = _optimize_classwise_alpha(
            prob_a=prob_a[train_mask, :],
            prob_b=prob_b[train_mask, :],
            true_idx=true_idx[train_mask],
            class_names=class_names,
            alpha_grid=alpha_grid,
        )
        train_blend = _blend_classwise(
            prob_a=prob_a[train_mask, :],
            prob_b=prob_b[train_mask, :],
            alpha_by_class=alpha,
        )
        valid_blend = _blend_classwise(
            prob_a=prob_a[valid_mask, :],
            prob_b=prob_b[valid_mask, :],
            alpha_by_class=alpha,
        )
        thresholds, threshold_metrics = optimize_class_thresholds(
            prob_matrix=train_blend,
            true_idx=true_idx[train_mask],
            class_names=class_names,
            grid=threshold_grid,
        )
        train_pred = _prediction_indices_with_thresholds(
            prob_matrix=train_blend,
            thresholds=thresholds,
        )
        valid_pred = _prediction_indices_with_thresholds(
            prob_matrix=valid_blend,
            thresholds=thresholds,
        )
        specialist_train = (
            train_mask
            & plant_mask
            & np.isin(true_idx, [ctp_idx, ltp_idx])
        )
        score_threshold = None
        specialist_train_macro = None
        train_override_count = 0
        valid_override_count = 0
        if len(set(true_idx[specialist_train].tolist())) >= 2:
            classifier = make_targetp_stack_classifier(
                model_kind=model_kind,
                n_estimators=n_estimators,
                random_state=int(random_state) + int(fold_i),
                class_weight=ltp_ctp_class_weight,
                max_features=max_features,
                min_samples_leaf=ltp_ctp_min_samples_leaf,
            )
            classifier.fit(
                features[specialist_train, :],
                (true_idx[specialist_train] == ltp_idx).astype(np.int64),
            )
            classes = [int(value) for value in list(classifier.classes_)]
            if 1 in classes:
                train_score = np.asarray(
                    classifier.predict_proba(features[train_mask, :]),
                    dtype=np.float64,
                )[:, classes.index(1)]
                valid_score = np.asarray(
                    classifier.predict_proba(features[valid_mask, :]),
                    dtype=np.float64,
                )[:, classes.index(1)]
                train_plant = plant_mask[train_mask]
                valid_plant = plant_mask[valid_mask]
                best_threshold = None
                best_metrics = None
                best_override_count = 0
                for trial in score_grid:
                    trial_pred = train_pred.copy()
                    override_mask = (
                        train_plant
                        & np.isin(train_pred, ltp_source_idx)
                        & (train_score >= float(trial))
                    )
                    trial_pred[override_mask] = ltp_idx
                    metrics = _metrics_from_prediction_indices(
                        pred_idx=trial_pred,
                        true_idx=true_idx[train_mask],
                        class_names=class_names,
                    )
                    if best_metrics is None or metrics['macro_f1'] > best_metrics['macro_f1']:
                        best_threshold = float(trial)
                        best_metrics = metrics
                        best_override_count = int(np.sum(override_mask))
                valid_override = (
                    valid_plant
                    & np.isin(valid_pred, ltp_source_idx)
                    & (valid_score >= float(best_threshold))
                )
                valid_pred[valid_override] = ltp_idx
                score_threshold = float(best_threshold)
                specialist_train_macro = float(best_metrics['macro_f1'])
                train_override_count = int(best_override_count)
                valid_override_count = int(np.sum(valid_override))
        pred_idx[valid_mask] = valid_pred
        fold_rows.append({
            'fold_id': str(fold_id),
            'alpha_by_class': {
                class_names[i]: float(alpha[i]) for i in range(len(class_names))
            },
            'class_thresholds': {
                class_names[i]: float(thresholds[i]) for i in range(len(class_names))
            },
            'train_alpha_macro_f1': float(alpha_metrics['macro_f1']),
            'train_threshold_macro_f1': float(threshold_metrics['macro_f1']),
            'ltp_ctp_score_threshold': score_threshold,
            'ltp_ctp_train_macro_f1': specialist_train_macro,
            'ltp_ctp_train_override_count': int(train_override_count),
            'ltp_ctp_valid_override_count': int(valid_override_count),
            'n_train': int(np.sum(train_mask)),
            'n_valid': int(np.sum(valid_mask)),
            'n_ltp_ctp_specialist_train': int(np.sum(specialist_train)),
        })
    return {
        'description': (
            'Each held-out fold is predicted using classwise OOF blending and '
            'class thresholds optimized on the other folds, followed by a '
            'plant cTP-to-lTP specialist trained and thresholded only on the '
            'other folds.'
        ),
        'metrics': _metrics_from_prediction_indices(
            pred_idx=pred_idx,
            true_idx=true_idx,
            class_names=class_names,
        ),
        'folds': fold_rows,
        'profile': {
            'ltp_ctp_feature_profile': 'targetp_ltp_signal_v1',
            'model_kind': str(model_kind),
            'n_estimators': int(n_estimators),
            'random_state': int(random_state),
            'class_weight': str(ltp_ctp_class_weight),
            'max_features': str(max_features),
            'min_samples_leaf': int(ltp_ctp_min_samples_leaf),
            'alpha_grid': [float(value) for value in alpha_grid],
            'ltp_source_classes': [class_names[i] for i in ltp_source_idx],
        },
    }


def _targetp_macro_f1(class_names):
    return float(np.mean(np.asarray([
        TARGETP_TABLE1_REFERENCE[class_name]['f1'] for class_name in class_names
    ], dtype=np.float64)))


def render_markdown(out):
    results = list(out['results'].items())
    headers = ['Class', 'TargetP F1'] + ['{} F1'.format(key) for key, _ in results]
    lines = ['| {} |'.format(' | '.join(headers))]
    lines.append('|{}|'.format('|'.join(['---'] + ['---:'] * (len(headers) - 1))))
    for row in out['class_rows']:
        values = [row['class'], '{:.3f}'.format(row['targetp_f1'])]
        for key, _ in results:
            values.append('{:.3f}'.format(row[key]))
        lines.append('| {} |'.format(' | '.join(values)))
    lines.append('')
    lines.append('| Metric | TargetP | {} |'.format(' | '.join([key for key, _ in results])))
    lines.append('|{}|'.format('|'.join(['---'] + ['---:'] * (len(results) + 1))))
    macro = ['Macro F1', '{:.3f}'.format(out['targetp_macro_f1'])]
    acc = ['Overall accuracy', '-']
    for _, result in results:
        macro.append('{:.3f}'.format(result['metrics']['macro_f1']))
        acc.append('{:.3f}'.format(result['metrics']['overall_accuracy']))
    lines.append('| {} |'.format(' | '.join(macro)))
    lines.append('| {} |'.format(' | '.join(acc)))
    lines.append('')
    lines.append('stack profile: {}'.format(out['stack_profile']))
    return '\n'.join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate a foldwise TargetP stacking model from fair base OOF probabilities.',
    )
    parser.add_argument('--training_tsv', default='data/localize_bench/targetp2_benchmark.tsv', type=str)
    parser.add_argument('--base_oof_npzs', required=True, type=str)
    parser.add_argument('--stack_oof_npz', default='data/localize_bench/targetp2_oof_stack.npz', type=str)
    parser.add_argument('--model_kind', default=TARGETP_STACK_DEFAULTS['model_kind'], choices=['random_forest', 'extra_trees', 'hist_gradient_boosting'], type=str)
    parser.add_argument('--n_estimators', default=TARGETP_STACK_DEFAULTS['n_estimators'], type=int)
    parser.add_argument('--random_state', default=TARGETP_STACK_DEFAULTS['random_state'], type=int)
    parser.add_argument('--class_weight', default=TARGETP_STACK_DEFAULTS['class_weight'], type=str)
    parser.add_argument('--max_features', default=TARGETP_STACK_DEFAULTS['max_features'], type=str)
    parser.add_argument('--min_samples_leaf', default=TARGETP_STACK_DEFAULTS['min_samples_leaf'], type=int)
    parser.add_argument('--include_sequence_features', default='yes', choices=['yes', 'no'], type=str)
    parser.add_argument(
        '--ltp_ctp_override',
        default='yes' if TARGETP_STACK_LTP_CTP_DEFAULTS['ltp_ctp_override'] else 'no',
        choices=['yes', 'no'],
        type=str,
    )
    parser.add_argument(
        '--ltp_ctp_model_kind',
        default=TARGETP_STACK_LTP_CTP_DEFAULTS['ltp_ctp_model_kind'],
        choices=['random_forest', 'extra_trees', 'hist_gradient_boosting'],
        type=str,
    )
    parser.add_argument('--ltp_ctp_n_estimators', default=TARGETP_STACK_LTP_CTP_DEFAULTS['ltp_ctp_n_estimators'], type=int)
    parser.add_argument('--ltp_ctp_random_state', default='', type=str)
    parser.add_argument(
        '--ltp_ctp_class_weight',
        default=TARGETP_STACK_LTP_CTP_DEFAULTS['ltp_ctp_class_weight'],
        type=str,
    )
    parser.add_argument(
        '--ltp_ctp_min_samples_leaf',
        default=TARGETP_STACK_LTP_CTP_DEFAULTS['ltp_ctp_min_samples_leaf'],
        type=int,
    )
    parser.add_argument('--ltp_ctp_score_min', default=0.02, type=float)
    parser.add_argument('--ltp_ctp_score_max', default=0.80, type=float)
    parser.add_argument('--ltp_ctp_score_step', default=0.01, type=float)
    parser.add_argument('--ltp_source_classes', default='cTP', type=str)
    parser.add_argument(
        '--notp_ctp_ltp_override',
        default='yes' if TARGETP_STACK_NOTP_CTP_LTP_DEFAULTS['notp_ctp_ltp_override'] else 'no',
        choices=['yes', 'no'],
        type=str,
    )
    parser.add_argument(
        '--notp_ctp_model_kind',
        default=TARGETP_STACK_NOTP_CTP_LTP_DEFAULTS['notp_ctp_model_kind'],
        choices=['random_forest', 'extra_trees', 'hist_gradient_boosting'],
        type=str,
    )
    parser.add_argument('--notp_ctp_n_estimators', default=TARGETP_STACK_NOTP_CTP_LTP_DEFAULTS['notp_ctp_n_estimators'], type=int)
    parser.add_argument('--notp_ctp_random_state', default='', type=str)
    parser.add_argument(
        '--notp_ctp_class_weight',
        default=TARGETP_STACK_NOTP_CTP_LTP_DEFAULTS['notp_ctp_class_weight'],
        type=str,
    )
    parser.add_argument(
        '--notp_ctp_min_samples_leaf',
        default=TARGETP_STACK_NOTP_CTP_LTP_DEFAULTS['notp_ctp_min_samples_leaf'],
        type=int,
    )
    parser.add_argument(
        '--organism_gate',
        default='yes' if TARGETP_STACK_DEFAULTS['organism_gate'] else 'no',
        choices=['yes', 'no'],
        type=str,
    )
    parser.add_argument(
        '--organism_specialized_stack',
        default='yes' if TARGETP_STACK_DEFAULTS['organism_specialized_stack'] else 'no',
        choices=['yes', 'no'],
        type=str,
    )
    parser.add_argument('--post_blend_oof_npz', default='', type=str)
    parser.add_argument('--post_blend_oof_npzs', default='', type=str)
    parser.add_argument('--post_blend_label', default='post_blend', type=str)
    parser.add_argument('--post_blend_grid_step', default=0.10, type=float)
    parser.add_argument('--post_blend_ltp_ctp_override', default='yes', choices=['yes', 'no'], type=str)
    parser.add_argument('--threshold_grid', default='0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.5,0.65,0.8,1.0,1.25,1.5,2.0,3.0,5.0', type=str)
    parser.add_argument('--out_json', default='data/localize_bench/targetp2_stack_eval.json', type=str)
    parser.add_argument('--out_md', default='data/localize_bench/targetp2_stack_eval.md', type=str)
    return parser


def _to_bool(value):
    return str(value).strip().lower() in ['yes', 'y', 'true', '1']


def main():
    args = build_parser().parse_args()
    base_oof_npzs = [
        value.strip() for value in str(args.base_oof_npzs).split(',')
        if value.strip() != ''
    ]
    if len(base_oof_npzs) == 0:
        raise ValueError('--base_oof_npzs should contain at least one path.')
    oof = run_targetp_stack_oof(
        training_tsv=args.training_tsv,
        base_oof_npzs=base_oof_npzs,
        model_kind=args.model_kind,
        n_estimators=int(args.n_estimators),
        random_state=int(args.random_state),
        class_weight=args.class_weight,
        max_features=args.max_features,
        min_samples_leaf=int(args.min_samples_leaf),
        include_sequence_features=_to_bool(args.include_sequence_features),
        organism_gate=_to_bool(args.organism_gate),
        organism_specialized_stack=_to_bool(args.organism_specialized_stack),
    )
    out_dir = os.path.dirname(str(args.stack_oof_npz))
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        args.stack_oof_npz,
        prob_matrix=oof['prob_matrix'],
        true_idx=oof['true_idx'],
        class_names=np.asarray(oof['class_names']),
        fold_ids=np.asarray(oof['fold_ids']),
        stack_profile_json=json.dumps(oof['profile'], sort_keys=True),
    )
    rows = read_training_rows(path=args.training_tsv)
    threshold_grid = sorted(set([
        float(v.strip()) for v in str(args.threshold_grid).split(',')
        if v.strip() != ''
    ]))
    if float(args.ltp_ctp_score_step) <= 0.0:
        raise ValueError('--ltp_ctp_score_step should be positive.')
    score_grid = np.arange(
        float(args.ltp_ctp_score_min),
        float(args.ltp_ctp_score_max) + (0.5 * float(args.ltp_ctp_score_step)),
        float(args.ltp_ctp_score_step),
        dtype=np.float64,
    )
    ltp_ctp_random_state = (
        int(args.ltp_ctp_random_state)
        if str(args.ltp_ctp_random_state).strip() != ''
        else int(args.random_state) + 90
    )
    notp_ctp_random_state = (
        int(args.notp_ctp_random_state)
        if str(args.notp_ctp_random_state).strip() != ''
        else int(args.random_state) + 389
    )
    ltp_ctp_class_weight = (
        args.class_weight
        if str(args.ltp_ctp_class_weight).strip() == ''
        else str(args.ltp_ctp_class_weight)
    )
    notp_ctp_class_weight = (
        args.class_weight
        if str(args.notp_ctp_class_weight).strip() == ''
        else str(args.notp_ctp_class_weight)
    )
    ltp_ctp_min_samples_leaf = (
        int(args.min_samples_leaf)
        if int(args.ltp_ctp_min_samples_leaf) <= 0
        else int(args.ltp_ctp_min_samples_leaf)
    )
    notp_ctp_min_samples_leaf = (
        int(args.min_samples_leaf)
        if int(args.notp_ctp_min_samples_leaf) <= 0
        else int(args.notp_ctp_min_samples_leaf)
    )
    results = {
        'stack_argmax': {
            'metrics': _metrics_from_prob_matrix(
                prob_matrix=oof['prob_matrix'],
                true_idx=oof['true_idx'],
                class_names=oof['class_names'],
            ),
        },
        'stack_foldwise_threshold': evaluate_foldwise_thresholds(
            prob_matrix=oof['prob_matrix'],
            true_idx=oof['true_idx'],
            fold_ids=oof['fold_ids'],
            class_names=oof['class_names'],
            threshold_grid=threshold_grid,
        ),
    }
    if _to_bool(args.ltp_ctp_override):
        results['stack_foldwise_ltp_ctp_override'] = evaluate_foldwise_ltp_ctp_override(
            prob_matrix=oof['prob_matrix'],
            true_idx=oof['true_idx'],
            fold_ids=oof['fold_ids'],
            rows=rows,
            class_names=oof['class_names'],
            threshold_grid=threshold_grid,
            score_grid=score_grid,
            model_kind=args.ltp_ctp_model_kind,
            n_estimators=int(args.ltp_ctp_n_estimators),
            random_state=ltp_ctp_random_state,
            class_weight=args.class_weight,
            max_features=args.max_features,
            min_samples_leaf=int(args.min_samples_leaf),
            ltp_ctp_class_weight=ltp_ctp_class_weight,
            ltp_ctp_min_samples_leaf=ltp_ctp_min_samples_leaf,
            ltp_source_classes=args.ltp_source_classes,
        )
        if _to_bool(args.notp_ctp_ltp_override):
            results['stack_foldwise_notp_ctp_ltp_override'] = evaluate_foldwise_notp_ctp_ltp_override(
                prob_matrix=oof['prob_matrix'],
                true_idx=oof['true_idx'],
                fold_ids=oof['fold_ids'],
                rows=rows,
                class_names=oof['class_names'],
                threshold_grid=threshold_grid,
                score_grid=score_grid,
                notp_ctp_model_kind=args.notp_ctp_model_kind,
                notp_ctp_n_estimators=int(args.notp_ctp_n_estimators),
                notp_ctp_random_state=notp_ctp_random_state,
                ltp_ctp_model_kind=args.ltp_ctp_model_kind,
                ltp_ctp_n_estimators=int(args.ltp_ctp_n_estimators),
                ltp_ctp_random_state=ltp_ctp_random_state,
                class_weight=args.class_weight,
                max_features=args.max_features,
                min_samples_leaf=int(args.min_samples_leaf),
                notp_ctp_class_weight=notp_ctp_class_weight,
                notp_ctp_min_samples_leaf=notp_ctp_min_samples_leaf,
                ltp_ctp_class_weight=ltp_ctp_class_weight,
                ltp_ctp_min_samples_leaf=ltp_ctp_min_samples_leaf,
            )
    post_blend_paths = [
        value.strip() for value in str(args.post_blend_oof_npzs).split(',')
        if value.strip() != ''
    ]
    if len(post_blend_paths) == 0 and str(args.post_blend_oof_npz).strip() != '':
        post_blend_paths = [str(args.post_blend_oof_npz).strip()]
    if len(post_blend_paths) > 0:
        post_blend_probs = list()
        for post_blend_path in post_blend_paths:
            post_blend_prob, post_blend_true_idx, post_blend_class_names = _load_oof_npz(
                path=post_blend_path,
                fallback_true_idx=oof['true_idx'],
            )
            if post_blend_class_names != list(oof['class_names']):
                raise ValueError('Class names in {} do not match LOCALIZATION_CLASSES.'.format(post_blend_path))
            if np.any(post_blend_true_idx != oof['true_idx']):
                raise ValueError('True labels in {} do not match training_tsv.'.format(post_blend_path))
            post_blend_probs.append(post_blend_prob)
        post_blend_step = float(args.post_blend_grid_step)
        if post_blend_step <= 0.0:
            raise ValueError('--post_blend_grid_step should be positive.')
        alpha_grid = sorted(set([
            round(i * post_blend_step, 10)
            for i in range(int(np.floor(1.0 / post_blend_step)) + 1)
        ] + [1.0]))
        post_blend_label = str(args.post_blend_label).strip() or 'post_blend'
        post_blend_key = 'stack_{}_foldwise_blend'.format(post_blend_label)
        if len(post_blend_probs) == 1:
            results[post_blend_key] = evaluate_foldwise_classwise_blend(
                prob_a=oof['prob_matrix'],
                prob_b=post_blend_probs[0],
                true_idx=oof['true_idx'],
                fold_ids=oof['fold_ids'],
                class_names=oof['class_names'],
                alpha_grid=alpha_grid,
                threshold_grid=threshold_grid,
            )
        else:
            source_labels = ['stack'] + [
                'post_blend_{}'.format(path_i + 1)
                for path_i in range(len(post_blend_probs))
            ]
            results[post_blend_key] = evaluate_foldwise_classwise_multi_blend(
                prob_matrices=[oof['prob_matrix']] + post_blend_probs,
                true_idx=oof['true_idx'],
                fold_ids=oof['fold_ids'],
                class_names=oof['class_names'],
                source_labels=source_labels,
                weight_grid=_convex_weight_grid(
                    n_sources=1 + len(post_blend_probs),
                    step=post_blend_step,
                ),
                threshold_grid=threshold_grid,
            )
            results[post_blend_key]['profile']['post_blend_oof_npzs'] = list(post_blend_paths)
        if len(post_blend_probs) == 1 and _to_bool(args.post_blend_ltp_ctp_override):
            post_blend_ltp_key = '{}_ltp_ctp_override'.format(post_blend_key)
            results[post_blend_ltp_key] = evaluate_foldwise_classwise_blend_ltp_ctp_override(
                prob_a=oof['prob_matrix'],
                prob_b=post_blend_probs[0],
                true_idx=oof['true_idx'],
                fold_ids=oof['fold_ids'],
                rows=rows,
                class_names=oof['class_names'],
                alpha_grid=alpha_grid,
                threshold_grid=threshold_grid,
                score_grid=score_grid,
                model_kind=args.ltp_ctp_model_kind,
                n_estimators=int(args.ltp_ctp_n_estimators),
                random_state=ltp_ctp_random_state,
                class_weight=args.class_weight,
                max_features=args.max_features,
                min_samples_leaf=int(args.min_samples_leaf),
                ltp_ctp_class_weight=ltp_ctp_class_weight,
                ltp_ctp_min_samples_leaf=ltp_ctp_min_samples_leaf,
                ltp_source_classes=args.ltp_source_classes,
            )
        if len(post_blend_probs) > 1 and _to_bool(args.post_blend_ltp_ctp_override):
            post_blend_ltp_key = '{}_ltp_ctp_override'.format(post_blend_key)
            results[post_blend_ltp_key] = evaluate_foldwise_classwise_multi_blend_ltp_ctp_override(
                prob_matrices=[oof['prob_matrix']] + post_blend_probs,
                true_idx=oof['true_idx'],
                fold_ids=oof['fold_ids'],
                rows=rows,
                class_names=oof['class_names'],
                source_labels=source_labels,
                weight_grid=_convex_weight_grid(
                    n_sources=1 + len(post_blend_probs),
                    step=post_blend_step,
                ),
                threshold_grid=threshold_grid,
                score_grid=score_grid,
                model_kind=args.ltp_ctp_model_kind,
                n_estimators=int(args.ltp_ctp_n_estimators),
                random_state=ltp_ctp_random_state,
                class_weight=args.class_weight,
                max_features=args.max_features,
                min_samples_leaf=int(args.min_samples_leaf),
                ltp_ctp_class_weight=ltp_ctp_class_weight,
                ltp_ctp_min_samples_leaf=ltp_ctp_min_samples_leaf,
                ltp_source_classes=args.ltp_source_classes,
                fixed_fold_rows=results[post_blend_key]['folds'],
            )
    out = {
        'training_tsv': str(args.training_tsv),
        'stack_oof_npz': str(args.stack_oof_npz),
        'class_names': list(oof['class_names']),
        'targetp_reference': TARGETP_TABLE1_REFERENCE,
        'targetp_macro_f1': _targetp_macro_f1(oof['class_names']),
        'stack_profile': oof['profile'],
        'results': results,
    }
    out['class_rows'] = _class_rows(
        class_names=oof['class_names'],
        results=list(results.items()),
    )
    out_json_dir = os.path.dirname(str(args.out_json))
    if out_json_dir != '':
        os.makedirs(out_json_dir, exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    md = render_markdown(out)
    out_md_dir = os.path.dirname(str(args.out_md))
    if out_md_dir != '':
        os.makedirs(out_md_dir, exist_ok=True)
    with open(args.out_md, 'w', encoding='utf-8') as fh:
        fh.write(md + '\n')
    print(md)
    return out


if __name__ == '__main__':
    main()

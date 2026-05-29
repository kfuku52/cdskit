import argparse
import csv
import json
import os

import numpy as np

from cdskit.localize_model import (
    FEATURE_NAMES,
    TARGETP_FEATURE_ENSEMBLE_PROFILE,
    extract_targetp_feature_ensemble_features,
    fit_perox_binary_classifier,
    load_localize_model,
    save_localize_model,
)
from cdskit.localize_learn import (
    LOCALIZATION_CLASSES,
    build_training_matrix,
)
from cdskit.targetp_benchmark import TARGETP_TABLE1_REFERENCE, compute_prf_by_class
from cdskit.targetp_blend import (
    _apply_organism_gate,
    _blend_classwise,
    _load_oof_npz,
    _read_organism_group_mask,
    _read_true_idx_from_training_tsv,
    _save_oof_npz,
)


TARGETP_FEATURE_ENSEMBLE_DEFAULTS = {
    'model_kind': 'extra_trees',
    'n_estimators': 300,
    'random_state': 1,
    'class_weight': 'balanced',
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
}

TARGETP_FEATURE_BINARY_MODEL_KINDS = frozenset([
    'binary_extra_trees',
    'extra_trees_ovr',
])


def read_training_rows(path):
    with open(path, 'r', encoding='utf-8', newline='') as inp:
        return list(csv.DictReader(inp, delimiter='\t'))


def build_targetp_feature_matrix(rows):
    features = [
        extract_targetp_feature_ensemble_features(
            aa_seq=row.get('sequence', ''),
            organism_group=row.get('organism_group', ''),
        )
        for row in rows
    ]
    if len(features) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    return np.vstack(features).astype(np.float64)


def _true_idx_from_rows(rows, class_names):
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    return np.asarray([
        int(class_to_idx[str(row.get('localization', ''))])
        for row in rows
    ], dtype=np.int64)


def _fold_ids_from_rows(rows):
    return np.asarray([str(row.get('fold_id', '')) for row in rows])


def make_targetp_feature_classifier(
    model_kind='extra_trees',
    n_estimators=300,
    random_state=1,
    class_weight='balanced',
    max_features='sqrt',
    min_samples_leaf=1,
):
    model_kind = str(model_kind).strip().lower()
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
    raise ValueError('Unsupported targetp feature model_kind: {}'.format(model_kind))


def _is_binary_feature_model_kind(model_kind):
    return str(model_kind).strip().lower() in TARGETP_FEATURE_BINARY_MODEL_KINDS


def make_targetp_binary_feature_classifiers(
    features,
    true_idx,
    class_names,
    model_kind='binary_extra_trees',
    n_estimators=300,
    random_state=1,
    class_weight='balanced',
    max_features='sqrt',
    min_samples_leaf=1,
):
    model_kind = str(model_kind).strip().lower()
    if not _is_binary_feature_model_kind(model_kind):
        raise ValueError('Unsupported binary targetp feature model_kind: {}'.format(model_kind))
    classifiers = list()
    for class_i, _ in enumerate(class_names):
        classifier = make_targetp_feature_classifier(
            model_kind='extra_trees',
            n_estimators=n_estimators,
            random_state=int(random_state) + int(class_i),
            class_weight=class_weight,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
        )
        binary_y = (np.asarray(true_idx, dtype=np.int64) == int(class_i)).astype(np.int64)
        classifier.fit(features, binary_y)
        classifiers.append(classifier)
    return classifiers


def predict_feature_classifier_prob_matrix(classifier, feature_matrix, class_names):
    prob = np.asarray(classifier.predict_proba(feature_matrix), dtype=np.float64)
    out = np.zeros((feature_matrix.shape[0], len(class_names)), dtype=np.float64)
    class_to_col = {int(cls): i for i, cls in enumerate(list(classifier.classes_))}
    for class_i in range(len(class_names)):
        if class_i in class_to_col:
            out[:, class_i] = prob[:, class_to_col[class_i]]
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    return out / row_sum


def predict_binary_feature_classifiers_prob_matrix(classifiers, feature_matrix, class_names):
    if not isinstance(classifiers, list) or len(classifiers) != len(class_names):
        raise ValueError('binary feature classifiers should be a list matching class_names.')
    out = np.zeros((feature_matrix.shape[0], len(class_names)), dtype=np.float64)
    for class_i, classifier in enumerate(classifiers):
        proba = np.asarray(classifier.predict_proba(feature_matrix), dtype=np.float64)
        classes = [int(v) for v in list(getattr(classifier, 'classes_', []))]
        if 1 in classes:
            out[:, class_i] = proba[:, classes.index(1)]
        elif len(classes) == 1 and classes[0] == 1:
            out[:, class_i] = 1.0
        else:
            out[:, class_i] = 0.0
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    return out / row_sum


def run_targetp_feature_ensemble_oof(
    training_tsv,
    model_kind='extra_trees',
    n_estimators=300,
    random_state=1,
    class_weight='balanced',
    max_features='sqrt',
    min_samples_leaf=1,
):
    rows = read_training_rows(path=training_tsv)
    class_names = list(LOCALIZATION_CLASSES)
    true_idx = _true_idx_from_rows(rows=rows, class_names=class_names)
    fold_ids = _fold_ids_from_rows(rows=rows)
    if np.any(fold_ids == ''):
        raise ValueError('TargetP feature ensemble OOF requires fold_id in every row.')
    features = build_targetp_feature_matrix(rows=rows)
    prob_matrix = np.zeros((features.shape[0], len(class_names)), dtype=np.float64)
    fold_rows = list()
    for fold_id in sorted(set(fold_ids.tolist())):
        valid_mask = fold_ids == fold_id
        train_mask = ~valid_mask
        if _is_binary_feature_model_kind(model_kind):
            classifiers = make_targetp_binary_feature_classifiers(
                features=features[train_mask, :],
                true_idx=true_idx[train_mask],
                class_names=class_names,
                model_kind=model_kind,
                n_estimators=n_estimators,
                random_state=random_state,
                class_weight=class_weight,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
            )
            prob_matrix[valid_mask, :] = predict_binary_feature_classifiers_prob_matrix(
                classifiers=classifiers,
                feature_matrix=features[valid_mask, :],
                class_names=class_names,
            )
        else:
            classifier = make_targetp_feature_classifier(
                model_kind=model_kind,
                n_estimators=n_estimators,
                random_state=random_state,
                class_weight=class_weight,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
            )
            classifier.fit(features[train_mask, :], true_idx[train_mask])
            prob_matrix[valid_mask, :] = predict_feature_classifier_prob_matrix(
                classifier=classifier,
                feature_matrix=features[valid_mask, :],
                class_names=class_names,
            )
        fold_rows.append({
            'fold_id': str(fold_id),
            'n_train': int(np.sum(train_mask)),
            'n_valid': int(np.sum(valid_mask)),
        })
    return {
        'prob_matrix': prob_matrix,
        'true_idx': true_idx,
        'class_names': class_names,
        'fold_ids': fold_ids,
        'folds': fold_rows,
        'feature_dim': int(features.shape[1]),
        'profile': dict(TARGETP_FEATURE_ENSEMBLE_DEFAULTS, **{
            'model_kind': str(model_kind),
            'n_estimators': int(n_estimators),
            'random_state': int(random_state),
            'class_weight': str(class_weight),
            'max_features': str(max_features),
            'min_samples_leaf': int(min_samples_leaf),
            'feature_profile': TARGETP_FEATURE_ENSEMBLE_PROFILE['name'],
        }),
    }


def fit_targetp_feature_runtime_model(
    training_tsv,
    class_thresholds=None,
    model_kind='extra_trees',
    n_estimators=300,
    random_state=1,
    class_weight='balanced',
    max_features='sqrt',
    min_samples_leaf=1,
):
    rows = read_training_rows(path=training_tsv)
    class_names = list(LOCALIZATION_CLASSES)
    true_idx = _true_idx_from_rows(rows=rows, class_names=class_names)
    features = build_targetp_feature_matrix(rows=rows)
    if _is_binary_feature_model_kind(model_kind):
        classifier = None
        binary_classifiers = make_targetp_binary_feature_classifiers(
            features=features,
            true_idx=true_idx,
            class_names=class_names,
            model_kind=model_kind,
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
        )
    else:
        binary_classifiers = None
        classifier = make_targetp_feature_classifier(
            model_kind=model_kind,
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
        )
        classifier.fit(features, true_idx)
    broad_features, _, _, perox_labels, skipped, _ = build_training_matrix(
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
        raise ValueError('TargetP feature runtime export requires no skipped rows.')
    if class_thresholds is None:
        class_thresholds = {class_name: 1.0 for class_name in class_names}
    model = {
        'model_type': 'targetp_feature_ensemble_v1',
        'feature_names': list(FEATURE_NAMES),
        'localization_model': {
            'mode': 'targetp_feature_ensemble',
            'class_order': class_names,
            'classifier': classifier,
            'binary_classifiers': binary_classifiers,
            'class_thresholds': dict(class_thresholds),
            'feature_dim': int(features.shape[1]),
            'feature_profile': TARGETP_FEATURE_ENSEMBLE_PROFILE['name'],
            'classifier_profile': dict(TARGETP_FEATURE_ENSEMBLE_DEFAULTS, **{
                'model_kind': str(model_kind),
                'n_estimators': int(n_estimators),
                'random_state': int(random_state),
                'class_weight': str(class_weight),
                'max_features': str(max_features),
                'min_samples_leaf': int(min_samples_leaf),
            }),
        },
        'perox_model': fit_perox_binary_classifier(
            features=broad_features,
            labels=perox_labels,
        ),
        'metadata': {
            'training_tsv': str(training_tsv),
            'num_training_rows': int(len(rows)),
            'model_arch': 'targetp_feature_ensemble_v1',
        },
    }
    return model


def build_targetp_feature_blend_runtime_model(
    feature_model,
    blend_source_model,
    alpha_by_class,
    class_thresholds,
    blend_base_index=1,
    metadata=None,
):
    if feature_model.get('model_type') != 'targetp_feature_ensemble_v1':
        raise ValueError('feature_model should have model_type targetp_feature_ensemble_v1.')
    if blend_source_model.get('model_type') == 'targetp_blend_v1':
        base_models = blend_source_model['localization_model'].get('base_models', [])
        blend_base_index = int(blend_base_index)
        if blend_base_index < 0 or blend_base_index >= len(base_models):
            raise ValueError('blend_base_index is outside the source targetp_blend base_models.')
        blend_base = base_models[blend_base_index]
    else:
        blend_base = {
            'model_type': str(blend_source_model.get('model_type', '')),
            'localization_model': blend_source_model.get('localization_model', {}),
        }
    model = {
        'model_type': 'targetp_blend_v1',
        'feature_names': list(FEATURE_NAMES),
        'localization_model': {
            'class_order': list(LOCALIZATION_CLASSES),
            'base_models': [
                {
                    'model_type': 'targetp_feature_ensemble_v1',
                    'localization_model': feature_model['localization_model'],
                },
                blend_base,
            ],
            'alpha_by_class': dict(alpha_by_class),
            'class_thresholds': dict(class_thresholds),
        },
        'perox_model': feature_model['perox_model'],
        'metadata': {} if metadata is None else dict(metadata),
    }
    return model


def _prediction_indices_with_thresholds(prob_matrix, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float64).reshape((1, -1))
    thresholds[thresholds <= 0.0] = 1.0
    return np.argmax(np.asarray(prob_matrix, dtype=np.float64) / thresholds, axis=1).astype(np.int64)


def _metrics_from_prediction_indices(pred_idx, true_idx, class_names):
    true_names = [class_names[int(i)] for i in true_idx.tolist()]
    pred_names = [class_names[int(i)] for i in pred_idx.tolist()]
    by_class = compute_prf_by_class(
        true_classes=true_names,
        pred_classes=pred_names,
        class_names=class_names,
    )
    macro_f1 = float(np.mean(np.asarray([
        by_class[class_name]['f1'] for class_name in class_names
    ], dtype=np.float64)))
    return {
        'overall_accuracy': float(np.mean(pred_idx == true_idx)),
        'macro_f1': macro_f1,
        'by_class': by_class,
    }


def optimize_class_thresholds(prob_matrix, true_idx, class_names, grid):
    thresholds = np.ones((len(class_names),), dtype=np.float64)
    best_pred = _prediction_indices_with_thresholds(
        prob_matrix=prob_matrix,
        thresholds=thresholds,
    )
    best_metrics = _metrics_from_prediction_indices(
        pred_idx=best_pred,
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
                pred = _prediction_indices_with_thresholds(
                    prob_matrix=prob_matrix,
                    thresholds=tmp,
                )
                metrics = _metrics_from_prediction_indices(
                    pred_idx=pred,
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


def _optimize_classwise_alpha(prob_a, prob_b, true_idx, class_names, alpha_grid):
    alpha = np.ones((len(class_names),), dtype=np.float64)
    best_metrics = None
    best_global = 1.0
    for trial in alpha_grid:
        pred = np.argmax(_blend_classwise(
            prob_a=prob_a,
            prob_b=prob_b,
            alpha_by_class=np.full((len(class_names),), float(trial), dtype=np.float64),
        ), axis=1).astype(np.int64)
        metrics = _metrics_from_prediction_indices(
            pred_idx=pred,
            true_idx=true_idx,
            class_names=class_names,
        )
        if best_metrics is None or metrics['macro_f1'] > best_metrics['macro_f1']:
            best_metrics = metrics
            best_global = float(trial)
    alpha[:] = best_global
    improved = True
    while improved:
        improved = False
        for class_i in range(len(class_names)):
            best_local = float(alpha[class_i])
            best_local_metrics = best_metrics
            for trial in alpha_grid:
                tmp = alpha.copy()
                tmp[class_i] = float(trial)
                pred = np.argmax(_blend_classwise(
                    prob_a=prob_a,
                    prob_b=prob_b,
                    alpha_by_class=tmp,
                ), axis=1).astype(np.int64)
                metrics = _metrics_from_prediction_indices(
                    pred_idx=pred,
                    true_idx=true_idx,
                    class_names=class_names,
                )
                if metrics['macro_f1'] > best_local_metrics['macro_f1']:
                    best_local = float(trial)
                    best_local_metrics = metrics
            if best_local != float(alpha[class_i]):
                alpha[class_i] = best_local
                best_metrics = best_local_metrics
                improved = True
    return alpha, best_metrics


def evaluate_foldwise_thresholds(prob_matrix, true_idx, fold_ids, class_names, threshold_grid):
    pred_idx = np.zeros((np.asarray(true_idx).shape[0],), dtype=np.int64)
    fold_rows = list()
    for fold_id in sorted(set([str(v) for v in np.asarray(fold_ids).tolist()])):
        valid_mask = np.asarray([str(v) == fold_id for v in np.asarray(fold_ids).tolist()], dtype=bool)
        train_mask = ~valid_mask
        thresholds, train_metrics = optimize_class_thresholds(
            prob_matrix=np.asarray(prob_matrix)[train_mask, :],
            true_idx=np.asarray(true_idx)[train_mask],
            class_names=class_names,
            grid=threshold_grid,
        )
        pred_idx[valid_mask] = _prediction_indices_with_thresholds(
            prob_matrix=np.asarray(prob_matrix)[valid_mask, :],
            thresholds=thresholds,
        )
        fold_rows.append({
            'fold_id': str(fold_id),
            'class_thresholds': {
                class_names[i]: float(thresholds[i]) for i in range(len(class_names))
            },
            'train_macro_f1': float(train_metrics['macro_f1']),
            'n_train': int(np.sum(train_mask)),
            'n_valid': int(np.sum(valid_mask)),
        })
    return {
        'description': 'Each held-out fold is predicted using class thresholds optimized on the other folds.',
        'metrics': _metrics_from_prediction_indices(
            pred_idx=pred_idx,
            true_idx=np.asarray(true_idx, dtype=np.int64),
            class_names=class_names,
        ),
        'folds': fold_rows,
    }


def evaluate_foldwise_classwise_blend(
    prob_a,
    prob_b,
    true_idx,
    fold_ids,
    class_names,
    alpha_grid,
    threshold_grid,
):
    pred_idx = np.zeros((np.asarray(true_idx).shape[0],), dtype=np.int64)
    fold_rows = list()
    fold_ids = np.asarray(fold_ids)
    for fold_id in sorted(set([str(v) for v in fold_ids.tolist()])):
        valid_mask = np.asarray([str(v) == fold_id for v in fold_ids.tolist()], dtype=bool)
        train_mask = ~valid_mask
        alpha, alpha_metrics = _optimize_classwise_alpha(
            prob_a=np.asarray(prob_a)[train_mask, :],
            prob_b=np.asarray(prob_b)[train_mask, :],
            true_idx=np.asarray(true_idx)[train_mask],
            class_names=class_names,
            alpha_grid=alpha_grid,
        )
        train_blend = _blend_classwise(
            prob_a=np.asarray(prob_a)[train_mask, :],
            prob_b=np.asarray(prob_b)[train_mask, :],
            alpha_by_class=alpha,
        )
        thresholds, threshold_metrics = optimize_class_thresholds(
            prob_matrix=train_blend,
            true_idx=np.asarray(true_idx)[train_mask],
            class_names=class_names,
            grid=threshold_grid,
        )
        valid_blend = _blend_classwise(
            prob_a=np.asarray(prob_a)[valid_mask, :],
            prob_b=np.asarray(prob_b)[valid_mask, :],
            alpha_by_class=alpha,
        )
        pred_idx[valid_mask] = _prediction_indices_with_thresholds(
            prob_matrix=valid_blend,
            thresholds=thresholds,
        )
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
            'n_train': int(np.sum(train_mask)),
            'n_valid': int(np.sum(valid_mask)),
        })
    return {
        'description': 'Each held-out fold is predicted using classwise alpha and class thresholds optimized on the other folds.',
        'metrics': _metrics_from_prediction_indices(
            pred_idx=pred_idx,
            true_idx=np.asarray(true_idx, dtype=np.int64),
            class_names=class_names,
        ),
        'folds': fold_rows,
    }


def _metrics_from_prob_matrix(prob_matrix, true_idx, class_names):
    return _metrics_from_prediction_indices(
        pred_idx=np.argmax(np.asarray(prob_matrix), axis=1).astype(np.int64),
        true_idx=np.asarray(true_idx, dtype=np.int64),
        class_names=class_names,
    )


def _targetp_macro_f1(class_names):
    return float(np.mean(np.asarray([
        TARGETP_TABLE1_REFERENCE[class_name]['f1'] for class_name in class_names
    ], dtype=np.float64)))


def _class_rows(class_names, results):
    rows = list()
    for class_name in class_names:
        row = {
            'class': class_name,
            'targetp_f1': float(TARGETP_TABLE1_REFERENCE[class_name]['f1']),
        }
        for key, result in results:
            row[key] = float(result['metrics']['by_class'][class_name]['f1'])
        rows.append(row)
    return rows


def _render_markdown(out):
    results = list(out['results'].items())
    headers = ['Class', 'TargetP F1'] + [
        '{} F1'.format(key) for key, _ in results
    ]
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
    if 'feature_thresholds' in out:
        lines.append('feature thresholds: {}'.format(out['feature_thresholds']))
    if 'blend_alpha_by_class' in out:
        lines.append('feature/blend alpha by class: {}'.format(out['blend_alpha_by_class']))
    if 'blend_thresholds' in out:
        lines.append('feature/blend thresholds: {}'.format(out['blend_thresholds']))
    return '\n'.join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate a CPU TargetP feature ensemble on the fold-fixed TargetP benchmark.',
    )
    parser.add_argument('--training_tsv', default='data/localize_bench/targetp2_benchmark.tsv', type=str)
    parser.add_argument('--reuse_oof_cache', default='yes', choices=['yes', 'no'], type=str)
    parser.add_argument('--feature_oof_npz', default='data/localize_bench/targetp2_oof_feature_ensemble.npz', type=str)
    parser.add_argument('--organism_gate', default='yes', choices=['yes', 'no'], type=str)
    parser.add_argument('--model_kind', default=TARGETP_FEATURE_ENSEMBLE_DEFAULTS['model_kind'], choices=['extra_trees', 'random_forest', 'binary_extra_trees', 'extra_trees_ovr'], type=str)
    parser.add_argument('--n_estimators', default=TARGETP_FEATURE_ENSEMBLE_DEFAULTS['n_estimators'], type=int)
    parser.add_argument('--random_state', default=TARGETP_FEATURE_ENSEMBLE_DEFAULTS['random_state'], type=int)
    parser.add_argument('--class_weight', default=TARGETP_FEATURE_ENSEMBLE_DEFAULTS['class_weight'], type=str)
    parser.add_argument('--max_features', default=TARGETP_FEATURE_ENSEMBLE_DEFAULTS['max_features'], type=str)
    parser.add_argument('--min_samples_leaf', default=TARGETP_FEATURE_ENSEMBLE_DEFAULTS['min_samples_leaf'], type=int)
    parser.add_argument('--threshold_grid', default='0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.5,0.65,0.8,1.0,1.25,1.5,2.0,3.0,5.0', type=str)
    parser.add_argument('--blend_oof_npz', default='', type=str)
    parser.add_argument('--blend_label', default='blend_model', type=str)
    parser.add_argument('--blend_grid_step', default=0.05, type=float)
    parser.add_argument('--model_out', default='', type=str)
    parser.add_argument('--blend_model', default='', type=str)
    parser.add_argument('--blend_model_base_index', default=1, type=int)
    parser.add_argument('--blend_model_out', default='', type=str)
    parser.add_argument('--out_json', default='data/localize_bench/targetp2_feature_ensemble_eval.json', type=str)
    parser.add_argument('--out_md', default='data/localize_bench/targetp2_feature_ensemble_eval.md', type=str)
    return parser


def _to_bool_yes_no(value):
    return str(value).strip().lower() in ['yes', 'y', 'true', '1']


def main():
    args = build_parser().parse_args()
    class_names = list(LOCALIZATION_CLASSES)
    fallback_true_idx = _read_true_idx_from_training_tsv(
        training_tsv=args.training_tsv,
        class_names=class_names,
    )
    reuse_cache = _to_bool_yes_no(args.reuse_oof_cache)
    if reuse_cache and os.path.exists(args.feature_oof_npz):
        feature_prob, true_idx, names = _load_oof_npz(
            path=args.feature_oof_npz,
            fallback_true_idx=fallback_true_idx,
        )
        if names != class_names:
            raise ValueError('Class names in feature_oof_npz do not match LOCALIZATION_CLASSES.')
        fold_ids = _fold_ids_from_rows(read_training_rows(path=args.training_tsv))
        feature_profile = {'used_cache': True}
    else:
        oof = run_targetp_feature_ensemble_oof(
            training_tsv=args.training_tsv,
            model_kind=args.model_kind,
            n_estimators=int(args.n_estimators),
            random_state=int(args.random_state),
            class_weight=args.class_weight,
            max_features=args.max_features,
            min_samples_leaf=int(args.min_samples_leaf),
        )
        feature_prob = oof['prob_matrix']
        true_idx = oof['true_idx']
        fold_ids = oof['fold_ids']
        feature_profile = dict(oof['profile'])
        feature_profile['used_cache'] = False
        _save_oof_npz(
            path=args.feature_oof_npz,
            prob_matrix=feature_prob,
            true_idx=true_idx,
            class_names=class_names,
        )

    organism_gate = _to_bool_yes_no(args.organism_gate)
    if organism_gate:
        plant_mask = _read_organism_group_mask(training_tsv=args.training_tsv)
        feature_prob = _apply_organism_gate(
            prob_matrix=feature_prob,
            plant_mask=plant_mask,
            class_names=class_names,
        )

    threshold_grid = [
        float(v.strip()) for v in str(args.threshold_grid).split(',')
        if str(v).strip() != ''
    ]
    threshold_grid = sorted(set(threshold_grid))
    feature_thresholds, feature_threshold_metrics = optimize_class_thresholds(
        prob_matrix=feature_prob,
        true_idx=true_idx,
        class_names=class_names,
        grid=threshold_grid,
    )
    feature_foldwise = evaluate_foldwise_thresholds(
        prob_matrix=feature_prob,
        true_idx=true_idx,
        fold_ids=fold_ids,
        class_names=class_names,
        threshold_grid=threshold_grid,
    )
    results = {
        'feature_argmax': {
            'metrics': _metrics_from_prob_matrix(
                prob_matrix=feature_prob,
                true_idx=true_idx,
                class_names=class_names,
            ),
        },
        'feature_threshold': {
            'metrics': feature_threshold_metrics,
        },
        'feature_foldwise_threshold': feature_foldwise,
    }

    blend_prob = None
    if str(args.blend_oof_npz).strip() != '':
        blend_prob, blend_true, blend_names = _load_oof_npz(
            path=args.blend_oof_npz,
            fallback_true_idx=true_idx,
        )
        if blend_names != class_names:
            raise ValueError('Class names in blend_oof_npz do not match LOCALIZATION_CLASSES.')
        if np.any(blend_true != true_idx):
            raise ValueError('True labels in blend_oof_npz do not match feature OOF labels.')
        if organism_gate:
            plant_mask = _read_organism_group_mask(training_tsv=args.training_tsv)
            blend_prob = _apply_organism_gate(
                prob_matrix=blend_prob,
                plant_mask=plant_mask,
                class_names=class_names,
            )
        step = float(args.blend_grid_step)
        alpha_grid = sorted(set([round(i * step, 10) for i in range(int(np.floor(1.0 / step)) + 1)] + [1.0]))
        alpha_by_class, blend_alpha_metrics = _optimize_classwise_alpha(
            prob_a=feature_prob,
            prob_b=blend_prob,
            true_idx=true_idx,
            class_names=class_names,
            alpha_grid=alpha_grid,
        )
        classwise_blend = _blend_classwise(
            prob_a=feature_prob,
            prob_b=blend_prob,
            alpha_by_class=alpha_by_class,
        )
        blend_thresholds, blend_threshold_metrics = optimize_class_thresholds(
            prob_matrix=classwise_blend,
            true_idx=true_idx,
            class_names=class_names,
            grid=threshold_grid,
        )
        results['feature_{}_classwise'.format(args.blend_label)] = {
            'metrics': blend_alpha_metrics,
        }
        results['feature_{}_threshold'.format(args.blend_label)] = {
            'metrics': blend_threshold_metrics,
        }
        results['feature_{}_foldwise'.format(args.blend_label)] = evaluate_foldwise_classwise_blend(
            prob_a=feature_prob,
            prob_b=blend_prob,
            true_idx=true_idx,
            fold_ids=fold_ids,
            class_names=class_names,
            alpha_grid=alpha_grid,
            threshold_grid=threshold_grid,
        )

    feature_threshold_dict = {
        class_names[i]: float(feature_thresholds[i]) for i in range(len(class_names))
    }
    out = {
        'training_tsv': str(args.training_tsv),
        'feature_oof_npz': str(args.feature_oof_npz),
        'organism_gate': bool(organism_gate),
        'class_names': class_names,
        'targetp_reference': TARGETP_TABLE1_REFERENCE,
        'targetp_macro_f1': _targetp_macro_f1(class_names=class_names),
        'feature_profile': feature_profile,
        'feature_thresholds': feature_threshold_dict,
        'results': results,
    }
    if blend_prob is not None:
        out['blend_oof_npz'] = str(args.blend_oof_npz)
        out['blend_label'] = str(args.blend_label)
        out['blend_alpha_by_class'] = {
            class_names[i]: float(alpha_by_class[i]) for i in range(len(class_names))
        }
        out['blend_thresholds'] = {
            class_names[i]: float(blend_thresholds[i]) for i in range(len(class_names))
        }
    out['class_rows'] = _class_rows(
        class_names=class_names,
        results=list(results.items()),
    )

    export_feature_model = None
    if str(args.model_out).strip() != '' or str(args.blend_model_out).strip() != '':
        export_feature_model = fit_targetp_feature_runtime_model(
            training_tsv=args.training_tsv,
            class_thresholds=feature_threshold_dict,
            model_kind=args.model_kind,
            n_estimators=int(args.n_estimators),
            random_state=int(args.random_state),
            class_weight=args.class_weight,
            max_features=args.max_features,
            min_samples_leaf=int(args.min_samples_leaf),
        )
        export_feature_model['metadata']['benchmark_feature_threshold_macro_f1'] = float(
            feature_threshold_metrics['macro_f1']
        )
        export_feature_model['metadata']['benchmark_feature_foldwise_threshold_macro_f1'] = float(
            feature_foldwise['metrics']['macro_f1']
        )
    if str(args.model_out).strip() != '':
        save_localize_model(model=export_feature_model, path=str(args.model_out))
        out['model_out'] = str(args.model_out)
    if str(args.blend_model_out).strip() != '':
        if blend_prob is None:
            raise ValueError('--blend_model_out requires --blend_oof_npz for calibration.')
        if str(args.blend_model).strip() == '':
            raise ValueError('--blend_model_out requires --blend_model as the runtime source.')
        blend_source = load_localize_model(str(args.blend_model))
        blend_runtime = build_targetp_feature_blend_runtime_model(
            feature_model=export_feature_model,
            blend_source_model=blend_source,
            alpha_by_class=out['blend_alpha_by_class'],
            class_thresholds=out['blend_thresholds'],
            blend_base_index=int(args.blend_model_base_index),
            metadata={
                'model_arch': 'targetp_feature_blend_v1',
                'training_tsv': str(args.training_tsv),
                'feature_oof_npz': str(args.feature_oof_npz),
                'blend_oof_npz': str(args.blend_oof_npz),
                'blend_model': str(args.blend_model),
                'blend_model_base_index': int(args.blend_model_base_index),
                'benchmark_feature_blend_threshold_macro_f1': float(
                    results['feature_{}_threshold'.format(args.blend_label)]['metrics']['macro_f1']
                ),
                'benchmark_feature_blend_foldwise_macro_f1': float(
                    results['feature_{}_foldwise'.format(args.blend_label)]['metrics']['macro_f1']
                ),
            },
        )
        save_localize_model(model=blend_runtime, path=str(args.blend_model_out))
        out['blend_model_out'] = str(args.blend_model_out)

    out_dir = os.path.dirname(str(args.out_json))
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    md = _render_markdown(out)
    md_dir = os.path.dirname(str(args.out_md))
    if md_dir != '':
        os.makedirs(md_dir, exist_ok=True)
    with open(args.out_md, 'w', encoding='utf-8') as fh:
        fh.write(md + '\n')

    print(md)
    return out


if __name__ == '__main__':
    main()

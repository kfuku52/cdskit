import argparse
import csv
import json
import os

import numpy as np

from cdskit.localize_learn import LOCALIZATION_CLASSES
from cdskit.targetp_benchmark import TARGETP_TABLE1_REFERENCE
from cdskit.targetp_feature_ensemble import (
    _class_rows,
    _metrics_from_prob_matrix,
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
}


def read_training_rows(path):
    with open(path, 'r', encoding='utf-8', newline='') as inp:
        return list(csv.DictReader(inp, delimiter='\t'))


def fold_ids_from_rows(rows):
    return np.asarray([str(row.get('fold_id', '')) for row in rows])


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
    for fold_id in sorted(set(fold_ids.tolist())):
        valid_mask = fold_ids == fold_id
        train_mask = ~valid_mask
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
        'profile': dict(TARGETP_STACK_DEFAULTS, **{
            'model_kind': str(model_kind),
            'n_estimators': int(n_estimators),
            'random_state': int(random_state),
            'class_weight': str(class_weight),
            'max_features': str(max_features),
            'min_samples_leaf': int(min_samples_leaf),
            'include_sequence_features': bool(include_sequence_features),
            'organism_gate': bool(organism_gate),
            'base_oof_npzs': [str(path) for path in base_oof_npzs],
        }),
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
        '--organism_gate',
        default='yes' if TARGETP_STACK_DEFAULTS['organism_gate'] else 'no',
        choices=['yes', 'no'],
        type=str,
    )
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
    threshold_grid = sorted(set([
        float(v.strip()) for v in str(args.threshold_grid).split(',')
        if v.strip() != ''
    ]))
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

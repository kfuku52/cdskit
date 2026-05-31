import argparse
import json

import numpy as np

from cdskit.localize_model import (
    LOCALIZATION_CLASSES,
    _targetp_notp_specialist_feature_vector,
    _targetp_reranker_feature_vector,
    load_localize_model,
    predict_localization_and_peroxisome,
    save_localize_model,
)
from cdskit.targetp_external_eval import load_fixed_uniprot_holdout_rows, read_tsv
from cdskit.targetp_blend import _fit_sklearn_model, build_targetp_pair_blend_runtime_model


TARGETP_NOTP_SPECIALIST_PROFILE = {
    'name': 'targetp_notp_probability_sequence_v1',
    'model_kind': 'HistGradientBoostingClassifier',
    'max_iter': 120,
    'learning_rate': 0.04,
    'l2_regularization': 0.02,
    'random_state': 99,
    'class_weight': 'balanced',
    'threshold': 0.54,
}


TARGETP_RERANKER_PROFILE = {
    'name': 'targetp_probability_pair_sequence_reranker_v2',
    'model_kind': 'HistGradientBoostingClassifier',
    'max_iter': 120,
    'learning_rate': 0.04,
    'l2_regularization': 0.02,
    'random_state': 111,
    'class_weight': 'balanced',
    'threshold': 0.50,
}


TARGETP_MTP_NOTP_SPECIALIST_PROFILE = {
    'name': 'targetp_mtp_notp_pair_sequence_v1',
    'feature_profile': TARGETP_RERANKER_PROFILE['name'],
    'model_kind': 'HistGradientBoostingClassifier',
    'max_iter': 120,
    'learning_rate': 0.04,
    'l2_regularization': 0.02,
    'random_state': 811,
    'class_weight': 'balanced',
    'threshold': 0.60,
}


def _normalize_class_weight(value):
    text = str(value or '').strip()
    if text == '' or text.lower() in ['none', 'null']:
        return None
    return text


def _class_weight_metadata_value(value):
    resolved = _normalize_class_weight(value)
    return 'none' if resolved is None else str(resolved)


def _parse_class_values(text, default):
    text = str(text or '').strip()
    if text == '':
        return {class_name: float(default) for class_name in LOCALIZATION_CLASSES}
    if '=' not in text:
        value = float(text)
        return {class_name: value for class_name in LOCALIZATION_CLASSES}
    out = {class_name: float(default) for class_name in LOCALIZATION_CLASSES}
    for part in text.split(','):
        part = part.strip()
        if part == '':
            continue
        if '=' not in part:
            raise ValueError('Class-specific values should use CLASS=VALUE entries.')
        class_name, value = part.split('=', 1)
        class_name = class_name.strip()
        if class_name not in LOCALIZATION_CLASSES:
            raise ValueError('Unknown class: {}'.format(class_name))
        out[class_name] = float(value.strip())
    return out


def _targetp_notp_specialist_training_matrix(model, rows):
    return _targetp_notp_specialist_training_matrix_from_predictions(
        model=model,
        rows=rows,
        prediction_rows=None,
    )


def _probabilities_from_prediction_row(row):
    return {
        class_name: float(row.get('p_{}'.format(class_name), 0.0))
        for class_name in LOCALIZATION_CLASSES
    }


def _base_probabilities_from_prediction_row(row, prefix):
    prefix = str(prefix or '').strip()
    if prefix == '':
        return _probabilities_from_prediction_row(row)
    keys = ['p_{}_{}'.format(prefix, class_name) for class_name in LOCALIZATION_CLASSES]
    if not all(key in row for key in keys):
        return _probabilities_from_prediction_row(row)
    return {
        class_name: float(row.get('p_{}_{}'.format(prefix, class_name), 0.0))
        for class_name in LOCALIZATION_CLASSES
    }


def _targetp_notp_specialist_training_matrix_from_predictions(model, rows, prediction_rows=None):
    if str(model.get('model_type', '')).strip() != 'targetp_blend_v1':
        raise ValueError('noTP specialist training requires a targetp_blend_v1 model.')
    localization_model = model.get('localization_model', {})
    if prediction_rows is not None and len(prediction_rows) != len(rows):
        raise ValueError('noTP specialist rows and prediction rows differ in length.')
    features = list()
    labels = list()
    for row_i, row in enumerate(rows):
        true_class = str(row.get('true_class', row.get('localization', '')) or '').strip()
        if true_class not in LOCALIZATION_CLASSES:
            raise ValueError('Unknown true class for noTP specialist: {}'.format(true_class))
        sequence = row.get('sequence', '')
        organism_group = row.get('organism_group', '')
        if prediction_rows is None:
            pred = predict_localization_and_peroxisome(
                aa_seq=sequence,
                model=model,
                organism_group=organism_group,
            )
            blend_details = pred.get('targetp_blend_details', {})
            base_model_probabilities = blend_details.get('base_model_probabilities', [])
            if len(base_model_probabilities) != 2:
                raise ValueError('targetp_blend_v1 did not return two base probability vectors.')
            base_probs = pred['class_probabilities']
            prob_a = base_model_probabilities[0]
            prob_b = base_model_probabilities[1]
        else:
            pred_row = prediction_rows[row_i]
            pred_true = str(pred_row.get('true_class', '') or '').strip()
            pred_acc = str(pred_row.get('accession', '') or '').strip()
            row_acc = str(row.get('accession', '') or '').strip()
            if pred_true != '' and pred_true != true_class:
                raise ValueError('noTP specialist prediction row true_class mismatch.')
            if pred_acc != '' and row_acc != '' and pred_acc != row_acc:
                raise ValueError('noTP specialist prediction row accession mismatch.')
            base_probs = _probabilities_from_prediction_row(pred_row)
            prob_a = base_probs
            prob_b = base_probs
        features.append(_targetp_notp_specialist_feature_vector(
            aa_seq=sequence,
            base_probs=base_probs,
            prob_a=prob_a,
            prob_b=prob_b,
            organism_group=organism_group,
            class_thresholds=localization_model.get('class_thresholds', {}),
        ))
        labels.append(1 if true_class == 'noTP' else 0)
    if len(features) == 0:
        raise ValueError('No rows were available for noTP specialist training.')
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _targetp_reranker_training_matrix_from_predictions(model, rows, prediction_rows=None):
    if str(model.get('model_type', '')).strip() != 'targetp_blend_v1':
        raise ValueError('TargetP reranker training requires a targetp_blend_v1 model.')
    localization_model = model.get('localization_model', {})
    if prediction_rows is not None and len(prediction_rows) != len(rows):
        raise ValueError('TargetP reranker rows and prediction rows differ in length.')
    class_to_idx = {class_name: i for i, class_name in enumerate(LOCALIZATION_CLASSES)}
    features = list()
    labels = list()
    for row_i, row in enumerate(rows):
        true_class = str(row.get('true_class', row.get('localization', '')) or '').strip()
        if true_class not in LOCALIZATION_CLASSES:
            raise ValueError('Unknown true class for TargetP reranker: {}'.format(true_class))
        sequence = row.get('sequence', '')
        organism_group = row.get('organism_group', '')
        if prediction_rows is None:
            pred = predict_localization_and_peroxisome(
                aa_seq=sequence,
                model=model,
                organism_group=organism_group,
            )
            blend_details = pred.get('targetp_blend_details', {})
            base_model_probabilities = blend_details.get('base_model_probabilities', [])
            if len(base_model_probabilities) != 2:
                raise ValueError('targetp_blend_v1 did not return two base probability vectors.')
            base_probs = pred['class_probabilities']
            prob_a = base_model_probabilities[0]
            prob_b = base_model_probabilities[1]
        else:
            pred_row = prediction_rows[row_i]
            pred_true = str(pred_row.get('true_class', '') or '').strip()
            pred_acc = str(pred_row.get('accession', '') or '').strip()
            row_acc = str(row.get('accession', '') or '').strip()
            if pred_true != '' and pred_true != true_class:
                raise ValueError('TargetP reranker prediction row true_class mismatch.')
            if pred_acc != '' and row_acc != '' and pred_acc != row_acc:
                raise ValueError('TargetP reranker prediction row accession mismatch.')
            base_probs = _probabilities_from_prediction_row(pred_row)
            prob_a = _base_probabilities_from_prediction_row(pred_row, 'a')
            prob_b = _base_probabilities_from_prediction_row(pred_row, 'b')
        features.append(_targetp_reranker_feature_vector(
            aa_seq=sequence,
            base_probs=base_probs,
            prob_a=prob_a,
            prob_b=prob_b,
            organism_group=organism_group,
            class_thresholds=localization_model.get('class_thresholds', {}),
            feature_profile=TARGETP_RERANKER_PROFILE['name'],
        ))
        labels.append(class_to_idx[true_class])
    if len(features) == 0:
        raise ValueError('No rows were available for TargetP reranker training.')
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _targetp_mtp_notp_specialist_training_matrix_from_predictions(
    model,
    rows,
    prediction_rows=None,
):
    if str(model.get('model_type', '')).strip() != 'targetp_blend_v1':
        raise ValueError('mTP/noTP specialist training requires a targetp_blend_v1 model.')
    localization_model = model.get('localization_model', {})
    if prediction_rows is not None and len(prediction_rows) != len(rows):
        raise ValueError('mTP/noTP specialist rows and prediction rows differ in length.')
    features = list()
    labels = list()
    for row_i, row in enumerate(rows):
        true_class = str(row.get('true_class', row.get('localization', '')) or '').strip()
        if true_class not in LOCALIZATION_CLASSES:
            raise ValueError('Unknown true class for mTP/noTP specialist: {}'.format(true_class))
        if true_class not in ['noTP', 'mTP']:
            continue
        sequence = row.get('sequence', '')
        organism_group = row.get('organism_group', '')
        if prediction_rows is None:
            pred = predict_localization_and_peroxisome(
                aa_seq=sequence,
                model=model,
                organism_group=organism_group,
            )
            blend_details = pred.get('targetp_blend_details', {})
            base_model_probabilities = blend_details.get('base_model_probabilities', [])
            if len(base_model_probabilities) != 2:
                raise ValueError('targetp_blend_v1 did not return two base probability vectors.')
            base_probs = pred['class_probabilities']
            prob_a = base_model_probabilities[0]
            prob_b = base_model_probabilities[1]
        else:
            pred_row = prediction_rows[row_i]
            pred_true = str(pred_row.get('true_class', '') or '').strip()
            pred_acc = str(pred_row.get('accession', '') or '').strip()
            row_acc = str(row.get('accession', '') or '').strip()
            if pred_true != '' and pred_true != true_class:
                raise ValueError('mTP/noTP specialist prediction row true_class mismatch.')
            if pred_acc != '' and row_acc != '' and pred_acc != row_acc:
                raise ValueError('mTP/noTP specialist prediction row accession mismatch.')
            base_probs = _probabilities_from_prediction_row(pred_row)
            prob_a = _base_probabilities_from_prediction_row(pred_row, 'a')
            prob_b = _base_probabilities_from_prediction_row(pred_row, 'b')
        features.append(_targetp_reranker_feature_vector(
            aa_seq=sequence,
            base_probs=base_probs,
            prob_a=prob_a,
            prob_b=prob_b,
            organism_group=organism_group,
            class_thresholds=localization_model.get('class_thresholds', {}),
            feature_profile=TARGETP_MTP_NOTP_SPECIALIST_PROFILE['feature_profile'],
        ))
        labels.append(1 if true_class == 'mTP' else 0)
    if len(features) == 0:
        raise ValueError('No rows were available for mTP/noTP specialist training.')
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def attach_targetp_notp_specialist(
    model,
    rows,
    prediction_rows=None,
    threshold=TARGETP_NOTP_SPECIALIST_PROFILE['threshold'],
    max_iter=TARGETP_NOTP_SPECIALIST_PROFILE['max_iter'],
    learning_rate=TARGETP_NOTP_SPECIALIST_PROFILE['learning_rate'],
    l2_regularization=TARGETP_NOTP_SPECIALIST_PROFILE['l2_regularization'],
    random_state=TARGETP_NOTP_SPECIALIST_PROFILE['random_state'],
):
    """Attach a CPU-runtime noTP rescue classifier trained on a development set."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:
        raise RuntimeError('noTP specialist training requires scikit-learn.') from exc

    features, labels = _targetp_notp_specialist_training_matrix_from_predictions(
        model=model,
        rows=rows,
        prediction_rows=prediction_rows,
    )
    if len(set(labels.tolist())) < 2:
        raise ValueError('noTP specialist training requires both noTP and non-noTP rows.')
    classifier = HistGradientBoostingClassifier(
        max_iter=int(max_iter),
        learning_rate=float(learning_rate),
        l2_regularization=float(l2_regularization),
        random_state=int(random_state),
        class_weight=TARGETP_NOTP_SPECIALIST_PROFILE['class_weight'],
    )
    _fit_sklearn_model(classifier, features, labels)

    localization_model = model['localization_model']
    specialist = dict(localization_model.get('targetp_specialist_postprocess', {}))
    specialist.update({
        'enabled': True,
        'notp_feature_profile': TARGETP_NOTP_SPECIALIST_PROFILE['name'],
        'notp_models': [classifier],
        'notp_weights': [1.0],
        'notp_threshold': float(threshold),
        'notp_model_kind': TARGETP_NOTP_SPECIALIST_PROFILE['model_kind'],
        'notp_max_iter': int(max_iter),
        'notp_learning_rate': float(learning_rate),
        'notp_l2_regularization': float(l2_regularization),
        'notp_random_state': int(random_state),
        'notp_class_weight': TARGETP_NOTP_SPECIALIST_PROFILE['class_weight'],
        'notp_training_rows': int(features.shape[0]),
        'notp_feature_dim': int(features.shape[1]),
    })
    localization_model['targetp_specialist_postprocess'] = specialist
    model.setdefault('metadata', {})
    model['metadata']['targetp_notp_specialist'] = {
        'feature_profile': TARGETP_NOTP_SPECIALIST_PROFILE['name'],
        'model_kind': TARGETP_NOTP_SPECIALIST_PROFILE['model_kind'],
        'threshold': float(threshold),
        'training_rows': int(features.shape[0]),
        'positive_rows': int(np.sum(labels == 1)),
        'negative_rows': int(np.sum(labels == 0)),
        'feature_dim': int(features.shape[1]),
        'max_iter': int(max_iter),
        'learning_rate': float(learning_rate),
        'l2_regularization': float(l2_regularization),
        'random_state': int(random_state),
        'class_weight': TARGETP_NOTP_SPECIALIST_PROFILE['class_weight'],
    }
    return model


def attach_targetp_mtp_notp_specialist(
    model,
    rows,
    prediction_rows=None,
    threshold=TARGETP_MTP_NOTP_SPECIALIST_PROFILE['threshold'],
    max_iter=TARGETP_MTP_NOTP_SPECIALIST_PROFILE['max_iter'],
    learning_rate=TARGETP_MTP_NOTP_SPECIALIST_PROFILE['learning_rate'],
    l2_regularization=TARGETP_MTP_NOTP_SPECIALIST_PROFILE['l2_regularization'],
    random_state=TARGETP_MTP_NOTP_SPECIALIST_PROFILE['random_state'],
    class_weight=TARGETP_MTP_NOTP_SPECIALIST_PROFILE['class_weight'],
):
    """Attach a CPU-runtime mTP/noTP resolver trained on a development set."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:
        raise RuntimeError('mTP/noTP specialist training requires scikit-learn.') from exc

    features, labels = _targetp_mtp_notp_specialist_training_matrix_from_predictions(
        model=model,
        rows=rows,
        prediction_rows=prediction_rows,
    )
    if len(set(labels.tolist())) < 2:
        raise ValueError('mTP/noTP specialist training requires both noTP and mTP rows.')
    resolved_class_weight = _normalize_class_weight(class_weight)
    class_weight_text = _class_weight_metadata_value(class_weight)
    classifier = HistGradientBoostingClassifier(
        max_iter=int(max_iter),
        learning_rate=float(learning_rate),
        l2_regularization=float(l2_regularization),
        random_state=int(random_state),
        class_weight=resolved_class_weight,
    )
    _fit_sklearn_model(classifier, features, labels)

    localization_model = model['localization_model']
    specialist = dict(localization_model.get('targetp_specialist_postprocess', {}))
    specialist.update({
        'enabled': True,
        'mtp_notp_feature_profile': TARGETP_MTP_NOTP_SPECIALIST_PROFILE['feature_profile'],
        'mtp_notp_models': [classifier],
        'mtp_notp_weights': [1.0],
        'mtp_notp_threshold': float(threshold),
        'mtp_notp_model_kind': TARGETP_MTP_NOTP_SPECIALIST_PROFILE['model_kind'],
        'mtp_notp_max_iter': int(max_iter),
        'mtp_notp_learning_rate': float(learning_rate),
        'mtp_notp_l2_regularization': float(l2_regularization),
        'mtp_notp_random_state': int(random_state),
        'mtp_notp_class_weight': class_weight_text,
        'mtp_notp_training_rows': int(features.shape[0]),
        'mtp_notp_feature_dim': int(features.shape[1]),
    })
    localization_model['targetp_specialist_postprocess'] = specialist
    model.setdefault('metadata', {})
    model['metadata']['targetp_mtp_notp_specialist'] = {
        'feature_profile': TARGETP_MTP_NOTP_SPECIALIST_PROFILE['feature_profile'],
        'model_kind': TARGETP_MTP_NOTP_SPECIALIST_PROFILE['model_kind'],
        'threshold': float(threshold),
        'training_rows': int(features.shape[0]),
        'mtp_rows': int(np.sum(labels == 1)),
        'notp_rows': int(np.sum(labels == 0)),
        'feature_dim': int(features.shape[1]),
        'max_iter': int(max_iter),
        'learning_rate': float(learning_rate),
        'l2_regularization': float(l2_regularization),
        'random_state': int(random_state),
        'class_weight': class_weight_text,
    }
    return model


def attach_targetp_reranker(
    model,
    rows,
    prediction_rows=None,
    threshold=TARGETP_RERANKER_PROFILE['threshold'],
    class_thresholds=None,
    max_iter=TARGETP_RERANKER_PROFILE['max_iter'],
    learning_rate=TARGETP_RERANKER_PROFILE['learning_rate'],
    l2_regularization=TARGETP_RERANKER_PROFILE['l2_regularization'],
    random_state=TARGETP_RERANKER_PROFILE['random_state'],
):
    """Attach a CPU-runtime multiclass TargetP reranker trained on a development set."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:
        raise RuntimeError('TargetP reranker training requires scikit-learn.') from exc

    features, labels = _targetp_reranker_training_matrix_from_predictions(
        model=model,
        rows=rows,
        prediction_rows=prediction_rows,
    )
    if len(set(labels.tolist())) < 2:
        raise ValueError('TargetP reranker training requires at least two classes.')
    classifier = HistGradientBoostingClassifier(
        max_iter=int(max_iter),
        learning_rate=float(learning_rate),
        l2_regularization=float(l2_regularization),
        random_state=int(random_state),
        class_weight=TARGETP_RERANKER_PROFILE['class_weight'],
    )
    _fit_sklearn_model(classifier, features, labels)

    localization_model = model['localization_model']
    specialist = dict(localization_model.get('targetp_specialist_postprocess', {}))
    specialist.update({
        'enabled': True,
        'reranker_feature_profile': TARGETP_RERANKER_PROFILE['name'],
        'reranker_models': [classifier],
        'reranker_weights': [1.0],
        'reranker_threshold': float(threshold),
        'reranker_model_kind': TARGETP_RERANKER_PROFILE['model_kind'],
        'reranker_max_iter': int(max_iter),
        'reranker_learning_rate': float(learning_rate),
        'reranker_l2_regularization': float(l2_regularization),
        'reranker_random_state': int(random_state),
        'reranker_class_weight': TARGETP_RERANKER_PROFILE['class_weight'],
        'reranker_training_rows': int(features.shape[0]),
        'reranker_feature_dim': int(features.shape[1]),
    })
    if class_thresholds is not None:
        specialist['reranker_thresholds'] = {
            class_name: float(class_thresholds[class_name])
            for class_name in LOCALIZATION_CLASSES
        }
    localization_model['targetp_specialist_postprocess'] = specialist
    model.setdefault('metadata', {})
    counts = {
        class_name: int(np.sum(labels == class_i))
        for class_i, class_name in enumerate(LOCALIZATION_CLASSES)
    }
    model['metadata']['targetp_reranker'] = {
        'feature_profile': TARGETP_RERANKER_PROFILE['name'],
        'model_kind': TARGETP_RERANKER_PROFILE['model_kind'],
        'threshold': float(threshold),
        'training_rows': int(features.shape[0]),
        'class_counts': counts,
        'feature_dim': int(features.shape[1]),
        'max_iter': int(max_iter),
        'learning_rate': float(learning_rate),
        'l2_regularization': float(l2_regularization),
        'random_state': int(random_state),
        'class_weight': TARGETP_RERANKER_PROFILE['class_weight'],
    }
    if class_thresholds is not None:
        model['metadata']['targetp_reranker']['class_thresholds'] = {
            class_name: float(class_thresholds[class_name])
            for class_name in LOCALIZATION_CLASSES
        }
    return model


def build_parser():
    parser = argparse.ArgumentParser(
        description='Build a CPU-runtime TargetP pair blend from two trained cdskit localize models.',
    )
    parser.add_argument('--model_a', required=True, type=str)
    parser.add_argument('--model_b', required=True, type=str)
    parser.add_argument('--alpha', default='0.5', type=str)
    parser.add_argument('--class_thresholds', default='', type=str)
    parser.add_argument('--perox_source', default='a', choices=['a', 'b'], type=str)
    parser.add_argument('--metadata_json', default='', type=str)
    parser.add_argument('--notp_specialist_tsv', default='', type=str)
    parser.add_argument('--notp_specialist_predictions_tsv', default='', type=str)
    parser.add_argument('--reranker_tsv', default='', type=str)
    parser.add_argument('--reranker_predictions_tsv', default='', type=str)
    parser.add_argument('--mtp_notp_specialist_tsv', default='', type=str)
    parser.add_argument('--mtp_notp_specialist_predictions_tsv', default='', type=str)
    parser.add_argument(
        '--reranker_threshold',
        default=TARGETP_RERANKER_PROFILE['threshold'],
        type=float,
    )
    parser.add_argument('--reranker_class_thresholds', default='', type=str)
    parser.add_argument(
        '--reranker_max_iter',
        default=TARGETP_RERANKER_PROFILE['max_iter'],
        type=int,
    )
    parser.add_argument(
        '--reranker_learning_rate',
        default=TARGETP_RERANKER_PROFILE['learning_rate'],
        type=float,
    )
    parser.add_argument(
        '--reranker_l2',
        default=TARGETP_RERANKER_PROFILE['l2_regularization'],
        type=float,
    )
    parser.add_argument(
        '--reranker_random_state',
        default=TARGETP_RERANKER_PROFILE['random_state'],
        type=int,
    )
    parser.add_argument(
        '--notp_specialist_threshold',
        default=TARGETP_NOTP_SPECIALIST_PROFILE['threshold'],
        type=float,
    )
    parser.add_argument(
        '--notp_specialist_max_iter',
        default=TARGETP_NOTP_SPECIALIST_PROFILE['max_iter'],
        type=int,
    )
    parser.add_argument(
        '--notp_specialist_learning_rate',
        default=TARGETP_NOTP_SPECIALIST_PROFILE['learning_rate'],
        type=float,
    )
    parser.add_argument(
        '--notp_specialist_l2',
        default=TARGETP_NOTP_SPECIALIST_PROFILE['l2_regularization'],
        type=float,
    )
    parser.add_argument(
        '--notp_specialist_random_state',
        default=TARGETP_NOTP_SPECIALIST_PROFILE['random_state'],
        type=int,
    )
    parser.add_argument(
        '--mtp_notp_specialist_threshold',
        default=TARGETP_MTP_NOTP_SPECIALIST_PROFILE['threshold'],
        type=float,
    )
    parser.add_argument(
        '--mtp_notp_specialist_max_iter',
        default=TARGETP_MTP_NOTP_SPECIALIST_PROFILE['max_iter'],
        type=int,
    )
    parser.add_argument(
        '--mtp_notp_specialist_learning_rate',
        default=TARGETP_MTP_NOTP_SPECIALIST_PROFILE['learning_rate'],
        type=float,
    )
    parser.add_argument(
        '--mtp_notp_specialist_l2',
        default=TARGETP_MTP_NOTP_SPECIALIST_PROFILE['l2_regularization'],
        type=float,
    )
    parser.add_argument(
        '--mtp_notp_specialist_random_state',
        default=TARGETP_MTP_NOTP_SPECIALIST_PROFILE['random_state'],
        type=int,
    )
    parser.add_argument(
        '--mtp_notp_specialist_class_weight',
        default=TARGETP_MTP_NOTP_SPECIALIST_PROFILE['class_weight'],
        type=str,
    )
    parser.add_argument('--model_out', required=True, type=str)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    model_a = load_localize_model(args.model_a)
    model_b = load_localize_model(args.model_b)
    metadata = {}
    if str(args.metadata_json).strip() != '':
        metadata = json.loads(str(args.metadata_json))
        if not isinstance(metadata, dict):
            raise ValueError('--metadata_json should decode to an object.')
    metadata.update({
        'base_model_a': str(args.model_a),
        'base_model_b': str(args.model_b),
        'alpha_by_class': _parse_class_values(args.alpha, default=0.5),
        'class_thresholds': _parse_class_values(args.class_thresholds, default=1.0),
    })
    model = build_targetp_pair_blend_runtime_model(
        base_model_a=model_a,
        base_model_b=model_b,
        alpha_by_class=metadata['alpha_by_class'],
        class_thresholds=metadata['class_thresholds'],
        perox_source=args.perox_source,
        metadata=metadata,
    )
    notp_report = None
    if str(args.notp_specialist_tsv).strip() != '':
        notp_rows, notp_skipped = load_fixed_uniprot_holdout_rows(
            path=str(args.notp_specialist_tsv),
            strict_targetp_organism_labels=True,
        )
        notp_prediction_rows = None
        if str(args.notp_specialist_predictions_tsv).strip() != '':
            notp_prediction_rows = read_tsv(path=str(args.notp_specialist_predictions_tsv))
        model = attach_targetp_notp_specialist(
            model=model,
            rows=notp_rows,
            prediction_rows=notp_prediction_rows,
            threshold=float(args.notp_specialist_threshold),
            max_iter=int(args.notp_specialist_max_iter),
            learning_rate=float(args.notp_specialist_learning_rate),
            l2_regularization=float(args.notp_specialist_l2),
            random_state=int(args.notp_specialist_random_state),
        )
        model['metadata']['targetp_notp_specialist']['training_tsv'] = str(
            args.notp_specialist_tsv
        )
        model['metadata']['targetp_notp_specialist']['predictions_tsv'] = str(
            args.notp_specialist_predictions_tsv
        )
        model['metadata']['targetp_notp_specialist']['skipped_rows'] = dict(notp_skipped)
        notp_report = dict(model['metadata']['targetp_notp_specialist'])
    reranker_report = None
    if str(args.reranker_tsv).strip() != '':
        reranker_rows, reranker_skipped = load_fixed_uniprot_holdout_rows(
            path=str(args.reranker_tsv),
            strict_targetp_organism_labels=True,
        )
        reranker_prediction_rows = None
        if str(args.reranker_predictions_tsv).strip() != '':
            reranker_prediction_rows = read_tsv(path=str(args.reranker_predictions_tsv))
        model = attach_targetp_reranker(
            model=model,
            rows=reranker_rows,
            prediction_rows=reranker_prediction_rows,
            threshold=float(args.reranker_threshold),
            class_thresholds=(
                None
                if str(args.reranker_class_thresholds).strip() == ''
                else _parse_class_values(
                    args.reranker_class_thresholds,
                    default=float(args.reranker_threshold),
                )
            ),
            max_iter=int(args.reranker_max_iter),
            learning_rate=float(args.reranker_learning_rate),
            l2_regularization=float(args.reranker_l2),
            random_state=int(args.reranker_random_state),
        )
        model['metadata']['targetp_reranker']['training_tsv'] = str(args.reranker_tsv)
        model['metadata']['targetp_reranker']['predictions_tsv'] = str(
            args.reranker_predictions_tsv
        )
        model['metadata']['targetp_reranker']['skipped_rows'] = dict(reranker_skipped)
        reranker_report = dict(model['metadata']['targetp_reranker'])
    mtp_notp_report = None
    if str(args.mtp_notp_specialist_tsv).strip() != '':
        mtp_notp_rows, mtp_notp_skipped = load_fixed_uniprot_holdout_rows(
            path=str(args.mtp_notp_specialist_tsv),
            strict_targetp_organism_labels=True,
        )
        mtp_notp_prediction_rows = None
        if str(args.mtp_notp_specialist_predictions_tsv).strip() != '':
            mtp_notp_prediction_rows = read_tsv(
                path=str(args.mtp_notp_specialist_predictions_tsv)
            )
        model = attach_targetp_mtp_notp_specialist(
            model=model,
            rows=mtp_notp_rows,
            prediction_rows=mtp_notp_prediction_rows,
            threshold=float(args.mtp_notp_specialist_threshold),
            max_iter=int(args.mtp_notp_specialist_max_iter),
            learning_rate=float(args.mtp_notp_specialist_learning_rate),
            l2_regularization=float(args.mtp_notp_specialist_l2),
            random_state=int(args.mtp_notp_specialist_random_state),
            class_weight=args.mtp_notp_specialist_class_weight,
        )
        model['metadata']['targetp_mtp_notp_specialist']['training_tsv'] = str(
            args.mtp_notp_specialist_tsv
        )
        model['metadata']['targetp_mtp_notp_specialist']['predictions_tsv'] = str(
            args.mtp_notp_specialist_predictions_tsv
        )
        model['metadata']['targetp_mtp_notp_specialist']['skipped_rows'] = dict(
            mtp_notp_skipped
        )
        mtp_notp_report = dict(model['metadata']['targetp_mtp_notp_specialist'])
    save_localize_model(model=model, path=str(args.model_out))
    print(json.dumps({
        'model_out': str(args.model_out),
        'model_type': model['model_type'],
        'alpha_by_class': metadata['alpha_by_class'],
        'class_thresholds': metadata['class_thresholds'],
        'notp_specialist': notp_report,
        'reranker': reranker_report,
        'mtp_notp_specialist': mtp_notp_report,
    }, indent=2, sort_keys=True))
    return model


if __name__ == '__main__':
    main()

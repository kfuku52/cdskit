"""
Build and evaluate a CPU peroxisome head for cdskit localize.

The default workflow trains an ExtraTrees head on DeepLoc 2.1 Swiss-Prot
train/validation rows, tunes the operating threshold on one predefined
partition, and evaluates the final head on an independent external table.
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from cdskit.localize_model import (
    BROAD_FEATURE_NAMES,
    PEROX_FEATURE_NAMES,
    TARGETP_FEATURE_ENSEMBLE_PROFILE,
    detect_perox_signals,
    extract_broad_localize_features,
    extract_perox_features,
    extract_targetp_feature_ensemble_features,
    load_localize_model,
    save_localize_model,
)
from cdskit.localize_models import resolve_localize_model_path


DEFAULT_TRAIN_TSV = 'data/localize_bench/deeploc21/deeploc21_localization_train_validation.tsv'
DEFAULT_EXTERNAL_TEST_TSV = 'data/localize_bench/deeploc21/deeploc21_hpa_test.tsv'


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value or '').strip().lower() in ['1', 'yes', 'y', 'true', 't', 'on']


def _label_to_int(value):
    txt = str(value or '').strip().lower()
    if txt in ['1', 'yes', 'true', 't', 'peroxisome', 'peroxisomal']:
        return 1
    if txt in ['0', 'no', 'false', 'f', 'non-peroxisomal', 'not_peroxisomal']:
        return 0
    raise ValueError('Unsupported peroxisome label: {}'.format(value))


def read_perox_rows(path):
    rows = list()
    with open(path, 'r', encoding='utf-8', newline='') as inp:
        reader = csv.DictReader(inp, delimiter='\t')
        required = ['accession', 'kingdom', 'partition', 'sequence', 'peroxisome']
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError('Missing required columns in {}: {}'.format(path, ', '.join(missing)))
        for row in reader:
            seq = str(row.get('sequence', '') or '').strip()
            if seq == '':
                continue
            rows.append({
                'accession': str(row.get('accession', '') or '').strip(),
                'kingdom': str(row.get('kingdom', '') or '').strip(),
                'partition': str(row.get('partition', '') or '').strip(),
                'sequence': seq,
                'peroxisome': int(_label_to_int(row.get('peroxisome', '0'))),
                'localization_labels': str(row.get('localization_labels', '') or '').strip(),
                'source': str(row.get('source', '') or '').strip(),
            })
    return rows


def _sequence_digest(seq):
    return hashlib.sha256(str(seq or '').encode('utf-8')).hexdigest()


def _kingdom_from_lineage_ids(lineage_ids):
    txt = str(lineage_ids or '')
    if '33090 ' in txt or 'Viridiplantae' in txt:
        return 'Viridiplantae'
    if '33208 ' in txt or 'Metazoa' in txt:
        return 'Metazoa'
    if '4751 ' in txt or 'Fungi' in txt:
        return 'Fungi'
    return 'Other'


def read_uniprot_exp_cc_perox_rows(path, exclude_rows=None):
    exclude_rows = list(exclude_rows or [])
    exclude_accessions = {
        row.get('accession', '')
        for row in exclude_rows
        if str(row.get('accession', '') or '') != ''
    }
    exclude_sequences = {
        _sequence_digest(row.get('sequence', ''))
        for row in exclude_rows
        if str(row.get('sequence', '') or '') != ''
    }
    rows = list()
    skipped = Counter()
    with open(path, 'r', encoding='utf-8', newline='') as inp:
        reader = csv.DictReader(inp, delimiter='\t')
        required = ['accession', 'sequence', 'cc_subcellular_location', 'lineage_ids']
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError('Missing required columns in {}: {}'.format(path, ', '.join(missing)))
        for row in reader:
            acc = str(row.get('accession', '') or '').strip()
            seq = str(row.get('sequence', '') or '').strip()
            cc = str(row.get('cc_subcellular_location', '') or '').strip()
            if seq == '' or cc == '':
                skipped['missing_sequence_or_cc'] += 1
                continue
            if 'ECO:0000269' not in cc:
                skipped['non_experimental'] += 1
                continue
            if acc in exclude_accessions:
                skipped['accession_overlap'] += 1
                continue
            if _sequence_digest(seq) in exclude_sequences:
                skipped['exact_sequence_overlap'] += 1
                continue
            is_perox = 'peroxisom' in cc.lower()
            rows.append({
                'accession': acc,
                'kingdom': _kingdom_from_lineage_ids(row.get('lineage_ids', '')),
                'partition': 'external_uniprot_exp_cc',
                'sequence': seq,
                'peroxisome': 1 if is_perox else 0,
                'localization_labels': 'peroxisome' if is_perox else 'non_peroxisome_exp_cc',
                'source': 'uniprot_exp_cc',
            })
    return rows, dict(skipped)


def load_external_perox_rows(path, external_format='prepared', exclude_rows=None):
    external_format = str(external_format or 'prepared').strip().lower()
    if external_format == 'prepared':
        return read_perox_rows(path), {'format': 'prepared'}
    if external_format == 'uniprot_exp_cc':
        rows, skipped = read_uniprot_exp_cc_perox_rows(
            path=path,
            exclude_rows=exclude_rows,
        )
        return rows, {
            'format': 'uniprot_exp_cc',
            'skipped': skipped,
        }
    raise ValueError('Unsupported external_format: {}'.format(external_format))


def _feature_names_for_profile(feature_profile):
    profile = str(feature_profile or '').strip().lower()
    if profile in ['perox_sequence_v1', 'perox_features_v1']:
        return list(PEROX_FEATURE_NAMES)
    if profile in ['broad_localize_v1', 'broad_localize_features_v1']:
        return list(BROAD_FEATURE_NAMES)
    if profile == TARGETP_FEATURE_ENSEMBLE_PROFILE['name']:
        return ['targetp_feature_{}'.format(i) for i in range(_targetp_feature_dim_probe())]
    raise ValueError('Unsupported perox feature_profile: {}'.format(feature_profile))


def _targetp_feature_dim_probe():
    return int(extract_targetp_feature_ensemble_features('MAAAAAAAAAASKL', 'Metazoa').shape[0])


def extract_perox_model_features(row, feature_profile='perox_sequence_v1'):
    profile = str(feature_profile or '').strip().lower()
    seq = row.get('sequence', '')
    kingdom = row.get('kingdom', '')
    if profile in ['perox_sequence_v1', 'perox_features_v1']:
        return extract_perox_features(aa_seq=seq, kingdom=kingdom)[0]
    if profile in ['broad_localize_v1', 'broad_localize_features_v1']:
        return extract_broad_localize_features(aa_seq=seq, kingdom=kingdom)[0]
    if profile == TARGETP_FEATURE_ENSEMBLE_PROFILE['name']:
        return extract_targetp_feature_ensemble_features(
            aa_seq=seq,
            organism_group=kingdom,
        )
    raise ValueError('Unsupported perox feature_profile: {}'.format(feature_profile))


def build_perox_feature_matrix(rows, feature_profile='perox_sequence_v1'):
    features = [
        extract_perox_model_features(row=row, feature_profile=feature_profile)
        for row in rows
    ]
    if len(features) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    return np.vstack(features).astype(np.float64)


def make_perox_classifier(
    model_kind='extra_trees',
    random_state=1,
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=0.0,
    n_estimators=300,
    min_samples_leaf=10,
):
    model_kind = str(model_kind or '').strip().lower()
    if model_kind in ['hist_gradient_boosting', 'hgb']:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            learning_rate=float(learning_rate),
            max_iter=int(max_iter),
            max_leaf_nodes=int(max_leaf_nodes),
            min_samples_leaf=int(min_samples_leaf),
            l2_regularization=float(l2_regularization),
            class_weight='balanced',
            random_state=int(random_state),
        )
    if model_kind in ['extra_trees', 'et']:
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(
            n_estimators=int(n_estimators),
            random_state=int(random_state),
            class_weight='balanced',
            max_features='sqrt',
            min_samples_leaf=max(1, int(min_samples_leaf)),
            n_jobs=1,
        )
    raise ValueError('Unsupported perox model_kind: {}'.format(model_kind))


def fit_sklearn_perox_binary_model(
    rows,
    feature_profile='perox_sequence_v1',
    model_kind='extra_trees',
    threshold=0.5,
    random_state=1,
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=0.0,
    n_estimators=300,
    min_samples_leaf=10,
):
    y = np.asarray([int(row['peroxisome']) for row in rows], dtype=np.int64)
    if y.shape[0] == 0:
        raise ValueError('No training rows were provided.')
    if len(set(y.tolist())) == 1:
        return {
            'mode': 'constant',
            'yes_probability': float(y[0]),
            'feature_profile': str(feature_profile),
            'prediction_scope': 'peroxisome sequence-label probability; strongest for PTS-like targeting signals',
            'threshold': float(threshold),
            'n_training_rows': int(y.shape[0]),
            'n_positive': int(np.sum(y == 1)),
            'n_negative': int(np.sum(y == 0)),
        }
    x = build_perox_feature_matrix(rows=rows, feature_profile=feature_profile)
    classifier = make_perox_classifier(
        model_kind=model_kind,
        random_state=random_state,
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
    )
    classifier.fit(x, y)
    return {
        'mode': 'sklearn_binary',
        'classifier': classifier,
        'classes': [int(v) for v in list(getattr(classifier, 'classes_', [0, 1]))],
        'positive_class': 1,
        'feature_profile': str(feature_profile),
        'feature_names': _feature_names_for_profile(feature_profile),
        'feature_dim': int(x.shape[1]),
        'prediction_scope': 'peroxisome sequence-label probability; strongest for PTS-like targeting signals',
        'threshold': float(threshold),
        'classifier_profile': {
            'model_kind': str(model_kind),
            'random_state': int(random_state),
            'max_iter': int(max_iter),
            'learning_rate': float(learning_rate),
            'max_leaf_nodes': int(max_leaf_nodes),
            'l2_regularization': float(l2_regularization),
            'n_estimators': int(n_estimators),
            'min_samples_leaf': int(min_samples_leaf),
            'class_weight': 'balanced',
        },
        'n_training_rows': int(y.shape[0]),
        'n_positive': int(np.sum(y == 1)),
        'n_negative': int(np.sum(y == 0)),
    }


def predict_perox_yes_probabilities(rows, perox_model):
    mode = str(perox_model.get('mode', '')).strip().lower()
    if mode == 'constant':
        p_yes = float(perox_model.get('yes_probability', 0.0))
        return np.full((len(rows),), p_yes, dtype=np.float64)
    if mode != 'sklearn_binary':
        raise ValueError('Unsupported perox_model mode for batch prediction: {}'.format(mode))
    classifier = perox_model.get('classifier', None)
    if classifier is None or not hasattr(classifier, 'predict_proba'):
        raise ValueError('sklearn_binary perox_model requires predict_proba.')
    x = build_perox_feature_matrix(
        rows=rows,
        feature_profile=perox_model.get('feature_profile', 'perox_sequence_v1'),
    )
    proba = np.asarray(classifier.predict_proba(x), dtype=np.float64)
    classes = list(getattr(classifier, 'classes_', perox_model.get('classes', [0, 1])))
    positive_col = None
    for i, class_value in enumerate(classes):
        if int(class_value) == 1:
            positive_col = i
            break
    if positive_col is None:
        raise ValueError('Could not find positive class in perox_model classifier.')
    return np.clip(proba[:, int(positive_col)], 0.0, 1.0)


def regex_perox_probabilities(rows):
    out = list()
    for row in rows:
        signals = detect_perox_signals(row.get('sequence', ''))
        out.append(1.0 if str(signals.get('signal_type', 'none')) != 'none' else 0.0)
    return np.asarray(out, dtype=np.float64)


def _binary_confusion(y_true, y_prob, threshold):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= float(threshold)).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def _safe_div(num, den):
    return 0.0 if den <= 0 else float(num) / float(den)


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    tp, fp, fn, tn = _binary_confusion(y_true=y_true, y_prob=y_prob, threshold=threshold)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    accuracy = _safe_div(tp + tn, tp + fp + fn + tn)
    f1 = 0.0 if precision + recall <= 0.0 else (2.0 * precision * recall) / (precision + recall)
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = 0.0 if denom <= 0.0 else float(((tp * tn) - (fp * fn)) / np.sqrt(denom))
    metrics = {
        'rows': int(y_true.shape[0]),
        'positive_count': int(np.sum(y_true == 1)),
        'negative_count': int(np.sum(y_true == 0)),
        'threshold': float(threshold),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'accuracy': float(accuracy),
        'f1': float(f1),
        'mcc': float(mcc),
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        if len(set(y_true.tolist())) > 1:
            metrics['auprc'] = float(average_precision_score(y_true, y_prob))
            metrics['auroc'] = float(roc_auc_score(y_true, y_prob))
        else:
            metrics['auprc'] = None
            metrics['auroc'] = None
    except Exception:
        metrics['auprc'] = None
        metrics['auroc'] = None
    return metrics


def tune_threshold(y_true, y_prob, objective='f1', threshold_grid=None):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if threshold_grid is None:
        threshold_grid = np.linspace(0.01, 0.99, 99)
    objective = str(objective or 'f1').strip().lower()
    if objective not in ['f1', 'mcc', 'precision', 'recall']:
        raise ValueError('Unsupported threshold objective: {}'.format(objective))
    best_threshold = 0.5
    best_score = -1.0e9
    for threshold in threshold_grid:
        row = binary_metrics(y_true=y_true, y_prob=y_prob, threshold=float(threshold))
        score = float(row.get(objective, 0.0))
        if score > best_score + 1.0e-12:
            best_score = score
            best_threshold = float(threshold)
        elif abs(score - best_score) <= 1.0e-12:
            if abs(float(threshold) - 0.5) < abs(best_threshold - 0.5):
                best_threshold = float(threshold)
    return float(best_threshold)


def label_vector(rows):
    return np.asarray([int(row['peroxisome']) for row in rows], dtype=np.int64)


def _subset_by_indices(rows, values, indices, invert=False):
    rows = list(rows or [])
    values = np.asarray(values, dtype=np.float64)
    index_set = set(int(i) for i in list(indices or []))
    keep = [
        i for i in range(len(rows))
        if ((i in index_set) and not invert) or ((i not in index_set) and invert)
    ]
    subset_rows = [rows[i] for i in keep]
    subset_values = values[keep] if len(keep) > 0 else np.zeros((0,), dtype=np.float64)
    return subset_rows, subset_values


def add_homology_subset_metrics(metrics, prefix, rows, probabilities, hit_indices, threshold):
    hit_rows, hit_prob = _subset_by_indices(
        rows=rows,
        values=probabilities,
        indices=hit_indices,
        invert=False,
    )
    nohit_rows, nohit_prob = _subset_by_indices(
        rows=rows,
        values=probabilities,
        indices=hit_indices,
        invert=True,
    )
    if len(hit_rows) > 0:
        metrics['{}_homology_hit'.format(prefix)] = binary_metrics(
            y_true=label_vector(hit_rows),
            y_prob=hit_prob,
            threshold=threshold,
        )
    if len(nohit_rows) > 0:
        metrics['{}_homology_nohit'.format(prefix)] = binary_metrics(
            y_true=label_vector(nohit_rows),
            y_prob=nohit_prob,
            threshold=threshold,
        )


def _hash_sort_key_for_group(group):
    seq_digest = str(group.get('sequence_digest', ''))
    accessions = sorted(str(row.get('accession', '')) for row in group.get('rows', []))
    txt = '{}\t{}'.format(seq_digest, '\t'.join(accessions))
    return hashlib.sha256(txt.encode('utf-8')).hexdigest()


def split_train_valid_hash_stratified(rows, validation_fraction=0.2):
    rows = list(rows or [])
    if len(rows) < 2:
        raise ValueError('At least two rows are required for a hash-stratified split.')
    fraction = float(validation_fraction)
    if fraction <= 0.0 or fraction >= 1.0:
        raise ValueError('validation_fraction should be between 0 and 1.')

    sequence_groups = {}
    for row in rows:
        digest = _sequence_digest(row.get('sequence', ''))
        group = sequence_groups.setdefault(
            digest,
            {
                'sequence_digest': digest,
                'rows': [],
                'positive': 0,
                'negative': 0,
            },
        )
        group['rows'].append(row)
        if int(row.get('peroxisome', 0)) == 1:
            group['positive'] += 1
        else:
            group['negative'] += 1

    by_label = {}
    for group in sequence_groups.values():
        label = 1 if int(group['positive']) > 0 else 0
        by_label.setdefault(label, []).append(group)

    train_rows = list()
    valid_rows = list()
    for label_groups in by_label.values():
        ordered = sorted(label_groups, key=_hash_sort_key_for_group)
        total_rows = sum(len(group['rows']) for group in ordered)
        if len(ordered) <= 1:
            for group in ordered:
                train_rows.extend(group['rows'])
            continue
        target_valid_rows = int(round(float(total_rows) * fraction))
        target_valid_rows = max(1, min(total_rows - 1, target_valid_rows))
        selected_valid_groups = set()
        selected_count = 0
        for group in ordered:
            if len(selected_valid_groups) >= len(ordered) - 1:
                break
            if selected_count >= target_valid_rows:
                break
            selected_valid_groups.add(group['sequence_digest'])
            selected_count += len(group['rows'])
        if len(selected_valid_groups) == 0:
            selected_valid_groups.add(ordered[0]['sequence_digest'])
        for group in ordered:
            if group['sequence_digest'] in selected_valid_groups:
                valid_rows.extend(group['rows'])
            else:
                train_rows.extend(group['rows'])

    if len(train_rows) == 0 or len(valid_rows) == 0:
        raise ValueError('Could not create a non-empty hash-stratified split.')
    return train_rows, valid_rows


def split_train_valid(rows, validation_partition='4', validation_fraction=0.2):
    validation_partition = str(validation_partition)
    if validation_partition.lower() in ['hash_stratified', 'auto_hash', 'stratified_hash']:
        return split_train_valid_hash_stratified(
            rows=rows,
            validation_fraction=validation_fraction,
        )
    train_rows = [row for row in rows if str(row.get('partition', '')) != validation_partition]
    valid_rows = [row for row in rows if str(row.get('partition', '')) == validation_partition]
    if len(train_rows) == 0 or len(valid_rows) == 0:
        raise ValueError(
            'Could not split rows with validation_partition={}. '
            'Use validation_partition=hash_stratified for datasets without predefined folds.'.format(
                validation_partition
            )
        )
    return train_rows, valid_rows


def leakage_report(train_rows, eval_rows):
    train_acc = {row['accession'] for row in train_rows if row.get('accession', '') != ''}
    eval_acc = {row['accession'] for row in eval_rows if row.get('accession', '') != ''}
    train_seq = {_sequence_digest(row['sequence']) for row in train_rows}
    eval_seq = {_sequence_digest(row['sequence']) for row in eval_rows}
    return {
        'train_rows': int(len(train_rows)),
        'eval_rows': int(len(eval_rows)),
        'accession_overlap_count': int(len(train_acc & eval_acc)),
        'exact_sequence_overlap_count': int(len(train_seq & eval_seq)),
    }


def sequence_label_conflict_report(rows):
    rows = list(rows or [])
    by_seq = {}
    for row in rows:
        digest = _sequence_digest(row.get('sequence', ''))
        entry = by_seq.setdefault(
            digest,
            {
                'labels': set(),
                'accessions': set(),
                'row_count': 0,
            },
        )
        entry['labels'].add(int(row.get('peroxisome', 0)))
        acc = str(row.get('accession', '') or '').strip()
        if acc:
            entry['accessions'].add(acc)
        entry['row_count'] += 1

    conflicts = [
        dict(
            sequence_digest=digest,
            row_count=int(entry['row_count']),
            labels=sorted(int(label) for label in entry['labels']),
            accession_count=int(len(entry['accessions'])),
            example_accessions=sorted(entry['accessions'])[:10],
        )
        for digest, entry in by_seq.items()
        if len(entry['labels']) > 1
    ]
    conflicts.sort(key=lambda row: (row['row_count'], row['sequence_digest']), reverse=True)
    duplicate_groups = [
        entry
        for entry in by_seq.values()
        if entry['row_count'] > 1
    ]
    return {
        'rows': int(len(rows)),
        'unique_sequences': int(len(by_seq)),
        'duplicate_sequence_count': int(len(duplicate_groups)),
        'duplicate_sequence_rows': int(sum(entry['row_count'] for entry in duplicate_groups)),
        'duplicate_sequence_excess_rows': int(sum(entry['row_count'] - 1 for entry in duplicate_groups)),
        'conflicting_sequence_count': int(len(conflicts)),
        'conflicting_row_count': int(sum(row['row_count'] for row in conflicts)),
        'examples': conflicts[:10],
    }


def exclude_rows_overlapping_eval(train_rows, eval_rows):
    train_rows = list(train_rows or [])
    eval_rows = list(eval_rows or [])
    eval_acc = {row['accession'] for row in eval_rows if row.get('accession', '') != ''}
    eval_seq = {_sequence_digest(row['sequence']) for row in eval_rows}
    kept = list()
    skipped = Counter()
    for row in train_rows:
        acc = row.get('accession', '')
        seq_digest = _sequence_digest(row.get('sequence', ''))
        acc_overlap = acc != '' and acc in eval_acc
        seq_overlap = seq_digest in eval_seq
        if acc_overlap or seq_overlap:
            if acc_overlap:
                skipped['accession_overlap'] += 1
            if seq_overlap:
                skipped['exact_sequence_overlap'] += 1
            skipped['total_removed'] += 1
            continue
        kept.append(row)
    return kept, {
        'enabled': True,
        'input_rows': int(len(train_rows)),
        'eval_rows': int(len(eval_rows)),
        'kept_rows': int(len(kept)),
        'removed_rows': int(skipped.get('total_removed', 0)),
        'accession_overlap_rows': int(skipped.get('accession_overlap', 0)),
        'exact_sequence_overlap_rows': int(skipped.get('exact_sequence_overlap', 0)),
    }


def _sanitize_fasta_id(text):
    out = ''.join(ch if ch.isalnum() or ch in ['_', '.', '-'] else '_' for ch in str(text or ''))
    return out or 'seq'


def _write_rows_fasta(rows, path, prefix):
    with open(path, 'w', encoding='utf-8') as out:
        for i, row in enumerate(rows):
            acc = _sanitize_fasta_id(row.get('accession', ''))
            seq_id = '{}{}_{}'.format(prefix, int(i), acc)
            seq = str(row.get('sequence', '') or '').strip()
            out.write('>{}\n'.format(seq_id))
            for start in range(0, len(seq), 80):
                out.write(seq[start:start + 80] + '\n')


def _write_rows_fasta_with_ids(rows, path, ids):
    with open(path, 'w', encoding='utf-8') as out:
        for row, seq_id in zip(rows, ids):
            seq = str(row.get('sequence', '') or '').strip()
            out.write('>{}\n'.format(_sanitize_fasta_id(seq_id)))
            for start in range(0, len(seq), 80):
                out.write(seq[start:start + 80] + '\n')


def _empty_homology_report(train_rows, eval_rows, status, reason='', tool='mmseqs'):
    return {
        'status': str(status),
        'reason': str(reason),
        'tool': str(tool),
        'train_rows': int(len(train_rows)),
        'eval_rows': int(len(eval_rows)),
    }


def _query_index_from_mmseqs_id(query_id):
    match = re.match(r'^q([0-9]+)_', str(query_id or ''))
    if match is None:
        return None
    return int(match.group(1))


def mmseqs_homology_report(
    train_rows,
    eval_rows,
    min_seq_id=0.3,
    coverage=0.8,
    cov_mode=0,
    threads=1,
    tmp_root='',
    include_hit_indices=False,
):
    train_rows = list(train_rows or [])
    eval_rows = list(eval_rows or [])
    if len(train_rows) == 0 or len(eval_rows) == 0:
        return _empty_homology_report(
            train_rows=train_rows,
            eval_rows=eval_rows,
            status='skipped',
            reason='empty train or eval rows',
        )

    mmseqs_path = shutil.which('mmseqs')
    if mmseqs_path is None:
        return _empty_homology_report(
            train_rows=train_rows,
            eval_rows=eval_rows,
            status='unavailable',
            reason='mmseqs not found on PATH',
        )

    tmp_kwargs = {}
    if str(tmp_root or '').strip():
        Path(tmp_root).mkdir(parents=True, exist_ok=True)
        tmp_kwargs['dir'] = str(tmp_root)
    with tempfile.TemporaryDirectory(prefix='cdskit_perox_mmseqs_', **tmp_kwargs) as tmp_dir:
        tmp_path = Path(tmp_dir)
        query_fasta = tmp_path / 'query.faa'
        target_fasta = tmp_path / 'target.faa'
        result_path = tmp_path / 'result.m8'
        mmseqs_tmp = tmp_path / 'tmp'
        _write_rows_fasta(eval_rows, query_fasta, 'q')
        _write_rows_fasta(train_rows, target_fasta, 't')
        cmd = [
            mmseqs_path,
            'easy-search',
            str(query_fasta),
            str(target_fasta),
            str(result_path),
            str(mmseqs_tmp),
            '--min-seq-id',
            str(float(min_seq_id)),
            '-c',
            str(float(coverage)),
            '--cov-mode',
            str(int(cov_mode)),
            '--format-output',
            'query,target,pident,alnlen,qlen,tlen,qcov,tcov,evalue,bits',
            '--threads',
            str(max(1, int(threads))),
            '-v',
            '1',
        ]
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return _empty_homology_report(
                train_rows=train_rows,
                eval_rows=eval_rows,
                status='failed',
                reason=completed.stderr.strip()[-500:],
            )

        best_by_query = {}
        hit_rows = 0
        if result_path.exists():
            with open(result_path, 'r', encoding='utf-8') as inp:
                for line in inp:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 10:
                        continue
                    query_id, target_id = parts[0], parts[1]
                    try:
                        pident = float(parts[2])
                        alnlen = int(float(parts[3]))
                        qlen = int(float(parts[4]))
                        tlen = int(float(parts[5]))
                        qcov = float(parts[6])
                        tcov = float(parts[7])
                        evalue = float(parts[8])
                        bits = float(parts[9])
                    except Exception:
                        continue
                    hit_rows += 1
                    current = best_by_query.get(query_id)
                    if current is None or bits > current['bits']:
                        best_by_query[query_id] = {
                            'query': query_id,
                            'target': target_id,
                            'pident': pident,
                            'alnlen': alnlen,
                            'qlen': qlen,
                            'tlen': tlen,
                            'qcov': qcov,
                            'tcov': tcov,
                            'evalue': evalue,
                            'bits': bits,
                        }

    hit_query_ids = set(best_by_query.keys())
    hit_eval_indices = sorted(
        idx for idx in (_query_index_from_mmseqs_id(query_id) for query_id in hit_query_ids)
        if idx is not None
    )
    positive_query_ids = {
        'q{}_{}'.format(i, _sanitize_fasta_id(row.get('accession', '')))
        for i, row in enumerate(eval_rows)
        if int(row.get('peroxisome', 0)) == 1
    }
    negative_query_ids = {
        'q{}_{}'.format(i, _sanitize_fasta_id(row.get('accession', '')))
        for i, row in enumerate(eval_rows)
        if int(row.get('peroxisome', 0)) == 0
    }
    pidents = [row['pident'] for row in best_by_query.values()]
    top_hits = sorted(
        best_by_query.values(),
        key=lambda row: (row['bits'], row['pident']),
        reverse=True,
    )[:10]
    n_pos = len(positive_query_ids)
    n_neg = len(negative_query_ids)
    report = {
        'status': 'ok',
        'tool': 'mmseqs',
        'min_seq_id': float(min_seq_id),
        'coverage': float(coverage),
        'cov_mode': int(cov_mode),
        'train_rows': int(len(train_rows)),
        'eval_rows': int(len(eval_rows)),
        'eval_positive_count': int(n_pos),
        'eval_negative_count': int(n_neg),
        'raw_hit_rows': int(hit_rows),
        'hit_query_count': int(len(hit_query_ids)),
        'hit_query_fraction': 0.0 if len(eval_rows) == 0 else float(len(hit_query_ids)) / float(len(eval_rows)),
        'positive_hit_query_count': int(len(hit_query_ids & positive_query_ids)),
        'positive_hit_query_fraction': 0.0 if n_pos == 0 else float(len(hit_query_ids & positive_query_ids)) / float(n_pos),
        'negative_hit_query_count': int(len(hit_query_ids & negative_query_ids)),
        'negative_hit_query_fraction': 0.0 if n_neg == 0 else float(len(hit_query_ids & negative_query_ids)) / float(n_neg),
        'best_hit_pident_max': None if len(pidents) == 0 else float(np.max(pidents)),
        'best_hit_pident_median': None if len(pidents) == 0 else float(np.median(pidents)),
        'top_hits': top_hits,
    }
    if include_hit_indices:
        report['_hit_eval_indices'] = hit_eval_indices
    return report


def maybe_homology_report(
    train_rows,
    eval_rows,
    enabled=False,
    min_seq_id=0.3,
    coverage=0.8,
    cov_mode=0,
    threads=1,
    tmp_root='',
    include_hit_indices=False,
):
    if not _str_to_bool(enabled):
        return _empty_homology_report(
            train_rows=train_rows,
            eval_rows=eval_rows,
            status='skipped',
            reason='homology_check disabled',
        )
    return mmseqs_homology_report(
        train_rows=train_rows,
        eval_rows=eval_rows,
        min_seq_id=min_seq_id,
        coverage=coverage,
        cov_mode=cov_mode,
        threads=threads,
        tmp_root=tmp_root,
        include_hit_indices=include_hit_indices,
    )


def _cluster_id_for_singletons(index):
    return 'singleton_{}'.format(int(index))


def mmseqs_cluster_assignments(
    rows,
    min_seq_id=0.3,
    coverage=0.8,
    cov_mode=0,
    threads=1,
    tmp_root='',
):
    rows = list(rows or [])
    if len(rows) == 0:
        return [], {
            'status': 'skipped',
            'reason': 'empty rows',
            'tool': 'mmseqs',
            'rows': 0,
        }

    mmseqs_path = shutil.which('mmseqs')
    if mmseqs_path is None:
        return [_cluster_id_for_singletons(i) for i in range(len(rows))], {
            'status': 'unavailable',
            'reason': 'mmseqs not found on PATH',
            'tool': 'mmseqs',
            'rows': int(len(rows)),
        }

    seq_ids = ['s{}_{}'.format(i, _sanitize_fasta_id(row.get('accession', ''))) for i, row in enumerate(rows)]
    id_to_index = {seq_id: i for i, seq_id in enumerate(seq_ids)}
    tmp_kwargs = {}
    if str(tmp_root or '').strip():
        Path(tmp_root).mkdir(parents=True, exist_ok=True)
        tmp_kwargs['dir'] = str(tmp_root)
    with tempfile.TemporaryDirectory(prefix='cdskit_perox_cluster_', **tmp_kwargs) as tmp_dir:
        tmp_path = Path(tmp_dir)
        fasta_path = tmp_path / 'input.faa'
        cluster_prefix = tmp_path / 'clustered'
        mmseqs_tmp = tmp_path / 'tmp'
        _write_rows_fasta_with_ids(rows=rows, path=fasta_path, ids=seq_ids)
        cmd = [
            mmseqs_path,
            'easy-cluster',
            str(fasta_path),
            str(cluster_prefix),
            str(mmseqs_tmp),
            '--min-seq-id',
            str(float(min_seq_id)),
            '-c',
            str(float(coverage)),
            '--cov-mode',
            str(int(cov_mode)),
            '--cluster-reassign',
            '1',
            '--threads',
            str(max(1, int(threads))),
            '-v',
            '1',
        ]
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return [_cluster_id_for_singletons(i) for i in range(len(rows))], {
                'status': 'failed',
                'reason': completed.stderr.strip()[-500:],
                'tool': 'mmseqs',
                'rows': int(len(rows)),
            }
        cluster_tsv = Path(str(cluster_prefix) + '_cluster.tsv')
        if not cluster_tsv.exists():
            return [_cluster_id_for_singletons(i) for i in range(len(rows))], {
                'status': 'failed',
                'reason': 'MMseqs cluster TSV was not created.',
                'tool': 'mmseqs',
                'rows': int(len(rows)),
            }
        assignments = [_cluster_id_for_singletons(i) for i in range(len(rows))]
        with open(cluster_tsv, 'r', encoding='utf-8') as inp:
            for line in inp:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 2:
                    continue
                representative, member = parts[0], parts[1]
                if member in id_to_index:
                    assignments[id_to_index[member]] = representative

    counts = Counter(assignments)
    return assignments, {
        'status': 'ok',
        'tool': 'mmseqs',
        'rows': int(len(rows)),
        'cluster_count': int(len(counts)),
        'largest_cluster_size': int(max(counts.values())) if counts else 0,
        'singleton_cluster_count': int(sum(1 for size in counts.values() if size == 1)),
        'min_seq_id': float(min_seq_id),
        'coverage': float(coverage),
        'cov_mode': int(cov_mode),
    }


def assign_cluster_folds(rows, cluster_ids, n_folds=5):
    rows = list(rows or [])
    cluster_ids = list(cluster_ids or [])
    if len(rows) != len(cluster_ids):
        raise ValueError('rows and cluster_ids should have the same length.')
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError('n_folds should be at least 2.')
    cluster_stats = {}
    for i, (row, cluster_id) in enumerate(zip(rows, cluster_ids)):
        stats = cluster_stats.setdefault(
            str(cluster_id),
            {'indices': [], 'positive': 0, 'negative': 0},
        )
        stats['indices'].append(i)
        if int(row.get('peroxisome', 0)) == 1:
            stats['positive'] += 1
        else:
            stats['negative'] += 1
    ordered = sorted(
        cluster_stats.items(),
        key=lambda item: (item[1]['positive'], len(item[1]['indices']), item[0]),
        reverse=True,
    )
    fold_stats = [
        {'positive': 0, 'negative': 0, 'total': 0}
        for _ in range(n_folds)
    ]
    fold_ids = [0 for _ in rows]
    for cluster_id, stats in ordered:
        if int(stats['positive']) > 0:
            best_fold = min(
                range(n_folds),
                key=lambda fold: (
                    fold_stats[fold]['positive'],
                    fold_stats[fold]['total'],
                    fold,
                ),
            )
        else:
            best_fold = min(
                range(n_folds),
                key=lambda fold: (
                    fold_stats[fold]['negative'],
                    fold_stats[fold]['total'],
                    fold,
                ),
            )
        for idx in stats['indices']:
            fold_ids[idx] = best_fold
        fold_stats[best_fold]['positive'] += int(stats['positive'])
        fold_stats[best_fold]['negative'] += int(stats['negative'])
        fold_stats[best_fold]['total'] += len(stats['indices'])
    return fold_ids, {
        'n_folds': int(n_folds),
        'cluster_count': int(len(cluster_stats)),
        'folds': [
            dict({'fold_id': int(fold)}, **fold_stats[fold])
            for fold in range(n_folds)
        ],
    }


def _rows_by_mask(rows, mask):
    return [row for row, keep in zip(rows, mask.tolist()) if bool(keep)]


def run_cluster_oof_perox_benchmark(
    rows,
    feature_profile='perox_sequence_v1',
    model_kind='extra_trees',
    n_folds=5,
    classification_threshold=0.5,
    cluster_method='mmseqs',
    cluster_min_seq_id=0.3,
    cluster_coverage=0.8,
    cluster_cov_mode=0,
    cluster_threads=1,
    cluster_tmp_dir='',
    random_state=1,
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=0.0,
    n_estimators=300,
    min_samples_leaf=10,
):
    rows = list(rows or [])
    if len(rows) == 0:
        return {
            'status': 'skipped',
            'reason': 'empty rows',
            'rows': 0,
        }
    method = str(cluster_method or 'mmseqs').strip().lower()
    if method == 'singleton':
        cluster_ids = [_cluster_id_for_singletons(i) for i in range(len(rows))]
        cluster_report = {
            'status': 'singleton',
            'tool': 'none',
            'rows': int(len(rows)),
            'cluster_count': int(len(rows)),
            'largest_cluster_size': 1,
            'singleton_cluster_count': int(len(rows)),
        }
    elif method == 'mmseqs':
        cluster_ids, cluster_report = mmseqs_cluster_assignments(
            rows=rows,
            min_seq_id=cluster_min_seq_id,
            coverage=cluster_coverage,
            cov_mode=cluster_cov_mode,
            threads=cluster_threads,
            tmp_root=cluster_tmp_dir,
        )
    else:
        raise ValueError('Unsupported cluster_method: {}'.format(cluster_method))

    fold_ids, fold_report = assign_cluster_folds(
        rows=rows,
        cluster_ids=cluster_ids,
        n_folds=n_folds,
    )
    fold_ids_arr = np.asarray(fold_ids, dtype=np.int64)
    prob = np.zeros((len(rows),), dtype=np.float64)
    fold_metrics = list()
    for fold_id in sorted(set(fold_ids)):
        valid_mask = fold_ids_arr == int(fold_id)
        train_mask = ~valid_mask
        train_rows = _rows_by_mask(rows, train_mask)
        valid_rows = _rows_by_mask(rows, valid_mask)
        model = fit_sklearn_perox_binary_model(
            rows=train_rows,
            feature_profile=feature_profile,
            model_kind=model_kind,
            threshold=classification_threshold,
            random_state=random_state + int(fold_id),
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
        )
        fold_prob = predict_perox_yes_probabilities(rows=valid_rows, perox_model=model)
        prob[valid_mask] = fold_prob
        row = binary_metrics(
            y_true=label_vector(valid_rows),
            y_prob=fold_prob,
            threshold=classification_threshold,
        )
        row['fold_id'] = int(fold_id)
        row['train_rows'] = int(len(train_rows))
        fold_metrics.append(row)

    metrics = binary_metrics(
        y_true=label_vector(rows),
        y_prob=prob,
        threshold=classification_threshold,
    )
    regex_metrics = binary_metrics(
        y_true=label_vector(rows),
        y_prob=regex_perox_probabilities(rows),
        threshold=0.5,
    )
    constant_zero_metrics = binary_metrics(
        y_true=label_vector(rows),
        y_prob=np.zeros((len(rows),), dtype=np.float64),
        threshold=0.5,
    )
    return {
        'status': 'ok',
        'rows': int(len(rows)),
        'positive_count': int(np.sum(label_vector(rows) == 1)),
        'feature_profile': str(feature_profile),
        'model_kind': str(model_kind),
        'classification_threshold': float(classification_threshold),
        'cluster_method': str(method),
        'cluster_report': cluster_report,
        'fold_assignment': fold_report,
        'metrics': metrics,
        'regex_metrics': regex_metrics,
        'constant_zero_metrics': constant_zero_metrics,
        'fold_metrics': fold_metrics,
    }


def _serializable_perox_model_summary(perox_model):
    return {
        key: value
        for key, value in perox_model.items()
        if key not in ['classifier']
    }


def attach_perox_model_to_base(base_model, perox_model, metadata):
    out = dict(base_model)
    out['perox_model'] = perox_model
    out_metadata = dict(out.get('metadata', {}))
    out_metadata.update(metadata)
    out['metadata'] = out_metadata
    return out


def write_predictions_tsv(path, rows, probabilities, threshold):
    with open(path, 'w', encoding='utf-8', newline='') as out:
        fieldnames = [
            'accession',
            'source',
            'kingdom',
            'partition',
            'true_peroxisome',
            'p_peroxisome',
            'predicted_peroxisome',
            'signal_type',
            'sequence_tail',
        ]
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for row, prob in zip(rows, probabilities.tolist()):
            signals = detect_perox_signals(row.get('sequence', ''))
            writer.writerow({
                'accession': row.get('accession', ''),
                'source': row.get('source', ''),
                'kingdom': row.get('kingdom', ''),
                'partition': row.get('partition', ''),
                'true_peroxisome': int(row.get('peroxisome', 0)),
                'p_peroxisome': float(prob),
                'predicted_peroxisome': 1 if float(prob) >= float(threshold) else 0,
                'signal_type': signals.get('signal_type', 'none'),
                'sequence_tail': str(row.get('sequence', ''))[-20:],
            })


def run_deeploc21_perox_benchmark(
    train_tsv=DEFAULT_TRAIN_TSV,
    external_test_tsv=DEFAULT_EXTERNAL_TEST_TSV,
    external_format='prepared',
    exclude_external_from_train=True,
    validation_partition='4',
    validation_fraction=0.2,
    feature_profile='perox_sequence_v1',
    model_kind='extra_trees',
    threshold_objective='f1',
    random_state=1,
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=0.0,
    n_estimators=300,
    min_samples_leaf=10,
    base_model='',
    model_out='',
    report_json='',
    report_md='',
    predictions_prefix='',
    homology_check=False,
    homology_min_seq_id=0.3,
    homology_coverage=0.8,
    homology_cov_mode=0,
    homology_threads=1,
    homology_tmp_dir='',
    cluster_oof=False,
    cluster_oof_source='external',
    cluster_oof_folds=5,
    cluster_oof_threshold=0.5,
    cluster_oof_method='mmseqs',
):
    original_all_rows = read_perox_rows(train_tsv)
    if external_test_tsv:
        external_rows, external_info = load_external_perox_rows(
            path=external_test_tsv,
            external_format=external_format,
            exclude_rows=original_all_rows,
        )
    else:
        external_rows = []
        external_info = {'format': str(external_format)}
    if _str_to_bool(exclude_external_from_train) and external_rows:
        all_rows, train_exclusion_report = exclude_rows_overlapping_eval(
            train_rows=original_all_rows,
            eval_rows=external_rows,
        )
    else:
        all_rows = original_all_rows
        train_exclusion_report = {
            'enabled': False,
            'input_rows': int(len(original_all_rows)),
            'eval_rows': int(len(external_rows)),
            'kept_rows': int(len(original_all_rows)),
            'removed_rows': 0,
            'accession_overlap_rows': 0,
            'exact_sequence_overlap_rows': 0,
        }

    train_rows, valid_rows = split_train_valid(
        rows=all_rows,
        validation_partition=validation_partition,
        validation_fraction=validation_fraction,
    )

    fold_model = fit_sklearn_perox_binary_model(
        rows=train_rows,
        feature_profile=feature_profile,
        model_kind=model_kind,
        threshold=0.5,
        random_state=random_state,
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
    )
    valid_prob = predict_perox_yes_probabilities(rows=valid_rows, perox_model=fold_model)
    threshold = tune_threshold(
        y_true=label_vector(valid_rows),
        y_prob=valid_prob,
        objective=threshold_objective,
    )
    fold_model['threshold'] = float(threshold)

    final_model = fit_sklearn_perox_binary_model(
        rows=all_rows,
        feature_profile=feature_profile,
        model_kind=model_kind,
        threshold=threshold,
        random_state=random_state,
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
    )
    final_model['threshold'] = float(threshold)

    homology_validation = maybe_homology_report(
        train_rows=train_rows,
        eval_rows=valid_rows,
        enabled=homology_check,
        min_seq_id=homology_min_seq_id,
        coverage=homology_coverage,
        cov_mode=homology_cov_mode,
        threads=homology_threads,
        tmp_root=homology_tmp_dir,
        include_hit_indices=True,
    )
    validation_homology_hit_indices = homology_validation.pop('_hit_eval_indices', [])
    if external_rows:
        homology_external = maybe_homology_report(
            train_rows=all_rows,
            eval_rows=external_rows,
            enabled=homology_check,
            min_seq_id=homology_min_seq_id,
            coverage=homology_coverage,
            cov_mode=homology_cov_mode,
            threads=homology_threads,
            tmp_root=homology_tmp_dir,
            include_hit_indices=True,
        )
        external_homology_hit_indices = homology_external.pop('_hit_eval_indices', [])
    else:
        homology_external = None
        external_homology_hit_indices = []

    report = {
        'dataset': {
            'train_tsv': str(train_tsv),
            'external_test_tsv': str(external_test_tsv),
            'external_format': str(external_info.get('format', external_format)),
            'external_filter_report': external_info,
            'train_exclusion_report': train_exclusion_report,
            'validation_partition': str(validation_partition),
            'validation_fraction': float(validation_fraction),
            'all_train_validation_rows': int(len(all_rows)),
            'train_rows_for_threshold_model': int(len(train_rows)),
            'validation_rows': int(len(valid_rows)),
            'external_rows': int(len(external_rows)),
            'train_validation_positive_count': int(np.sum(label_vector(all_rows) == 1)),
            'validation_positive_count': int(np.sum(label_vector(valid_rows) == 1)),
            'external_positive_count': int(np.sum(label_vector(external_rows) == 1)) if external_rows else 0,
        },
        'model': _serializable_perox_model_summary(final_model),
        'threshold_selection': {
            'objective': str(threshold_objective),
            'threshold': float(threshold),
            'tuned_on': 'validation partition {}'.format(validation_partition),
        },
        'leakage_checks': {
            'threshold_train_vs_validation': leakage_report(train_rows, valid_rows),
            'all_train_validation_vs_external': leakage_report(all_rows, external_rows) if external_rows else None,
            'homology_threshold_train_vs_validation': homology_validation,
            'homology_all_train_validation_vs_external': homology_external,
        },
        'label_quality': {
            'all_train_validation': sequence_label_conflict_report(all_rows),
            'threshold_train': sequence_label_conflict_report(train_rows),
            'validation': sequence_label_conflict_report(valid_rows),
            'external': sequence_label_conflict_report(external_rows) if external_rows else None,
        },
        'metrics': {
            'validation_model': binary_metrics(
                y_true=label_vector(valid_rows),
                y_prob=valid_prob,
                threshold=threshold,
            ),
            'validation_regex_pts': binary_metrics(
                y_true=label_vector(valid_rows),
                y_prob=regex_perox_probabilities(valid_rows),
                threshold=0.5,
            ),
            'validation_constant_zero': binary_metrics(
                y_true=label_vector(valid_rows),
                y_prob=np.zeros((len(valid_rows),), dtype=np.float64),
                threshold=0.5,
            ),
        },
    }
    if str(validation_partition).strip().lower() in ['hash_stratified', 'auto_hash', 'stratified_hash']:
        report['threshold_selection']['tuned_on'] = (
            'hash-stratified validation split (fraction {:.3f})'.format(float(validation_fraction))
        )
    if _str_to_bool(cluster_oof):
        source = str(cluster_oof_source or 'external').strip().lower()
        if source == 'external':
            oof_rows = external_rows
        elif source == 'train_validation':
            oof_rows = all_rows
        else:
            raise ValueError('Unsupported cluster_oof_source: {}'.format(cluster_oof_source))
        report['cluster_oof'] = run_cluster_oof_perox_benchmark(
            rows=oof_rows,
            feature_profile=feature_profile,
            model_kind=model_kind,
            n_folds=cluster_oof_folds,
            classification_threshold=cluster_oof_threshold,
            cluster_method=cluster_oof_method,
            cluster_min_seq_id=homology_min_seq_id,
            cluster_coverage=homology_coverage,
            cluster_cov_mode=homology_cov_mode,
            cluster_threads=homology_threads,
            cluster_tmp_dir=homology_tmp_dir,
            random_state=random_state,
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
        )
        report['cluster_oof']['source'] = source

    if _str_to_bool(homology_check) and homology_validation.get('status') == 'ok':
        add_homology_subset_metrics(
            metrics=report['metrics'],
            prefix='validation_model',
            rows=valid_rows,
            probabilities=valid_prob,
            hit_indices=validation_homology_hit_indices,
            threshold=threshold,
        )

    external_prob = None
    if external_rows:
        external_prob = predict_perox_yes_probabilities(rows=external_rows, perox_model=final_model)
        report['metrics']['external_final_model'] = binary_metrics(
            y_true=label_vector(external_rows),
            y_prob=external_prob,
            threshold=threshold,
        )
        report['metrics']['external_regex_pts'] = binary_metrics(
            y_true=label_vector(external_rows),
            y_prob=regex_perox_probabilities(external_rows),
            threshold=0.5,
        )
        report['metrics']['external_constant_zero'] = binary_metrics(
            y_true=label_vector(external_rows),
            y_prob=np.zeros((len(external_rows),), dtype=np.float64),
            threshold=0.5,
        )
        if _str_to_bool(homology_check) and homology_external is not None and homology_external.get('status') == 'ok':
            add_homology_subset_metrics(
                metrics=report['metrics'],
                prefix='external_final_model',
                rows=external_rows,
                probabilities=external_prob,
                hit_indices=external_homology_hit_indices,
                threshold=threshold,
            )

    if model_out:
        if not base_model:
            raise ValueError('--base_model is required when --model_out is set.')
        base_model_path = resolve_localize_model_path(base_model, allow_download=True)
        base_payload = load_localize_model(base_model_path)
        attached = attach_perox_model_to_base(
            base_model=base_payload,
            perox_model=final_model,
            metadata={
                'perox_model_source': 'perox_train_tsv',
                'perox_model_source_tsv': str(train_tsv),
                'perox_model_exclude_external_from_train': bool(_str_to_bool(exclude_external_from_train)),
                'perox_model_feature_profile': str(feature_profile),
                'perox_model_validation_partition': str(validation_partition),
                'perox_model_validation_fraction': float(validation_fraction),
                'perox_model_threshold': float(threshold),
                'perox_model_external_test': str(external_test_tsv),
                'perox_model_prediction_scope': final_model.get('prediction_scope', ''),
            },
        )
        Path(model_out).parent.mkdir(parents=True, exist_ok=True)
        save_localize_model(model=attached, path=str(model_out))
        report['model_out'] = str(model_out)

    if predictions_prefix:
        prefix = Path(predictions_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        write_predictions_tsv(
            path=str(prefix) + '.validation.tsv',
            rows=valid_rows,
            probabilities=valid_prob,
            threshold=threshold,
        )
        if external_rows and external_prob is not None:
            write_predictions_tsv(
                path=str(prefix) + '.external.tsv',
                rows=external_rows,
                probabilities=external_prob,
                threshold=threshold,
            )
        report['predictions_prefix'] = str(predictions_prefix)

    if report_json:
        Path(report_json).parent.mkdir(parents=True, exist_ok=True)
        with open(report_json, 'w', encoding='utf-8') as out:
            json.dump(report, out, indent=2, sort_keys=True)

    if report_md:
        Path(report_md).parent.mkdir(parents=True, exist_ok=True)
        with open(report_md, 'w', encoding='utf-8') as out:
            out.write(format_report_markdown(report))

    return report


def _metric_row(name, metrics):
    values = [
        name,
        str(metrics.get('rows', '')),
        str(metrics.get('positive_count', '')),
        '{:.4f}'.format(float(metrics.get('auprc', 0.0) or 0.0)) if metrics.get('auprc') is not None else 'NA',
        '{:.4f}'.format(float(metrics.get('auroc', 0.0) or 0.0)) if metrics.get('auroc') is not None else 'NA',
        '{:.4f}'.format(float(metrics.get('precision', 0.0))),
        '{:.4f}'.format(float(metrics.get('recall', 0.0))),
        '{:.4f}'.format(float(metrics.get('f1', 0.0))),
        '{:.4f}'.format(float(metrics.get('mcc', 0.0))),
        str(metrics.get('tp', '')),
        str(metrics.get('fp', '')),
        str(metrics.get('fn', '')),
        str(metrics.get('tn', '')),
    ]
    return '| ' + ' | '.join(values) + ' |'


def format_report_markdown(report):
    lines = [
        '# cdskit localize peroxisome head benchmark',
        '',
        '## Dataset',
        '',
        '- Train/validation TSV: `{}`'.format(report['dataset']['train_tsv']),
        '- External test TSV: `{}`'.format(report['dataset']['external_test_tsv']),
        '- External format: `{}`'.format(report['dataset']['external_format']),
        '- External-overlap rows removed from training: `{}` / `{}`'.format(
            report['dataset'].get('train_exclusion_report', {}).get('removed_rows', 0),
            report['dataset'].get('train_exclusion_report', {}).get('input_rows', 0),
        ),
        '- Validation partition for threshold tuning: `{}`'.format(
            report['dataset']['validation_partition']
        ),
        '- Validation fraction: `{:.3f}`'.format(
            float(report['dataset'].get('validation_fraction', 0.0))
        ),
        '- Train/validation positives: `{}` / `{}`'.format(
            report['dataset']['train_validation_positive_count'],
            report['dataset']['all_train_validation_rows'],
        ),
        '- External positives: `{}` / `{}`'.format(
            report['dataset']['external_positive_count'],
            report['dataset']['external_rows'],
        ),
        '',
        '## Threshold',
        '',
        '- Objective: `{}`'.format(report['threshold_selection']['objective']),
        '- Threshold: `{:.4f}`'.format(float(report['threshold_selection']['threshold'])),
        '- Tuned on: `{}`'.format(report['threshold_selection'].get('tuned_on', '')),
        '',
        '## Interpretation',
        '',
        '- The peroxisome head is a CPU-runtime sequence-feature model.',
        '- It is expected to be strongest for PTS-like peroxisomal targeting signals; broad peroxisome-associated localization should be judged by external and cluster-OOF metrics.',
        '- Constant-zero and regex PTS baselines are included so apparent gains can be checked against simple alternatives.',
        '',
        '## Metrics',
        '',
        '| split/model | rows | pos | AUPRC | AUROC | precision | recall | F1 | MCC | TP | FP | FN | TN |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for name, metrics in report.get('metrics', {}).items():
        lines.append(_metric_row(name, metrics))
    cluster_oof = report.get('cluster_oof', None)
    if isinstance(cluster_oof, dict):
        lines.extend([
            '',
            '## Cluster OOF',
            '',
            '- Source: `{}`'.format(cluster_oof.get('source', '')),
            '- Status: `{}`'.format(cluster_oof.get('status', '')),
            '- Cluster method: `{}`'.format(cluster_oof.get('cluster_method', '')),
            '- Classification threshold: `{}`'.format(cluster_oof.get('classification_threshold', '')),
            '',
            '| model | rows | pos | AUPRC | AUROC | precision | recall | F1 | MCC | TP | FP | FN | TN |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        ])
        if 'metrics' in cluster_oof:
            lines.append(_metric_row('cluster_oof_model', cluster_oof['metrics']))
        if 'regex_metrics' in cluster_oof:
            lines.append(_metric_row('cluster_oof_regex_pts', cluster_oof['regex_metrics']))
        if 'constant_zero_metrics' in cluster_oof:
            lines.append(_metric_row('cluster_oof_constant_zero', cluster_oof['constant_zero_metrics']))
        lines.extend([
            '',
            'Cluster report:',
            '',
            '```json',
            json.dumps(cluster_oof.get('cluster_report', {}), indent=2, sort_keys=True),
            '```',
        ])
    lines.extend([
        '',
        '## Leakage Checks',
        '',
        '```json',
        json.dumps(report.get('leakage_checks', {}), indent=2, sort_keys=True),
        '```',
        '',
        '## Label Quality',
        '',
        '```json',
        json.dumps(report.get('label_quality', {}), indent=2, sort_keys=True),
        '```',
        '',
        'Note: external-set recall/F1 can be unstable when the positive count is small; keep AUPRC, AUROC, and the positive count visible when comparing runs.',
        '',
    ])
    return '\n'.join(lines)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Train and evaluate a cdskit localize peroxisome head.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--train_tsv',
        default=DEFAULT_TRAIN_TSV,
        type=str,
        help='Prepared perox TSV with accession, kingdom, partition, sequence, and peroxisome columns.',
    )
    parser.add_argument(
        '--external_test_tsv',
        default=DEFAULT_EXTERNAL_TEST_TSV,
        type=str,
        help='Optional external evaluation TSV. Use an empty string to skip external evaluation.',
    )
    parser.add_argument(
        '--external_format',
        default='prepared',
        choices=['prepared', 'uniprot_exp_cc'],
        help='External table format. uniprot_exp_cc extracts ECO:0000269 experimental CC labels.',
    )
    parser.add_argument(
        '--exclude_external_from_train',
        default='yes',
        choices=['yes', 'no'],
        help='Remove training rows with accession or exact-sequence overlap against the external test set.',
    )
    parser.add_argument(
        '--validation_partition',
        default='4',
        type=str,
        help='Partition value used for threshold tuning, or hash_stratified for deterministic label-stratified holdout.',
    )
    parser.add_argument(
        '--validation_fraction',
        default=0.2,
        type=float,
        help='Validation fraction used only when --validation_partition hash_stratified is selected.',
    )
    parser.add_argument('--feature_profile', default='perox_sequence_v1', choices=[
        'perox_sequence_v1',
        'broad_localize_v1',
        TARGETP_FEATURE_ENSEMBLE_PROFILE['name'],
    ], help='Sequence feature profile used by the peroxisome head.')
    parser.add_argument('--model_kind', default='extra_trees', choices=[
        'hist_gradient_boosting',
        'hgb',
        'extra_trees',
        'et',
    ], help='CPU scikit-learn classifier family for the peroxisome head.')
    parser.add_argument('--threshold_objective', default='f1', choices=['f1', 'mcc', 'precision', 'recall'])
    parser.add_argument('--random_state', default=1, type=int)
    parser.add_argument('--max_iter', default=200, type=int)
    parser.add_argument('--learning_rate', default=0.05, type=float)
    parser.add_argument('--max_leaf_nodes', default=31, type=int)
    parser.add_argument('--l2_regularization', default=0.0, type=float)
    parser.add_argument('--n_estimators', default=300, type=int)
    parser.add_argument('--min_samples_leaf', default=10, type=int)
    parser.add_argument('--base_model', default='', type=str)
    parser.add_argument('--model_out', default='', type=str)
    parser.add_argument('--report_json', default='', type=str)
    parser.add_argument('--report_md', default='', type=str)
    parser.add_argument('--predictions_prefix', default='', type=str)
    parser.add_argument(
        '--homology_check',
        default='no',
        choices=['yes', 'no'],
        help='Run MMseqs train-vs-evaluation homology checks and subset metrics.',
    )
    parser.add_argument('--homology_min_seq_id', default=0.3, type=float)
    parser.add_argument('--homology_coverage', default=0.8, type=float)
    parser.add_argument('--homology_cov_mode', default=0, type=int)
    parser.add_argument('--homology_threads', default=1, type=int)
    parser.add_argument('--homology_tmp_dir', default='', type=str)
    parser.add_argument(
        '--cluster_oof',
        default='no',
        choices=['yes', 'no'],
        help='Run cluster-level out-of-fold evaluation to expose similarity-driven optimism.',
    )
    parser.add_argument('--cluster_oof_source', default='external', choices=['external', 'train_validation'])
    parser.add_argument('--cluster_oof_folds', default=5, type=int)
    parser.add_argument('--cluster_oof_threshold', default=0.5, type=float)
    parser.add_argument('--cluster_oof_method', default='mmseqs', choices=['mmseqs', 'singleton'])
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = run_deeploc21_perox_benchmark(
        train_tsv=args.train_tsv,
        external_test_tsv=args.external_test_tsv,
        external_format=args.external_format,
        exclude_external_from_train=args.exclude_external_from_train,
        validation_partition=args.validation_partition,
        validation_fraction=args.validation_fraction,
        feature_profile=args.feature_profile,
        model_kind=args.model_kind,
        threshold_objective=args.threshold_objective,
        random_state=args.random_state,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        base_model=args.base_model,
        model_out=args.model_out,
        report_json=args.report_json,
        report_md=args.report_md,
        predictions_prefix=args.predictions_prefix,
        homology_check=args.homology_check,
        homology_min_seq_id=args.homology_min_seq_id,
        homology_coverage=args.homology_coverage,
        homology_cov_mode=args.homology_cov_mode,
        homology_threads=args.homology_threads,
        homology_tmp_dir=args.homology_tmp_dir,
        cluster_oof=args.cluster_oof,
        cluster_oof_source=args.cluster_oof_source,
        cluster_oof_folds=args.cluster_oof_folds,
        cluster_oof_threshold=args.cluster_oof_threshold,
        cluster_oof_method=args.cluster_oof_method,
    )
    if not args.report_json:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

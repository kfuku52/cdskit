import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import tempfile
from collections import Counter

import numpy as np

from cdskit.localize_model import (
    LOCALIZATION_CLASSES,
    infer_labels_from_uniprot_cc,
    load_localize_model,
    predict_localization_and_peroxisome,
    to_canonical_aa_sequence,
)
from cdskit.targetp_feature_ensemble import (
    _metrics_from_prediction_indices,
    _prediction_indices_with_thresholds,
    optimize_class_thresholds,
)
from cdskit.uniprot_preset_split import classify_lineage_ids, parse_taxon_ids


DEFAULT_CLASS_THRESHOLD_GRID = [
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.65,
    0.8,
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
    5.0,
]


DEEPLOC_SORTING_TO_TARGETP = {
    'SP': 'SP',
    'MT': 'mTP',
    'CH': 'cTP',
    'TH': 'lTP',
}

DEEPLOC_LOCALIZATION_TO_TARGETP = {
    'extracellular': 'SP',
    'mitochondrion': 'mTP',
    'chloroplast': 'cTP',
}


def read_tsv(path):
    with open(path, 'r', encoding='utf-8', newline='') as inp:
        return list(csv.DictReader(inp, delimiter='\t'))


def write_tsv(path, rows, fieldnames):
    out_dir = os.path.dirname(path)
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, delimiter='\t', fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, '') for name in fieldnames})


def _accession_key(value):
    return str(value or '').strip().split('.')[0]


def _sequence_key(value):
    return str(value or '').strip().upper()


def load_targetp_exclusion_keys(targetp_tsv):
    rows = read_tsv(path=targetp_tsv)
    return {
        'rows': rows,
        'accessions': set(_accession_key(row.get('accession', '')) for row in rows),
        'sequences': set(_sequence_key(row.get('sequence', '')) for row in rows),
    }


def is_exact_targetp_overlap(row, targetp_keys):
    accession = _accession_key(row.get('accession', ''))
    sequence = _sequence_key(row.get('sequence', ''))
    return (
        (accession != '' and accession in targetp_keys['accessions'])
        or (sequence != '' and sequence in targetp_keys['sequences'])
    )


def organism_group_from_row(row):
    explicit = str(row.get('organism_group', '') or '').strip()
    if explicit != '':
        return explicit
    kingdom = str(row.get('kingdom', '') or '').strip().lower()
    if kingdom == 'viridiplantae':
        return 'plant'
    if kingdom in ['metazoa', 'fungi', 'other']:
        return 'non_plant'
    lineage_text = ''
    for key in ['lineage_ids', 'Taxonomic lineage (Ids)', 'Taxonomic lineage IDs']:
        if key in row:
            lineage_text = row.get(key, '')
            break
    if lineage_text != '':
        flags = classify_lineage_ids(taxon_ids=parse_taxon_ids(lineage_text))
        if flags.get('viridiplantae', False):
            return 'plant'
        if flags.get('eukaryota', False):
            return 'non_plant'
    return 'unknown'


def build_deeploc_sorting_rows(path, targetp_keys, exclude_exact=True):
    rows = read_tsv(path=path)
    out = list()
    skipped = Counter()
    for row in rows:
        if exclude_exact and is_exact_targetp_overlap(row=row, targetp_keys=targetp_keys):
            skipped['targetp_exact_overlap'] += 1
            continue
        active = [
            value for value in str(row.get('sorting_signal_labels', '')).split(';')
            if value != ''
        ]
        mapped = [DEEPLOC_SORTING_TO_TARGETP[value] for value in active if value in DEEPLOC_SORTING_TO_TARGETP]
        if len(mapped) == 0:
            skipped['no_targetp_equivalent_label'] += 1
            continue
        if len(mapped) > 1:
            skipped['ambiguous_targetp_equivalent_label'] += 1
            continue
        out.append({
            'source': 'deeploc21_sorting_signals',
            'accession': row.get('accession', ''),
            'sequence': row.get('sequence', ''),
            'organism_group': organism_group_from_row(row),
            'true_class': mapped[0],
            'external_labels': ';'.join(active),
        })
    return out, dict(skipped)


def _targetp_class_from_deeploc_localization_labels(label_text):
    active = [value for value in str(label_text or '').split(';') if value != '']
    mapped = [
        DEEPLOC_LOCALIZATION_TO_TARGETP[value]
        for value in active
        if value in DEEPLOC_LOCALIZATION_TO_TARGETP
    ]
    mapped = sorted(set(mapped))
    if len(mapped) > 1:
        return '', active, True
    if len(mapped) == 1:
        return mapped[0], active, False
    return 'noTP', active, False


def build_deeploc_hpa_broad_rows(path, targetp_keys, exclude_exact=True):
    rows = read_tsv(path=path)
    out = list()
    skipped = Counter()
    for row in rows:
        if exclude_exact and is_exact_targetp_overlap(row=row, targetp_keys=targetp_keys):
            skipped['targetp_exact_overlap'] += 1
            continue
        true_class, active, ambiguous = _targetp_class_from_deeploc_localization_labels(
            row.get('localization_labels', '')
        )
        if ambiguous:
            skipped['ambiguous_targetp_equivalent_label'] += 1
            continue
        out.append({
            'source': 'deeploc21_hpa_broad',
            'accession': row.get('accession', ''),
            'sequence': row.get('sequence', ''),
            'organism_group': organism_group_from_row(row),
            'true_class': true_class,
            'external_labels': ';'.join(active),
        })
    return out, dict(skipped)


def build_uniprot_holdout_rows(
    path,
    targetp_keys,
    exclude_exact=True,
    skip_ambiguous=True,
    strict_targetp_organism_labels=False,
):
    rows = read_tsv(path=path)
    out = list()
    skipped = Counter()
    for row in rows:
        if exclude_exact and is_exact_targetp_overlap(row=row, targetp_keys=targetp_keys):
            skipped['targetp_exact_overlap'] += 1
            continue
        cc_text = row.get('cc_subcellular_location', row.get('localization', ''))
        if str(cc_text or '').strip() == '':
            skipped['missing_location_text'] += 1
            continue
        true_class, _, ambiguous = infer_labels_from_uniprot_cc(location_text=cc_text)
        if ambiguous and skip_ambiguous:
            skipped['ambiguous_uniprot_cc'] += 1
            continue
        organism_group = organism_group_from_row(row)
        if (
            bool(strict_targetp_organism_labels)
            and true_class in ['cTP', 'lTP']
            and organism_group != 'plant'
        ):
            skipped['inconsistent_targetp_organism_label'] += 1
            continue
        out.append({
            'source': 'uniprot_cc_holdout',
            'accession': row.get('accession', ''),
            'sequence': row.get('sequence', ''),
            'organism_group': organism_group,
            'true_class': true_class,
            'external_labels': true_class,
        })
    return out, dict(skipped)


def stratified_sample_rows(rows, max_per_class=0, seed=1):
    if int(max_per_class) <= 0:
        return list(rows)
    by_class = {class_name: [] for class_name in LOCALIZATION_CLASSES}
    for row in rows:
        class_name = row.get('true_class', '')
        if class_name in by_class:
            by_class[class_name].append(row)
    rng = random.Random(int(seed))
    out = list()
    for class_name in LOCALIZATION_CLASSES:
        class_rows = list(by_class[class_name])
        rng.shuffle(class_rows)
        out.extend(class_rows[:int(max_per_class)])
    return out


def _write_fasta(path, rows):
    with open(path, 'w', encoding='utf-8') as out:
        for row_i, row in enumerate(rows):
            seq = to_canonical_aa_sequence(row.get('sequence', ''))
            if seq == '':
                continue
            out.write('>{}\n{}\n'.format(row.get('_mmseqs_id', 'seq{}'.format(row_i)), seq))


def filter_rows_by_mmseqs_similarity(
    rows,
    targetp_rows,
    min_seq_id=0.30,
    min_coverage=0.80,
    threads=1,
    enabled=True,
):
    report = {
        'requested': bool(enabled),
        'available': False,
        'min_seq_id': float(min_seq_id),
        'min_coverage': float(min_coverage),
        'removed': 0,
        'kept': int(len(rows)),
    }
    if not enabled:
        report['status'] = 'disabled'
        return list(rows), report
    mmseqs = shutil.which('mmseqs')
    if mmseqs is None:
        report['status'] = 'mmseqs_not_found'
        return list(rows), report

    query_rows = [dict(row, _mmseqs_id='q{}'.format(i)) for i, row in enumerate(rows)]
    target_rows = [dict(row, _mmseqs_id='t{}'.format(i)) for i, row in enumerate(targetp_rows)]
    with tempfile.TemporaryDirectory(prefix='cdskit_targetp_ext_') as tmpdir:
        query_fasta = os.path.join(tmpdir, 'query.fa')
        target_fasta = os.path.join(tmpdir, 'target.fa')
        out_m8 = os.path.join(tmpdir, 'hits.m8')
        mmseqs_tmp = os.path.join(tmpdir, 'mmseqs_tmp')
        _write_fasta(path=query_fasta, rows=query_rows)
        _write_fasta(path=target_fasta, rows=target_rows)
        cmd = [
            mmseqs,
            'easy-search',
            query_fasta,
            target_fasta,
            out_m8,
            mmseqs_tmp,
            '--min-seq-id',
            str(float(min_seq_id)),
            '-c',
            str(float(min_coverage)),
            '--cov-mode',
            '0',
            '--threads',
            str(max(1, int(threads))),
            '--format-output',
            'query,target,pident,alnlen,qlen,tlen,evalue,bits',
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        report['available'] = True
        report['command'] = ' '.join(cmd)
        report['returncode'] = int(proc.returncode)
        if proc.returncode != 0:
            report['status'] = 'mmseqs_failed'
            report['stderr'] = proc.stderr[-2000:]
            return list(rows), report
        hit_queries = set()
        if os.path.exists(out_m8):
            with open(out_m8, 'r', encoding='utf-8') as inp:
                for line in inp:
                    fields = line.rstrip('\n').split('\t')
                    if len(fields) >= 5:
                        hit_queries.add(fields[0])
        kept = [
            row for row in query_rows
            if row.get('_mmseqs_id', '') not in hit_queries
        ]
    for row in kept:
        row.pop('_mmseqs_id', None)
    report['removed'] = int(len(rows) - len(kept))
    report['kept'] = int(len(kept))
    report['status'] = 'ok'
    return kept, report


def compute_single_label_metrics(rows, class_names=LOCALIZATION_CLASSES):
    class_names = list(class_names)
    true = [row.get('true_class', '') for row in rows]
    pred = [row.get('predicted_class', '') for row in rows]
    by_class = dict()
    for class_name in class_names:
        tp = sum(1 for t, p in zip(true, pred) if t == class_name and p == class_name)
        fp = sum(1 for t, p in zip(true, pred) if t != class_name and p == class_name)
        fn = sum(1 for t, p in zip(true, pred) if t == class_name and p != class_name)
        support = sum(1 for t in true if t == class_name)
        precision = 0.0 if tp + fp == 0 else float(tp) / float(tp + fp)
        recall = 0.0 if tp + fn == 0 else float(tp) / float(tp + fn)
        f1 = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall)
        by_class[class_name] = {
            'support': int(support),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }
    accuracy = 0.0 if len(rows) == 0 else float(sum(1 for t, p in zip(true, pred) if t == p)) / float(len(rows))
    macro_f1 = float(np.mean([by_class[class_name]['f1'] for class_name in class_names]))
    observed = [class_name for class_name in class_names if by_class[class_name]['support'] > 0]
    observed_macro = 0.0 if len(observed) == 0 else float(np.mean([by_class[class_name]['f1'] for class_name in observed]))
    return {
        'n_rows': int(len(rows)),
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'macro_f1_observed_labels': float(observed_macro),
        'true_counts': dict(Counter(true)),
        'predicted_counts': dict(Counter(pred)),
        'by_class': by_class,
    }


def _prob_matrix_from_prediction_rows(rows, class_names=LOCALIZATION_CLASSES):
    class_names = list(class_names)
    out = list()
    for row in rows:
        values = list()
        for class_name in class_names:
            value = row.get('p_{}'.format(class_name), 0.0)
            try:
                value = float(value)
            except Exception:
                value = 0.0
            values.append(float(value))
        out.append(values)
    if len(out) == 0:
        return np.zeros((0, len(class_names)), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def _true_indices_from_prediction_rows(rows, class_names=LOCALIZATION_CLASSES):
    class_names = list(class_names)
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
    out = list()
    for row in rows:
        class_name = str(row.get('true_class', '')).strip()
        if class_name not in class_to_idx:
            raise ValueError('Unknown true_class in prediction rows: {}'.format(class_name))
        out.append(int(class_to_idx[class_name]))
    return np.asarray(out, dtype=np.int64)


def _stratified_fold_indices(true_idx, n_folds=5, seed=1):
    true_idx = np.asarray(true_idx, dtype=np.int64)
    if true_idx.shape[0] == 0:
        return []
    if true_idx.shape[0] == 1:
        return [np.arange(1, dtype=np.int64)]
    n_folds = int(max(2, min(int(n_folds), int(true_idx.shape[0]))))
    rng = random.Random(int(seed))
    by_class = dict()
    for row_i, class_i in enumerate(true_idx.tolist()):
        by_class.setdefault(int(class_i), []).append(int(row_i))
    folds = [[] for _ in range(n_folds)]
    for class_i in sorted(by_class.keys()):
        indices = list(by_class[class_i])
        rng.shuffle(indices)
        for pos, row_i in enumerate(indices):
            folds[pos % n_folds].append(int(row_i))
    return [
        np.asarray(sorted(fold), dtype=np.int64)
        for fold in folds
        if len(fold) > 0
    ]


def evaluate_prediction_threshold_calibration(
    rows,
    threshold_grid=None,
    cv_folds=5,
    seed=1,
    class_names=LOCALIZATION_CLASSES,
):
    class_names = list(class_names)
    threshold_grid = (
        list(DEFAULT_CLASS_THRESHOLD_GRID)
        if threshold_grid is None
        else sorted(set([float(value) for value in threshold_grid]))
    )
    prob_matrix = _prob_matrix_from_prediction_rows(rows=rows, class_names=class_names)
    true_idx = _true_indices_from_prediction_rows(rows=rows, class_names=class_names)
    if prob_matrix.shape[0] == 0:
        return {
            'n_rows': 0,
            'threshold_grid': [float(value) for value in threshold_grid],
        }
    argmax_metrics = _metrics_from_prediction_indices(
        pred_idx=np.argmax(prob_matrix, axis=1),
        true_idx=true_idx,
        class_names=class_names,
    )
    oracle_thresholds, oracle_train_metrics = optimize_class_thresholds(
        prob_matrix=prob_matrix,
        true_idx=true_idx,
        class_names=class_names,
        grid=threshold_grid,
    )
    oracle_pred = _prediction_indices_with_thresholds(
        prob_matrix=prob_matrix,
        thresholds=oracle_thresholds,
    )
    oracle_metrics = _metrics_from_prediction_indices(
        pred_idx=oracle_pred,
        true_idx=true_idx,
        class_names=class_names,
    )
    cv_pred = np.zeros((true_idx.shape[0],), dtype=np.int64)
    folds_out = list()
    all_idx = np.arange(true_idx.shape[0], dtype=np.int64)
    folds = _stratified_fold_indices(true_idx=true_idx, n_folds=cv_folds, seed=seed)
    for fold_i, valid_idx in enumerate(folds):
        valid_mask = np.zeros((true_idx.shape[0],), dtype=bool)
        valid_mask[valid_idx] = True
        train_idx = all_idx[~valid_mask]
        if train_idx.shape[0] == 0:
            cv_pred[valid_idx] = np.argmax(prob_matrix[valid_idx, :], axis=1)
            folds_out.append({
                'fold': int(fold_i + 1),
                'n_train': 0,
                'n_valid': int(valid_idx.shape[0]),
                'train_macro_f1': 0.0,
                'thresholds': {
                    class_name: 1.0 for class_name in class_names
                },
            })
            continue
        thresholds, train_metrics = optimize_class_thresholds(
            prob_matrix=prob_matrix[train_idx, :],
            true_idx=true_idx[train_idx],
            class_names=class_names,
            grid=threshold_grid,
        )
        cv_pred[valid_idx] = _prediction_indices_with_thresholds(
            prob_matrix=prob_matrix[valid_idx, :],
            thresholds=thresholds,
        )
        folds_out.append({
            'fold': int(fold_i + 1),
            'n_train': int(train_idx.shape[0]),
            'n_valid': int(valid_idx.shape[0]),
            'train_macro_f1': float(train_metrics['macro_f1']),
            'thresholds': {
                class_names[class_i]: float(thresholds[class_i])
                for class_i in range(len(class_names))
            },
        })
    cv_metrics = _metrics_from_prediction_indices(
        pred_idx=cv_pred,
        true_idx=true_idx,
        class_names=class_names,
    )
    return {
        'n_rows': int(true_idx.shape[0]),
        'threshold_grid': [float(value) for value in threshold_grid],
        'argmax_metrics': argmax_metrics,
        'oracle_thresholds': {
            class_names[class_i]: float(oracle_thresholds[class_i])
            for class_i in range(len(class_names))
        },
        'oracle_train_macro_f1': float(oracle_train_metrics['macro_f1']),
        'oracle_metrics': oracle_metrics,
        'cv_folds': int(len(folds)),
        'cv_seed': int(seed),
        'cv_metrics': cv_metrics,
        'folds': folds_out,
    }


def predict_rows(rows, model_path):
    model = load_localize_model(path=model_path)
    out = list()
    for row in rows:
        seq = to_canonical_aa_sequence(row.get('sequence', ''))
        if seq == '':
            continue
        pred = predict_localization_and_peroxisome(
            aa_seq=seq,
            model=model,
            organism_group=row.get('organism_group', ''),
        )
        pred_row = dict(row)
        pred_row['predicted_class'] = pred['predicted_class']
        for class_name in LOCALIZATION_CLASSES:
            pred_row['p_{}'.format(class_name)] = float(
                pred['class_probabilities'].get(class_name, 0.0)
            )
        pred_row['p_peroxisome'] = float(pred.get('perox_probability_yes', 0.0))
        pred_row['perox_signal_type'] = pred.get('perox_signal_type', '')
        out.append(pred_row)
    return out


def _dataset_report(rows, skipped):
    return {
        'n_rows': int(len(rows)),
        'true_counts': dict(Counter(row.get('true_class', '') for row in rows)),
        'skipped': dict(skipped),
    }


def render_markdown(result):
    lines = ['# TargetP external evaluation', '']
    lines.append('Model: `{}`'.format(result.get('model', '')))
    lines.append('')
    lines.append('| Dataset | Rows | Accuracy | Macro F1 | Observed macro F1 | Notes |')
    lines.append('| --- | ---: | ---: | ---: | ---: | --- |')
    for key, name in [
        ('deeploc_sorting', 'DeepLoc sorting signals'),
        ('deeploc_hpa_broad', 'DeepLoc HPA broad'),
        ('uniprot_holdout', 'UniProt CC holdout'),
    ]:
        item = result.get(key, {})
        metrics = item.get('metrics', {})
        if not metrics:
            continue
        notes = item.get('notes', '')
        lines.append('| {} | {} | {:.3f} | {:.3f} | {:.3f} | {} |'.format(
            name,
            int(metrics.get('n_rows', 0)),
            float(metrics.get('accuracy', 0.0)),
            float(metrics.get('macro_f1', 0.0)),
            float(metrics.get('macro_f1_observed_labels', 0.0)),
            notes,
        ))
    lines.append('')
    for key, name in [
        ('deeploc_sorting', 'DeepLoc sorting signals'),
        ('deeploc_hpa_broad', 'DeepLoc HPA broad'),
        ('uniprot_holdout', 'UniProt CC holdout'),
    ]:
        item = result.get(key, {})
        metrics = item.get('metrics', {})
        if not metrics:
            continue
        lines.append('## {}'.format(name))
        lines.append('')
        lines.append('| Class | Support | Precision | Recall | F1 |')
        lines.append('| --- | ---: | ---: | ---: | ---: |')
        for class_name in LOCALIZATION_CLASSES:
            cls = metrics['by_class'][class_name]
            lines.append('| {} | {} | {:.3f} | {:.3f} | {:.3f} |'.format(
                class_name,
                int(cls['support']),
                float(cls['precision']),
                float(cls['recall']),
                float(cls['f1']),
            ))
        lines.append('')
        calibration = item.get('threshold_calibration', {})
        if calibration:
            lines.append('### Class-threshold calibration')
            lines.append('')
            lines.append('| Evaluation | Accuracy | Macro F1 | Notes |')
            lines.append('| --- | ---: | ---: | --- |')
            argmax = calibration.get('argmax_metrics', {})
            oracle = calibration.get('oracle_metrics', {})
            cv = calibration.get('cv_metrics', {})
            if argmax:
                lines.append('| Argmax probabilities | {:.3f} | {:.3f} | Raw probability argmax from the model. |'.format(
                    float(argmax.get('overall_accuracy', 0.0)),
                    float(argmax.get('macro_f1', 0.0)),
                ))
            if oracle:
                lines.append('| Oracle thresholds | {:.3f} | {:.3f} | Diagnostic upper bound tuned on all rows; not a fair test estimate. |'.format(
                    float(oracle.get('overall_accuracy', 0.0)),
                    float(oracle.get('macro_f1', 0.0)),
                ))
            if cv:
                lines.append('| {}-fold thresholds | {:.3f} | {:.3f} | Thresholds selected on other folds only. |'.format(
                    int(calibration.get('cv_folds', 0)),
                    float(cv.get('overall_accuracy', 0.0)),
                    float(cv.get('macro_f1', 0.0)),
                ))
            lines.append('')
    return '\n'.join(lines)


def run_external_evaluation(
    model_path,
    targetp_tsv,
    deeploc_dir,
    uniprot_tsv,
    out_dir,
    max_uniprot_per_class=500,
    use_mmseqs=True,
    mmseqs_min_seq_id=0.30,
    mmseqs_min_coverage=0.80,
    seed=1,
    threads=1,
    threshold_calibration=True,
    threshold_cv_folds=5,
    threshold_grid=None,
    strict_targetp_organism_labels=False,
):
    os.makedirs(out_dir, exist_ok=True)
    targetp_keys = load_targetp_exclusion_keys(targetp_tsv=targetp_tsv)
    result = {
        'model': model_path,
        'targetp_tsv': targetp_tsv,
        'deeploc_dir': deeploc_dir,
        'uniprot_tsv': uniprot_tsv,
        'class_names': list(LOCALIZATION_CLASSES),
        'exact_overlap_filter': True,
        'strict_targetp_organism_labels': bool(strict_targetp_organism_labels),
    }

    sorting_rows, sorting_skipped = build_deeploc_sorting_rows(
        path=os.path.join(deeploc_dir, 'deeploc21_sorting_signals.tsv'),
        targetp_keys=targetp_keys,
        exclude_exact=True,
    )
    sorting_pred = predict_rows(rows=sorting_rows, model_path=model_path)
    sorting_path = os.path.join(out_dir, 'deeploc_sorting_predictions.tsv')
    write_tsv(
        path=sorting_path,
        rows=sorting_pred,
        fieldnames=[
            'source',
            'accession',
            'organism_group',
            'true_class',
            'predicted_class',
            'external_labels',
            'p_noTP',
            'p_SP',
            'p_mTP',
            'p_cTP',
            'p_lTP',
            'p_peroxisome',
            'perox_signal_type',
        ],
    )
    result['deeploc_sorting'] = {
        'dataset': os.path.join(deeploc_dir, 'deeploc21_sorting_signals.tsv'),
        'predictions_tsv': sorting_path,
        'candidate_report': _dataset_report(sorting_rows, sorting_skipped),
        'metrics': compute_single_label_metrics(sorting_pred),
        'notes': 'TargetP exact overlaps removed; only labels with TargetP equivalents are scored.',
    }

    hpa_rows, hpa_skipped = build_deeploc_hpa_broad_rows(
        path=os.path.join(deeploc_dir, 'deeploc21_hpa_test.tsv'),
        targetp_keys=targetp_keys,
        exclude_exact=True,
    )
    hpa_pred = predict_rows(rows=hpa_rows, model_path=model_path)
    hpa_path = os.path.join(out_dir, 'deeploc_hpa_broad_predictions.tsv')
    write_tsv(
        path=hpa_path,
        rows=hpa_pred,
        fieldnames=[
            'source',
            'accession',
            'organism_group',
            'true_class',
            'predicted_class',
            'external_labels',
            'p_noTP',
            'p_SP',
            'p_mTP',
            'p_cTP',
            'p_lTP',
            'p_peroxisome',
            'perox_signal_type',
        ],
    )
    result['deeploc_hpa_broad'] = {
        'dataset': os.path.join(deeploc_dir, 'deeploc21_hpa_test.tsv'),
        'predictions_tsv': hpa_path,
        'candidate_report': _dataset_report(hpa_rows, hpa_skipped),
        'metrics': compute_single_label_metrics(hpa_pred),
        'notes': 'Broad mature-localization proxy: mitochondrion/chloroplast/extracellular map to mTP/cTP/SP, others to noTP.',
    }

    uniprot_rows, uniprot_skipped = build_uniprot_holdout_rows(
        path=uniprot_tsv,
        targetp_keys=targetp_keys,
        exclude_exact=True,
        skip_ambiguous=True,
        strict_targetp_organism_labels=bool(strict_targetp_organism_labels),
    )
    sampled = stratified_sample_rows(
        rows=uniprot_rows,
        max_per_class=int(max_uniprot_per_class),
        seed=int(seed),
    )
    filtered, mmseqs_report = filter_rows_by_mmseqs_similarity(
        rows=sampled,
        targetp_rows=targetp_keys['rows'],
        min_seq_id=float(mmseqs_min_seq_id),
        min_coverage=float(mmseqs_min_coverage),
        threads=int(threads),
        enabled=bool(use_mmseqs),
    )
    holdout_path = os.path.join(out_dir, 'uniprot_targetp_holdout.tsv')
    write_tsv(
        path=holdout_path,
        rows=filtered,
        fieldnames=[
            'source',
            'accession',
            'organism_group',
            'true_class',
            'external_labels',
            'sequence',
        ],
    )
    uniprot_pred = predict_rows(rows=filtered, model_path=model_path)
    uniprot_pred_path = os.path.join(out_dir, 'uniprot_targetp_holdout_predictions.tsv')
    write_tsv(
        path=uniprot_pred_path,
        rows=uniprot_pred,
        fieldnames=[
            'source',
            'accession',
            'organism_group',
            'true_class',
            'predicted_class',
            'external_labels',
            'p_noTP',
            'p_SP',
            'p_mTP',
            'p_cTP',
            'p_lTP',
            'p_peroxisome',
            'perox_signal_type',
        ],
    )
    result['uniprot_holdout'] = {
        'dataset': uniprot_tsv,
        'holdout_tsv': holdout_path,
        'predictions_tsv': uniprot_pred_path,
        'candidate_report': _dataset_report(uniprot_rows, uniprot_skipped),
        'sampled_rows': int(len(sampled)),
        'mmseqs_similarity_filter': mmseqs_report,
        'metrics': compute_single_label_metrics(uniprot_pred),
        'notes': (
            'Weak labels from cdskit UniProt CC rules; TargetP exact overlaps '
            'and MMseqs similarities removed before scoring.'
            + (
                ' cTP/lTP rows outside plant organism groups are also removed.'
                if bool(strict_targetp_organism_labels) else ''
            )
        ),
    }
    if bool(threshold_calibration):
        result['uniprot_holdout']['threshold_calibration'] = evaluate_prediction_threshold_calibration(
            rows=uniprot_pred,
            threshold_grid=threshold_grid,
            cv_folds=int(threshold_cv_folds),
            seed=int(seed),
            class_names=LOCALIZATION_CLASSES,
        )

    out_json = os.path.join(out_dir, 'targetp_external_eval.json')
    out_md = os.path.join(out_dir, 'targetp_external_eval.md')
    result['out_json'] = out_json
    result['out_md'] = out_md
    with open(out_json, 'w', encoding='utf-8') as out:
        json.dump(result, out, indent=2, sort_keys=True)
    with open(out_md, 'w', encoding='utf-8') as out:
        out.write(render_markdown(result=result))
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate a TargetP-trained cdskit localize model on external non-overlapping data.'
    )
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--targetp_tsv', default='data/localize_bench/targetp2_benchmark.tsv', type=str)
    parser.add_argument('--deeploc_dir', default='data/localize_bench/deeploc21', type=str)
    parser.add_argument('--uniprot_tsv', default='data/localize_bench/eukaryota_full_with_lineage.tsv', type=str)
    parser.add_argument('--out_dir', default='data/localize_bench/targetp_external_eval', type=str)
    parser.add_argument('--max_uniprot_per_class', default=500, type=int)
    parser.add_argument('--mmseqs', default='yes', choices=['yes', 'no'], type=str)
    parser.add_argument('--mmseqs_min_seq_id', default=0.30, type=float)
    parser.add_argument('--mmseqs_min_coverage', default=0.80, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--threads', default=1, type=int)
    parser.add_argument('--threshold_calibration', default='yes', choices=['yes', 'no'], type=str)
    parser.add_argument('--threshold_cv_folds', default=5, type=int)
    parser.add_argument(
        '--threshold_grid',
        default=','.join(str(value) for value in DEFAULT_CLASS_THRESHOLD_GRID),
        type=str,
    )
    parser.add_argument('--strict_targetp_organism_labels', default='no', choices=['yes', 'no'], type=str)
    return parser


def _to_bool_yes_no(value):
    return str(value).strip().lower() in ['yes', 'y', 'true', '1']


def _parse_threshold_grid(value):
    out = [
        float(part.strip())
        for part in str(value).split(',')
        if part.strip() != ''
    ]
    if len(out) == 0:
        raise ValueError('--threshold_grid should contain at least one value.')
    return sorted(set(out))


def main():
    args = build_parser().parse_args()
    result = run_external_evaluation(
        model_path=args.model,
        targetp_tsv=args.targetp_tsv,
        deeploc_dir=args.deeploc_dir,
        uniprot_tsv=args.uniprot_tsv,
        out_dir=args.out_dir,
        max_uniprot_per_class=int(args.max_uniprot_per_class),
        use_mmseqs=_to_bool_yes_no(args.mmseqs),
        mmseqs_min_seq_id=float(args.mmseqs_min_seq_id),
        mmseqs_min_coverage=float(args.mmseqs_min_coverage),
        seed=int(args.seed),
        threads=int(args.threads),
        threshold_calibration=_to_bool_yes_no(args.threshold_calibration),
        threshold_cv_folds=int(args.threshold_cv_folds),
        threshold_grid=_parse_threshold_grid(args.threshold_grid),
        strict_targetp_organism_labels=_to_bool_yes_no(args.strict_targetp_organism_labels),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

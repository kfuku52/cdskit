import argparse
import csv
import json
import os
from urllib import request as urllib_request

import numpy as np

from cdskit.localize_model import (
    BROAD_FEATURE_NAMES,
    extract_broad_localize_features,
    fit_multilabel_centroid_classifier,
    predict_multilabel_centroid_matrix,
    save_localize_model,
)
from cdskit.localize_multilabel_cnn import (
    fit_multilabel_cnn_classifier,
    predict_multilabel_cnn_batch,
)


DEEPLOC21_URLS = {
    'localization_train_validation': (
        'https://services.healthtech.dtu.dk/services/DeepLoc-2.1/data/'
        'Swissprot_Train_Validation_dataset.csv'
    ),
    'membrane_train_validation': (
        'https://services.healthtech.dtu.dk/services/DeepLoc-2.1/data/'
        'Swissprot_Membrane_Train_Validation_dataset.csv'
    ),
    'hpa_test': (
        'https://services.healthtech.dtu.dk/services/DeepLoc-2.1/data/'
        'hpa_testset.csv'
    ),
    'sorting_signals': (
        'https://services.healthtech.dtu.dk/services/DeepLoc-2.1/data/'
        'SortingSignalsSwissprot.csv'
    ),
}

DEEPLOC_LOCALIZATION_LABELS = (
    'nucleus',
    'cytoplasm',
    'extracellular',
    'mitochondrion',
    'cell_membrane',
    'endoplasmic_reticulum',
    'chloroplast',
    'golgi_apparatus',
    'lysosome_vacuole',
    'peroxisome',
)

DEEPLOC_MEMBRANE_LABELS = (
    'peripheral',
    'transmembrane',
    'lipid_anchor',
    'soluble',
)

DEEPLOC_SORTING_SIGNAL_LABELS = (
    'SP',
    'MT',
    'CH',
    'TH',
    'GPI',
    'NLS',
    'NES',
    'PTS',
    'TM',
)

DEEPLOC20_LOCALIZATION_REFERENCE = {
    'source': (
        'DeepLoc 2.0: multi-label subcellular localization prediction using '
        'protein language models, Nucleic Acids Research 2022, '
        'doi:10.1093/nar/gkac278'
    ),
    'swissprot_cv': {
        'count': 28303,
        'deeploc20_esm1b': {
            'accuracy': 0.53,
            'jaccard': 0.68,
            'micro_f1': 0.72,
            'macro_f1': 0.64,
        },
        'deeploc20_prott5': {
            'accuracy': 0.55,
            'jaccard': 0.69,
            'micro_f1': 0.73,
            'macro_f1': 0.66,
        },
    },
    'hpa_independent': {
        'count': 1717,
        'yloc_plus': {'accuracy': 0.23, 'jaccard': 0.41, 'micro_f1': 0.51, 'macro_f1': 0.34},
        'deeploc10': {'accuracy': 0.37, 'jaccard': 0.42, 'micro_f1': 0.46, 'macro_f1': 0.35},
        'fuel_mloc': {'accuracy': 0.38, 'jaccard': 0.46, 'micro_f1': 0.52, 'macro_f1': 0.39},
        'laprott5': {'accuracy': 0.45, 'jaccard': 0.52, 'micro_f1': 0.56, 'macro_f1': 0.43},
        'deeploc20_esm1b': {'accuracy': 0.34, 'jaccard': 0.48, 'micro_f1': 0.57, 'macro_f1': 0.44},
        'deeploc20_prott5': {'accuracy': 0.39, 'jaccard': 0.53, 'micro_f1': 0.60, 'macro_f1': 0.46},
    },
}

DEEPLOC21_MEMBRANE_REFERENCE = {
    'source': (
        'DeepLoc 2.1: multi-label membrane protein type prediction using '
        'protein language models, Nucleic Acids Research 2024, '
        'doi:10.1093/nar/gkae237'
    ),
    'heldout_test': {
        'count': 4933,
        'deeploc21_esm1b': {
            'accuracy': 0.87,
            'subset_accuracy': 0.79,
            'jaccard': 0.83,
            'micro_f1': 0.88,
            'macro_f1': 0.74,
        },
        'deeploc21_prott5': {
            'accuracy': 0.88,
            'subset_accuracy': 0.80,
            'jaccard': 0.84,
            'micro_f1': 0.89,
            'macro_f1': 0.75,
        },
    },
}

LOCALIZATION_INPUT_COLUMNS = {
    'nucleus': ('Nucleus',),
    'cytoplasm': ('Cytoplasm',),
    'extracellular': ('Extracellular',),
    'mitochondrion': ('Mitochondrion',),
    'cell_membrane': ('Cell membrane',),
    'endoplasmic_reticulum': ('Endoplasmic reticulum',),
    'chloroplast': ('Chloroplast', 'Plastid'),
    'golgi_apparatus': ('Golgi apparatus',),
    'lysosome_vacuole': ('Lysosome/Vacuole',),
    'peroxisome': ('Peroxisome',),
}

MEMBRANE_INPUT_COLUMNS = {
    'peripheral': ('Peripheral',),
    'transmembrane': ('Transmembrane',),
    'lipid_anchor': ('LipidAnchor', 'Lipid anchor'),
    'soluble': ('Soluble',),
}


def _to_bool01(value):
    text = str(value or '').strip().lower()
    if text in ['1', '1.0', 'true', 'yes', 'y']:
        return 1
    return 0


def _read_csv_rows(path):
    with open(path, 'r', encoding='utf-8', newline='') as inp:
        return list(csv.DictReader(inp))


def _write_tsv_rows(path, rows, fieldnames):
    out_dir = os.path.dirname(path)
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(
            out,
            fieldnames=fieldnames,
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _collect_labels(row, labels, input_columns):
    out = list()
    for label in labels:
        columns = input_columns[label]
        active = 0
        for column in columns:
            if column in row:
                active = max(active, _to_bool01(row.get(column, '0')))
        if active:
            out.append(label)
    return out


def _label_columns_to_row(label_values, labels):
    values = set(label_values)
    return {label: int(label in values) for label in labels}


def _summarize_label_rows(rows, labels, label_list_col):
    counts = {label: 0 for label in labels}
    multi_label = 0
    for row in rows:
        active = [v for v in str(row[label_list_col]).split(';') if v != '']
        if len(active) > 1:
            multi_label += 1
        for label in active:
            if label in counts:
                counts[label] += 1
    return {
        'n_rows': int(len(rows)),
        'n_multi_label_rows': int(multi_label),
        'label_counts': counts,
    }


def download_deeploc21_data(out_dir, names=None, timeout_sec=120):
    if names is None:
        names = sorted(DEEPLOC21_URLS.keys())
    os.makedirs(out_dir, exist_ok=True)
    out = dict()
    for name in names:
        if name not in DEEPLOC21_URLS:
            raise ValueError('Unsupported DeepLoc 2.1 dataset name: {}'.format(name))
        url = DEEPLOC21_URLS[name]
        filename = os.path.basename(url)
        out_path = os.path.join(out_dir, filename)
        with urllib_request.urlopen(url, timeout=int(timeout_sec)) as resp:
            body = resp.read()
        with open(out_path, 'wb') as out_file:
            out_file.write(body)
        out[name] = {
            'url': url,
            'path': out_path,
            'bytes': int(len(body)),
        }
    return out


def prepare_deeploc21_localization_tsv(csv_path, out_tsv_path, source_name='swissprot'):
    rows = _read_csv_rows(path=csv_path)
    out_rows = list()
    for row in rows:
        labels = _collect_labels(
            row=row,
            labels=DEEPLOC_LOCALIZATION_LABELS,
            input_columns=LOCALIZATION_INPUT_COLUMNS,
        )
        out_row = {
            'source': source_name,
            'accession': str(row.get('ACC', row.get('sid', ''))).strip(),
            'kingdom': str(row.get('Kingdom', '')).strip(),
            'partition': str(row.get('Partition', '')).strip(),
            'sequence': str(row.get('Sequence', row.get('fasta', ''))).strip(),
            'localization_labels': ';'.join(labels),
        }
        out_row.update(_label_columns_to_row(labels, DEEPLOC_LOCALIZATION_LABELS))
        out_rows.append(out_row)
    fieldnames = [
        'source',
        'accession',
        'kingdom',
        'partition',
        'sequence',
        'localization_labels',
    ] + list(DEEPLOC_LOCALIZATION_LABELS)
    _write_tsv_rows(path=out_tsv_path, rows=out_rows, fieldnames=fieldnames)
    return _summarize_label_rows(
        rows=out_rows,
        labels=DEEPLOC_LOCALIZATION_LABELS,
        label_list_col='localization_labels',
    )


def prepare_deeploc21_hpa_tsv(csv_path, out_tsv_path):
    rows = _read_csv_rows(path=csv_path)
    out_rows = list()
    for row in rows:
        labels = _collect_labels(
            row=row,
            labels=DEEPLOC_LOCALIZATION_LABELS,
            input_columns=LOCALIZATION_INPUT_COLUMNS,
        )
        out_row = {
            'source': 'hpa',
            'accession': str(row.get('sid', '')).strip(),
            'kingdom': 'Metazoa',
            'partition': 'test',
            'sequence': str(row.get('fasta', row.get('Sequence', ''))).strip(),
            'localization_labels': ';'.join(labels),
        }
        out_row.update(_label_columns_to_row(labels, DEEPLOC_LOCALIZATION_LABELS))
        out_rows.append(out_row)
    fieldnames = [
        'source',
        'accession',
        'kingdom',
        'partition',
        'sequence',
        'localization_labels',
    ] + list(DEEPLOC_LOCALIZATION_LABELS)
    _write_tsv_rows(path=out_tsv_path, rows=out_rows, fieldnames=fieldnames)
    return _summarize_label_rows(
        rows=out_rows,
        labels=DEEPLOC_LOCALIZATION_LABELS,
        label_list_col='localization_labels',
    )


def prepare_deeploc21_membrane_tsv(csv_path, out_tsv_path):
    rows = _read_csv_rows(path=csv_path)
    out_rows = list()
    for row in rows:
        labels = _collect_labels(
            row=row,
            labels=DEEPLOC_MEMBRANE_LABELS,
            input_columns=MEMBRANE_INPUT_COLUMNS,
        )
        out_row = {
            'source': 'swissprot_membrane',
            'accession': str(row.get('ACC', '')).strip(),
            'kingdom': str(row.get('Kingdom', '')).strip(),
            'partition': str(row.get('Partition', '')).strip(),
            'sequence': str(row.get('Sequence', '')).strip(),
            'membrane_labels': ';'.join(labels),
        }
        out_row.update(_label_columns_to_row(labels, DEEPLOC_MEMBRANE_LABELS))
        out_rows.append(out_row)
    fieldnames = [
        'source',
        'accession',
        'kingdom',
        'partition',
        'sequence',
        'membrane_labels',
    ] + list(DEEPLOC_MEMBRANE_LABELS)
    _write_tsv_rows(path=out_tsv_path, rows=out_rows, fieldnames=fieldnames)
    return _summarize_label_rows(
        rows=out_rows,
        labels=DEEPLOC_MEMBRANE_LABELS,
        label_list_col='membrane_labels',
    )


def prepare_deeploc21_sorting_signal_tsv(csv_path, out_tsv_path):
    rows = _read_csv_rows(path=csv_path)
    out_rows = list()
    for row in rows:
        raw_types = [
            value.strip() for value in str(row.get('Types', '')).split('_')
            if value.strip() != ''
        ]
        labels = [value for value in raw_types if value in DEEPLOC_SORTING_SIGNAL_LABELS]
        out_row = {
            'source': 'swissprot_sorting_signals',
            'accession': str(row.get('ACC', '')).strip(),
            'kingdom': str(row.get('Kingdom', '')).strip(),
            'sequence': str(row.get('Sequence', '')).strip(),
            'annotation_encoded': str(row.get('AnnotEncoded', '')).strip(),
            'sorting_signal_labels': ';'.join(labels),
            'sorting_signal_types_raw': str(row.get('Types', '')).strip(),
        }
        out_row.update(_label_columns_to_row(labels, DEEPLOC_SORTING_SIGNAL_LABELS))
        out_rows.append(out_row)
    fieldnames = [
        'source',
        'accession',
        'kingdom',
        'sequence',
        'annotation_encoded',
        'sorting_signal_labels',
        'sorting_signal_types_raw',
    ] + list(DEEPLOC_SORTING_SIGNAL_LABELS)
    _write_tsv_rows(path=out_tsv_path, rows=out_rows, fieldnames=fieldnames)
    return _summarize_label_rows(
        rows=out_rows,
        labels=DEEPLOC_SORTING_SIGNAL_LABELS,
        label_list_col='sorting_signal_labels',
    )


def prepare_all_deeploc21(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    inputs = {
        'localization_train_validation': os.path.join(
            data_dir,
            'Swissprot_Train_Validation_dataset.csv',
        ),
        'membrane_train_validation': os.path.join(
            data_dir,
            'Swissprot_Membrane_Train_Validation_dataset.csv',
        ),
        'hpa_test': os.path.join(data_dir, 'hpa_testset.csv'),
        'sorting_signals': os.path.join(data_dir, 'SortingSignalsSwissprot.csv'),
    }
    report = dict()
    report['localization_train_validation'] = prepare_deeploc21_localization_tsv(
        csv_path=inputs['localization_train_validation'],
        out_tsv_path=os.path.join(out_dir, 'deeploc21_localization_train_validation.tsv'),
        source_name='swissprot',
    )
    report['membrane_train_validation'] = prepare_deeploc21_membrane_tsv(
        csv_path=inputs['membrane_train_validation'],
        out_tsv_path=os.path.join(out_dir, 'deeploc21_membrane_train_validation.tsv'),
    )
    report['hpa_test'] = prepare_deeploc21_hpa_tsv(
        csv_path=inputs['hpa_test'],
        out_tsv_path=os.path.join(out_dir, 'deeploc21_hpa_test.tsv'),
    )
    report['sorting_signals'] = prepare_deeploc21_sorting_signal_tsv(
        csv_path=inputs['sorting_signals'],
        out_tsv_path=os.path.join(out_dir, 'deeploc21_sorting_signals.tsv'),
    )
    report['data_dir'] = data_dir
    report['out_dir'] = out_dir
    return report


def _read_prepared_tsv(path):
    with open(path, 'r', encoding='utf-8', newline='') as inp:
        return list(csv.DictReader(inp, delimiter='\t'))


def _active_labels_from_row(row, labels, label_col):
    label_text = str(row.get(label_col, '')).strip()
    if label_text != '':
        return [value for value in label_text.split(';') if value in labels]
    out = list()
    for label in labels:
        if _to_bool01(row.get(label, '0')):
            out.append(label)
    return out


def _filter_multilabel_rows(rows, labels, label_col):
    out = list()
    for row in rows:
        seq = str(row.get('sequence', '')).strip()
        if seq == '':
            continue
        active = _active_labels_from_row(
            row=row,
            labels=labels,
            label_col=label_col,
        )
        if len(active) == 0:
            continue
        out.append(row)
    return out


def build_deeploc_feature_matrix(rows):
    features = list()
    for row in rows:
        feats, _ = extract_broad_localize_features(
            aa_seq=row.get('sequence', ''),
            kingdom=row.get('kingdom', ''),
        )
        features.append(feats)
    if len(features) == 0:
        return np.zeros((0, len(BROAD_FEATURE_NAMES)), dtype=np.float64)
    return np.asarray(features, dtype=np.float64)


def build_label_matrix(rows, labels, label_col):
    labels = list(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    mat = np.zeros((len(rows), len(labels)), dtype=np.int64)
    for row_i, row in enumerate(rows):
        active = _active_labels_from_row(
            row=row,
            labels=labels,
            label_col=label_col,
        )
        for label in active:
            mat[row_i, label_to_idx[label]] = 1
    return mat


def _normalize_model_arch(model_arch):
    value = str(model_arch or 'centroid').strip().lower()
    if value in ['centroid', 'multilabel_centroid', 'multilabel_centroid_v1']:
        return 'centroid'
    if value in ['cnn', 'multilabel_cnn', 'multilabel_cnn_v1']:
        return 'cnn'
    raise ValueError('Unsupported --model_arch: {}'.format(model_arch))


def _model_type_from_arch(model_arch):
    arch = _normalize_model_arch(model_arch=model_arch)
    if arch == 'cnn':
        return 'multilabel_cnn_v1'
    return 'multilabel_centroid_v1'


def _deeploc_cnn_params(dl_params=None):
    params = dict(dl_params or {})
    return {
        'seq_len': int(params.get('seq_len', 512)),
        'embed_dim': int(params.get('embed_dim', 32)),
        'num_filters': int(params.get('num_filters', 64)),
        'kernel_sizes': params.get('kernel_sizes', '3,5,9,15'),
        'dropout': float(params.get('dropout', 0.25)),
        'epochs': int(params.get('epochs', 6)),
        'batch_size': int(params.get('batch_size', 256)),
        'learning_rate': float(params.get('learning_rate', 1.0e-3)),
        'weight_decay': float(params.get('weight_decay', 1.0e-4)),
        'seed': int(params.get('seed', 1)),
        'class_weight': _to_bool_yes_no(params.get('class_weight', 'yes')),
        'feature_fusion': _to_bool_yes_no(params.get('feature_fusion', 'no')),
        'device': str(params.get('device', 'auto')),
        'threshold_objective': str(params.get('threshold_objective', 'f1')),
    }


def _safe_div(num, denom):
    if denom <= 0:
        return 0.0
    return float(num) / float(denom)


def compute_multilabel_metrics(y_true, y_pred, labels):
    labels = list(labels)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError('Shape mismatch between true and predicted labels.')
    if y_true.ndim != 2:
        raise ValueError('Multilabel matrices should be 2D.')

    by_label = dict()
    macro_f1_values = list()
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    for label_i, label in enumerate(labels):
        true_col = y_true[:, label_i]
        pred_col = y_pred[:, label_i]
        tp = int(np.sum((true_col == 1) & (pred_col == 1)))
        fp = int(np.sum((true_col == 0) & (pred_col == 1)))
        fn = int(np.sum((true_col == 1) & (pred_col == 0)))
        tn = int(np.sum((true_col == 0) & (pred_col == 0)))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        f1 = 0.0 if precision + recall <= 0.0 else (
            (2.0 * precision * recall) / (precision + recall)
        )
        denom_mcc = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denom_mcc <= 0.0:
            mcc = 0.0
        else:
            mcc = float(((tp * tn) - (fp * fn)) / np.sqrt(denom_mcc))
        by_label[label] = {
            'support': int(np.sum(true_col == 1)),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'precision': float(precision),
            'recall': float(recall),
            'sensitivity': float(recall),
            'specificity': float(specificity),
            'f1': float(f1),
            'mcc': float(mcc),
        }
        macro_f1_values.append(float(f1))
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    micro_precision = _safe_div(total_tp, total_tp + total_fp)
    micro_recall = _safe_div(total_tp, total_tp + total_fn)
    micro_f1 = 0.0
    if micro_precision + micro_recall > 0.0:
        micro_f1 = (2.0 * micro_precision * micro_recall) / (micro_precision + micro_recall)

    exact_match = np.all(y_true == y_pred, axis=1)
    intersections = np.sum((y_true == 1) & (y_pred == 1), axis=1).astype(np.float64)
    unions = np.sum((y_true == 1) | (y_pred == 1), axis=1).astype(np.float64)
    jaccard = np.ones_like(unions, dtype=np.float64)
    np.divide(intersections, unions, out=jaccard, where=(unions > 0.0))
    row_true = np.sum(y_true == 1, axis=1).astype(np.float64)
    row_pred = np.sum(y_pred == 1, axis=1).astype(np.float64)
    sample_f1 = np.ones_like(row_true, dtype=np.float64)
    sample_denom = row_true + row_pred
    np.divide(
        2.0 * intersections,
        sample_denom,
        out=sample_f1,
        where=(sample_denom > 0.0),
    )
    true_label_count = int(np.sum(y_true == 1))
    predicted_label_count = int(np.sum(y_pred == 1))
    observed_f1_values = [
        float(by_label[label]['f1'])
        for label in labels
        if int(by_label[label]['support']) > 0
    ]
    if len(observed_f1_values) == 0:
        observed_macro_f1 = float(np.mean(np.asarray(macro_f1_values, dtype=np.float64)))
    else:
        observed_macro_f1 = float(np.mean(np.asarray(observed_f1_values, dtype=np.float64)))

    return {
        'n_rows': int(y_true.shape[0]),
        'n_labels': int(y_true.shape[1]),
        'subset_accuracy': float(np.mean(exact_match)) if y_true.shape[0] > 0 else 0.0,
        'accuracy': float(np.mean(sample_f1)) if y_true.shape[0] > 0 else 0.0,
        'sample_f1': float(np.mean(sample_f1)) if y_true.shape[0] > 0 else 0.0,
        'jaccard': float(np.mean(jaccard)) if y_true.shape[0] > 0 else 0.0,
        'micro_precision': float(micro_precision),
        'micro_recall': float(micro_recall),
        'micro_f1': float(micro_f1),
        'macro_f1': float(np.mean(np.asarray(macro_f1_values, dtype=np.float64))),
        'macro_f1_observed_labels': float(observed_macro_f1),
        'hamming_loss': float(np.mean(y_true != y_pred)) if y_true.size > 0 else 0.0,
        'predicted_label_count': int(predicted_label_count),
        'true_label_count': int(true_label_count),
        'predicted_per_true': _safe_div(predicted_label_count, true_label_count),
        'by_label': by_label,
    }


def _fold_ids_from_rows(rows, fold_col='partition', n_folds=5, seed=1):
    values = [str(row.get(fold_col, '')).strip() for row in rows]
    if all(value != '' for value in values):
        return np.asarray(values)
    n_rows = len(rows)
    n_folds = max(2, int(n_folds))
    rng = np.random.default_rng(int(seed))
    order = np.arange(n_rows, dtype=np.int64)
    rng.shuffle(order)
    folds = np.empty((n_rows,), dtype=object)
    for pos, row_idx in enumerate(order.tolist()):
        folds[int(row_idx)] = 'fold{}'.format(int(pos % n_folds) + 1)
    return folds


def fit_deeploc_multilabel_model(
    rows,
    labels,
    label_col,
    task_name,
    model_arch='centroid',
    dl_params=None,
):
    rows = _filter_multilabel_rows(
        rows=rows,
        labels=labels,
        label_col=label_col,
    )
    x = build_deeploc_feature_matrix(rows=rows)
    y = build_label_matrix(
        rows=rows,
        labels=labels,
        label_col=label_col,
    )
    arch = _normalize_model_arch(model_arch=model_arch)
    cnn_params = None
    if arch == 'cnn':
        cnn_params = _deeploc_cnn_params(dl_params=dl_params)
        feature_matrix = x if bool(cnn_params['feature_fusion']) else None
        localization_model = fit_multilabel_cnn_classifier(
            aa_sequences=[row.get('sequence', '') for row in rows],
            label_matrix=y,
            class_order=labels,
            seq_len=cnn_params['seq_len'],
            embed_dim=cnn_params['embed_dim'],
            num_filters=cnn_params['num_filters'],
            kernel_sizes=cnn_params['kernel_sizes'],
            dropout=cnn_params['dropout'],
            epochs=cnn_params['epochs'],
            batch_size=cnn_params['batch_size'],
            learning_rate=cnn_params['learning_rate'],
            weight_decay=cnn_params['weight_decay'],
            seed=cnn_params['seed'],
            use_class_weight=cnn_params['class_weight'],
            device=cnn_params['device'],
            feature_matrix=feature_matrix,
            threshold_objective=cnn_params['threshold_objective'],
            ensure_one_label=True,
        )
    else:
        localization_model = fit_multilabel_centroid_classifier(
            features=x,
            label_matrix=y,
            class_order=labels,
            ensure_one_label=True,
        )
    localization_model['task'] = str(task_name)
    localization_model['feature_names'] = list(BROAD_FEATURE_NAMES)
    model_type = _model_type_from_arch(model_arch=arch)
    metadata = {
        'task': str(task_name),
        'num_training_rows': int(len(rows)),
        'class_counts': {
            labels[i]: int(np.sum(y[:, i] == 1)) for i in range(len(labels))
        },
        'model_arch': 'multilabel_cnn' if arch == 'cnn' else 'multilabel_centroid',
        'seqtype': 'protein',
    }
    if cnn_params is not None:
        metadata['cnn_params'] = dict(cnn_params)
    return {
        'model_type': model_type,
        'feature_names': list(BROAD_FEATURE_NAMES),
        'localization_model': localization_model,
        'perox_model': {'mode': 'embedded_multilabel'},
        'metadata': metadata,
    }


def _predict_model_on_rows(model, rows):
    model_type = str(model.get('model_type', ''))
    if model_type == 'multilabel_cnn_v1':
        localization_model = model['localization_model']
        feature_matrix = None
        if int(localization_model.get('feature_dim', 0)) > 0:
            feature_matrix = build_deeploc_feature_matrix(rows=rows)
        batch_size = int(
            model.get('metadata', {})
            .get('cnn_params', {})
            .get('batch_size', 512)
        )
        return predict_multilabel_cnn_batch(
            aa_sequences=[row.get('sequence', '') for row in rows],
            localization_model=localization_model,
            device='cpu',
            batch_size=batch_size,
            feature_matrix=feature_matrix,
            apply_thresholds=True,
        )
    x = build_deeploc_feature_matrix(rows=rows)
    return predict_multilabel_centroid_matrix(
        features=x,
        localization_model=model['localization_model'],
        apply_thresholds=True,
    )


def evaluate_deeploc21_task_cv(
    tsv_path,
    labels,
    label_col,
    task_name='localization',
    fold_col='partition',
    n_folds=5,
    seed=1,
    model_arch='centroid',
    dl_params=None,
):
    rows = _filter_multilabel_rows(
        rows=_read_prepared_tsv(path=tsv_path),
        labels=labels,
        label_col=label_col,
    )
    fold_ids = _fold_ids_from_rows(
        rows=rows,
        fold_col=fold_col,
        n_folds=n_folds,
        seed=seed,
    )
    y_true = build_label_matrix(
        rows=rows,
        labels=labels,
        label_col=label_col,
    )
    y_pred = np.zeros_like(y_true, dtype=np.int64)
    prob = np.zeros_like(y_true, dtype=np.float64)
    fold_rows = list()
    for fold_id in sorted(set([str(value) for value in fold_ids.tolist()])):
        test_mask = np.asarray([str(value) == fold_id for value in fold_ids.tolist()], dtype=bool)
        train_rows = [rows[i] for i in range(len(rows)) if not bool(test_mask[i])]
        test_rows = [rows[i] for i in range(len(rows)) if bool(test_mask[i])]
        if len(train_rows) == 0 or len(test_rows) == 0:
            continue
        model = fit_deeploc_multilabel_model(
            rows=train_rows,
            labels=labels,
            label_col=label_col,
            task_name=task_name,
            model_arch=model_arch,
            dl_params=dl_params,
        )
        pred = _predict_model_on_rows(model=model, rows=test_rows)
        test_indices = np.where(test_mask)[0]
        y_pred[test_indices, :] = pred['prediction_matrix']
        prob[test_indices, :] = pred['prob_matrix']
        fold_metrics = compute_multilabel_metrics(
            y_true=y_true[test_indices, :],
            y_pred=pred['prediction_matrix'],
            labels=labels,
        )
        fold_rows.append({
            'fold_id': str(fold_id),
            'n_train': int(len(train_rows)),
            'n_test': int(len(test_rows)),
            'macro_f1': float(fold_metrics['macro_f1']),
            'micro_f1': float(fold_metrics['micro_f1']),
            'jaccard': float(fold_metrics['jaccard']),
            'subset_accuracy': float(fold_metrics['subset_accuracy']),
        })
    metrics = compute_multilabel_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
    )
    metrics['folds'] = fold_rows
    metrics['task'] = str(task_name)
    metrics['tsv_path'] = tsv_path
    metrics['fold_col'] = fold_col
    metrics['model'] = _model_type_from_arch(model_arch=model_arch)
    metrics['n_input_rows'] = int(len(rows))
    metrics['mean_probability_by_label'] = {
        labels[i]: float(np.mean(prob[:, i])) for i in range(len(labels))
    }
    return metrics


def evaluate_deeploc21_train_test(
    train_tsv_path,
    test_tsv_path,
    labels,
    label_col,
    task_name='localization',
    model_arch='centroid',
    dl_params=None,
):
    train_rows = _filter_multilabel_rows(
        rows=_read_prepared_tsv(path=train_tsv_path),
        labels=labels,
        label_col=label_col,
    )
    test_rows = _filter_multilabel_rows(
        rows=_read_prepared_tsv(path=test_tsv_path),
        labels=labels,
        label_col=label_col,
    )
    model = fit_deeploc_multilabel_model(
        rows=train_rows,
        labels=labels,
        label_col=label_col,
        task_name=task_name,
        model_arch=model_arch,
        dl_params=dl_params,
    )
    pred = _predict_model_on_rows(model=model, rows=test_rows)
    y_true = build_label_matrix(
        rows=test_rows,
        labels=labels,
        label_col=label_col,
    )
    metrics = compute_multilabel_metrics(
        y_true=y_true,
        y_pred=pred['prediction_matrix'],
        labels=labels,
    )
    metrics['task'] = str(task_name)
    metrics['train_tsv_path'] = train_tsv_path
    metrics['test_tsv_path'] = test_tsv_path
    metrics['model'] = _model_type_from_arch(model_arch=model_arch)
    metrics['n_train_rows'] = int(len(train_rows))
    metrics['n_test_rows'] = int(len(test_rows))
    return metrics


def _task_config(task_name, prepared_dir):
    task = str(task_name or 'localization').strip().lower()
    if task == 'localization':
        return {
            'task': 'localization',
            'labels': list(DEEPLOC_LOCALIZATION_LABELS),
            'label_col': 'localization_labels',
            'train_tsv': os.path.join(prepared_dir, 'deeploc21_localization_train_validation.tsv'),
            'test_tsv': os.path.join(prepared_dir, 'deeploc21_hpa_test.tsv'),
            'reference': DEEPLOC20_LOCALIZATION_REFERENCE,
        }
    if task == 'membrane':
        return {
            'task': 'membrane',
            'labels': list(DEEPLOC_MEMBRANE_LABELS),
            'label_col': 'membrane_labels',
            'train_tsv': os.path.join(prepared_dir, 'deeploc21_membrane_train_validation.tsv'),
            'test_tsv': '',
            'reference': DEEPLOC21_MEMBRANE_REFERENCE,
        }
    if task in ['sorting_signal', 'sorting_signals']:
        return {
            'task': 'sorting_signals',
            'labels': list(DEEPLOC_SORTING_SIGNAL_LABELS),
            'label_col': 'sorting_signal_labels',
            'train_tsv': os.path.join(prepared_dir, 'deeploc21_sorting_signals.tsv'),
            'test_tsv': '',
            'reference': {},
        }
    raise ValueError('Unsupported --task: {}'.format(task_name))


def render_deeploc_benchmark_markdown(result):
    lines = list()
    task = result.get('task', '')
    lines.append('# DeepLoc benchmark: {}'.format(task))
    lines.append('')
    lines.append('| Split | Rows | Subset acc. | Jaccard | Micro F1 | Macro F1 | Macro F1 observed | Pred/true |')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for key, name in [('cross_validation', 'SwissProt CV'), ('independent_test', 'Independent test')]:
        metrics = result.get(key, None)
        if not isinstance(metrics, dict):
            continue
        lines.append(
            '| {name} | {rows} | {subset:.4f} | {jacc:.4f} | {micro:.4f} | {macro:.4f} | {obs:.4f} | {ptr:.4f} |'.format(
                name=name,
                rows=int(metrics.get('n_rows', metrics.get('n_test_rows', 0))),
                subset=float(metrics.get('subset_accuracy', 0.0)),
                jacc=float(metrics.get('jaccard', 0.0)),
                micro=float(metrics.get('micro_f1', 0.0)),
                macro=float(metrics.get('macro_f1', 0.0)),
                obs=float(metrics.get('macro_f1_observed_labels', metrics.get('macro_f1', 0.0))),
                ptr=float(metrics.get('predicted_per_true', 0.0)),
            )
        )
    reference = result.get('published_reference', {})
    if isinstance(reference, dict) and len(reference) > 0:
        lines.append('')
        lines.append('Published reference: {}'.format(reference.get('source', '')))
        for split_name in ['swissprot_cv', 'hpa_independent', 'heldout_test']:
            split = reference.get(split_name, None)
            if not isinstance(split, dict):
                continue
            lines.append('')
            lines.append('## {}'.format(split_name))
            lines.append('| Model | Count | Jaccard | Micro F1 | Macro F1 |')
            lines.append('| --- | ---: | ---: | ---: | ---: |')
            for model_name, vals in sorted(split.items()):
                if not isinstance(vals, dict):
                    continue
                lines.append(
                    '| {model} | {count} | {jacc:.4f} | {micro:.4f} | {macro:.4f} |'.format(
                        model=model_name,
                        count=int(split.get('count', 0)),
                        jacc=float(vals.get('jaccard', 0.0)),
                        micro=float(vals.get('micro_f1', 0.0)),
                        macro=float(vals.get('macro_f1', 0.0)),
                    )
                )
    lines.append('')
    return '\n'.join(lines)


def run_deeploc21_benchmark(
    prepared_dir,
    task_name='localization',
    comparison_json='',
    comparison_md='',
    model_out='',
    n_folds=5,
    seed=1,
    model_arch='centroid',
    dl_params=None,
):
    cfg = _task_config(task_name=task_name, prepared_dir=prepared_dir)
    model_type = _model_type_from_arch(model_arch=model_arch)
    result = {
        'task': cfg['task'],
        'labels': list(cfg['labels']),
        'model': model_type,
        'model_arch': _normalize_model_arch(model_arch=model_arch),
        'feature_names': list(BROAD_FEATURE_NAMES),
        'train_tsv': cfg['train_tsv'],
        'published_reference': cfg.get('reference', {}),
    }
    result['cross_validation'] = evaluate_deeploc21_task_cv(
        tsv_path=cfg['train_tsv'],
        labels=cfg['labels'],
        label_col=cfg['label_col'],
        task_name=cfg['task'],
        fold_col='partition',
        n_folds=n_folds,
        seed=seed,
        model_arch=model_arch,
        dl_params=dl_params,
    )
    if cfg.get('test_tsv', '') != '':
        result['independent_test'] = evaluate_deeploc21_train_test(
            train_tsv_path=cfg['train_tsv'],
            test_tsv_path=cfg['test_tsv'],
            labels=cfg['labels'],
            label_col=cfg['label_col'],
            task_name=cfg['task'],
            model_arch=model_arch,
            dl_params=dl_params,
        )
    if str(model_out or '').strip() != '':
        train_rows = _read_prepared_tsv(path=cfg['train_tsv'])
        model = fit_deeploc_multilabel_model(
            rows=train_rows,
            labels=cfg['labels'],
            label_col=cfg['label_col'],
            task_name=cfg['task'],
            model_arch=model_arch,
            dl_params=dl_params,
        )
        model['metadata']['benchmark_task'] = cfg['task']
        model['metadata']['benchmark_train_tsv'] = cfg['train_tsv']
        save_localize_model(model=model, path=model_out)
        result['model_out'] = model_out

    if str(comparison_json or '').strip() != '':
        out_dir = os.path.dirname(comparison_json)
        if out_dir != '':
            os.makedirs(out_dir, exist_ok=True)
        with open(comparison_json, 'w', encoding='utf-8') as out_json:
            json.dump(result, out_json, indent=2, sort_keys=True)
    if str(comparison_md or '').strip() != '':
        out_dir = os.path.dirname(comparison_md)
        if out_dir != '':
            os.makedirs(out_dir, exist_ok=True)
        with open(comparison_md, 'w', encoding='utf-8') as out_md:
            out_md.write(render_deeploc_benchmark_markdown(result=result))
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description='Prepare DeepLoc 2.1 public benchmark datasets for cdskit localization work.',
    )
    parser.add_argument('--download', default='no', choices=['yes', 'no'], type=str)
    parser.add_argument('--prepare', default='yes', choices=['yes', 'no'], type=str)
    parser.add_argument('--benchmark', default='no', choices=['yes', 'no'], type=str)
    parser.add_argument('--task', default='localization', choices=['localization', 'membrane', 'sorting_signals'], type=str)
    parser.add_argument('--data_dir', default='data/deeploc21_raw', type=str)
    parser.add_argument('--out_dir', default='data/localize_bench/deeploc21', type=str)
    parser.add_argument('--report_json', default='data/localize_bench/deeploc21/prepare_report.json', type=str)
    parser.add_argument('--comparison_json', default='data/localize_bench/deeploc21/comparison.json', type=str)
    parser.add_argument('--comparison_md', default='data/localize_bench/deeploc21/comparison.md', type=str)
    parser.add_argument('--model_out', default='', type=str)
    parser.add_argument('--model_arch', default='centroid', choices=['centroid', 'cnn'], type=str)
    parser.add_argument('--dl_seq_len', default=512, type=int)
    parser.add_argument('--dl_embed_dim', default=32, type=int)
    parser.add_argument('--dl_num_filters', default=64, type=int)
    parser.add_argument('--dl_kernel_sizes', default='3,5,9,15', type=str)
    parser.add_argument('--dl_dropout', default=0.25, type=float)
    parser.add_argument('--dl_epochs', default=6, type=int)
    parser.add_argument('--dl_batch_size', default=256, type=int)
    parser.add_argument('--dl_lr', default=1.0e-3, type=float)
    parser.add_argument('--dl_weight_decay', default=1.0e-4, type=float)
    parser.add_argument('--dl_class_weight', default='yes', choices=['yes', 'no'], type=str)
    parser.add_argument('--dl_feature_fusion', default='no', choices=['yes', 'no'], type=str)
    parser.add_argument('--dl_threshold_objective', default='f1', choices=['f1', 'mcc'], type=str)
    parser.add_argument('--dl_seed', default=1, type=int)
    parser.add_argument('--dl_device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], type=str)
    parser.add_argument('--cv_folds', default=5, type=int)
    parser.add_argument('--cv_seed', default=1, type=int)
    parser.add_argument('--timeout_sec', default=120, type=int)
    return parser


def _to_bool_yes_no(value):
    return str(value).strip().lower() in ['yes', 'y', 'true', '1']


def main():
    args = build_parser().parse_args()
    out = dict()
    if _to_bool_yes_no(args.download):
        out['download'] = download_deeploc21_data(
            out_dir=args.data_dir,
            timeout_sec=int(args.timeout_sec),
        )
    if _to_bool_yes_no(args.prepare):
        out['prepare'] = prepare_all_deeploc21(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
        )
    if _to_bool_yes_no(args.benchmark):
        dl_params = {
            'seq_len': int(args.dl_seq_len),
            'embed_dim': int(args.dl_embed_dim),
            'num_filters': int(args.dl_num_filters),
            'kernel_sizes': args.dl_kernel_sizes,
            'dropout': float(args.dl_dropout),
            'epochs': int(args.dl_epochs),
            'batch_size': int(args.dl_batch_size),
            'learning_rate': float(args.dl_lr),
            'weight_decay': float(args.dl_weight_decay),
            'class_weight': args.dl_class_weight,
            'feature_fusion': args.dl_feature_fusion,
            'threshold_objective': args.dl_threshold_objective,
            'seed': int(args.dl_seed),
            'device': args.dl_device,
        }
        out['benchmark'] = run_deeploc21_benchmark(
            prepared_dir=args.out_dir,
            task_name=args.task,
            comparison_json=args.comparison_json,
            comparison_md=args.comparison_md,
            model_out=args.model_out,
            n_folds=int(args.cv_folds),
            seed=int(args.cv_seed),
            model_arch=args.model_arch,
            dl_params=dl_params,
        )
    report_dir = os.path.dirname(args.report_json)
    if report_dir != '':
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report_json, 'w', encoding='utf-8') as out_json:
        json.dump(out, out_json, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

import csv
import io
import re
import time
from collections import Counter
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import numpy as np

from cdskit.localize_model import (
    FEATURE_NAMES,
    LOCALIZATION_CLASSES,
    TP_STAGE_CLASSES,
    extract_localize_features,
    fit_nearest_centroid_classifier,
    fit_perox_binary_classifier,
    infer_labels_from_uniprot_cc,
    is_dna_like,
    normalize_class_probabilities,
    predict_localization_and_peroxisome,
    normalize_localization_label,
    normalize_yes_no,
    save_localize_model,
    to_canonical_aa_sequence,
    translate_inframe_cds_to_aa,
    write_rows_json,
    write_rows_tsv,
)
from cdskit.util import stop_if_invalid_codontable

UNIPROT_SEARCH_URL = 'https://rest.uniprot.org/uniprotkb/search'
UNIPROT_MAX_PAGE_SIZE = 500
UNIPROT_DEFAULT_FIELDS = ('accession', 'sequence', 'cc_subcellular_location')
UNIPROT_PRESET_QUERIES = {
    'none': '',
    'viridiplantae': 'taxonomy_id:33090',
    'eukaryota': 'taxonomy_id:2759',
    'metazoa': 'taxonomy_id:33208',
    'fungi': 'taxonomy_id:4751',
    'non_viridiplantae_euk': '(taxonomy_id:2759) AND (NOT taxonomy_id:33090)',
    'protist_core': (
        '(taxonomy_id:2759) AND (NOT taxonomy_id:33090) '
        'AND (NOT taxonomy_id:33208) AND (NOT taxonomy_id:4751)'
    ),
    'bacteria_hard_negative': 'taxonomy_id:2',
}


def read_training_tsv(path):
    with open(path, 'r', encoding='utf-8', newline='') as inp:
        reader = csv.DictReader(inp, delimiter='\t')
        return list(reader)


def parse_uniprot_fields(field_text):
    if field_text is None:
        return list(UNIPROT_DEFAULT_FIELDS)
    fields = [f.strip() for f in str(field_text).split(',') if f.strip() != '']
    if len(fields) == 0:
        return list(UNIPROT_DEFAULT_FIELDS)
    out = list()
    seen = set()
    for field_name in fields:
        if field_name in seen:
            continue
        seen.add(field_name)
        out.append(field_name)
    return out


def resolve_uniprot_query(uniprot_query, uniprot_preset):
    preset_name = str(uniprot_preset or 'none').strip().lower()
    if preset_name == '':
        preset_name = 'none'
    if preset_name not in UNIPROT_PRESET_QUERIES:
        valid = ','.join(sorted(UNIPROT_PRESET_QUERIES.keys()))
        txt = 'Invalid --uniprot_preset: {}. Supported: {}.'
        raise ValueError(txt.format(uniprot_preset, valid))
    preset_query = UNIPROT_PRESET_QUERIES[preset_name]

    query = str(uniprot_query or '').strip()
    if (preset_query != '') and (query != ''):
        return '({}) AND ({})'.format(preset_query, query), preset_name
    if preset_query != '':
        return preset_query, preset_name
    return query, preset_name


def parse_uniprot_next_link(link_header):
    if (link_header is None) or (link_header == ''):
        return None
    match = re.search(r'<([^>]+)>;\s*rel=\"next\"', str(link_header))
    if match is None:
        return None
    return match.group(1)


def _fetch_url_text(url, timeout_sec, retries):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            with urllib_request.urlopen(url, timeout=timeout_sec) as response:
                body = response.read().decode('utf-8')
                link_header = response.headers.get('Link', '')
                return body, link_header
        except urllib_error.URLError as exc:
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(1.0 + attempt)
    txt = 'Failed to download UniProt data: {}'
    raise Exception(txt.format(str(last_exc)))


def parse_uniprot_tsv_text(tsv_text, field_order):
    rows = list()
    if tsv_text.strip() == '':
        return rows
    reader = csv.reader(io.StringIO(tsv_text), delimiter='\t')
    try:
        _ = next(reader)  # Skip header row from UniProt.
    except StopIteration:
        return rows
    num_field = len(field_order)
    for raw_row in reader:
        if len(raw_row) == 0:
            continue
        row = dict()
        for i, field_name in enumerate(field_order):
            if i < len(raw_row):
                row[field_name] = raw_row[i]
            else:
                row[field_name] = ''
        rows.append(row)
    return rows


def fetch_uniprot_training_rows(
    query,
    fields,
    reviewed,
    exclude_fragments,
    page_size,
    max_rows,
    timeout_sec,
    retries,
    sampling_mode='head',
    sampling_seed=1,
):
    if query is None:
        query = ''
    query = str(query).strip()
    if query == '':
        query = '*'
    if reviewed:
        query = '({}) AND (reviewed:true)'.format(query)
    if exclude_fragments:
        query = '({}) AND (NOT fragment:true)'.format(query)

    page_size = int(page_size)
    if page_size < 1:
        raise ValueError('--uniprot_page_size should be >= 1.')
    if page_size > UNIPROT_MAX_PAGE_SIZE:
        page_size = UNIPROT_MAX_PAGE_SIZE

    max_rows = int(max_rows)
    if max_rows < 0:
        raise ValueError('--uniprot_max_rows should be >= 0.')
    sampling_mode = str(sampling_mode).strip().lower()
    if sampling_mode not in ['head', 'random']:
        raise ValueError('--uniprot_sampling should be head or random.')
    sampling_seed = int(sampling_seed)

    params = {
        'query': query,
        'format': 'tsv',
        'fields': ','.join(fields),
        'size': str(page_size),
    }
    next_url = '{}?{}'.format(UNIPROT_SEARCH_URL, urllib_parse.urlencode(params))
    all_rows = list()
    rng = np.random.default_rng(sampling_seed)
    seen_count = 0
    while next_url is not None:
        body, link_header = _fetch_url_text(
            url=next_url,
            timeout_sec=timeout_sec,
            retries=retries,
        )
        new_rows = parse_uniprot_tsv_text(tsv_text=body, field_order=fields)
        if max_rows == 0:
            all_rows.extend(new_rows)
        elif sampling_mode == 'head':
            remain = max_rows - len(all_rows)
            if remain > 0:
                all_rows.extend(new_rows[:remain])
            if len(all_rows) >= max_rows:
                all_rows = all_rows[:max_rows]
                break
        else:
            for row in new_rows:
                seen_count += 1
                if len(all_rows) < max_rows:
                    all_rows.append(row)
                else:
                    j = int(rng.integers(0, seen_count))
                    if j < max_rows:
                        all_rows[j] = row
        next_url = parse_uniprot_next_link(link_header=link_header)
    return all_rows


def write_uniprot_rows_tsv(rows, fields, out_path):
    with open(out_path, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(
            out,
            fieldnames=fields,
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_sequence_mode(seq_str, seqtype):
    if seqtype == 'dna':
        return 'dna'
    if seqtype == 'protein':
        return 'protein'
    if is_dna_like(seq_str):
        return 'dna'
    return 'protein'


def parse_training_row(
    row,
    row_index,
    seq_col,
    seqtype,
    codontable,
    label_mode,
    localization_col,
    perox_col,
    skip_ambiguous,
    cv_fold_col='',
):
    seq_raw = row.get(seq_col, '')
    if seq_raw is None:
        seq_raw = ''
    seq_raw = str(seq_raw).strip()
    if seq_raw == '':
        txt = 'Empty sequence at row {} in training table.'
        raise ValueError(txt.format(row_index + 1))

    seq_mode = resolve_sequence_mode(seq_str=seq_raw, seqtype=seqtype)
    if seq_mode == 'dna':
        aa_seq = translate_inframe_cds_to_aa(
            cds_seq=seq_raw,
            codontable=codontable,
            seq_id='training_row_{}'.format(row_index + 1),
        )
    else:
        aa_seq = to_canonical_aa_sequence(seq_raw)

    if label_mode == 'explicit':
        class_label = normalize_localization_label(row.get(localization_col, ''))
        perox_label = normalize_yes_no(row.get(perox_col, 'no'), default='no')
    elif label_mode == 'uniprot_cc':
        location_text = row.get(localization_col, '')
        if (location_text is None) or (str(location_text).strip() == ''):
            for alt_key in (
                'cc_subcellular_location',
                'Subcellular location [CC]',
                'subcellular_location',
                'localization',
            ):
                if alt_key in row and str(row.get(alt_key, '')).strip() != '':
                    location_text = row.get(alt_key, '')
                    break
        class_label, perox_label, ambiguous = infer_labels_from_uniprot_cc(
            location_text=location_text,
        )
        if ambiguous and skip_ambiguous:
            return None
        perox_label = normalize_yes_no(perox_label, default='no')
    else:
        raise ValueError('Unsupported --label_mode: {}'.format(label_mode))

    fold_id = None
    cv_fold_col = str(cv_fold_col or '').strip()
    if cv_fold_col != '':
        fold_raw = row.get(cv_fold_col, None)
        if (fold_raw is None) or (str(fold_raw).strip() == ''):
            txt = 'Missing fold value in column "{}" at row {} in training table.'
            raise ValueError(txt.format(cv_fold_col, row_index + 1))
        fold_id = str(fold_raw).strip()

    feats, _ = extract_localize_features(aa_seq=aa_seq)
    return aa_seq, feats, class_label, perox_label, fold_id


def build_training_matrix(
    rows,
    seq_col,
    seqtype,
    codontable,
    label_mode,
    localization_col,
    perox_col,
    skip_ambiguous,
    cv_fold_col='',
):
    features = list()
    aa_sequences = list()
    class_labels = list()
    perox_labels = list()
    fold_ids = None
    if str(cv_fold_col or '').strip() != '':
        fold_ids = list()
    skipped = 0
    for i, row in enumerate(rows):
        parsed = parse_training_row(
            row=row,
            row_index=i,
            seq_col=seq_col,
            seqtype=seqtype,
            codontable=codontable,
            label_mode=label_mode,
            localization_col=localization_col,
            perox_col=perox_col,
            skip_ambiguous=skip_ambiguous,
            cv_fold_col=cv_fold_col,
        )
        if parsed is None:
            skipped += 1
            continue
        aa_seq, feat_vec, class_label, perox_label, fold_id = parsed
        aa_sequences.append(aa_seq)
        features.append(feat_vec)
        class_labels.append(class_label)
        perox_labels.append(perox_label)
        if fold_ids is not None:
            fold_ids.append(fold_id)
    if len(features) == 0:
        raise ValueError('No valid training sample remained after filtering.')
    x = np.asarray(features, dtype=np.float64)
    return x, aa_sequences, class_labels, perox_labels, skipped, fold_ids


def calculate_training_metrics(
    x,
    aa_sequences,
    class_labels,
    perox_labels,
    model,
    model_arch,
    dl_device,
):
    rows = list()
    correct_class = 0
    correct_perox = 0
    class_total = {class_name: 0 for class_name in LOCALIZATION_CLASSES}
    class_correct = {class_name: 0 for class_name in LOCALIZATION_CLASSES}
    for i in range(x.shape[0]):
        _ = model_arch
        _ = dl_device
        pred = predict_localization_and_peroxisome(
            aa_seq=aa_sequences[i],
            model=model,
        )
        true_class = class_labels[i]
        true_perox = perox_labels[i]
        class_total[true_class] = class_total.get(true_class, 0) + 1
        if pred['predicted_class'] == true_class:
            correct_class += 1
            class_correct[true_class] = class_correct.get(true_class, 0) + 1
        pred_perox = 'yes' if pred['perox_probability_yes'] >= 0.5 else 'no'
        if pred_perox == true_perox:
            correct_perox += 1
        rows.append({
            'index': i,
            'true_class': true_class,
            'predicted_class': pred['predicted_class'],
            'true_perox': true_perox,
            'predicted_perox': pred_perox,
        })
    n = float(x.shape[0])
    class_accuracy_by_class = dict()
    for class_name in LOCALIZATION_CLASSES:
        denom = float(class_total.get(class_name, 0))
        if denom <= 0:
            class_accuracy_by_class[class_name] = 0.0
        else:
            class_accuracy_by_class[class_name] = float(class_correct.get(class_name, 0)) / denom
    return {
        'class_train_accuracy': float(correct_class) / n,
        'perox_train_accuracy': float(correct_perox) / n,
        'class_accuracy_by_class': class_accuracy_by_class,
        'rows': rows,
    }


def _fit_constant_localization_model(class_label):
    class_label = str(class_label)
    return {
        'mode': 'constant',
        'class_label': class_label,
        'class_order': [class_label],
    }


def _fit_arch_specific_localization_model(
    x,
    aa_sequences,
    labels,
    class_order,
    model_arch,
    dl_train_params,
):
    labels = list(labels)
    class_order = list(class_order)
    if len(labels) == 0:
        raise ValueError('No training sample was provided to localization classifier.')
    observed = sorted(set(labels))
    if len(observed) == 1:
        return _fit_constant_localization_model(class_label=observed[0])
    if model_arch == 'nearest_centroid':
        return fit_nearest_centroid_classifier(
            features=x,
            labels=labels,
            class_order=class_order,
        )
    if model_arch == 'bilstm_attention':
        from cdskit.localize_bilstm import fit_bilstm_attention_classifier
        return fit_bilstm_attention_classifier(
            aa_sequences=aa_sequences,
            labels=labels,
            class_order=class_order,
            seq_len=dl_train_params['seq_len'],
            embed_dim=dl_train_params['embed_dim'],
            hidden_dim=dl_train_params['hidden_dim'],
            num_layers=dl_train_params['num_layers'],
            dropout=dl_train_params['dropout'],
            epochs=dl_train_params['epochs'],
            batch_size=dl_train_params['batch_size'],
            learning_rate=dl_train_params['learning_rate'],
            weight_decay=dl_train_params['weight_decay'],
            seed=dl_train_params['seed'],
            use_class_weight=dl_train_params['use_class_weight'],
            device=dl_train_params['device'],
            loss_name=dl_train_params['loss_name'],
            balanced_batch=dl_train_params['balanced_batch'],
        )
    raise ValueError('Unsupported model_arch: {}'.format(model_arch))


def fit_localization_model(
    x,
    aa_sequences,
    class_labels,
    model_arch,
    dl_train_params,
    localize_strategy='single_stage',
):
    localize_strategy = str(localize_strategy or 'single_stage').strip().lower()
    if localize_strategy not in ['single_stage', 'two_stage']:
        raise ValueError('Unsupported localize_strategy: {}'.format(localize_strategy))

    if localize_strategy == 'single_stage':
        return _fit_arch_specific_localization_model(
            x=x,
            aa_sequences=aa_sequences,
            labels=class_labels,
            class_order=LOCALIZATION_CLASSES,
            model_arch=model_arch,
            dl_train_params=dl_train_params,
        )

    stage1_labels = ['noTP' if cls == 'noTP' else 'TP' for cls in class_labels]
    stage1_model = _fit_arch_specific_localization_model(
        x=x,
        aa_sequences=aa_sequences,
        labels=stage1_labels,
        class_order=('noTP', 'TP'),
        model_arch=model_arch,
        dl_train_params=dl_train_params,
    )

    tp_indices = [i for i, cls in enumerate(class_labels) if cls != 'noTP']
    if len(tp_indices) == 0:
        raise ValueError('two_stage strategy requires at least one TP sample.')
    x_tp = x[tp_indices, :]
    aa_tp = [aa_sequences[i] for i in tp_indices]
    labels_tp = [class_labels[i] for i in tp_indices]
    stage2_class_order = [c for c in TP_STAGE_CLASSES if c in set(labels_tp)]
    stage2_model = _fit_arch_specific_localization_model(
        x=x_tp,
        aa_sequences=aa_tp,
        labels=labels_tp,
        class_order=stage2_class_order,
        model_arch=model_arch,
        dl_train_params=dl_train_params,
    )
    return {
        'strategy': 'two_stage',
        'class_order': list(LOCALIZATION_CLASSES),
        'stage1_class_order': ['noTP', 'TP'],
        'stage2_class_order': list(stage2_class_order),
        'stage1_model': stage1_model,
        'stage2_model': stage2_model,
    }


def _validate_cross_validation_labels(class_labels):
    counts = Counter(class_labels)
    present_classes = [c for c in LOCALIZATION_CLASSES if counts.get(c, 0) > 0]
    if len(present_classes) < 2:
        txt = (
            'Cross validation requires at least 2 localization classes '
            'with at least one sample each.'
        )
        raise ValueError(txt)
    insufficient = [c for c in present_classes if counts.get(c, 0) < 2]
    if len(insufficient) > 0:
        txt = (
            'Cross validation requires at least 2 samples for each observed class. '
            'Insufficient classes: {}.'
        )
        raise ValueError(txt.format(', '.join(insufficient)))


def build_stratified_folds(class_labels, n_folds, seed):
    if n_folds < 2:
        raise ValueError('--cv_folds should be >= 2 when cross validation is enabled.')
    labels = list(class_labels)
    n_sample = len(labels)
    if n_sample < n_folds:
        txt = '--cv_folds ({}) should be <= number of samples ({}).'
        raise ValueError(txt.format(n_folds, n_sample))

    _validate_cross_validation_labels(class_labels=labels)

    rng = np.random.default_rng(int(seed))
    fold_buckets = [list() for _ in range(n_folds)]
    for class_name in LOCALIZATION_CLASSES:
        class_indices = [i for i, lab in enumerate(labels) if lab == class_name]
        class_indices = np.asarray(class_indices, dtype=np.int64)
        rng.shuffle(class_indices)
        for pos, idx in enumerate(class_indices.tolist()):
            fold_buckets[pos % n_folds].append(int(idx))

    folds = list()
    for idx_list in fold_buckets:
        if len(idx_list) == 0:
            raise ValueError('At least one fold became empty. Reduce --cv_folds.')
        fold = np.asarray(sorted(idx_list), dtype=np.int64)
        folds.append(fold)
    return folds


def build_predefined_folds(class_labels, fold_ids):
    labels = list(class_labels)
    fold_ids = list(fold_ids)
    if len(labels) != len(fold_ids):
        raise ValueError('Length mismatch between class labels and fold ids.')
    _validate_cross_validation_labels(class_labels=labels)

    fold_to_indices = dict()
    for i, fold_id in enumerate(fold_ids):
        key = str(fold_id).strip()
        if key == '':
            raise ValueError('Empty fold id was detected in predefined folds.')
        if key not in fold_to_indices:
            fold_to_indices[key] = list()
        fold_to_indices[key].append(int(i))

    ordered_keys = sorted(fold_to_indices.keys())
    if len(ordered_keys) < 2:
        raise ValueError('Predefined folds should contain at least 2 unique fold ids.')

    folds = list()
    for key in ordered_keys:
        idx_list = fold_to_indices[key]
        if len(idx_list) == 0:
            raise ValueError('At least one predefined fold became empty.')
        folds.append(np.asarray(sorted(idx_list), dtype=np.int64))
    return folds


def _safe_probability_matrix_from_oof(oof_rows):
    class_to_idx = {class_name: i for i, class_name in enumerate(LOCALIZATION_CLASSES)}
    probs = list()
    true_idx = list()
    fold_idx = list()
    for row in oof_rows:
        true_class = str(row.get('true_class', '')).strip()
        if true_class not in class_to_idx:
            continue
        class_probs = normalize_class_probabilities(
            class_probs=row.get('class_probabilities', {}),
        )
        probs.append([float(class_probs[class_name]) for class_name in LOCALIZATION_CLASSES])
        true_idx.append(int(class_to_idx[true_class]))
        fold_idx.append(int(row.get('fold', 0)))
    if len(probs) == 0:
        raise ValueError('No valid out-of-fold predictions were available for postprocessing.')
    return (
        np.asarray(probs, dtype=np.float64),
        np.asarray(true_idx, dtype=np.int64),
        np.asarray(fold_idx, dtype=np.int64),
    )


def _apply_temperature_to_matrix(prob_matrix, temperature):
    try:
        temp = float(temperature)
    except Exception:
        temp = 1.0
    if (not np.isfinite(temp)) or (temp <= 0.0) or (abs(temp - 1.0) < 1.0e-12):
        return np.asarray(prob_matrix, dtype=np.float64)
    clipped = np.clip(np.asarray(prob_matrix, dtype=np.float64), 1.0e-12, 1.0)
    logits = np.log(clipped) / temp
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_vals = np.exp(logits)
    denom = exp_vals.sum(axis=1, keepdims=True)
    denom[denom <= 0.0] = 1.0
    return exp_vals / denom


def _class_threshold_vector(class_thresholds):
    out = np.ones(len(LOCALIZATION_CLASSES), dtype=np.float64)
    if not isinstance(class_thresholds, dict):
        return out
    for i, class_name in enumerate(LOCALIZATION_CLASSES):
        value = class_thresholds.get(class_name, 1.0)
        try:
            value = float(value)
        except Exception:
            value = 1.0
        if (not np.isfinite(value)) or (value <= 0.0):
            value = 1.0
        out[i] = value
    return out


def _prediction_indices_from_scores(prob_matrix, class_thresholds):
    thresholds = _class_threshold_vector(class_thresholds=class_thresholds)
    scores = np.asarray(prob_matrix, dtype=np.float64) / thresholds[np.newaxis, :]
    return np.argmax(scores, axis=1).astype(np.int64)


def _accuracy_from_indices(true_idx, pred_idx):
    true_idx = np.asarray(true_idx, dtype=np.int64)
    pred_idx = np.asarray(pred_idx, dtype=np.int64)
    n = int(true_idx.shape[0])
    if n == 0:
        raise ValueError('No prediction was available to compute accuracy.')
    overall = float(np.mean(pred_idx == true_idx))
    by_class = dict()
    class_acc = list()
    for class_i, class_name in enumerate(LOCALIZATION_CLASSES):
        mask = (true_idx == class_i)
        denom = int(np.sum(mask))
        if denom <= 0:
            acc = 0.0
        else:
            acc = float(np.mean(pred_idx[mask] == class_i))
        by_class[class_name] = acc
        class_acc.append(acc)
    macro = float(np.mean(np.asarray(class_acc, dtype=np.float64)))
    return overall, macro, by_class


def fit_temperature_from_oof(oof_rows):
    prob_matrix, true_idx, _ = _safe_probability_matrix_from_oof(oof_rows=oof_rows)
    base = np.clip(prob_matrix, 1.0e-12, 1.0)
    candidates = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]
    best_temp = 1.0
    best_nll = None
    row_idx = np.arange(true_idx.shape[0], dtype=np.int64)
    for temp in candidates:
        scaled = _apply_temperature_to_matrix(prob_matrix=base, temperature=temp)
        true_prob = np.clip(scaled[row_idx, true_idx], 1.0e-12, 1.0)
        nll = float(-np.mean(np.log(true_prob)))
        if (best_nll is None) or (nll < best_nll - 1.0e-12):
            best_nll = nll
            best_temp = float(temp)
            continue
        if abs(nll - best_nll) <= 1.0e-12:
            if abs(float(temp) - 1.0) < abs(best_temp - 1.0):
                best_temp = float(temp)
    return float(best_temp)


def optimize_class_thresholds_from_oof(oof_rows, temperature, objective):
    obj = str(objective or 'macro').strip().lower()
    if obj not in ['overall', 'macro']:
        raise ValueError('Invalid threshold objective: {}'.format(objective))
    prob_matrix, true_idx, _ = _safe_probability_matrix_from_oof(oof_rows=oof_rows)
    prob_matrix = _apply_temperature_to_matrix(prob_matrix=prob_matrix, temperature=temperature)

    thresholds = {class_name: 1.0 for class_name in LOCALIZATION_CLASSES}
    candidates = [0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.40]

    pred_idx = _prediction_indices_from_scores(
        prob_matrix=prob_matrix,
        class_thresholds=thresholds,
    )
    best_overall, best_macro, _ = _accuracy_from_indices(
        true_idx=true_idx,
        pred_idx=pred_idx,
    )
    best_score = best_overall if obj == 'overall' else best_macro

    for _ in range(2):
        improved = False
        for class_name in LOCALIZATION_CLASSES:
            current_value = float(thresholds[class_name])
            class_best_value = current_value
            class_best_overall = best_overall
            class_best_macro = best_macro
            class_best_score = best_score
            for cand in candidates:
                trial = dict(thresholds)
                trial[class_name] = float(cand)
                trial_pred = _prediction_indices_from_scores(
                    prob_matrix=prob_matrix,
                    class_thresholds=trial,
                )
                trial_overall, trial_macro, _ = _accuracy_from_indices(
                    true_idx=true_idx,
                    pred_idx=trial_pred,
                )
                trial_score = trial_overall if obj == 'overall' else trial_macro
                if trial_score > class_best_score + 1.0e-12:
                    class_best_value = float(cand)
                    class_best_overall = trial_overall
                    class_best_macro = trial_macro
                    class_best_score = trial_score
                    continue
                if abs(trial_score - class_best_score) <= 1.0e-12:
                    if abs(float(cand) - 1.0) < abs(class_best_value - 1.0):
                        class_best_value = float(cand)
                        class_best_overall = trial_overall
                        class_best_macro = trial_macro
                        class_best_score = trial_score
            if abs(class_best_value - current_value) > 1.0e-12:
                thresholds[class_name] = float(class_best_value)
                best_overall = class_best_overall
                best_macro = class_best_macro
                best_score = class_best_score
                improved = True
        if not improved:
            break
    return {class_name: float(thresholds[class_name]) for class_name in LOCALIZATION_CLASSES}


def evaluate_oof_postprocess(oof_rows, temperature=1.0, class_thresholds=None):
    prob_matrix, true_idx, fold_idx = _safe_probability_matrix_from_oof(oof_rows=oof_rows)
    prob_matrix = _apply_temperature_to_matrix(prob_matrix=prob_matrix, temperature=temperature)
    pred_idx = _prediction_indices_from_scores(
        prob_matrix=prob_matrix,
        class_thresholds=class_thresholds,
    )
    overall, macro, by_class = _accuracy_from_indices(
        true_idx=true_idx,
        pred_idx=pred_idx,
    )

    fold_rows = list()
    unique_folds = sorted(set(fold_idx.tolist()))
    fold_acc = list()
    for fold_id in unique_folds:
        mask = (fold_idx == int(fold_id))
        if int(np.sum(mask)) <= 0:
            continue
        fold_acc_value = float(np.mean(pred_idx[mask] == true_idx[mask]))
        fold_acc.append(fold_acc_value)
        fold_rows.append({
            'fold': int(fold_id),
            'class_accuracy': fold_acc_value,
        })
    if len(fold_acc) == 0:
        fold_mean = overall
        fold_std = 0.0
    else:
        fold_arr = np.asarray(fold_acc, dtype=np.float64)
        fold_mean = float(fold_arr.mean())
        fold_std = float(fold_arr.std())
    return {
        'class_accuracy_overall': float(overall),
        'class_accuracy_macro5': float(macro),
        'class_accuracy_by_class': dict(by_class),
        'fold_class_accuracy_mean': float(fold_mean),
        'fold_class_accuracy_std': float(fold_std),
        'folds': fold_rows,
    }


def evaluate_cross_validation(
    x,
    aa_sequences,
    class_labels,
    perox_labels,
    n_folds,
    seed,
    model_arch,
    dl_train_params,
    dl_device,
    localize_strategy='single_stage',
    fold_ids=None,
):
    if fold_ids is None:
        folds = build_stratified_folds(
            class_labels=class_labels,
            n_folds=n_folds,
            seed=seed,
        )
    else:
        folds = build_predefined_folds(
            class_labels=class_labels,
            fold_ids=fold_ids,
        )
    n_sample = int(x.shape[0])
    fold_rows = list()
    class_accs = list()
    perox_accs = list()
    oof_rows = list()
    class_total = {class_name: 0 for class_name in LOCALIZATION_CLASSES}
    class_correct_by_class = {class_name: 0 for class_name in LOCALIZATION_CLASSES}
    for fold_i, test_idx in enumerate(folds):
        train_mask = np.ones(n_sample, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]
        if len(train_idx) == 0:
            raise ValueError('Empty train split detected in cross validation.')
        x_train = x[train_idx, :]
        x_test = x[test_idx, :]
        aa_train = [aa_sequences[i] for i in train_idx.tolist()]
        aa_test = [aa_sequences[i] for i in test_idx.tolist()]
        class_train = [class_labels[i] for i in train_idx.tolist()]
        class_test = [class_labels[i] for i in test_idx.tolist()]
        perox_train = [perox_labels[i] for i in train_idx.tolist()]
        perox_test = [perox_labels[i] for i in test_idx.tolist()]

        local_model = fit_localization_model(
            x=x_train,
            aa_sequences=aa_train,
            class_labels=class_train,
            model_arch=model_arch,
            dl_train_params=dl_train_params,
            localize_strategy=localize_strategy,
        )
        perox_model = fit_perox_binary_classifier(
            features=x_train,
            labels=perox_train,
        )
        if model_arch == 'nearest_centroid':
            tmp_model_type = 'nearest_centroid_v1'
        elif model_arch == 'bilstm_attention':
            tmp_model_type = 'bilstm_attention_v1'
        else:
            raise ValueError('Unsupported model_arch: {}'.format(model_arch))
        tmp_model = {
            'model_type': tmp_model_type,
            'localization_model': local_model,
            'perox_model': perox_model,
        }

        class_correct_fold = 0
        perox_correct = 0
        for row_i in range(x_test.shape[0]):
            _ = dl_device
            pred = predict_localization_and_peroxisome(
                aa_seq=aa_test[row_i],
                model=tmp_model,
            )
            true_class = class_test[row_i]
            pred_class = pred['predicted_class']
            oof_rows.append({
                'index': int(test_idx[row_i]),
                'fold': int(fold_i + 1),
                'true_class': true_class,
                'class_probabilities': normalize_class_probabilities(
                    class_probs=pred.get('class_probabilities', {}),
                ),
            })
            class_total[true_class] = class_total.get(true_class, 0) + 1
            if pred_class == true_class:
                class_correct_fold += 1
                class_correct_by_class[true_class] = class_correct_by_class.get(true_class, 0) + 1
            pred_perox = 'yes' if pred['perox_probability_yes'] >= 0.5 else 'no'
            if pred_perox == perox_test[row_i]:
                perox_correct += 1

        test_n = float(x_test.shape[0])
        class_acc = float(class_correct_fold) / test_n
        perox_acc = float(perox_correct) / test_n
        class_accs.append(class_acc)
        perox_accs.append(perox_acc)
        fold_rows.append({
            'fold': int(fold_i + 1),
            'n_train': int(x_train.shape[0]),
            'n_test': int(x_test.shape[0]),
            'class_accuracy': class_acc,
            'perox_accuracy': perox_acc,
        })

    class_arr = np.asarray(class_accs, dtype=np.float64)
    perox_arr = np.asarray(perox_accs, dtype=np.float64)
    class_accuracy_by_class = dict()
    for class_name in LOCALIZATION_CLASSES:
        denom = float(class_total.get(class_name, 0))
        if denom <= 0:
            class_accuracy_by_class[class_name] = 0.0
        else:
            class_accuracy_by_class[class_name] = float(class_correct_by_class.get(class_name, 0)) / denom
    return {
        'n_folds': int(len(folds)),
        'class_accuracy_mean': float(class_arr.mean()),
        'class_accuracy_std': float(class_arr.std()),
        'class_accuracy_by_class': class_accuracy_by_class,
        'perox_accuracy_mean': float(perox_arr.mean()),
        'perox_accuracy_std': float(perox_arr.std()),
        'folds': fold_rows,
        'oof_rows': oof_rows,
    }


def localize_learn_main(args):
    stop_if_invalid_codontable(codontable=args.codontable, label='--codontable')
    training_tsv = getattr(args, 'training_tsv', '')
    uniprot_query = getattr(args, 'uniprot_query', '')
    uniprot_preset = getattr(args, 'uniprot_preset', 'none')
    uniprot_reviewed = getattr(args, 'uniprot_reviewed', True)
    uniprot_exclude_fragments = getattr(args, 'uniprot_exclude_fragments', True)
    uniprot_fields_text = getattr(args, 'uniprot_fields', ','.join(UNIPROT_DEFAULT_FIELDS))
    uniprot_page_size = getattr(args, 'uniprot_page_size', UNIPROT_MAX_PAGE_SIZE)
    uniprot_max_rows = getattr(args, 'uniprot_max_rows', 0)
    uniprot_sampling = getattr(args, 'uniprot_sampling', 'head')
    uniprot_sampling_seed = int(getattr(args, 'uniprot_sampling_seed', 1))
    uniprot_timeout_sec = getattr(args, 'uniprot_timeout_sec', 60)
    uniprot_retries = getattr(args, 'uniprot_retries', 2)
    uniprot_out_tsv = getattr(args, 'uniprot_out_tsv', '')
    cv_folds = int(getattr(args, 'cv_folds', 0))
    cv_seed = int(getattr(args, 'cv_seed', 1))
    cv_fold_col = str(getattr(args, 'cv_fold_col', '')).strip()
    model_arch = str(getattr(args, 'model_arch', 'nearest_centroid')).strip().lower()
    if model_arch not in ['nearest_centroid', 'bilstm_attention']:
        raise ValueError('Unsupported --model_arch: {}'.format(model_arch))
    localize_strategy = str(getattr(args, 'localize_strategy', 'single_stage')).strip().lower()
    if localize_strategy not in ['single_stage', 'two_stage']:
        raise ValueError('--localize_strategy should be single_stage or two_stage.')
    localize_temperature_scale = bool(getattr(args, 'localize_temperature_scale', False))
    localize_threshold_tune = bool(getattr(args, 'localize_threshold_tune', False))
    localize_threshold_objective = str(
        getattr(args, 'localize_threshold_objective', 'macro')
    ).strip().lower()
    if localize_threshold_objective not in ['overall', 'macro']:
        raise ValueError('--localize_threshold_objective should be overall or macro.')

    dl_seq_len = int(getattr(args, 'dl_seq_len', 200))
    dl_embed_dim = int(getattr(args, 'dl_embed_dim', 32))
    dl_hidden_dim = int(getattr(args, 'dl_hidden_dim', 64))
    dl_num_layers = int(getattr(args, 'dl_num_layers', 1))
    dl_dropout = float(getattr(args, 'dl_dropout', 0.2))
    dl_epochs = int(getattr(args, 'dl_epochs', 15))
    dl_batch_size = int(getattr(args, 'dl_batch_size', 128))
    dl_lr = float(getattr(args, 'dl_lr', 1.0e-3))
    dl_weight_decay = float(getattr(args, 'dl_weight_decay', 1.0e-4))
    dl_class_weight = bool(getattr(args, 'dl_class_weight', True))
    dl_loss = str(getattr(args, 'dl_loss', 'ce')).strip().lower()
    dl_balanced_batch = bool(getattr(args, 'dl_balanced_batch', False))
    dl_seed = int(getattr(args, 'dl_seed', 1))
    dl_device = str(getattr(args, 'dl_device', 'auto')).strip().lower()
    if dl_seq_len < 1:
        raise ValueError('--dl_seq_len should be >= 1.')
    if dl_embed_dim < 1:
        raise ValueError('--dl_embed_dim should be >= 1.')
    if dl_hidden_dim < 1:
        raise ValueError('--dl_hidden_dim should be >= 1.')
    if dl_num_layers < 1:
        raise ValueError('--dl_num_layers should be >= 1.')
    if dl_dropout < 0:
        raise ValueError('--dl_dropout should be >= 0.')
    if dl_epochs < 1:
        raise ValueError('--dl_epochs should be >= 1.')
    if dl_batch_size < 1:
        raise ValueError('--dl_batch_size should be >= 1.')
    if dl_lr <= 0:
        raise ValueError('--dl_lr should be > 0.')
    if dl_weight_decay < 0:
        raise ValueError('--dl_weight_decay should be >= 0.')
    if dl_loss not in ['ce', 'focal']:
        raise ValueError('--dl_loss should be ce or focal.')
    if model_arch == 'bilstm_attention':
        from cdskit.localize_bilstm import require_torch
        require_torch()
    if cv_folds < 0:
        raise ValueError('--cv_folds should be >= 0.')
    if (cv_fold_col == '') and (cv_folds == 1):
        raise ValueError('--cv_folds should be 0 (disabled) or >= 2.')

    has_training_tsv = (str(training_tsv).strip() != '')
    resolved_uniprot_query, preset_name = resolve_uniprot_query(
        uniprot_query=uniprot_query,
        uniprot_preset=uniprot_preset,
    )
    has_uniprot_source = (str(resolved_uniprot_query).strip() != '')
    if has_training_tsv and has_uniprot_source:
        txt = 'Use either --training_tsv or --uniprot_query/--uniprot_preset, not both.'
        raise Exception(txt)
    if (not has_training_tsv) and (not has_uniprot_source):
        txt = 'Either --training_tsv or --uniprot_query/--uniprot_preset should be specified.'
        raise Exception(txt)

    if has_training_tsv:
        rows = read_training_tsv(path=training_tsv)
    else:
        fields = parse_uniprot_fields(field_text=uniprot_fields_text)
        if args.seq_col not in fields:
            fields.append(args.seq_col)
        if args.label_mode == 'explicit':
            missing = list()
            if args.localization_col not in fields:
                missing.append(args.localization_col)
            if args.perox_col not in fields:
                missing.append(args.perox_col)
            if len(missing) > 0:
                txt = (
                    'In --label_mode explicit with UniProt source, '
                    '--uniprot_fields should include: {}.'
                )
                raise ValueError(txt.format(', '.join(missing)))
        else:
            # Keep UniProt request valid with default CLI arguments.
            if 'cc_subcellular_location' not in fields:
                fields.append('cc_subcellular_location')
        rows = fetch_uniprot_training_rows(
            query=resolved_uniprot_query,
            fields=fields,
            reviewed=uniprot_reviewed,
            exclude_fragments=uniprot_exclude_fragments,
            page_size=uniprot_page_size,
            max_rows=uniprot_max_rows,
            timeout_sec=uniprot_timeout_sec,
            retries=uniprot_retries,
            sampling_mode=uniprot_sampling,
            sampling_seed=uniprot_sampling_seed,
        )
        if len(rows) == 0:
            raise Exception('No row was downloaded from UniProt. Exiting.')
        if str(uniprot_out_tsv).strip() != '':
            write_uniprot_rows_tsv(
                rows=rows,
                fields=fields,
                out_path=uniprot_out_tsv,
            )

    x, aa_sequences, class_labels, perox_labels, skipped, fold_ids = build_training_matrix(
        rows=rows,
        seq_col=args.seq_col,
        seqtype=args.seqtype,
        codontable=args.codontable,
        label_mode=args.label_mode,
        localization_col=args.localization_col,
        perox_col=args.perox_col,
        skip_ambiguous=args.skip_ambiguous,
        cv_fold_col=cv_fold_col,
    )

    dl_train_params = {
        'seq_len': dl_seq_len,
        'embed_dim': dl_embed_dim,
        'hidden_dim': dl_hidden_dim,
        'num_layers': dl_num_layers,
        'dropout': dl_dropout,
        'epochs': dl_epochs,
        'batch_size': dl_batch_size,
        'learning_rate': dl_lr,
        'weight_decay': dl_weight_decay,
        'seed': dl_seed,
        'use_class_weight': dl_class_weight,
        'device': dl_device,
        'loss_name': dl_loss,
        'balanced_batch': dl_balanced_batch,
    }
    localization_model = fit_localization_model(
        x=x,
        aa_sequences=aa_sequences,
        class_labels=class_labels,
        model_arch=model_arch,
        dl_train_params=dl_train_params,
        localize_strategy=localize_strategy,
    )
    if model_arch == 'nearest_centroid':
        model_type = 'nearest_centroid_v1'
    elif model_arch == 'bilstm_attention':
        model_type = 'bilstm_attention_v1'
    else:
        raise ValueError('Unsupported --model_arch: {}'.format(model_arch))
    perox_model = fit_perox_binary_classifier(
        features=x,
        labels=perox_labels,
    )

    effective_cv_folds = int(cv_folds)
    predefined_folds_active = (cv_fold_col != '')
    if predefined_folds_active:
        unique_folds = sorted(set(fold_ids))
        if len(unique_folds) < 2:
            raise ValueError('--cv_fold_col should contain at least 2 unique fold ids.')
        if cv_folds not in [0, len(unique_folds)]:
            txt = (
                'When --cv_fold_col is used, --cv_folds should be 0 or match '
                'the number of unique fold ids ({}).'
            )
            raise ValueError(txt.format(len(unique_folds)))
        effective_cv_folds = int(len(unique_folds))
    elif cv_folds == 1:
        raise ValueError('--cv_folds should be 0 (disabled) or >= 2.')
    if (localize_temperature_scale or localize_threshold_tune) and (effective_cv_folds < 2):
        txt = (
            '--localize_temperature_scale/--localize_threshold_tune requires '
            'cross validation (--cv_folds >= 2 or --cv_fold_col).'
        )
        raise ValueError(txt)

    model = {
        'model_type': model_type,
        'feature_names': list(FEATURE_NAMES),
        'localization_model': localization_model,
        'perox_model': perox_model,
        'metadata': {
            'num_training_rows': int(len(rows)),
            'num_used_rows': int(x.shape[0]),
            'num_skipped_rows': int(skipped),
            'seq_col': args.seq_col,
            'seqtype': args.seqtype,
            'label_mode': args.label_mode,
            'localization_col': args.localization_col,
            'perox_col': args.perox_col,
            'codontable': int(args.codontable),
            'model_arch': model_arch,
            'localize_strategy': localize_strategy,
            'localize_temperature_scale': bool(localize_temperature_scale),
            'localize_threshold_tune': bool(localize_threshold_tune),
            'localize_threshold_objective': str(localize_threshold_objective),
            'data_source': 'training_tsv' if has_training_tsv else 'uniprot_query',
            'uniprot_query': '' if has_training_tsv else str(resolved_uniprot_query),
            'uniprot_preset': '' if has_training_tsv else preset_name,
            'uniprot_reviewed': bool(uniprot_reviewed),
            'uniprot_exclude_fragments': bool(uniprot_exclude_fragments),
            'uniprot_sampling': str(uniprot_sampling),
            'uniprot_sampling_seed': int(uniprot_sampling_seed),
            'cv_fold_col': cv_fold_col,
            'cv_predefined_folds': bool(predefined_folds_active),
            'class_counts': dict(Counter(class_labels)),
            'perox_counts': dict(Counter(perox_labels)),
        },
    }
    metrics = calculate_training_metrics(
        x=x,
        aa_sequences=aa_sequences,
        class_labels=class_labels,
        perox_labels=perox_labels,
        model=model,
        model_arch=model_arch,
        dl_device=dl_device,
    )
    model['metadata']['class_train_accuracy_by_class'] = dict(metrics['class_accuracy_by_class'])
    cv_metrics = None
    postproc_metrics = None
    if effective_cv_folds >= 2:
        cv_metrics = evaluate_cross_validation(
            x=x,
            aa_sequences=aa_sequences,
            class_labels=class_labels,
            perox_labels=perox_labels,
            n_folds=effective_cv_folds,
            seed=cv_seed,
            model_arch=model_arch,
            dl_train_params=dl_train_params,
            dl_device=dl_device,
            localize_strategy=localize_strategy,
            fold_ids=fold_ids if predefined_folds_active else None,
        )
        model['metadata']['cv_folds'] = int(cv_metrics['n_folds'])
        model['metadata']['cv_seed'] = int(cv_seed)
        model['metadata']['cv_class_accuracy_mean'] = float(cv_metrics['class_accuracy_mean'])
        model['metadata']['cv_class_accuracy_std'] = float(cv_metrics['class_accuracy_std'])
        model['metadata']['cv_class_accuracy_by_class'] = dict(cv_metrics['class_accuracy_by_class'])
        model['metadata']['cv_perox_accuracy_mean'] = float(cv_metrics['perox_accuracy_mean'])
        model['metadata']['cv_perox_accuracy_std'] = float(cv_metrics['perox_accuracy_std'])
    else:
        model['metadata']['cv_folds'] = 0
        model['metadata']['cv_seed'] = int(cv_seed)
        model['metadata']['cv_class_accuracy_by_class'] = dict()
    if cv_metrics is not None and (localize_temperature_scale or localize_threshold_tune):
        oof_rows = cv_metrics.get('oof_rows', [])
        tuned_temperature = 1.0
        if localize_temperature_scale:
            tuned_temperature = fit_temperature_from_oof(oof_rows=oof_rows)
            localization_model['probability_calibration'] = {
                'method': 'temperature',
                'temperature': float(tuned_temperature),
            }
            model['metadata']['localize_temperature'] = float(tuned_temperature)
        tuned_thresholds = None
        if localize_threshold_tune:
            tuned_thresholds = optimize_class_thresholds_from_oof(
                oof_rows=oof_rows,
                temperature=tuned_temperature,
                objective=localize_threshold_objective,
            )
            localization_model['class_thresholds'] = dict(tuned_thresholds)
            model['metadata']['localize_class_thresholds'] = dict(tuned_thresholds)
        postproc_metrics = evaluate_oof_postprocess(
            oof_rows=oof_rows,
            temperature=tuned_temperature,
            class_thresholds=tuned_thresholds,
        )
        model['metadata']['cv_postproc_class_accuracy_overall'] = float(
            postproc_metrics['class_accuracy_overall']
        )
        model['metadata']['cv_postproc_class_accuracy_macro5'] = float(
            postproc_metrics['class_accuracy_macro5']
        )
        model['metadata']['cv_postproc_class_accuracy_by_class'] = dict(
            postproc_metrics['class_accuracy_by_class']
        )
        metrics = calculate_training_metrics(
            x=x,
            aa_sequences=aa_sequences,
            class_labels=class_labels,
            perox_labels=perox_labels,
            model=model,
            model_arch=model_arch,
            dl_device=dl_device,
        )
        model['metadata']['class_train_accuracy_by_class'] = dict(
            metrics['class_accuracy_by_class']
        )
    if model_arch == 'bilstm_attention':
        model['metadata']['dl_seq_len'] = int(dl_seq_len)
        model['metadata']['dl_embed_dim'] = int(dl_embed_dim)
        model['metadata']['dl_hidden_dim'] = int(dl_hidden_dim)
        model['metadata']['dl_num_layers'] = int(dl_num_layers)
        model['metadata']['dl_dropout'] = float(dl_dropout)
        model['metadata']['dl_epochs'] = int(dl_epochs)
        model['metadata']['dl_batch_size'] = int(dl_batch_size)
        model['metadata']['dl_lr'] = float(dl_lr)
        model['metadata']['dl_weight_decay'] = float(dl_weight_decay)
        model['metadata']['dl_class_weight'] = bool(dl_class_weight)
        model['metadata']['dl_loss'] = str(dl_loss)
        model['metadata']['dl_balanced_batch'] = bool(dl_balanced_batch)
        model['metadata']['dl_seed'] = int(dl_seed)
        model['metadata']['dl_device'] = str(dl_device)

    save_localize_model(model=model, path=args.model_out)
    report_rows = list()
    report_rows.append({
        'metric': 'num_training_rows',
        'value': int(len(rows)),
    })
    report_rows.append({
        'metric': 'num_used_rows',
        'value': int(x.shape[0]),
    })
    report_rows.append({
        'metric': 'num_skipped_rows',
        'value': int(skipped),
    })
    report_rows.append({
        'metric': 'class_train_accuracy',
        'value': float(metrics['class_train_accuracy']),
    })
    report_rows.append({
        'metric': 'perox_train_accuracy',
        'value': float(metrics['perox_train_accuracy']),
    })
    for class_name in LOCALIZATION_CLASSES:
        report_rows.append({
            'metric': 'class_train_accuracy_{}'.format(class_name),
            'value': float(metrics['class_accuracy_by_class'].get(class_name, 0.0)),
        })
    if cv_metrics is not None:
        report_rows.append({
            'metric': 'cv_folds',
            'value': int(cv_metrics['n_folds']),
        })
        report_rows.append({
            'metric': 'cv_class_accuracy_mean',
            'value': float(cv_metrics['class_accuracy_mean']),
        })
        report_rows.append({
            'metric': 'cv_class_accuracy_std',
            'value': float(cv_metrics['class_accuracy_std']),
        })
        report_rows.append({
            'metric': 'cv_perox_accuracy_mean',
            'value': float(cv_metrics['perox_accuracy_mean']),
        })
        report_rows.append({
            'metric': 'cv_perox_accuracy_std',
            'value': float(cv_metrics['perox_accuracy_std']),
        })
        for class_name in LOCALIZATION_CLASSES:
            report_rows.append({
                'metric': 'cv_class_accuracy_{}'.format(class_name),
                'value': float(cv_metrics['class_accuracy_by_class'].get(class_name, 0.0)),
            })
        for fold_row in cv_metrics['folds']:
            fold_id = int(fold_row['fold'])
            report_rows.append({
                'metric': 'cv_fold{}_class_accuracy'.format(fold_id),
                'value': float(fold_row['class_accuracy']),
            })
            report_rows.append({
                'metric': 'cv_fold{}_perox_accuracy'.format(fold_id),
                'value': float(fold_row['perox_accuracy']),
            })
    if postproc_metrics is not None:
        if localize_temperature_scale:
            report_rows.append({
                'metric': 'postproc_temperature',
                'value': float(model['metadata'].get('localize_temperature', 1.0)),
            })
        if localize_threshold_tune:
            saved_thresholds = model['metadata'].get('localize_class_thresholds', {})
            for class_name in LOCALIZATION_CLASSES:
                report_rows.append({
                    'metric': 'postproc_threshold_{}'.format(class_name),
                    'value': float(saved_thresholds.get(class_name, 1.0)),
                })
        report_rows.append({
            'metric': 'cv_postproc_class_accuracy_overall',
            'value': float(postproc_metrics['class_accuracy_overall']),
        })
        report_rows.append({
            'metric': 'cv_postproc_class_accuracy_macro5',
            'value': float(postproc_metrics['class_accuracy_macro5']),
        })
        report_rows.append({
            'metric': 'cv_postproc_fold_class_accuracy_mean',
            'value': float(postproc_metrics['fold_class_accuracy_mean']),
        })
        report_rows.append({
            'metric': 'cv_postproc_fold_class_accuracy_std',
            'value': float(postproc_metrics['fold_class_accuracy_std']),
        })
        for class_name in LOCALIZATION_CLASSES:
            report_rows.append({
                'metric': 'cv_postproc_class_accuracy_{}'.format(class_name),
                'value': float(postproc_metrics['class_accuracy_by_class'].get(class_name, 0.0)),
            })
        for fold_row in postproc_metrics['folds']:
            fold_id = int(fold_row['fold'])
            report_rows.append({
                'metric': 'cv_postproc_fold{}_class_accuracy'.format(fold_id),
                'value': float(fold_row['class_accuracy']),
            })
    for class_name in LOCALIZATION_CLASSES:
        report_rows.append({
            'metric': 'count_class_{}'.format(class_name),
            'value': int(model['metadata']['class_counts'].get(class_name, 0)),
        })
    report_rows.append({
        'metric': 'count_perox_yes',
        'value': int(model['metadata']['perox_counts'].get('yes', 0)),
    })
    report_rows.append({
        'metric': 'count_perox_no',
        'value': int(model['metadata']['perox_counts'].get('no', 0)),
    })

    if args.report != '':
        if args.report.endswith('.json'):
            write_rows_json(rows=report_rows, output_path=args.report)
        else:
            write_rows_tsv(
                rows=report_rows,
                output_path=args.report,
                fieldnames=['metric', 'value'],
            )

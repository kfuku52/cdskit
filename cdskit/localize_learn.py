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
    extract_localize_features,
    fit_nearest_centroid_classifier,
    fit_perox_binary_classifier,
    infer_labels_from_uniprot_cc,
    is_dna_like,
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

    feats, _ = extract_localize_features(aa_seq=aa_seq)
    return aa_seq, feats, class_label, perox_label


def build_training_matrix(
    rows,
    seq_col,
    seqtype,
    codontable,
    label_mode,
    localization_col,
    perox_col,
    skip_ambiguous,
):
    features = list()
    aa_sequences = list()
    class_labels = list()
    perox_labels = list()
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
        )
        if parsed is None:
            skipped += 1
            continue
        aa_seq, feat_vec, class_label, perox_label = parsed
        aa_sequences.append(aa_seq)
        features.append(feat_vec)
        class_labels.append(class_label)
        perox_labels.append(perox_label)
    if len(features) == 0:
        raise ValueError('No valid training sample remained after filtering.')
    x = np.asarray(features, dtype=np.float64)
    return x, aa_sequences, class_labels, perox_labels, skipped


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
        if model_arch == 'nearest_centroid':
            pred = predict_localization_and_peroxisome_from_features(
                feature_vec=x[i, :],
                model=model,
            )
        elif model_arch == 'bilstm_attention':
            pred = predict_localization_and_peroxisome_from_sequence(
                aa_seq=aa_sequences[i],
                feature_vec=x[i, :],
                model=model,
                device=dl_device,
            )
        else:
            raise ValueError('Unsupported model_arch: {}'.format(model_arch))
        true_class = class_labels[i]
        true_perox = perox_labels[i]
        class_total[true_class] = class_total.get(true_class, 0) + 1
        if pred['predicted_class'] == true_class:
            correct_class += 1
            class_correct[true_class] = class_correct.get(true_class, 0) + 1
        pred_perox = 'yes' if pred['p_peroxisome'] >= 0.5 else 'no'
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


def build_stratified_folds(class_labels, n_folds, seed):
    if n_folds < 2:
        raise ValueError('--cv_folds should be >= 2 when cross validation is enabled.')
    labels = list(class_labels)
    n_sample = len(labels)
    if n_sample < n_folds:
        txt = '--cv_folds ({}) should be <= number of samples ({}).'
        raise ValueError(txt.format(n_folds, n_sample))

    counts = Counter(labels)
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
):
    folds = build_stratified_folds(
        class_labels=class_labels,
        n_folds=n_folds,
        seed=seed,
    )
    n_sample = int(x.shape[0])
    fold_rows = list()
    class_accs = list()
    perox_accs = list()
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

        if model_arch == 'nearest_centroid':
            local_model = fit_nearest_centroid_classifier(
                features=x_train,
                labels=class_train,
                class_order=LOCALIZATION_CLASSES,
            )
        elif model_arch == 'bilstm_attention':
            from cdskit.localize_bilstm import fit_bilstm_attention_classifier
            local_model = fit_bilstm_attention_classifier(
                aa_sequences=aa_train,
                labels=class_train,
                class_order=LOCALIZATION_CLASSES,
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
            )
        else:
            raise ValueError('Unsupported model_arch: {}'.format(model_arch))
        perox_model = fit_perox_binary_classifier(
            features=x_train,
            labels=perox_train,
        )
        tmp_model = {
            'localization_model': local_model,
            'perox_model': perox_model,
        }

        class_correct_fold = 0
        perox_correct = 0
        for row_i in range(x_test.shape[0]):
            if model_arch == 'nearest_centroid':
                pred = predict_localization_and_peroxisome_from_features(
                    feature_vec=x_test[row_i, :],
                    model=tmp_model,
                )
            else:
                pred = predict_localization_and_peroxisome_from_sequence(
                    aa_seq=aa_test[row_i],
                    feature_vec=x_test[row_i, :],
                    model=tmp_model,
                    device=dl_device,
                )
            true_class = class_test[row_i]
            pred_class = pred['predicted_class']
            class_total[true_class] = class_total.get(true_class, 0) + 1
            if pred_class == true_class:
                class_correct_fold += 1
                class_correct_by_class[true_class] = class_correct_by_class.get(true_class, 0) + 1
            pred_perox = 'yes' if pred['p_peroxisome'] >= 0.5 else 'no'
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
        'n_folds': int(n_folds),
        'class_accuracy_mean': float(class_arr.mean()),
        'class_accuracy_std': float(class_arr.std()),
        'class_accuracy_by_class': class_accuracy_by_class,
        'perox_accuracy_mean': float(perox_arr.mean()),
        'perox_accuracy_std': float(perox_arr.std()),
        'folds': fold_rows,
    }


def predict_localization_and_peroxisome_from_features(feature_vec, model):
    from cdskit.localize_model import predict_nearest_centroid, predict_perox

    pred_class, class_probs = predict_nearest_centroid(
        feature_vec=feature_vec,
        model=model['localization_model'],
    )
    _, perox_probs = predict_perox(
        feature_vec=feature_vec,
        perox_model=model['perox_model'],
    )
    return {
        'predicted_class': pred_class,
        'class_probabilities': class_probs,
        'p_peroxisome': float(perox_probs.get('yes', 0.0)),
    }


def predict_localization_and_peroxisome_from_sequence(aa_seq, feature_vec, model, device='cpu'):
    from cdskit.localize_bilstm import predict_bilstm_attention
    from cdskit.localize_model import predict_perox

    pred_class, class_probs = predict_bilstm_attention(
        aa_seq=aa_seq,
        localization_model=model['localization_model'],
        device=device,
    )
    _, perox_probs = predict_perox(
        feature_vec=feature_vec,
        perox_model=model['perox_model'],
    )
    return {
        'predicted_class': pred_class,
        'class_probabilities': class_probs,
        'p_peroxisome': float(perox_probs.get('yes', 0.0)),
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
    model_arch = str(getattr(args, 'model_arch', 'nearest_centroid')).strip().lower()
    if model_arch not in ['nearest_centroid', 'bilstm_attention']:
        raise ValueError('Unsupported --model_arch: {}'.format(model_arch))

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
    if model_arch == 'bilstm_attention':
        from cdskit.localize_bilstm import require_torch
        require_torch()
    if cv_folds < 0:
        raise ValueError('--cv_folds should be >= 0.')
    if cv_folds == 1:
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

    x, aa_sequences, class_labels, perox_labels, skipped = build_training_matrix(
        rows=rows,
        seq_col=args.seq_col,
        seqtype=args.seqtype,
        codontable=args.codontable,
        label_mode=args.label_mode,
        localization_col=args.localization_col,
        perox_col=args.perox_col,
        skip_ambiguous=args.skip_ambiguous,
    )

    if model_arch == 'nearest_centroid':
        localization_model = fit_nearest_centroid_classifier(
            features=x,
            labels=class_labels,
            class_order=LOCALIZATION_CLASSES,
        )
        model_type = 'nearest_centroid_v1'
    elif model_arch == 'bilstm_attention':
        from cdskit.localize_bilstm import fit_bilstm_attention_classifier
        localization_model = fit_bilstm_attention_classifier(
            aa_sequences=aa_sequences,
            labels=class_labels,
            class_order=LOCALIZATION_CLASSES,
            seq_len=dl_seq_len,
            embed_dim=dl_embed_dim,
            hidden_dim=dl_hidden_dim,
            num_layers=dl_num_layers,
            dropout=dl_dropout,
            epochs=dl_epochs,
            batch_size=dl_batch_size,
            learning_rate=dl_lr,
            weight_decay=dl_weight_decay,
            seed=dl_seed,
            use_class_weight=dl_class_weight,
            device=dl_device,
        )
        model_type = 'bilstm_attention_v1'
    else:
        raise ValueError('Unsupported --model_arch: {}'.format(model_arch))
    perox_model = fit_perox_binary_classifier(
        features=x,
        labels=perox_labels,
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
    }
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
            'data_source': 'training_tsv' if has_training_tsv else 'uniprot_query',
            'uniprot_query': '' if has_training_tsv else str(resolved_uniprot_query),
            'uniprot_preset': '' if has_training_tsv else preset_name,
            'uniprot_reviewed': bool(uniprot_reviewed),
            'uniprot_exclude_fragments': bool(uniprot_exclude_fragments),
            'uniprot_sampling': str(uniprot_sampling),
            'uniprot_sampling_seed': int(uniprot_sampling_seed),
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
    if cv_folds >= 2:
        cv_metrics = evaluate_cross_validation(
            x=x,
            aa_sequences=aa_sequences,
            class_labels=class_labels,
            perox_labels=perox_labels,
            n_folds=cv_folds,
            seed=cv_seed,
            model_arch=model_arch,
            dl_train_params=dl_train_params,
            dl_device=dl_device,
        )
        model['metadata']['cv_folds'] = int(cv_folds)
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

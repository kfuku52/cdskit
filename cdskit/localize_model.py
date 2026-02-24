import csv
import json
import math
import re

import numpy as np

from cdskit.translate import translate_sequence_string
from cdskit.util import DNA_ALLOWED_CHARS

LOCALIZATION_CLASSES = ('noTP', 'SP', 'mTP', 'cTP', 'lTP')
TP_STAGE_CLASSES = ('SP', 'mTP', 'cTP', 'lTP')

AA_HYDROPHOBIC = frozenset('AILMFWVY')
AA_BASIC = frozenset('KRH')
AA_BASIC_STRICT = frozenset('KR')
AA_ACIDIC = frozenset('DE')
AA_SER_THR = frozenset('ST')
AA_AROMATIC = frozenset('FWY')
AA_SMALL = frozenset('AGSTCP')

AA_HYDROPATHY = {
    'A': 1.8,
    'C': 2.5,
    'D': -3.5,
    'E': -3.5,
    'F': 2.8,
    'G': -0.4,
    'H': -3.2,
    'I': 4.5,
    'K': -3.9,
    'L': 3.8,
    'M': 1.9,
    'N': -3.5,
    'P': -1.6,
    'Q': -3.5,
    'R': -4.5,
    'S': -0.8,
    'T': -0.7,
    'V': 4.2,
    'W': -0.9,
    'Y': -1.3,
}

PTS1_REGEX = re.compile(r'[ASNCGTP][KRHQ][LIVMF]$')
PTS2_REGEX = re.compile(r'[RK][LIVQ].{4}[HQ][LA]')
LTP_RR_HYDRO_REGEX = re.compile(r'RR.{0,12}[AILMFWVY]{5,}')

FEATURE_NAMES = [
    'aa_len',
    'n20_basic_frac',
    'n20_acidic_frac',
    'n20_hydrophobic_frac',
    'n20_ser_thr_frac',
    'n20_pro_frac',
    'n40_basic_frac',
    'n40_acidic_frac',
    'n40_hydrophobic_frac',
    'n40_ser_thr_frac',
    'n40_arg_frac',
    'n40_lys_frac',
    'n40_ala_frac',
    'n40_pro_frac',
    'n40_gly_frac',
    'n40_aromatic_frac',
    'n40_hydropathy_mean',
    'n40_hydrophobic_run',
    'signal_peptide_like',
    'rr_motif',
    'rr_hydrophobic_after',
    'cleavage_zone_small_frac',
    'global_basic_frac',
    'global_acidic_frac',
    'global_hydrophobic_frac',
    'global_ser_thr_frac',
    'global_x_frac',
    'c10_basic_frac',
    'c10_acidic_frac',
    'c10_hydrophobic_frac',
    'pts1_match',
    'pts1_skl',
    'pts2_match',
]


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.size == 0:
        return logits
    max_logit = float(np.max(logits))
    shifted = logits - max_logit
    exp_vals = np.exp(shifted)
    denom = float(np.sum(exp_vals))
    if denom <= 0:
        return np.zeros_like(exp_vals)
    return exp_vals / denom


def _sanitize_probability(value):
    try:
        out = float(value)
    except Exception:
        return 0.0
    if (not np.isfinite(out)) or (out < 0.0):
        return 0.0
    return out


def normalize_class_probabilities(class_probs):
    out_probs = {class_name: 0.0 for class_name in LOCALIZATION_CLASSES}
    if isinstance(class_probs, dict):
        for class_name in LOCALIZATION_CLASSES:
            out_probs[class_name] = _sanitize_probability(class_probs.get(class_name, 0.0))
    total = float(sum(out_probs.values()))
    if total <= 0.0:
        out_probs['noTP'] = 1.0
        return out_probs
    for class_name in LOCALIZATION_CLASSES:
        out_probs[class_name] = out_probs[class_name] / total
    return out_probs


def apply_temperature_scaling(class_probs, temperature):
    probs = normalize_class_probabilities(class_probs=class_probs)
    try:
        temp = float(temperature)
    except Exception:
        temp = 1.0
    if (not np.isfinite(temp)) or (temp <= 0.0) or (abs(temp - 1.0) < 1.0e-12):
        return probs
    vec = np.asarray([probs[class_name] for class_name in LOCALIZATION_CLASSES], dtype=np.float64)
    vec = np.clip(vec, 1.0e-12, 1.0)
    logits = np.log(vec) / temp
    scaled = softmax(logits)
    return {LOCALIZATION_CLASSES[i]: float(scaled[i]) for i in range(len(LOCALIZATION_CLASSES))}


def _predict_class_with_thresholds(class_probs, class_thresholds):
    probs = normalize_class_probabilities(class_probs=class_probs)
    scores = list()
    for class_name in LOCALIZATION_CLASSES:
        threshold = 1.0
        if isinstance(class_thresholds, dict):
            threshold = class_thresholds.get(class_name, 1.0)
        try:
            threshold = float(threshold)
        except Exception:
            threshold = 1.0
        if (not np.isfinite(threshold)) or (threshold <= 0.0):
            threshold = 1.0
        scores.append(float(probs[class_name]) / threshold)
    pred_idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    return LOCALIZATION_CLASSES[pred_idx], probs


def postprocess_localization_probabilities(class_probs, localization_model):
    probs = normalize_class_probabilities(class_probs=class_probs)
    calibration = localization_model.get('probability_calibration', {})
    if isinstance(calibration, dict):
        method = str(calibration.get('method', '')).strip().lower()
        if method == 'temperature':
            probs = apply_temperature_scaling(
                class_probs=probs,
                temperature=calibration.get('temperature', 1.0),
            )
    pred_class, probs = _predict_class_with_thresholds(
        class_probs=probs,
        class_thresholds=localization_model.get('class_thresholds', None),
    )
    return pred_class, probs


def fraction_in_set(seq, chars):
    if len(seq) == 0:
        return 0.0
    count = 0
    for ch in seq:
        if ch in chars:
            count += 1
    return float(count) / float(len(seq))


def mean_hydropathy(seq):
    if len(seq) == 0:
        return 0.0
    total = 0.0
    for ch in seq:
        total += AA_HYDROPATHY.get(ch, 0.0)
    return total / float(len(seq))


def longest_hydrophobic_run(seq):
    best = 0
    current = 0
    for ch in seq:
        if ch in AA_HYDROPHOBIC:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return float(best)


def to_canonical_aa_sequence(aa_seq):
    aa_seq = str(aa_seq).upper().replace(' ', '')
    if aa_seq.endswith('*'):
        aa_seq = aa_seq[:-1]
    if '*' in aa_seq:
        raise Exception('Internal stop codon detected in translated peptide sequence. Exiting.')
    out = list()
    for ch in aa_seq:
        if ch in AA_HYDROPATHY:
            out.append(ch)
        elif ch == 'X':
            out.append('X')
        elif ch in '-?.':
            out.append('X')
        else:
            out.append('X')
    return ''.join(out)


def translate_inframe_cds_to_aa(cds_seq, codontable, seq_id=''):
    seq_upper = str(cds_seq).upper()
    if len(seq_upper) % 3 != 0:
        txt = 'Sequence length is not multiple of three: {}. Exiting.'
        raise Exception(txt.format(seq_id))
    translated = translate_sequence_string(
        seq_str=seq_upper,
        codontable=codontable,
        to_stop=False,
    )
    if translated.endswith('*'):
        translated = translated[:-1]
    if '*' in translated:
        txt = 'Internal stop codon was detected: {}. Exiting.'
        raise Exception(txt.format(seq_id))
    return to_canonical_aa_sequence(translated)


def is_dna_like(seq):
    if len(seq) == 0:
        return False
    for ch in seq:
        if ch not in DNA_ALLOWED_CHARS:
            return False
    return True


def detect_perox_signals(aa_seq):
    seq = aa_seq
    tail3 = seq[-3:] if len(seq) >= 3 else ''
    n40 = seq[:40]
    pts1_match = bool(PTS1_REGEX.search(seq))
    pts1_skl = (tail3 == 'SKL')
    pts2_match = bool(PTS2_REGEX.search(n40))
    if pts1_match:
        signal_type = 'PTS1'
    elif pts2_match:
        signal_type = 'PTS2'
    else:
        signal_type = 'none'
    return {
        'pts1_match': pts1_match,
        'pts1_skl': pts1_skl,
        'pts2_match': pts2_match,
        'signal_type': signal_type,
    }


def extract_localize_features(aa_seq):
    seq = to_canonical_aa_sequence(aa_seq)
    n20 = seq[:20]
    n40 = seq[:40]
    n60 = seq[:60]
    c10 = seq[-10:] if len(seq) >= 10 else seq
    cleavage_zone = seq[18:35]
    perox = detect_perox_signals(seq)

    signal_window = seq[6:30] if len(seq) > 6 else ''
    signal_peptide_like = 1.0 if longest_hydrophobic_run(signal_window) >= 7 else 0.0
    rr_motif = 1.0 if ('RR' in n40) else 0.0
    rr_hydrophobic_after = 1.0 if LTP_RR_HYDRO_REGEX.search(n60) else 0.0

    feats = np.array([
        float(len(seq)),
        fraction_in_set(n20, AA_BASIC),
        fraction_in_set(n20, AA_ACIDIC),
        fraction_in_set(n20, AA_HYDROPHOBIC),
        fraction_in_set(n20, AA_SER_THR),
        fraction_in_set(n20, frozenset('P')),
        fraction_in_set(n40, AA_BASIC),
        fraction_in_set(n40, AA_ACIDIC),
        fraction_in_set(n40, AA_HYDROPHOBIC),
        fraction_in_set(n40, AA_SER_THR),
        fraction_in_set(n40, frozenset('R')),
        fraction_in_set(n40, frozenset('K')),
        fraction_in_set(n40, frozenset('A')),
        fraction_in_set(n40, frozenset('P')),
        fraction_in_set(n40, frozenset('G')),
        fraction_in_set(n40, AA_AROMATIC),
        mean_hydropathy(n40),
        longest_hydrophobic_run(n40),
        signal_peptide_like,
        rr_motif,
        rr_hydrophobic_after,
        fraction_in_set(cleavage_zone, AA_SMALL),
        fraction_in_set(seq, AA_BASIC),
        fraction_in_set(seq, AA_ACIDIC),
        fraction_in_set(seq, AA_HYDROPHOBIC),
        fraction_in_set(seq, AA_SER_THR),
        fraction_in_set(seq, frozenset('X')),
        fraction_in_set(c10, AA_BASIC_STRICT),
        fraction_in_set(c10, AA_ACIDIC),
        fraction_in_set(c10, AA_HYDROPHOBIC),
        1.0 if perox['pts1_match'] else 0.0,
        1.0 if perox['pts1_skl'] else 0.0,
        1.0 if perox['pts2_match'] else 0.0,
    ], dtype=np.float64)
    return feats, perox


def normalize_localization_label(label):
    if label is None:
        raise ValueError('Missing localization label.')
    txt = str(label).strip().lower()
    mapping = {
        'notp': 'noTP',
        'none': 'noTP',
        'cytosol': 'noTP',
        'cytoplasm': 'noTP',
        'sptp': 'SP',
        'sp': 'SP',
        'signalpeptide': 'SP',
        'secreted': 'SP',
        'mtp': 'mTP',
        'mitochondria': 'mTP',
        'mitochondrion': 'mTP',
        'ctp': 'cTP',
        'chloroplast': 'cTP',
        'plastid': 'cTP',
        'ltp': 'lTP',
        'thylakoid': 'lTP',
        'lumen': 'lTP',
    }
    key = re.sub(r'[\s_\-]+', '', txt)
    out = mapping.get(key)
    if out is None:
        raise ValueError('Unsupported localization label: {}'.format(label))
    return out


def normalize_yes_no(value, default='no'):
    if value is None:
        return default
    txt = str(value).strip().lower()
    if txt in ['', 'na', 'nan', 'none']:
        return default
    if txt in ['1', 'y', 'yes', 'true', 't', 'peroxisome', 'peroxisomal']:
        return 'yes'
    if txt in ['0', 'n', 'no', 'false', 'f', 'non-peroxisomal', 'not_peroxisomal']:
        return 'no'
    raise ValueError('Unsupported yes/no value: {}'.format(value))


def infer_labels_from_uniprot_cc(location_text):
    txt = str(location_text or '').lower()
    has_sp = ('secreted' in txt) or ('signal peptide' in txt)
    has_mtp = ('mitochond' in txt)
    has_ctp = ('chloroplast' in txt) or ('plastid' in txt)
    has_ltp = ('thylakoid' in txt) or ('lumen' in txt)
    has_perox = ('peroxisom' in txt)

    if has_ltp:
        class_label = 'lTP'
        active = [has_sp, has_mtp]
    elif has_ctp:
        class_label = 'cTP'
        active = [has_sp, has_mtp]
    elif has_mtp:
        class_label = 'mTP'
        active = [has_sp]
    elif has_sp:
        class_label = 'SP'
        active = []
    else:
        class_label = 'noTP'
        active = []

    ambiguous = sum(1 for v in [has_sp, has_mtp, has_ctp, has_ltp] if v) > 1
    if class_label in ['lTP', 'cTP'] and has_ltp and has_ctp:
        # "thylakoid/chloroplast" co-annotations are common and biologically compatible.
        ambiguous = sum(1 for v in active if v) > 0

    perox_label = 'yes' if has_perox else 'no'
    return class_label, perox_label, ambiguous


def safe_log(x):
    if x <= 0:
        return -1.0e9
    return math.log(x)


def fit_nearest_centroid_classifier(features, labels, class_order):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError('Feature matrix should be 2D.')
    if x.shape[0] == 0:
        raise ValueError('No training samples were provided.')

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    z = (x - mean) / std

    labels = list(labels)
    centroids = list()
    priors = list()
    total = float(len(labels))
    for class_name in class_order:
        indices = [i for i, lab in enumerate(labels) if lab == class_name]
        if len(indices) == 0:
            raise ValueError('No training sample for class: {}'.format(class_name))
        class_z = z[indices, :]
        centroids.append(class_z.mean(axis=0))
        priors.append((len(indices) + 1.0) / (total + float(len(class_order))))
    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'class_order': list(class_order),
        'centroids': np.asarray(centroids, dtype=np.float64).tolist(),
        'log_priors': [safe_log(p) for p in priors],
    }


def fit_perox_binary_classifier(features, labels):
    labels = [normalize_yes_no(v, default='no') for v in labels]
    unique = sorted(set(labels))
    if len(unique) == 1:
        yes_fraction = 1.0 if unique[0] == 'yes' else 0.0
        return {
            'mode': 'constant',
            'yes_probability': yes_fraction,
        }
    model = fit_nearest_centroid_classifier(
        features=features,
        labels=labels,
        class_order=('no', 'yes'),
    )
    model['mode'] = 'centroid'
    return model


def predict_nearest_centroid(feature_vec, model):
    mean = np.asarray(model['mean'], dtype=np.float64)
    std = np.asarray(model['std'], dtype=np.float64)
    centroids = np.asarray(model['centroids'], dtype=np.float64)
    log_priors = np.asarray(model['log_priors'], dtype=np.float64)
    class_order = list(model['class_order'])

    x = np.asarray(feature_vec, dtype=np.float64)
    z = (x - mean) / std
    diff = centroids - z
    sq_dist = np.sum(diff * diff, axis=1)
    logits = (-0.5 * sq_dist) + log_priors
    probs = softmax(logits)
    pred_index = int(np.argmax(probs))
    pred_label = class_order[pred_index]
    out_probs = {class_order[i]: float(probs[i]) for i in range(len(class_order))}
    return pred_label, out_probs


def predict_perox(feature_vec, perox_model):
    if perox_model.get('mode') == 'constant':
        p_yes = float(perox_model['yes_probability'])
        if p_yes < 0:
            p_yes = 0.0
        if p_yes > 1:
            p_yes = 1.0
        return ('yes' if p_yes >= 0.5 else 'no', {'yes': p_yes, 'no': 1.0 - p_yes})
    pred, probs = predict_nearest_centroid(feature_vec=feature_vec, model=perox_model)
    return pred, probs


def _predict_constant_localization(localization_model):
    class_order = list(localization_model.get('class_order', []))
    class_label = str(localization_model.get('class_label', '')).strip()
    if class_label == '':
        if len(class_order) == 1:
            class_label = class_order[0]
        else:
            raise ValueError('Constant localization model is missing class_label.')
    if len(class_order) == 0:
        class_order = [class_label]
    if class_label not in class_order:
        class_order.append(class_label)
    probs = {name: 0.0 for name in class_order}
    probs[class_label] = 1.0
    return class_label, probs


def _predict_localization_from_model(aa_seq, feature_vec, localization_model, model_type):
    if str(localization_model.get('mode', '')).strip().lower() == 'constant':
        return _predict_constant_localization(localization_model=localization_model)
    if model_type == 'nearest_centroid_v1':
        return predict_nearest_centroid(
            feature_vec=feature_vec,
            model=localization_model,
        )
    if model_type == 'bilstm_attention_v1':
        from cdskit.localize_bilstm import predict_bilstm_attention
        return predict_bilstm_attention(
            aa_seq=aa_seq,
            localization_model=localization_model,
            device='cpu',
        )
    raise ValueError('Unsupported model_type: {}'.format(model_type))


def predict_two_stage_localization(aa_seq, feature_vec, localization_model, model_type):
    stage1_model = localization_model.get('stage1_model', {})
    stage2_model = localization_model.get('stage2_model', {})
    if (not isinstance(stage1_model, dict)) or (not isinstance(stage2_model, dict)):
        raise ValueError('Invalid two-stage localization model payload.')

    _, stage1_probs = _predict_localization_from_model(
        aa_seq=aa_seq,
        feature_vec=feature_vec,
        localization_model=stage1_model,
        model_type=model_type,
    )
    _, stage2_probs = _predict_localization_from_model(
        aa_seq=aa_seq,
        feature_vec=feature_vec,
        localization_model=stage2_model,
        model_type=model_type,
    )

    out_probs = {class_name: 0.0 for class_name in LOCALIZATION_CLASSES}
    p_no_tp = float(stage1_probs.get('noTP', 0.0))
    p_tp = float(stage1_probs.get('TP', max(0.0, 1.0 - p_no_tp)))
    out_probs['noTP'] = p_no_tp
    for class_name in TP_STAGE_CLASSES:
        out_probs[class_name] = p_tp * float(stage2_probs.get(class_name, 0.0))

    total = float(sum(out_probs.values()))
    if total <= 0.0:
        out_probs = {class_name: 0.0 for class_name in LOCALIZATION_CLASSES}
        out_probs['noTP'] = 1.0
    else:
        for class_name in LOCALIZATION_CLASSES:
            out_probs[class_name] = out_probs[class_name] / total

    pred_idx = int(np.argmax([out_probs[class_name] for class_name in LOCALIZATION_CLASSES]))
    pred_class = LOCALIZATION_CLASSES[pred_idx]
    return pred_class, out_probs


def predict_localization_and_peroxisome(aa_seq, model):
    feats, perox_signals = extract_localize_features(aa_seq=aa_seq)
    model_type = str(model.get('model_type', ''))
    localization_model = model['localization_model']
    localization_strategy = str(localization_model.get('strategy', 'single_stage')).strip().lower()
    if localization_strategy == 'two_stage':
        _, class_probs = predict_two_stage_localization(
            aa_seq=aa_seq,
            feature_vec=feats,
            localization_model=localization_model,
            model_type=model_type,
        )
    else:
        _, class_probs = _predict_localization_from_model(
            aa_seq=aa_seq,
            feature_vec=feats,
            localization_model=localization_model,
            model_type=model_type,
        )
    pred_class, class_probs = postprocess_localization_probabilities(
        class_probs=class_probs,
        localization_model=localization_model,
    )
    _, perox_probs = predict_perox(
        feature_vec=feats,
        perox_model=model['perox_model'],
    )
    return {
        'predicted_class': pred_class,
        'class_probabilities': class_probs,
        'perox_probability_yes': float(perox_probs.get('yes', 0.0)),
        'perox_signal_type': perox_signals['signal_type'],
        'feature_values': feats,
        'feature_names': list(FEATURE_NAMES),
        'pts1_match': bool(perox_signals['pts1_match']),
        'pts2_match': bool(perox_signals['pts2_match']),
    }


def _strip_runtime_caches(value):
    if isinstance(value, dict):
        out = dict()
        for key, val in value.items():
            if key == '_runtime_model_cache':
                continue
            out[key] = _strip_runtime_caches(val)
        return out
    if isinstance(value, list):
        return [_strip_runtime_caches(v) for v in value]
    return value


def save_localize_model(model, path):
    model_type = str(model.get('model_type', ''))
    if model_type == 'nearest_centroid_v1':
        with open(path, 'w', encoding='utf-8') as out:
            json.dump(model, out, indent=2, sort_keys=True)
        return
    if model_type == 'bilstm_attention_v1':
        from cdskit.localize_bilstm import require_torch
        torch, _ = require_torch()
        to_save = _strip_runtime_caches(dict(model))
        torch.save({'model': to_save}, path)
        return
    raise ValueError('Unsupported model_type: {}'.format(model_type))


def load_localize_model(path):
    model = None
    json_error = None
    try:
        with open(path, 'r', encoding='utf-8') as inp:
            model = json.load(inp)
    except Exception as exc:
        json_error = exc

    if model is None:
        try:
            from cdskit.localize_bilstm import require_torch
            torch, _ = require_torch()
            payload = torch.load(path, map_location='cpu')
            if isinstance(payload, dict) and ('model' in payload):
                model = payload['model']
            elif isinstance(payload, dict):
                model = payload
            else:
                raise ValueError('Unsupported model payload type.')
        except Exception as exc:
            txt = 'Failed to load model from {}. json_error={}, torch_error={}'
            raise ValueError(txt.format(path, str(json_error), str(exc)))

    required = ['model_type', 'localization_model', 'perox_model', 'feature_names']
    for key in required:
        if key not in model:
            raise ValueError('Invalid model file. Missing key: {}'.format(key))
    if model['model_type'] not in ['nearest_centroid_v1', 'bilstm_attention_v1']:
        raise ValueError('Unsupported model_type: {}'.format(model['model_type']))
    return model


def write_rows_tsv(rows, output_path, fieldnames):
    if output_path == '-':
        import sys
        out = sys.stdout
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return
    with open(output_path, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_rows_json(rows, output_path):
    if output_path == '-':
        import sys
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write('\n')
        return
    with open(output_path, 'w', encoding='utf-8') as out:
        json.dump(rows, out, indent=2)

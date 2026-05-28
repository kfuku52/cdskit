import csv
import json
import math
import re

import numpy as np

from cdskit.translate import translate_sequence_string
from cdskit.util import DNA_ALLOWED_CHARS

LOCALIZATION_CLASSES = ('noTP', 'SP', 'mTP', 'cTP', 'lTP')
TP_STAGE_CLASSES = ('SP', 'mTP', 'cTP', 'lTP')
CTP_LTP_STAGE_CLASSES = ('cTP', 'lTP')
SUBCELLULAR_LOCALIZATION_CLASSES = (
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
NLS_BASIC_CLUSTER_REGEX = re.compile(r'[KR]{3,}')
NLS_BIPARTITE_REGEX = re.compile(r'[KR]{2}.{8,12}[KR]{3,}')
NES_LIKE_REGEX = re.compile(r'[LIVMF].{1,4}[LIVMF].{1,4}[LIVMF].{1,4}[LIVMF]')

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

BROAD_FEATURE_NAMES = FEATURE_NAMES + [
    'n60_basic_frac',
    'n60_acidic_frac',
    'n60_hydrophobic_frac',
    'n60_ser_thr_frac',
    'n100_basic_frac',
    'n100_acidic_frac',
    'n100_hydrophobic_frac',
    'n100_ser_thr_frac',
    'global_hydropathy_mean',
    'global_hydrophobic_run',
    'hydrophobic_segment_count',
    'nls_basic_cluster_count',
    'nls_basic_cluster_match',
    'nls_bipartite_match',
    'nes_like_match',
    'er_kdel_hdel',
    'er_kkxx',
    'gpi_like_ctail',
    'c20_hydrophobic_run',
    'c20_basic_frac',
    'c20_acidic_frac',
    'kingdom_metazoa',
    'kingdom_viridiplantae',
    'kingdom_fungi',
    'kingdom_other',
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


def normalize_organism_group(value):
    txt = str(value or '').strip().lower()
    txt = re.sub(r'[\s\-]+', '_', txt)
    mapping = {
        '': '',
        'unknown': '',
        'auto': '',
        'plant': 'plant',
        'plants': 'plant',
        'viridiplantae': 'plant',
        'nonplant': 'non_plant',
        'non_plant': 'non_plant',
        'non_plants': 'non_plant',
        'other': 'non_plant',
        'metazoa': 'non_plant',
        'fungi': 'non_plant',
        'animal': 'non_plant',
        'animals': 'non_plant',
    }
    if txt in mapping:
        return mapping[txt]
    raise ValueError('Unsupported organism_group: {}'.format(value))


def apply_organism_group_constraints(class_probs, organism_group=''):
    group = normalize_organism_group(organism_group)
    probs = normalize_class_probabilities(class_probs=class_probs)
    if group == 'non_plant':
        probs['cTP'] = 0.0
        probs['lTP'] = 0.0
        probs = normalize_class_probabilities(class_probs=probs)
    return probs


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


def hydrophobic_segment_count(seq, min_run=17):
    count = 0
    current = 0
    for ch in seq:
        if ch in AA_HYDROPHOBIC:
            current += 1
        else:
            if current >= int(min_run):
                count += 1
            current = 0
    if current >= int(min_run):
        count += 1
    return float(count)


def _kingdom_feature_flags(kingdom):
    txt = str(kingdom or '').strip().lower()
    txt = re.sub(r'[\s\-]+', '_', txt)
    is_metazoa = txt in ['metazoa', 'animal', 'animals'] or ('metazoa' in txt)
    is_plant = (
        txt in ['plant', 'plants', 'viridiplantae', 'plantae']
        or ('viridiplantae' in txt)
        or ('plantae' in txt)
    )
    is_fungi = txt in ['fungi', 'fungus'] or ('fungi' in txt)
    has_known = is_metazoa or is_plant or is_fungi
    is_other = (txt not in ['', 'unknown', 'auto']) and (not has_known)
    if txt in ['non_plant', 'nonplant', 'other']:
        is_other = True
    return [
        1.0 if is_metazoa else 0.0,
        1.0 if is_plant else 0.0,
        1.0 if is_fungi else 0.0,
        1.0 if is_other else 0.0,
    ]


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


def extract_broad_localize_features(aa_seq, kingdom=''):
    seq = to_canonical_aa_sequence(aa_seq)
    base_feats, perox = extract_localize_features(aa_seq=seq)
    n60 = seq[:60]
    n100 = seq[:100]
    c20 = seq[-20:] if len(seq) >= 20 else seq
    c40 = seq[-40:] if len(seq) >= 40 else seq
    tail4 = seq[-4:] if len(seq) >= 4 else ''

    nls_count = float(len(NLS_BASIC_CLUSTER_REGEX.findall(seq)))
    extra_feats = np.array([
        fraction_in_set(n60, AA_BASIC),
        fraction_in_set(n60, AA_ACIDIC),
        fraction_in_set(n60, AA_HYDROPHOBIC),
        fraction_in_set(n60, AA_SER_THR),
        fraction_in_set(n100, AA_BASIC),
        fraction_in_set(n100, AA_ACIDIC),
        fraction_in_set(n100, AA_HYDROPHOBIC),
        fraction_in_set(n100, AA_SER_THR),
        mean_hydropathy(seq),
        longest_hydrophobic_run(seq),
        hydrophobic_segment_count(seq),
        nls_count,
        1.0 if nls_count > 0.0 else 0.0,
        1.0 if NLS_BIPARTITE_REGEX.search(seq) else 0.0,
        1.0 if NES_LIKE_REGEX.search(seq) else 0.0,
        1.0 if seq.endswith('KDEL') or seq.endswith('HDEL') else 0.0,
        1.0 if (len(tail4) == 4 and tail4[0] == 'K' and tail4[1] == 'K') else 0.0,
        1.0 if longest_hydrophobic_run(c40) >= 12.0 else 0.0,
        longest_hydrophobic_run(c20),
        fraction_in_set(c20, AA_BASIC_STRICT),
        fraction_in_set(c20, AA_ACIDIC),
    ] + _kingdom_feature_flags(kingdom=kingdom), dtype=np.float64)
    return np.concatenate([base_feats, extra_feats]), perox


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


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _binary_fbeta_from_predictions(true_binary, pred_binary, beta=1.0):
    true_binary = np.asarray(true_binary, dtype=np.int64)
    pred_binary = np.asarray(pred_binary, dtype=np.int64)
    tp = int(np.sum((true_binary == 1) & (pred_binary == 1)))
    fp = int(np.sum((true_binary == 0) & (pred_binary == 1)))
    fn = int(np.sum((true_binary == 1) & (pred_binary == 0)))
    precision = 0.0 if (tp + fp) <= 0 else float(tp) / float(tp + fp)
    recall = 0.0 if (tp + fn) <= 0 else float(tp) / float(tp + fn)
    if precision + recall <= 0.0:
        return 0.0
    beta2 = float(beta) * float(beta)
    return float(((1.0 + beta2) * precision * recall) / ((beta2 * precision) + recall))


def _binary_f1_from_predictions(true_binary, pred_binary):
    return _binary_fbeta_from_predictions(
        true_binary=true_binary,
        pred_binary=pred_binary,
        beta=1.0,
    )


def _binary_mcc_from_predictions(true_binary, pred_binary):
    true_binary = np.asarray(true_binary, dtype=np.int64)
    pred_binary = np.asarray(pred_binary, dtype=np.int64)
    tp = int(np.sum((true_binary == 1) & (pred_binary == 1)))
    fp = int(np.sum((true_binary == 0) & (pred_binary == 1)))
    fn = int(np.sum((true_binary == 1) & (pred_binary == 0)))
    tn = int(np.sum((true_binary == 0) & (pred_binary == 0)))
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom <= 0.0:
        return 0.0
    return float(((tp * tn) - (fp * fn)) / np.sqrt(denom))


def _tune_binary_threshold(prob_vec, true_binary, threshold_grid=None, objective='f1'):
    prob_vec = np.asarray(prob_vec, dtype=np.float64)
    true_binary = np.asarray(true_binary, dtype=np.int64)
    if threshold_grid is None:
        threshold_grid = np.linspace(0.05, 0.95, 19)
    objective = str(objective or 'f1').strip().lower()
    if objective not in ['mcc', 'f0.5', 'f1', 'f2']:
        raise ValueError('Unsupported threshold objective: {}'.format(objective))
    best_threshold = 0.5
    best_score = -1.0e9
    for threshold in threshold_grid:
        threshold = float(threshold)
        pred = (prob_vec >= threshold).astype(np.int64)
        if objective == 'mcc':
            score = _binary_mcc_from_predictions(
                true_binary=true_binary,
                pred_binary=pred,
            )
        elif objective == 'f0.5':
            score = _binary_fbeta_from_predictions(
                true_binary=true_binary,
                pred_binary=pred,
                beta=0.5,
            )
        elif objective == 'f2':
            score = _binary_fbeta_from_predictions(
                true_binary=true_binary,
                pred_binary=pred,
                beta=2.0,
            )
        else:
            score = _binary_f1_from_predictions(
                true_binary=true_binary,
                pred_binary=pred,
            )
        if score > best_score + 1.0e-12:
            best_score = float(score)
            best_threshold = threshold
            continue
        if abs(score - best_score) <= 1.0e-12:
            if abs(threshold - 0.5) < abs(best_threshold - 0.5):
                best_threshold = threshold
    return float(best_threshold)


def fit_multilabel_centroid_classifier(
    features,
    label_matrix,
    class_order,
    threshold_grid=None,
    threshold_objective='f1',
    threshold_objective_by_class=None,
    ensure_one_label=True,
):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(label_matrix, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError('Feature matrix should be 2D.')
    if y.ndim != 2:
        raise ValueError('Label matrix should be 2D.')
    if x.shape[0] == 0:
        raise ValueError('No training samples were provided.')
    if x.shape[0] != y.shape[0]:
        raise ValueError('Feature and label row counts do not match.')
    class_order = list(class_order)
    if y.shape[1] != len(class_order):
        raise ValueError('Label matrix column count does not match class_order.')

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    z = (x - mean) / std
    total = float(x.shape[0])

    label_models = list()
    for class_i, class_name in enumerate(class_order):
        y_col = y[:, class_i]
        pos_mask = (y_col == 1)
        neg_mask = ~pos_mask
        n_pos = int(np.sum(pos_mask))
        n_neg = int(np.sum(neg_mask))
        if n_pos == 0 or n_neg == 0:
            label_models.append({
                'class_name': class_name,
                'mode': 'constant',
                'probability': 1.0 if n_pos > 0 else 0.0,
                'n_positive': int(n_pos),
                'n_negative': int(n_neg),
            })
            continue
        prior_pos = (float(n_pos) + 1.0) / (total + 2.0)
        prior_neg = (float(n_neg) + 1.0) / (total + 2.0)
        label_models.append({
            'class_name': class_name,
            'mode': 'centroid',
            'positive_centroid': z[pos_mask, :].mean(axis=0).tolist(),
            'negative_centroid': z[neg_mask, :].mean(axis=0).tolist(),
            'log_prior_positive': safe_log(prior_pos),
            'log_prior_negative': safe_log(prior_neg),
            'n_positive': int(n_pos),
            'n_negative': int(n_neg),
        })

    model = {
        'mode': 'multilabel_centroid',
        'class_order': list(class_order),
        'mean': mean.tolist(),
        'std': std.tolist(),
        'label_models': label_models,
        'class_thresholds': {class_name: 0.5 for class_name in class_order},
        'ensure_one_label': bool(ensure_one_label),
    }
    train_prob = predict_multilabel_centroid_matrix(
        features=x,
        localization_model=model,
        apply_thresholds=False,
    )['prob_matrix']
    thresholds = dict()
    threshold_objective_by_class = dict(threshold_objective_by_class or {})
    for class_i, class_name in enumerate(class_order):
        objective = threshold_objective_by_class.get(class_name, threshold_objective)
        thresholds[class_name] = _tune_binary_threshold(
            prob_vec=train_prob[:, class_i],
            true_binary=y[:, class_i],
            threshold_grid=threshold_grid,
            objective=objective,
        )
    model['class_thresholds'] = thresholds
    return model


def predict_multilabel_centroid_matrix(features, localization_model, apply_thresholds=True):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape((1, -1))
    if x.ndim != 2:
        raise ValueError('Feature matrix should be 2D.')
    mean = np.asarray(localization_model['mean'], dtype=np.float64)
    std = np.asarray(localization_model['std'], dtype=np.float64)
    if x.shape[1] != mean.shape[0]:
        txt = 'Feature count mismatch: expected {}, got {}.'
        raise ValueError(txt.format(int(mean.shape[0]), int(x.shape[1])))
    std[std == 0.0] = 1.0
    z = (x - mean) / std
    label_models = list(localization_model.get('label_models', []))
    class_order = list(localization_model.get('class_order', []))
    if len(label_models) != len(class_order):
        raise ValueError('Invalid multilabel model: class_order and label_models differ.')
    prob = np.zeros((x.shape[0], len(class_order)), dtype=np.float64)
    for class_i, label_model in enumerate(label_models):
        mode = str(label_model.get('mode', '')).strip().lower()
        if mode == 'constant':
            prob[:, class_i] = float(label_model.get('probability', 0.0))
            continue
        if mode != 'centroid':
            raise ValueError('Unsupported multilabel label model mode: {}'.format(mode))
        pos_centroid = np.asarray(label_model['positive_centroid'], dtype=np.float64)
        neg_centroid = np.asarray(label_model['negative_centroid'], dtype=np.float64)
        pos_diff = z - pos_centroid.reshape((1, -1))
        neg_diff = z - neg_centroid.reshape((1, -1))
        pos_dist = np.sum(pos_diff * pos_diff, axis=1)
        neg_dist = np.sum(neg_diff * neg_diff, axis=1)
        logits = (
            (-0.5 * pos_dist)
            + float(label_model.get('log_prior_positive', 0.0))
            - (-0.5 * neg_dist)
            - float(label_model.get('log_prior_negative', 0.0))
        )
        prob[:, class_i] = _sigmoid(logits)
    prob = np.clip(prob, 0.0, 1.0)
    if not apply_thresholds:
        return {'prob_matrix': prob}

    thresholds = localization_model.get('class_thresholds', {})
    threshold_vec = np.asarray([
        float(thresholds.get(class_name, 0.5)) for class_name in class_order
    ], dtype=np.float64)
    threshold_vec[~np.isfinite(threshold_vec)] = 0.5
    threshold_vec[threshold_vec <= 0.0] = 0.5
    pred = (prob >= threshold_vec.reshape((1, -1))).astype(np.int64)
    if bool(localization_model.get('ensure_one_label', True)):
        empty = np.where(np.sum(pred, axis=1) == 0)[0]
        if empty.shape[0] > 0:
            scores = prob[empty, :] / threshold_vec.reshape((1, -1))
            best = np.argmax(scores, axis=1)
            pred[empty, best] = 1
    return {
        'prob_matrix': prob,
        'prediction_matrix': pred,
    }


def predict_multilabel_localization(aa_seq, model, kingdom=''):
    feats, perox_signals = extract_broad_localize_features(
        aa_seq=aa_seq,
        kingdom=kingdom,
    )
    model_type = str(model.get('model_type', ''))
    localization_model = model['localization_model']
    if model_type == 'multilabel_centroid_v1':
        pred = predict_multilabel_centroid_matrix(
            features=feats,
            localization_model=localization_model,
            apply_thresholds=True,
        )
    elif model_type == 'multilabel_cnn_v1':
        from cdskit.localize_multilabel_cnn import predict_multilabel_cnn_batch
        feature_matrix = None
        if int(localization_model.get('feature_dim', 0)) > 0:
            feature_matrix = np.asarray(feats, dtype=np.float32).reshape((1, -1))
        pred = predict_multilabel_cnn_batch(
            aa_sequences=[aa_seq],
            localization_model=localization_model,
            device='cpu',
            batch_size=1,
            feature_matrix=feature_matrix,
            apply_thresholds=True,
        )
    else:
        raise ValueError('Unsupported multilabel model_type: {}'.format(model_type))
    class_order = list(localization_model['class_order'])
    prob_vec = pred['prob_matrix'][0, :]
    pred_vec = pred['prediction_matrix'][0, :]
    labels = [class_order[i] for i in range(len(class_order)) if int(pred_vec[i]) == 1]
    return {
        'predicted_labels': labels,
        'class_probabilities': {
            class_order[i]: float(prob_vec[i]) for i in range(len(class_order))
        },
        'feature_values': feats,
        'feature_names': list(BROAD_FEATURE_NAMES),
        'perox_signal_type': perox_signals['signal_type'],
        'pts1_match': bool(perox_signals['pts1_match']),
        'pts2_match': bool(perox_signals['pts2_match']),
    }


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
            feature_vec=feature_vec,
        )
    if model_type == 'esm_head_v1':
        from cdskit.localize_esm_head import predict_esm_head
        return predict_esm_head(
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


def _normalize_ctp_ltp_probs(stage3_probs):
    p_ctp = float(stage3_probs.get('cTP', 0.0))
    p_ltp = float(stage3_probs.get('lTP', 0.0))
    if p_ctp < 0.0:
        p_ctp = 0.0
    if p_ltp < 0.0:
        p_ltp = 0.0
    total = p_ctp + p_ltp
    if total <= 0.0:
        return {'cTP': 0.5, 'lTP': 0.5}
    return {
        'cTP': p_ctp / total,
        'lTP': p_ltp / total,
    }


def _clip_float(value, lower, upper, default):
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if not np.isfinite(out):
        out = float(default)
    if out < lower:
        out = float(lower)
    if out > upper:
        out = float(upper)
    return float(out)


def compose_two_stage_ctp_ltp_probabilities(
    base_class_probs,
    stage3_ctp_ltp_probs,
    stage3_gate_threshold=0.0,
    stage3_blend_beta=1.0,
    stage3_ltp_threshold=0.5,
):
    base = normalize_class_probabilities(class_probs=base_class_probs)
    out = dict(base)
    stage3 = _normalize_ctp_ltp_probs(stage3_probs=stage3_ctp_ltp_probs)

    gate_threshold = _clip_float(
        value=stage3_gate_threshold,
        lower=0.0,
        upper=1.0,
        default=0.0,
    )
    blend_beta = _clip_float(
        value=stage3_blend_beta,
        lower=0.0,
        upper=1.0,
        default=1.0,
    )
    ltp_threshold = _clip_float(
        value=stage3_ltp_threshold,
        lower=1.0e-3,
        upper=1.0 - 1.0e-3,
        default=0.5,
    )

    ctp_ltp_mass = float(base.get('cTP', 0.0) + base.get('lTP', 0.0))
    gate_active = (ctp_ltp_mass > 0.0) and (ctp_ltp_mass >= gate_threshold)
    stage2_ctp_frac = 0.5
    stage2_ltp_frac = 0.5
    if ctp_ltp_mass > 0.0:
        stage2_ctp_frac = float(base.get('cTP', 0.0)) / ctp_ltp_mass
        stage2_ltp_frac = float(base.get('lTP', 0.0)) / ctp_ltp_mass

    if gate_active:
        blend_ctp = ((1.0 - blend_beta) * stage2_ctp_frac) + (blend_beta * float(stage3['cTP']))
        blend_ltp = ((1.0 - blend_beta) * stage2_ltp_frac) + (blend_beta * float(stage3['lTP']))
        denom_blend = blend_ctp + blend_ltp
        if denom_blend <= 0.0:
            blend_ctp = 0.5
            blend_ltp = 0.5
        else:
            blend_ctp = blend_ctp / denom_blend
            blend_ltp = blend_ltp / denom_blend

        # Soft threshold adjustment: larger stage3_ltp_threshold penalizes lTP.
        score_ctp = blend_ctp / max(1.0e-6, (1.0 - ltp_threshold))
        score_ltp = blend_ltp / max(1.0e-6, ltp_threshold)
        denom_score = score_ctp + score_ltp
        if denom_score <= 0.0:
            adj_ctp = 0.5
            adj_ltp = 0.5
        else:
            adj_ctp = score_ctp / denom_score
            adj_ltp = score_ltp / denom_score

        out['cTP'] = ctp_ltp_mass * adj_ctp
        out['lTP'] = ctp_ltp_mass * adj_ltp

    total = float(sum(out.values()))
    if total <= 0.0:
        out = {class_name: 0.0 for class_name in LOCALIZATION_CLASSES}
        out['noTP'] = 1.0
    else:
        for class_name in LOCALIZATION_CLASSES:
            out[class_name] = out[class_name] / total

    details = {
        'gate_threshold': float(gate_threshold),
        'blend_beta': float(blend_beta),
        'ltp_threshold': float(ltp_threshold),
        'ctp_ltp_mass': float(ctp_ltp_mass),
        'gate_active': bool(gate_active),
        'stage2_ctp_frac': float(stage2_ctp_frac),
        'stage2_ltp_frac': float(stage2_ltp_frac),
        'stage3_ctp_frac': float(stage3['cTP']),
        'stage3_ltp_frac': float(stage3['lTP']),
    }
    return out, details


def predict_two_stage_ctp_ltp_localization(
    aa_seq,
    feature_vec,
    localization_model,
    model_type,
    return_details=False,
):
    _, base_probs = predict_two_stage_localization(
        aa_seq=aa_seq,
        feature_vec=feature_vec,
        localization_model=localization_model,
        model_type=model_type,
    )
    stage3_model = localization_model.get('stage3_model', None)
    if not isinstance(stage3_model, dict):
        pred_idx = int(np.argmax([base_probs[class_name] for class_name in LOCALIZATION_CLASSES]))
        pred_class = LOCALIZATION_CLASSES[pred_idx]
        if return_details:
            return pred_class, base_probs, {
                'base_class_probabilities': dict(base_probs),
                'stage3_ctp_ltp_probabilities': {'cTP': 0.5, 'lTP': 0.5},
                'gate_threshold': 0.0,
                'blend_beta': 1.0,
                'ltp_threshold': 0.5,
                'ctp_ltp_mass': float(base_probs.get('cTP', 0.0) + base_probs.get('lTP', 0.0)),
                'gate_active': False,
            }
        return pred_class, base_probs
    if len(stage3_model) == 0:
        pred_idx = int(np.argmax([base_probs[class_name] for class_name in LOCALIZATION_CLASSES]))
        pred_class = LOCALIZATION_CLASSES[pred_idx]
        if return_details:
            return pred_class, base_probs, {
                'base_class_probabilities': dict(base_probs),
                'stage3_ctp_ltp_probabilities': {'cTP': 0.5, 'lTP': 0.5},
                'gate_threshold': 0.0,
                'blend_beta': 1.0,
                'ltp_threshold': 0.5,
                'ctp_ltp_mass': float(base_probs.get('cTP', 0.0) + base_probs.get('lTP', 0.0)),
                'gate_active': False,
            }
        return pred_class, base_probs

    _, stage3_probs = _predict_localization_from_model(
        aa_seq=aa_seq,
        feature_vec=feature_vec,
        localization_model=stage3_model,
        model_type=model_type,
    )
    stage3_probs = _normalize_ctp_ltp_probs(stage3_probs=stage3_probs)
    out_probs, details = compose_two_stage_ctp_ltp_probabilities(
        base_class_probs=base_probs,
        stage3_ctp_ltp_probs=stage3_probs,
        stage3_gate_threshold=localization_model.get('stage3_gate_threshold', 0.0),
        stage3_blend_beta=localization_model.get('stage3_blend_beta', 1.0),
        stage3_ltp_threshold=localization_model.get('stage3_ltp_threshold', 0.5),
    )
    details['base_class_probabilities'] = dict(base_probs)
    details['stage3_ctp_ltp_probabilities'] = dict(stage3_probs)

    pred_idx = int(np.argmax([out_probs[class_name] for class_name in LOCALIZATION_CLASSES]))
    pred_class = LOCALIZATION_CLASSES[pred_idx]
    if return_details:
        return pred_class, out_probs, details
    return pred_class, out_probs


def predict_localization_and_peroxisome(aa_seq, model, organism_group=''):
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
    elif localization_strategy == 'two_stage_ctp_ltp':
        _, class_probs, strategy_details = predict_two_stage_ctp_ltp_localization(
            aa_seq=aa_seq,
            feature_vec=feats,
            localization_model=localization_model,
            model_type=model_type,
            return_details=True,
        )
    else:
        _, class_probs = _predict_localization_from_model(
            aa_seq=aa_seq,
            feature_vec=feats,
            localization_model=localization_model,
            model_type=model_type,
        )
    class_probs = apply_organism_group_constraints(
        class_probs=class_probs,
        organism_group=organism_group,
    )
    pred_class, class_probs = postprocess_localization_probabilities(
        class_probs=class_probs,
        localization_model=localization_model,
    )
    _, perox_probs = predict_perox(
        feature_vec=feats,
        perox_model=model['perox_model'],
    )
    out = {
        'predicted_class': pred_class,
        'class_probabilities': class_probs,
        'perox_probability_yes': float(perox_probs.get('yes', 0.0)),
        'perox_signal_type': perox_signals['signal_type'],
        'feature_values': feats,
        'feature_names': list(FEATURE_NAMES),
        'pts1_match': bool(perox_signals['pts1_match']),
        'pts2_match': bool(perox_signals['pts2_match']),
    }
    if localization_strategy == 'two_stage_ctp_ltp':
        out['two_stage_ctp_ltp_details'] = strategy_details
    return out


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
    if model_type in ['nearest_centroid_v1', 'multilabel_centroid_v1']:
        with open(path, 'w', encoding='utf-8') as out:
            json.dump(model, out, indent=2, sort_keys=True)
        return
    if model_type in ['bilstm_attention_v1', 'esm_head_v1', 'multilabel_cnn_v1']:
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
    allowed_model_types = [
        'nearest_centroid_v1',
        'bilstm_attention_v1',
        'esm_head_v1',
        'multilabel_centroid_v1',
        'multilabel_cnn_v1',
    ]
    if model['model_type'] not in allowed_model_types:
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

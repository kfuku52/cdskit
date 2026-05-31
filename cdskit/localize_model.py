import csv
from contextlib import contextmanager
import json
import math
import re
import warnings

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
    has_ltp = ('thylakoid' in txt) and (
        ('lumen' in txt) or ('lumenal' in txt) or ('luminal' in txt)
    )
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


def _predict_localization_from_model(aa_seq, feature_vec, localization_model, model_type, organism_group=''):
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
    if model_type == 'targetp_feature_ensemble_v1':
        return predict_targetp_feature_ensemble_localization(
            aa_seq=aa_seq,
            localization_model=localization_model,
            organism_group=organism_group,
        )
    if model_type == 'targetp_torch_v1':
        from cdskit.targetp_torch import predict_targetp2_torch_localization
        return predict_targetp2_torch_localization(
            aa_seq=aa_seq,
            localization_model=localization_model,
            organism_group=organism_group,
        )
    if model_type == 'targetp_blend_v1':
        return predict_targetp_blend_localization(
            aa_seq=aa_seq,
            feature_vec=feature_vec,
            localization_model=localization_model,
            organism_group=organism_group,
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


def _class_probs_to_vector(class_probs):
    probs = normalize_class_probabilities(class_probs=class_probs)
    return np.asarray([probs[class_name] for class_name in LOCALIZATION_CLASSES], dtype=np.float64)


def _vector_to_class_probs(prob_vec):
    prob_vec = np.asarray(prob_vec, dtype=np.float64)
    prob_vec = np.clip(prob_vec, a_min=0.0, a_max=None)
    total = float(np.sum(prob_vec))
    if total <= 0.0:
        prob_vec = np.zeros((len(LOCALIZATION_CLASSES),), dtype=np.float64)
        prob_vec[0] = 1.0
    else:
        prob_vec = prob_vec / total
    return {
        LOCALIZATION_CLASSES[i]: float(prob_vec[i])
        for i in range(len(LOCALIZATION_CLASSES))
    }


def _targetp_blend_class_probabilities(prob_a, prob_b, alpha_by_class):
    vec_a = _class_probs_to_vector(prob_a)
    vec_b = _class_probs_to_vector(prob_b)
    alpha = np.ones((len(LOCALIZATION_CLASSES),), dtype=np.float64)
    if isinstance(alpha_by_class, dict):
        for i, class_name in enumerate(LOCALIZATION_CLASSES):
            try:
                alpha[i] = float(alpha_by_class.get(class_name, 1.0))
            except Exception:
                alpha[i] = 1.0
    else:
        try:
            alpha[:] = float(alpha_by_class)
        except Exception:
            alpha[:] = 1.0
    alpha = np.clip(alpha, a_min=0.0, a_max=1.0)
    return _vector_to_class_probs((alpha * vec_a) + ((1.0 - alpha) * vec_b))


def _targetp_sp_scan_features(seq):
    seq = str(seq or '')
    best_score = -99.0
    best_cut = 0
    best_parts = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for cut in range(12, min(45, len(seq) - 1)):
        pre = seq[:cut]
        nreg = seq[:max(1, cut - 18)]
        hreg = seq[max(0, cut - 18):max(0, cut - 7)]
        creg = seq[max(0, cut - 7):cut + 2]
        m3 = seq[cut - 3] if cut - 3 >= 0 else 'X'
        m2 = seq[cut - 2] if cut - 2 >= 0 else 'X'
        m1 = seq[cut - 1] if cut - 1 >= 0 else 'X'
        p1 = seq[cut] if cut < len(seq) else 'X'
        small_m3 = 1.0 if m3 in 'AVSGTC' else 0.0
        small_m1 = 1.0 if m1 in 'ASGTC' else 0.0
        ala_m1 = 1.0 if m1 == 'A' else 0.0
        pro_bad = 1.0 if 'P' in (m3 + m2 + m1 + p1) else 0.0
        hyd = fraction_in_set(hreg, AA_HYDROPHOBIC)
        run = longest_hydrophobic_run(hreg)
        small = fraction_in_set(creg, AA_SMALL)
        ncharge = fraction_in_set(nreg, AA_BASIC) - fraction_in_set(nreg, AA_ACIDIC)
        st_frac = fraction_in_set(pre, AA_SER_THR)
        score = (
            (2.2 * hyd)
            + (0.15 * run)
            + (0.8 * small_m3)
            + (1.0 * small_m1)
            + (0.4 * ala_m1)
            + (0.5 * small)
            + (0.4 * ncharge)
            - (0.9 * pro_bad)
            - (0.25 * st_frac)
        )
        if score > best_score:
            best_score = float(score)
            best_cut = int(cut)
            best_parts = (
                float(hyd),
                float(run),
                float(small_m3),
                float(small_m1),
                float(ala_m1),
                float(pro_bad),
                float(small),
                float(ncharge),
                float(st_frac),
            )

    out = [best_score, float(best_cut), float(best_cut) / float(max(1, len(seq)))]
    out.extend(best_parts)
    for window in [
        seq[:15],
        seq[:25],
        seq[:35],
        seq[:50],
        seq[:80],
        seq[5:30],
        seq[20:60],
        seq[40:100],
    ]:
        out.extend([
            mean_hydropathy(window),
            longest_hydrophobic_run(window),
            fraction_in_set(window, AA_HYDROPHOBIC),
            fraction_in_set(window, AA_BASIC),
            fraction_in_set(window, AA_ACIDIC),
            fraction_in_set(window, AA_SER_THR),
            fraction_in_set(window, AA_SMALL),
        ])
    return out


def _targetp_ctp_ltp_sequence_features(seq, organism_group):
    seq = str(seq or '')
    out = list(extract_broad_localize_features(seq, organism_group)[0])
    windows = [
        seq[:20],
        seq[:40],
        seq[:60],
        seq[:80],
        seq[:100],
        seq[:120],
        seq[20:80],
        seq[40:120],
    ]
    groups = [
        AA_BASIC,
        AA_ACIDIC,
        AA_HYDROPHOBIC,
        AA_SMALL,
        AA_SER_THR,
        AA_AROMATIC,
        frozenset('R'),
        frozenset('K'),
        frozenset('A'),
        frozenset('S'),
        frozenset('T'),
        frozenset('P'),
        frozenset('G'),
        frozenset('LIV'),
    ]
    for window in windows:
        out.extend([mean_hydropathy(window), longest_hydrophobic_run(window)])
        out.extend([fraction_in_set(window, group) for group in groups])
    n_term = seq[:140]
    for motif in ['RR', 'KR', 'RK', 'KK', 'RA', 'RS', 'SR', 'ST', 'TS', 'SS', 'TP', 'SP']:
        out.append(1.0 if motif in n_term else 0.0)
        out.append(float(n_term.find(motif) if motif in n_term else 999))
    return out


def _targetp_specialist_probability_features(base_probs, prob_a, prob_b):
    base_vec = _class_probs_to_vector(base_probs)
    vec_a = _class_probs_to_vector(prob_a)
    vec_b = _class_probs_to_vector(prob_b)
    ctp_idx = LOCALIZATION_CLASSES.index('cTP')
    ltp_idx = LOCALIZATION_CLASSES.index('lTP')
    denom = float(base_vec[ctp_idx] + base_vec[ltp_idx])
    ltp_ratio = 0.0 if denom <= 0.0 else float(base_vec[ltp_idx]) / denom
    return np.concatenate([base_vec, vec_a, vec_b, np.asarray([ltp_ratio], dtype=np.float64)])


def _targetp_sp_specialist_feature_vector(aa_seq, base_probs, prob_a, prob_b, organism_group):
    plant_flag = 1.0 if normalize_organism_group(organism_group) == 'plant' else 0.0
    return np.concatenate([
        np.asarray(_targetp_sp_scan_features(aa_seq), dtype=np.float64),
        _targetp_specialist_probability_features(base_probs, prob_a, prob_b),
        np.asarray([plant_flag], dtype=np.float64),
    ])


def _targetp_ctp_ltp_specialist_feature_vector(aa_seq, base_probs, prob_a, prob_b, organism_group):
    plant_flag = 1.0 if normalize_organism_group(organism_group) == 'plant' else 0.0
    return np.concatenate([
        np.asarray(_targetp_ctp_ltp_sequence_features(aa_seq, organism_group), dtype=np.float64),
        _targetp_specialist_probability_features(base_probs, prob_a, prob_b),
        np.asarray([plant_flag], dtype=np.float64),
    ])


def _targetp_threshold_vector(class_thresholds):
    thresholds = np.ones((len(LOCALIZATION_CLASSES),), dtype=np.float64)
    if isinstance(class_thresholds, dict):
        for i, class_name in enumerate(LOCALIZATION_CLASSES):
            try:
                thresholds[i] = float(class_thresholds.get(class_name, 1.0))
            except Exception:
                thresholds[i] = 1.0
            if (not np.isfinite(thresholds[i])) or thresholds[i] <= 0.0:
                thresholds[i] = 1.0
    return thresholds


def _targetp_probability_sequence_feature_vector(
    aa_seq,
    base_probs,
    prob_a,
    prob_b,
    organism_group,
    class_thresholds,
):
    del prob_a, prob_b
    base_vec = _class_probs_to_vector(base_probs)
    thresholds = _targetp_threshold_vector(class_thresholds)
    score_vec = base_vec / thresholds
    sorted_scores = np.sort(score_vec)
    top_score = float(sorted_scores[-1]) if sorted_scores.shape[0] > 0 else 0.0
    second_score = float(sorted_scores[-2]) if sorted_scores.shape[0] > 1 else 0.0
    notp_idx = LOCALIZATION_CLASSES.index('noTP')
    sp_idx = LOCALIZATION_CLASSES.index('SP')
    mtp_idx = LOCALIZATION_CLASSES.index('mTP')
    ctp_idx = LOCALIZATION_CLASSES.index('cTP')
    ltp_idx = LOCALIZATION_CLASSES.index('lTP')
    notp_score = float(score_vec[notp_idx])
    notp_top_ratio = 0.0 if top_score <= 0.0 else float(notp_score / top_score)
    pred_is_notp = 1.0 if int(np.argmax(score_vec)) == notp_idx else 0.0
    summary = np.asarray([
        top_score,
        float(top_score - notp_score),
        notp_top_ratio,
        float(base_vec[notp_idx]),
        float(base_vec[mtp_idx]),
        float(base_vec[sp_idx]),
        float(base_vec[ctp_idx] + base_vec[ltp_idx]),
        pred_is_notp,
        float(top_score - second_score),
    ], dtype=np.float64)
    return np.concatenate([
        base_vec,
        score_vec,
        summary,
        extract_targetp_feature_ensemble_features(
            aa_seq=aa_seq,
            organism_group=organism_group,
        ),
    ])


def _targetp_notp_specialist_feature_vector(
    aa_seq,
    base_probs,
    prob_a,
    prob_b,
    organism_group,
    class_thresholds,
):
    return _targetp_probability_sequence_feature_vector(
        aa_seq=aa_seq,
        base_probs=base_probs,
        prob_a=prob_a,
        prob_b=prob_b,
        organism_group=organism_group,
        class_thresholds=class_thresholds,
    )


def _targetp_feature_ltp_specialist_feature_vector(aa_seq, base_probs, organism_group):
    plant_flag = 1.0 if normalize_organism_group(organism_group) == 'plant' else 0.0
    base_vec = _class_probs_to_vector(base_probs)
    ctp_idx = LOCALIZATION_CLASSES.index('cTP')
    ltp_idx = LOCALIZATION_CLASSES.index('lTP')
    ctp_ltp_mass = float(base_vec[ctp_idx] + base_vec[ltp_idx])
    ltp_ratio = 0.0 if ctp_ltp_mass <= 0.0 else float(base_vec[ltp_idx]) / ctp_ltp_mass
    return np.concatenate([
        np.asarray(_targetp_ctp_ltp_sequence_features(aa_seq, organism_group), dtype=np.float64),
        base_vec,
        np.asarray([ctp_ltp_mass, ltp_ratio, plant_flag], dtype=np.float64),
    ])


TARGETP_FEATURE_ENSEMBLE_PROFILE = {
    'name': 'targetp_feature_ensemble_v1',
    'n_terminal_group_len': 100,
}


def _targetp_feature_window_features(seq):
    seq = str(seq or '')
    windows = [
        seq[:10],
        seq[:15],
        seq[:20],
        seq[:25],
        seq[:30],
        seq[:35],
        seq[:40],
        seq[:45],
        seq[:50],
        seq[:60],
        seq[:70],
        seq[:80],
        seq[:100],
        seq[:120],
        seq[5:35],
        seq[10:50],
        seq[20:80],
        seq[40:120],
        seq[-20:],
        seq[-40:],
    ]
    groups = [
        AA_BASIC,
        AA_ACIDIC,
        AA_HYDROPHOBIC,
        AA_SMALL,
        AA_SER_THR,
        AA_AROMATIC,
        frozenset('A'),
        frozenset('G'),
        frozenset('P'),
        frozenset('R'),
        frozenset('K'),
        frozenset('LIV'),
        frozenset('DE'),
        frozenset('STNQ'),
    ]
    out = list()
    for window in windows:
        out.extend([
            float(len(window)),
            mean_hydropathy(window),
            longest_hydrophobic_run(window),
        ])
        out.extend([fraction_in_set(window, group) for group in groups])
        out.extend([
            fraction_in_set(window, frozenset(aa))
            for aa in 'ACDEFGHIKLMNPQRSTVWY'
        ])
    return out


def _targetp_feature_positional_group_features(seq, n_terminal_len=100):
    seq = str(seq or '')[:int(n_terminal_len)]
    out = list()
    for i in range(int(n_terminal_len)):
        aa = seq[i] if i < len(seq) else 'X'
        out.extend([
            1.0 if aa in AA_HYDROPHOBIC else 0.0,
            1.0 if aa in AA_BASIC else 0.0,
            1.0 if aa in AA_ACIDIC else 0.0,
            1.0 if aa in AA_SMALL else 0.0,
            1.0 if aa in AA_SER_THR else 0.0,
            1.0 if aa == 'P' else 0.0,
            1.0 if aa == 'R' else 0.0,
            1.0 if aa == 'K' else 0.0,
            1.0 if aa == 'A' else 0.0,
            1.0 if aa == 'G' else 0.0,
        ])
    return out


def extract_targetp_feature_ensemble_features(aa_seq, organism_group=''):
    seq = to_canonical_aa_sequence(aa_seq)
    group = normalize_organism_group(organism_group)
    plant_flag = 1.0 if group == 'plant' else 0.0
    out = list(extract_broad_localize_features(seq, group)[0])
    out.extend(_targetp_sp_scan_features(seq))
    out.extend(_targetp_ctp_ltp_sequence_features(seq, group))
    out.append(plant_flag)
    out.extend(_targetp_feature_window_features(seq))
    out.extend(_targetp_feature_positional_group_features(
        seq=seq,
        n_terminal_len=TARGETP_FEATURE_ENSEMBLE_PROFILE['n_terminal_group_len'],
    ))
    return np.asarray(out, dtype=np.float64)


@contextmanager
def _targetp_sklearn_single_thread_context():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'\s*Found Intel OpenMP.*',
            category=RuntimeWarning,
        )
        with threadpool_limits(limits=1):
            yield


def _targetp_predict_sklearn_proba(model, features):
    with _targetp_sklearn_single_thread_context():
        return model.predict_proba(features)


def predict_targetp_feature_ensemble_localization(aa_seq, localization_model, organism_group=''):
    classifier = localization_model.get('classifier', None)
    binary_classifiers = localization_model.get('binary_classifiers', None)
    has_multiclass = classifier is not None and hasattr(classifier, 'predict_proba')
    has_binary = isinstance(binary_classifiers, list) and len(binary_classifiers) > 0
    if not has_multiclass and not has_binary:
        raise ValueError('targetp_feature_ensemble_v1 requires a sklearn classifier.')
    class_order = list(localization_model.get('class_order', LOCALIZATION_CLASSES))
    if class_order != list(LOCALIZATION_CLASSES):
        raise ValueError('targetp_feature_ensemble_v1 class_order should match LOCALIZATION_CLASSES.')
    feature_vec = extract_targetp_feature_ensemble_features(
        aa_seq=aa_seq,
        organism_group=organism_group,
    )
    expected_dim = int(localization_model.get('feature_dim', feature_vec.shape[0]))
    if feature_vec.shape[0] != expected_dim:
        txt = 'TargetP feature count mismatch: expected {}, got {}.'
        raise ValueError(txt.format(expected_dim, int(feature_vec.shape[0])))
    prob_vec = np.zeros((len(class_order),), dtype=np.float64)
    if has_binary:
        if len(binary_classifiers) != len(class_order):
            raise ValueError('targetp_feature_ensemble_v1 binary_classifiers should match class_order.')
        for class_i, binary_classifier in enumerate(binary_classifiers):
            if binary_classifier is None or not hasattr(binary_classifier, 'predict_proba'):
                raise ValueError('targetp_feature_ensemble_v1 binary classifier should support predict_proba.')
            proba = np.asarray(
                _targetp_predict_sklearn_proba(
                    binary_classifier,
                    feature_vec.reshape((1, -1)),
                ),
                dtype=np.float64,
            )
            classes = [int(cls) for cls in list(getattr(binary_classifier, 'classes_', []))]
            if 1 in classes:
                prob_vec[class_i] = float(proba[0, classes.index(1)])
    else:
        proba = np.asarray(
            _targetp_predict_sklearn_proba(
                classifier,
                feature_vec.reshape((1, -1)),
            ),
            dtype=np.float64,
        )
        classes = getattr(classifier, 'classes_', list(range(len(class_order))))
        class_to_col = {int(cls): i for i, cls in enumerate(list(classes))}
        for class_i in range(len(class_order)):
            if class_i in class_to_col:
                prob_vec[class_i] = float(proba[0, class_to_col[class_i]])
    probs = _vector_to_class_probs(prob_vec)
    pred_idx = int(np.argmax([probs[class_name] for class_name in LOCALIZATION_CLASSES]))
    return LOCALIZATION_CLASSES[pred_idx], probs


def _predict_binary_ensemble_score(feature_vec, models, weights=None):
    if not isinstance(models, list) or len(models) == 0:
        return 0.0
    scores = list()
    for model in models:
        if not hasattr(model, 'predict_proba'):
            raise TypeError('TargetP specialist model should support predict_proba.')
        proba = np.asarray(
            _targetp_predict_sklearn_proba(
                model,
                np.asarray(feature_vec, dtype=np.float64).reshape((1, -1)),
            ),
            dtype=np.float64,
        )
        classes = getattr(model, 'classes_', [0, 1])
        class_to_col = {int(cls): i for i, cls in enumerate(list(classes))}
        scores.append(float(proba[0, class_to_col.get(1, 0)]) if 1 in class_to_col else 0.0)
    if weights is None:
        return float(np.mean(np.asarray(scores, dtype=np.float64)))
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape[0] != len(scores):
        raise ValueError('TargetP specialist weights do not match model count.')
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError('TargetP specialist weights should sum to a positive value.')
    return float(np.average(np.asarray(scores, dtype=np.float64), weights=weights / total))


def _predict_multiclass_ensemble_probabilities(feature_vec, models, weights=None):
    if not isinstance(models, list) or len(models) == 0:
        return np.zeros((len(LOCALIZATION_CLASSES),), dtype=np.float64)
    prob_rows = list()
    for model in models:
        if not hasattr(model, 'predict_proba'):
            raise TypeError('TargetP reranker model should support predict_proba.')
        proba = np.asarray(
            _targetp_predict_sklearn_proba(
                model,
                np.asarray(feature_vec, dtype=np.float64).reshape((1, -1)),
            ),
            dtype=np.float64,
        )
        classes = getattr(model, 'classes_', list(range(len(LOCALIZATION_CLASSES))))
        class_to_col = {int(cls): i for i, cls in enumerate(list(classes))}
        row = np.zeros((len(LOCALIZATION_CLASSES),), dtype=np.float64)
        for class_i in range(len(LOCALIZATION_CLASSES)):
            if class_i in class_to_col:
                row[class_i] = float(proba[0, class_to_col[class_i]])
        prob_rows.append(row)
    if weights is None:
        probs = np.mean(np.asarray(prob_rows, dtype=np.float64), axis=0)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != len(prob_rows):
            raise ValueError('TargetP reranker weights do not match model count.')
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError('TargetP reranker weights should sum to a positive value.')
        probs = np.average(
            np.asarray(prob_rows, dtype=np.float64),
            axis=0,
            weights=weights / total,
        )
    total = float(np.sum(probs))
    if total <= 0.0:
        return probs
    return probs / total


def _targetp_specialist_model_list(specialist, plural_key, singular_key):
    models = specialist.get(plural_key, [])
    if models is None:
        models = []
    if not isinstance(models, list):
        models = [models]
    single_model = specialist.get(singular_key, None)
    if single_model is not None:
        models = [single_model] + list(models)
    return [model for model in models if model is not None]


def _prediction_index_with_thresholds(class_probs, class_thresholds):
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
        if (not np.isfinite(threshold)) or threshold <= 0.0:
            threshold = 1.0
        scores.append(float(probs[class_name]) / threshold)
    return int(np.argmax(np.asarray(scores, dtype=np.float64)))


def _apply_targetp_specialist_postprocess(
    aa_seq,
    base_probs,
    prob_a,
    prob_b,
    localization_model,
    organism_group,
):
    specialist = localization_model.get('targetp_specialist_postprocess', None)
    if not isinstance(specialist, dict) or not bool(specialist.get('enabled', True)):
        return None, {}

    class_thresholds = localization_model.get('class_thresholds', {})
    pred_idx = _prediction_index_with_thresholds(
        class_probs=base_probs,
        class_thresholds=class_thresholds,
    )
    scores = _class_probs_to_vector(base_probs)
    scores = scores / _targetp_threshold_vector(class_thresholds)
    sp_idx = LOCALIZATION_CLASSES.index('SP')
    ctp_idx = LOCALIZATION_CLASSES.index('cTP')
    ltp_idx = LOCALIZATION_CLASSES.index('lTP')
    notp_idx = LOCALIZATION_CLASSES.index('noTP')
    non_sp_scores = scores.copy()
    non_sp_scores[sp_idx] = -np.inf
    non_sp_pred_idx = int(np.argmax(non_sp_scores))

    reranker_models = _targetp_specialist_model_list(
        specialist,
        'reranker_models',
        'reranker_model',
    )
    reranker_threshold = float(specialist.get('reranker_threshold', 0.5))
    reranker_score = 0.0
    reranker_positive = False
    reranker_class = ''
    if len(reranker_models) > 0:
        reranker_feature_vec = _targetp_probability_sequence_feature_vector(
            aa_seq=aa_seq,
            base_probs=base_probs,
            prob_a=prob_a,
            prob_b=prob_b,
            organism_group=organism_group,
            class_thresholds=class_thresholds,
        )
        reranker_probs = _predict_multiclass_ensemble_probabilities(
            feature_vec=reranker_feature_vec,
            models=reranker_models,
            weights=specialist.get('reranker_weights', None),
        )
        constrained_reranker_probs = apply_organism_group_constraints(
            class_probs={
                LOCALIZATION_CLASSES[i]: float(reranker_probs[i])
                for i in range(len(LOCALIZATION_CLASSES))
            },
            organism_group=organism_group,
        )
        constrained_vec = _class_probs_to_vector(constrained_reranker_probs)
        reranker_idx = int(np.argmax(constrained_vec))
        reranker_score = float(constrained_vec[reranker_idx])
        reranker_class = LOCALIZATION_CLASSES[reranker_idx]
        reranker_positive = reranker_score >= reranker_threshold
        if reranker_positive:
            pred_idx = reranker_idx

    sp_models = _targetp_specialist_model_list(specialist, 'sp_models', 'sp_model')
    sp_threshold = float(specialist.get('sp_threshold', 0.5))
    sp_score = 0.0
    sp_positive = False
    if len(sp_models) > 0:
        sp_feature_vec = _targetp_sp_specialist_feature_vector(
            aa_seq=aa_seq,
            base_probs=base_probs,
            prob_a=prob_a,
            prob_b=prob_b,
            organism_group=organism_group,
        )
        sp_score = _predict_binary_ensemble_score(
            feature_vec=sp_feature_vec,
            models=sp_models,
            weights=specialist.get('sp_weights', None),
        )
        sp_positive = sp_score >= sp_threshold
        if sp_positive:
            pred_idx = sp_idx
        elif pred_idx == sp_idx:
            pred_idx = non_sp_pred_idx

    group = normalize_organism_group(organism_group)
    ctp_ltp_mass = float(base_probs.get('cTP', 0.0) + base_probs.get('lTP', 0.0))
    ltp_mass_threshold = float(specialist.get('ltp_mass_threshold', 0.20))
    ltp_models = _targetp_specialist_model_list(specialist, 'ltp_models', 'ltp_model')
    ltp_score = 0.0
    ltp_candidate = (
        group == 'plant'
        and (not sp_positive)
        and (ctp_ltp_mass > ltp_mass_threshold)
        and len(ltp_models) > 0
    )
    if ltp_candidate:
        ltp_feature_vec = _targetp_ctp_ltp_specialist_feature_vector(
            aa_seq=aa_seq,
            base_probs=base_probs,
            prob_a=prob_a,
            prob_b=prob_b,
            organism_group=organism_group,
        )
        ltp_score = _predict_binary_ensemble_score(
            feature_vec=ltp_feature_vec,
            models=ltp_models,
            weights=specialist.get('ltp_weights', None),
        )
        ltp_threshold = float(specialist.get('ltp_threshold', 0.5))
        if ltp_score >= ltp_threshold:
            pred_idx = ltp_idx
        elif pred_idx in [ctp_idx, ltp_idx]:
            pred_idx = ctp_idx

    notp_models = _targetp_specialist_model_list(specialist, 'notp_models', 'notp_model')
    notp_threshold = float(specialist.get('notp_threshold', 0.5))
    notp_score = 0.0
    notp_positive = False
    notp_applied = False
    if len(notp_models) > 0:
        notp_feature_vec = _targetp_notp_specialist_feature_vector(
            aa_seq=aa_seq,
            base_probs=base_probs,
            prob_a=prob_a,
            prob_b=prob_b,
            organism_group=organism_group,
            class_thresholds=class_thresholds,
        )
        notp_score = _predict_binary_ensemble_score(
            feature_vec=notp_feature_vec,
            models=notp_models,
            weights=specialist.get('notp_weights', None),
        )
        notp_positive = notp_score >= notp_threshold
        if pred_idx != notp_idx and notp_positive:
            pred_idx = notp_idx
            notp_applied = True

    details = {
        'sp_score': float(sp_score),
        'sp_threshold': float(sp_threshold),
        'sp_positive': bool(sp_positive),
        'ltp_score': float(ltp_score),
        'ltp_threshold': float(specialist.get('ltp_threshold', 0.5)),
        'ltp_mass_threshold': float(ltp_mass_threshold),
        'ltp_candidate': bool(ltp_candidate),
        'ctp_ltp_mass': float(ctp_ltp_mass),
        'reranker_score': float(reranker_score),
        'reranker_threshold': float(reranker_threshold),
        'reranker_positive': bool(reranker_positive),
        'reranker_class': str(reranker_class),
        'notp_score': float(notp_score),
        'notp_threshold': float(notp_threshold),
        'notp_positive': bool(notp_positive),
        'notp_applied': bool(notp_applied),
    }
    return LOCALIZATION_CLASSES[pred_idx], details


def _apply_targetp_feature_ltp_specialist_postprocess(
    aa_seq,
    base_probs,
    pred_class,
    localization_model,
    organism_group,
):
    specialist = localization_model.get('targetp_feature_ltp_specialist', None)
    if not isinstance(specialist, dict) or not bool(specialist.get('enabled', True)):
        return pred_class, {}
    group = normalize_organism_group(organism_group)
    if group != 'plant':
        return pred_class, {'enabled': True, 'applied': False, 'reason': 'non_plant'}

    source_classes = specialist.get('source_classes', ['cTP'])
    if isinstance(source_classes, str):
        source_classes = [
            value.strip() for value in str(source_classes).split(',')
            if value.strip() != ''
        ]
    source_classes = [class_name for class_name in source_classes if class_name in LOCALIZATION_CLASSES]
    if len(source_classes) == 0:
        source_classes = ['cTP']
    if pred_class not in source_classes:
        return pred_class, {
            'enabled': True,
            'applied': False,
            'reason': 'not_source_class',
            'source_classes': list(source_classes),
        }

    ctp_ltp_mass = float(base_probs.get('cTP', 0.0) + base_probs.get('lTP', 0.0))
    mass_threshold = float(specialist.get('mass_threshold', 0.0))
    if ctp_ltp_mass < mass_threshold:
        return pred_class, {
            'enabled': True,
            'applied': False,
            'reason': 'below_mass_threshold',
            'ctp_ltp_mass': float(ctp_ltp_mass),
            'mass_threshold': float(mass_threshold),
        }

    feature_vec = _targetp_feature_ltp_specialist_feature_vector(
        aa_seq=aa_seq,
        base_probs=base_probs,
        organism_group=organism_group,
    )
    score = _predict_binary_ensemble_score(
        feature_vec=feature_vec,
        models=specialist.get('models', []),
        weights=specialist.get('weights', None),
    )
    threshold = float(specialist.get('threshold', 0.5))
    out_class = 'lTP' if score >= threshold else pred_class
    return out_class, {
        'enabled': True,
        'applied': bool(out_class != pred_class),
        'score': float(score),
        'threshold': float(threshold),
        'source_classes': list(source_classes),
        'ctp_ltp_mass': float(ctp_ltp_mass),
        'mass_threshold': float(mass_threshold),
    }


def predict_targetp_blend_localization(
    aa_seq,
    feature_vec,
    localization_model,
    organism_group='',
    return_details=False,
):
    base_models = localization_model.get('base_models', [])
    if not isinstance(base_models, list) or len(base_models) != 2:
        raise ValueError('targetp_blend_v1 requires exactly two base_models.')
    base_probs = list()
    for base_model in base_models:
        if not isinstance(base_model, dict):
            raise ValueError('Invalid targetp_blend_v1 base model payload.')
        base_model_type = str(base_model.get('model_type', '')).strip()
        submodel = base_model.get('localization_model', {})
        _, probs = _predict_localization_from_model(
            aa_seq=aa_seq,
            feature_vec=feature_vec,
            localization_model=submodel,
            model_type=base_model_type,
            organism_group=organism_group,
        )
        base_probs.append(apply_organism_group_constraints(
            class_probs=probs,
            organism_group=organism_group,
        ))

    blend_probs = _targetp_blend_class_probabilities(
        prob_a=base_probs[0],
        prob_b=base_probs[1],
        alpha_by_class=localization_model.get('alpha_by_class', 1.0),
    )
    pred_class, out_probs = postprocess_localization_probabilities(
        class_probs=blend_probs,
        localization_model=localization_model,
    )
    specialist_pred, specialist_details = _apply_targetp_specialist_postprocess(
        aa_seq=aa_seq,
        base_probs=out_probs,
        prob_a=base_probs[0],
        prob_b=base_probs[1],
        localization_model=localization_model,
        organism_group=organism_group,
    )
    if specialist_pred is not None:
        pred_class = specialist_pred
    if return_details:
        return pred_class, out_probs, {
            'base_model_probabilities': [dict(base_probs[0]), dict(base_probs[1])],
            'blend_probabilities': dict(blend_probs),
            'specialist_postprocess': specialist_details,
        }
    return pred_class, out_probs


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
    strategy_details = None
    if model_type == 'targetp_blend_v1':
        pred_class, class_probs, strategy_details = predict_targetp_blend_localization(
            aa_seq=aa_seq,
            feature_vec=feats,
            localization_model=localization_model,
            organism_group=organism_group,
            return_details=True,
        )
    elif localization_strategy == 'two_stage':
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
            organism_group=organism_group,
        )
    if model_type != 'targetp_blend_v1':
        class_probs = apply_organism_group_constraints(
            class_probs=class_probs,
            organism_group=organism_group,
        )
        pred_class, class_probs = postprocess_localization_probabilities(
            class_probs=class_probs,
            localization_model=localization_model,
        )
        if model_type == 'targetp_feature_ensemble_v1':
            pred_class, strategy_details = _apply_targetp_feature_ltp_specialist_postprocess(
                aa_seq=aa_seq,
                base_probs=class_probs,
                pred_class=pred_class,
                localization_model=localization_model,
                organism_group=organism_group,
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
    if model_type == 'targetp_blend_v1':
        out['targetp_blend_details'] = strategy_details
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
    if model_type in [
        'bilstm_attention_v1',
        'esm_head_v1',
        'multilabel_cnn_v1',
        'targetp_blend_v1',
        'targetp_feature_ensemble_v1',
        'targetp_torch_v1',
    ]:
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
            try:
                payload = torch.load(path, map_location='cpu', weights_only=False)
            except TypeError:
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
        'targetp_blend_v1',
        'targetp_feature_ensemble_v1',
        'targetp_torch_v1',
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

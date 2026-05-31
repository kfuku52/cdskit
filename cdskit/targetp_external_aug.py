import argparse
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

from cdskit.localize_learn import LOCALIZATION_CLASSES, build_training_matrix
from cdskit.localize_model import (
    FEATURE_NAMES,
    _targetp_feature_ltp_specialist_feature_vector,
    fit_perox_binary_classifier,
    save_localize_model,
)
from cdskit.targetp_labeling import strict_uniprot_targetp_label
from cdskit.targetp_external_eval import (
    _targetp_class_from_deeploc_localization_labels,
    build_deeploc_sorting_rows,
    filter_rows_by_mmseqs_similarity,
    is_exact_targetp_overlap,
    load_targetp_exclusion_keys,
    organism_group_from_row,
    read_tsv,
    write_tsv,
)
from cdskit.targetp_feature_ensemble import (
    _metrics_from_prediction_indices,
    _prediction_indices_with_thresholds,
    build_targetp_feature_matrix,
    evaluate_foldwise_thresholds,
    make_targetp_feature_classifier,
    optimize_class_thresholds,
    predict_feature_classifier_prob_matrix,
)


TARGETP_EXTERNAL_AUG_DEFAULTS = {
    'model_kind': 'extra_trees',
    'n_estimators': 200,
    'random_state': 2100,
    'class_weight': 'balanced',
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'external_weight': 0.25,
    'max_external_per_class': 5000,
    'seed': 101,
}


def _blank_exclusion_keys():
    return {
        'accessions': set(),
        'sequences': set(),
    }


def _row_key(value):
    return str(value or '').strip().split('.')[0]


def _sequence_key(value):
    return str(value or '').strip().upper()


def load_external_exclusion_keys(paths):
    keys = _blank_exclusion_keys()
    for path in paths or []:
        if str(path).strip() == '':
            continue
        for row in read_tsv(str(path)):
            accession = _row_key(row.get('accession', ''))
            sequence = _sequence_key(row.get('sequence', ''))
            if accession != '':
                keys['accessions'].add(accession)
            if sequence != '':
                keys['sequences'].add(sequence)
    return keys


def load_external_exclusion_rows(paths):
    rows = list()
    for path in paths or []:
        if str(path).strip() == '':
            continue
        for row in read_tsv(str(path)):
            accession = _row_key(row.get('accession', ''))
            sequence = _sequence_key(row.get('sequence', ''))
            if accession == '' and sequence == '':
                continue
            rows.append({
                'accession': accession,
                'sequence': sequence,
            })
    return rows


def is_external_excluded_row(row, exclusion_keys):
    if not isinstance(exclusion_keys, dict):
        return False
    accession = _row_key(row.get('accession', ''))
    sequence = _sequence_key(row.get('sequence', ''))
    return (
        (accession != '' and accession in exclusion_keys.get('accessions', set()))
        or (sequence != '' and sequence in exclusion_keys.get('sequences', set()))
    )


def _external_training_row(source, accession, sequence, organism_group, localization):
    return {
        'source': source,
        'accession': accession,
        'sequence': sequence,
        'organism_group': organism_group,
        'localization': localization,
        'peroxisome': 'no',
    }


def build_strict_uniprot_external_rows(path, targetp_keys, exclude_exact=True, exclusion_keys=None):
    out = list()
    skipped = Counter()
    for row in read_tsv(path):
        if exclude_exact and is_exact_targetp_overlap(row=row, targetp_keys=targetp_keys):
            skipped['targetp_exact_overlap'] += 1
            continue
        if is_external_excluded_row(row=row, exclusion_keys=exclusion_keys):
            skipped['external_exclusion_overlap'] += 1
            continue
        organism_group = organism_group_from_row(row)
        class_name, reason = strict_uniprot_targetp_label(
            location_text=row.get('cc_subcellular_location', row.get('localization', '')),
            organism_group=organism_group,
        )
        if class_name is None:
            skipped[reason] += 1
            continue
        out.append(_external_training_row(
            source='uniprot_strict',
            accession=row.get('accession', ''),
            sequence=row.get('sequence', ''),
            organism_group=organism_group,
            localization=class_name,
        ))
    return out, dict(skipped)


def build_deeploc_localization_external_rows(path, targetp_keys, exclude_exact=True, exclusion_keys=None):
    out = list()
    skipped = Counter()
    for row in read_tsv(path):
        if exclude_exact and is_exact_targetp_overlap(row=row, targetp_keys=targetp_keys):
            skipped['targetp_exact_overlap'] += 1
            continue
        if is_external_excluded_row(row=row, exclusion_keys=exclusion_keys):
            skipped['external_exclusion_overlap'] += 1
            continue
        class_name, _, ambiguous = _targetp_class_from_deeploc_localization_labels(
            row.get('localization_labels', '')
        )
        if ambiguous:
            skipped['ambiguous_targetp_equivalent_label'] += 1
            continue
        organism_group = organism_group_from_row(row)
        if class_name == 'cTP' and organism_group != 'plant':
            skipped['nonplant_ctp_proxy'] += 1
            continue
        out.append(_external_training_row(
            source='deeploc21_localization',
            accession=row.get('accession', ''),
            sequence=row.get('sequence', ''),
            organism_group=organism_group,
            localization=class_name,
        ))
    return out, dict(skipped)


def _deduplicate_external_rows(rows):
    skipped = Counter()
    by_sequence = dict()
    conflicts = set()
    for row in rows:
        sequence = str(row.get('sequence', '') or '').strip().upper()
        if sequence == '':
            skipped['missing_sequence'] += 1
            continue
        previous = by_sequence.get(sequence)
        if previous is None:
            by_sequence[sequence] = dict(row)
        elif previous.get('localization', '') != row.get('localization', ''):
            conflicts.add(sequence)
        else:
            skipped['duplicate_same_label_sequence'] += 1
    skipped['conflicting_duplicate_sequence'] += len(conflicts)
    return [
        row for sequence, row in by_sequence.items()
        if sequence not in conflicts
    ], dict(skipped)


def _sample_external_rows(rows, max_per_class, seed):
    rng = random.Random(int(seed))
    by_class = defaultdict(list)
    for row in rows:
        by_class[row.get('localization', '')].append(row)
    sampled = list()
    sample_report = dict()
    for class_name in LOCALIZATION_CLASSES:
        class_rows = list(by_class[class_name])
        rng.shuffle(class_rows)
        if int(max_per_class) > 0:
            class_rows = class_rows[:int(max_per_class)]
        sampled.extend(class_rows)
        sample_report[class_name] = {
            'available': int(len(by_class[class_name])),
            'sampled': int(len(class_rows)),
        }
    rng.shuffle(sampled)
    return sampled, sample_report


def _prefilter_rows_for_similarity(rows, max_per_class, seed):
    max_per_class = int(max_per_class)
    if max_per_class <= 0:
        return list(rows), {
            'enabled': False,
            'reason': 'max_per_class_disabled',
            'input_rows': int(len(rows)),
            'rows': int(len(rows)),
        }
    counts = Counter(row.get('localization', '') for row in rows)
    if all(int(counts.get(class_name, 0)) <= max_per_class for class_name in LOCALIZATION_CLASSES):
        return list(rows), {
            'enabled': False,
            'reason': 'all_classes_within_limit',
            'input_rows': int(len(rows)),
            'rows': int(len(rows)),
            'max_per_class': int(max_per_class),
        }
    sampled, sample_report = _sample_external_rows(
        rows=rows,
        max_per_class=max_per_class,
        seed=seed,
    )
    return sampled, {
        'enabled': True,
        'input_rows': int(len(rows)),
        'rows': int(len(sampled)),
        'max_per_class': int(max_per_class),
        'sample_report': sample_report,
    }


def split_external_train_calibration_rows(rows, calibration_fraction=0.0, seed=101):
    fraction = float(calibration_fraction)
    if (not np.isfinite(fraction)) or fraction <= 0.0:
        return list(rows), [], {
            'enabled': False,
            'calibration_fraction': float(max(0.0, fraction if np.isfinite(fraction) else 0.0)),
            'train_counts': dict(Counter(row.get('localization', '') for row in rows)),
            'calibration_counts': {},
        }
    fraction = min(0.95, fraction)
    rng = random.Random(int(seed))
    by_class = defaultdict(list)
    for row in rows:
        by_class[row.get('localization', '')].append(row)
    train_rows = list()
    calibration_rows = list()
    for class_name in LOCALIZATION_CLASSES:
        class_rows = list(by_class[class_name])
        rng.shuffle(class_rows)
        if len(class_rows) <= 1:
            train_rows.extend(class_rows)
            continue
        n_calibration = int(round(float(len(class_rows)) * fraction))
        n_calibration = max(1, min(len(class_rows) - 1, n_calibration))
        calibration_rows.extend(class_rows[:n_calibration])
        train_rows.extend(class_rows[n_calibration:])
    rng.shuffle(train_rows)
    rng.shuffle(calibration_rows)
    return train_rows, calibration_rows, {
        'enabled': True,
        'calibration_fraction': float(fraction),
        'train_counts': dict(Counter(row.get('localization', '') for row in train_rows)),
        'calibration_counts': dict(Counter(row.get('localization', '') for row in calibration_rows)),
    }


def build_external_augmented_training_rows(
    targetp_tsv,
    uniprot_tsv,
    extra_uniprot_tsvs=None,
    exclusion_tsvs=None,
    deeploc_dir='',
    include_deeploc=True,
    max_per_class=5000,
    seed=101,
    use_mmseqs=False,
    exclusion_mmseqs=None,
    mmseqs_min_seq_id=0.30,
    mmseqs_min_coverage=0.80,
    threads=1,
):
    targetp_keys = load_targetp_exclusion_keys(targetp_tsv=targetp_tsv)
    targetp_rows = list(targetp_keys['rows'])
    exclusion_keys = load_external_exclusion_keys(exclusion_tsvs or [])
    exclusion_rows = load_external_exclusion_rows(exclusion_tsvs or [])
    use_exclusion_mmseqs = bool(use_mmseqs) if exclusion_mmseqs is None else bool(exclusion_mmseqs)
    rows = list()
    skipped = Counter()
    uniprot_paths = [str(uniprot_tsv)]
    if extra_uniprot_tsvs is not None:
        uniprot_paths.extend([
            str(path) for path in extra_uniprot_tsvs
            if str(path).strip() != ''
        ])
    for path_i, path in enumerate(uniprot_paths):
        uniprot_rows, uniprot_skipped = build_strict_uniprot_external_rows(
            path=path,
            targetp_keys=targetp_keys,
            exclude_exact=True,
            exclusion_keys=exclusion_keys,
        )
        rows.extend(uniprot_rows)
        prefix = 'uniprot' if path_i == 0 else 'extra_uniprot{}'.format(path_i)
        skipped.update({
            '{}_{}'.format(prefix, key): value
            for key, value in uniprot_skipped.items()
        })

    if bool(include_deeploc) and str(deeploc_dir).strip() != '':
        localization_rows, localization_skipped = build_deeploc_localization_external_rows(
            path=os.path.join(deeploc_dir, 'deeploc21_localization_train_validation.tsv'),
            targetp_keys=targetp_keys,
            exclude_exact=True,
            exclusion_keys=exclusion_keys,
        )
        rows.extend(localization_rows)
        skipped.update({'deeploc_localization_{}'.format(key): value for key, value in localization_skipped.items()})
        sorting_rows, sorting_skipped = build_deeploc_sorting_rows(
            path=os.path.join(deeploc_dir, 'deeploc21_sorting_signals.tsv'),
            targetp_keys=targetp_keys,
            exclude_exact=True,
        )
        for row in sorting_rows:
            if is_external_excluded_row(row=row, exclusion_keys=exclusion_keys):
                skipped['deeploc_sorting_external_exclusion_overlap'] += 1
                continue
            rows.append(_external_training_row(
                source='deeploc21_sorting_signals',
                accession=row.get('accession', ''),
                sequence=row.get('sequence', ''),
                organism_group=row.get('organism_group', ''),
                localization=row.get('true_class', ''),
            ))
        skipped.update({'deeploc_sorting_{}'.format(key): value for key, value in sorting_skipped.items()})

    deduped, dedup_skipped = _deduplicate_external_rows(rows=rows)
    skipped.update(dedup_skipped)
    needs_similarity_prefilter = bool(use_mmseqs) or bool(use_exclusion_mmseqs)
    similarity_rows, similarity_prefilter_report = _prefilter_rows_for_similarity(
        rows=deduped,
        max_per_class=max_per_class,
        seed=seed,
    ) if needs_similarity_prefilter else (list(deduped), {
        'enabled': False,
        'reason': 'mmseqs_disabled',
        'input_rows': int(len(deduped)),
        'rows': int(len(deduped)),
    })
    if bool(use_mmseqs):
        filtered, mmseqs_report = filter_rows_by_mmseqs_similarity(
            rows=similarity_rows,
            targetp_rows=targetp_rows,
            min_seq_id=float(mmseqs_min_seq_id),
            min_coverage=float(mmseqs_min_coverage),
            threads=int(threads),
            enabled=True,
        )
    else:
        filtered = list(similarity_rows)
        mmseqs_report = {
            'requested': False,
            'available': False,
            'min_seq_id': float(mmseqs_min_seq_id),
            'min_coverage': float(mmseqs_min_coverage),
            'removed': 0,
            'kept': int(len(filtered)),
            'status': 'disabled',
        }
    if bool(use_exclusion_mmseqs) and len(exclusion_rows) > 0:
        filtered, exclusion_mmseqs_report = filter_rows_by_mmseqs_similarity(
            rows=filtered,
            targetp_rows=exclusion_rows,
            min_seq_id=float(mmseqs_min_seq_id),
            min_coverage=float(mmseqs_min_coverage),
            threads=int(threads),
            enabled=True,
        )
    else:
        exclusion_mmseqs_report = {
            'requested': bool(use_exclusion_mmseqs),
            'available': False,
            'min_seq_id': float(mmseqs_min_seq_id),
            'min_coverage': float(mmseqs_min_coverage),
            'removed': 0,
            'kept': int(len(filtered)),
            'status': (
                'no_external_exclusion_rows'
                if len(exclusion_rows) == 0 else 'disabled'
            ),
        }
    sampled, sample_report = _sample_external_rows(
        rows=filtered,
        max_per_class=max_per_class,
        seed=seed,
    )
    return sampled, {
        'candidate_rows': int(len(rows)),
        'deduplicated_rows': int(len(deduped)),
        'similarity_prefilter': similarity_prefilter_report,
        'filtered_rows': int(len(filtered)),
        'sampled_rows': int(len(sampled)),
        'sample_report': sample_report,
        'mmseqs_similarity_filter': mmseqs_report,
        'external_exclusion_similarity_filter': exclusion_mmseqs_report,
        'skipped': dict(skipped),
        'sampled_counts': dict(Counter(row.get('localization', '') for row in sampled)),
        'external_exclusion_tsvs': [str(path) for path in (exclusion_tsvs or [])],
        'external_exclusion_accessions': int(len(exclusion_keys['accessions'])),
        'external_exclusion_sequences': int(len(exclusion_keys['sequences'])),
    }


def _true_idx_from_rows(rows, class_names):
    class_to_idx = {class_name: class_i for class_i, class_name in enumerate(class_names)}
    return np.asarray([
        int(class_to_idx[row.get('localization', '')])
        for row in rows
    ], dtype=np.int64)


def _external_sample_weight_vector(rows, external_weight, external_class_weights=None):
    base_weight = float(external_weight)
    class_weights = dict(external_class_weights or {})
    values = list()
    for row in rows:
        class_name = row.get('localization', '')
        class_weight = float(class_weights.get(class_name, 1.0))
        values.append(base_weight * class_weight)
    return np.asarray(values, dtype=np.float64)


def _parse_class_list(value, default):
    if value is None:
        return list(default)
    if isinstance(value, str):
        values = [
            part.strip() for part in str(value).split(',')
            if part.strip() != ''
        ]
    else:
        values = [str(part).strip() for part in value if str(part).strip() != '']
    out = list()
    for class_name in values:
        if class_name not in LOCALIZATION_CLASSES:
            raise ValueError('Unknown localization class: {}'.format(class_name))
        if class_name not in out:
            out.append(class_name)
    if len(out) == 0:
        return list(default)
    return out


def _feature_ltp_specialist_matrix(rows, prob_matrix, class_names):
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    class_names = list(class_names)
    out = list()
    for row_i, row in enumerate(rows):
        probs = {
            class_name: float(prob_matrix[row_i, class_i])
            for class_i, class_name in enumerate(class_names)
        }
        out.append(_targetp_feature_ltp_specialist_feature_vector(
            aa_seq=row.get('sequence', ''),
            base_probs=probs,
            organism_group=row.get('organism_group', ''),
        ))
    if len(out) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(out).astype(np.float32)


def _organism_group_is_plant(row):
    return str(row.get('organism_group', '') or '').strip().lower() == 'plant'


def _train_feature_ltp_specialist(
    classifier,
    training_rows,
    training_features,
    training_labels,
    training_sample_weight,
    calibration_rows,
    class_thresholds,
    class_names,
    source_classes=None,
    negative_classes=None,
    score_grid=None,
    model_kind='extra_trees',
    n_estimators=120,
    random_state=3100,
    class_weight='balanced',
    max_features='sqrt',
    min_samples_leaf=1,
    mass_threshold=0.0,
):
    class_names = list(class_names)
    source_classes = _parse_class_list(source_classes, default=['cTP'])
    negative_classes = _parse_class_list(negative_classes, default=['cTP'])
    ltp_idx = int(class_names.index('lTP'))
    fit_classes = set(negative_classes + ['lTP'])
    train_mask = np.asarray([
        _organism_group_is_plant(row) and row.get('localization', '') in fit_classes
        for row in training_rows
    ], dtype=bool)
    if len(set(np.asarray(training_labels, dtype=np.int64)[train_mask].tolist())) < 2:
        return None, {
            'enabled': False,
            'reason': 'specialist training split lacks cTP/lTP classes',
            'n_train': int(np.sum(train_mask)),
        }

    training_prob = predict_feature_classifier_prob_matrix(
        classifier=classifier,
        feature_matrix=training_features,
        class_names=class_names,
    )
    specialist_features = _feature_ltp_specialist_matrix(
        rows=training_rows,
        prob_matrix=training_prob,
        class_names=class_names,
    )
    specialist = make_targetp_feature_classifier(
        model_kind=model_kind,
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        class_weight=class_weight,
        max_features=max_features,
        min_samples_leaf=int(min_samples_leaf),
    )
    y = (np.asarray(training_labels, dtype=np.int64)[train_mask] == ltp_idx).astype(np.int64)
    specialist.fit(
        specialist_features[train_mask, :],
        y,
        sample_weight=np.asarray(training_sample_weight, dtype=np.float64)[train_mask],
    )

    threshold = 0.5
    threshold_metrics = None
    threshold_source = 'default'
    calibration_report = {
        'n_train': int(np.sum(train_mask)),
        'train_positive': int(np.sum(y == 1)),
        'train_negative': int(np.sum(y == 0)),
    }
    if len(calibration_rows) > 0:
        calibration_features = build_targetp_feature_matrix(rows=calibration_rows).astype(np.float32)
        calibration_prob = predict_feature_classifier_prob_matrix(
            classifier=classifier,
            feature_matrix=calibration_features,
            class_names=class_names,
        )
        calibration_idx = _true_idx_from_rows(rows=calibration_rows, class_names=class_names)
        threshold_vec = np.asarray([
            float(class_thresholds.get(class_name, 1.0))
            for class_name in class_names
        ], dtype=np.float64)
        base_pred = _prediction_indices_with_thresholds(
            prob_matrix=calibration_prob,
            thresholds=threshold_vec,
        )
        calibration_specialist_features = _feature_ltp_specialist_matrix(
            rows=calibration_rows,
            prob_matrix=calibration_prob,
            class_names=class_names,
        )
        proba = np.asarray(
            specialist.predict_proba(calibration_specialist_features),
            dtype=np.float64,
        )
        classes = [int(value) for value in list(getattr(specialist, 'classes_', []))]
        if 1 in classes:
            scores = proba[:, classes.index(1)]
            source_idx = np.asarray([
                int(class_names.index(class_name)) for class_name in source_classes
            ], dtype=np.int64)
            plant_mask = np.asarray([
                _organism_group_is_plant(row) for row in calibration_rows
            ], dtype=bool)
            if score_grid is None:
                score_grid = np.linspace(0.01, 0.99, 99)
            best_metrics = None
            best_threshold = float(threshold)
            best_override_count = 0
            for trial in score_grid:
                trial_pred = base_pred.copy()
                override = (
                    plant_mask
                    & np.isin(base_pred, source_idx)
                    & (scores >= float(trial))
                )
                trial_pred[override] = ltp_idx
                metrics = _metrics_from_prediction_indices(
                    pred_idx=trial_pred,
                    true_idx=calibration_idx,
                    class_names=class_names,
                )
                if best_metrics is None or metrics['macro_f1'] > best_metrics['macro_f1']:
                    best_threshold = float(trial)
                    best_metrics = metrics
                    best_override_count = int(np.sum(override))
            threshold = float(best_threshold)
            threshold_metrics = best_metrics
            threshold_source = 'external_calibration'
            calibration_report.update({
                'threshold_source': threshold_source,
                'threshold_macro_f1': float(best_metrics['macro_f1']),
                'threshold_overall_accuracy': float(best_metrics['overall_accuracy']),
                'threshold_override_count': int(best_override_count),
            })
    payload = {
        'enabled': True,
        'models': [specialist],
        'weights': [1.0],
        'threshold': float(threshold),
        'source_classes': list(source_classes),
        'negative_classes': list(negative_classes),
        'mass_threshold': float(mass_threshold),
        'feature_profile': 'targetp_feature_ltp_specialist_v1',
    }
    calibration_report.update({
        'enabled': True,
        'threshold': float(threshold),
        'threshold_source': threshold_source,
        'source_classes': list(source_classes),
        'negative_classes': list(negative_classes),
        'mass_threshold': float(mass_threshold),
    })
    if threshold_metrics is not None:
        calibration_report['threshold_metrics'] = threshold_metrics
    return payload, calibration_report


def _fold_ids_from_rows(rows):
    return np.asarray([str(row.get('fold_id', '')) for row in rows])


def run_external_augmented_feature_oof(
    training_tsv,
    uniprot_tsv,
    extra_uniprot_tsvs=None,
    exclusion_tsvs=None,
    deeploc_dir='',
    include_deeploc=True,
    max_external_per_class=5000,
    external_weight=0.25,
    external_class_weights=None,
    seed=101,
    use_mmseqs=False,
    exclusion_mmseqs=None,
    mmseqs_min_seq_id=0.30,
    mmseqs_min_coverage=0.80,
    threads=1,
    model_kind='extra_trees',
    n_estimators=200,
    random_state=2100,
    class_weight='balanced',
    max_features='sqrt',
    min_samples_leaf=1,
    threshold_grid=None,
):
    class_names = list(LOCALIZATION_CLASSES)
    target_rows = read_tsv(training_tsv)
    true_idx = _true_idx_from_rows(rows=target_rows, class_names=class_names)
    fold_ids = _fold_ids_from_rows(rows=target_rows)
    if np.any(fold_ids == ''):
        raise ValueError('TargetP external-augmented OOF requires fold_id in every target row.')
    external_rows, external_report = build_external_augmented_training_rows(
        targetp_tsv=training_tsv,
        uniprot_tsv=uniprot_tsv,
        extra_uniprot_tsvs=extra_uniprot_tsvs,
        exclusion_tsvs=exclusion_tsvs,
        deeploc_dir=deeploc_dir,
        include_deeploc=include_deeploc,
        max_per_class=int(max_external_per_class),
        seed=int(seed),
        use_mmseqs=bool(use_mmseqs),
        exclusion_mmseqs=exclusion_mmseqs,
        mmseqs_min_seq_id=float(mmseqs_min_seq_id),
        mmseqs_min_coverage=float(mmseqs_min_coverage),
        threads=int(threads),
    )
    if len(external_rows) == 0:
        raise ValueError('No external rows were available after filtering.')

    target_features = build_targetp_feature_matrix(rows=target_rows).astype(np.float32)
    external_features = build_targetp_feature_matrix(rows=external_rows).astype(np.float32)
    external_idx = _true_idx_from_rows(rows=external_rows, class_names=class_names)
    prob_matrix = np.zeros((target_features.shape[0], len(class_names)), dtype=np.float64)
    fold_report = list()
    for fold_i, fold_id in enumerate(sorted(set(fold_ids.tolist()))):
        valid_mask = fold_ids == fold_id
        train_mask = ~valid_mask
        features = np.vstack([target_features[train_mask, :], external_features])
        labels = np.concatenate([true_idx[train_mask], external_idx])
        external_sample_weight = _external_sample_weight_vector(
            rows=external_rows,
            external_weight=float(external_weight),
            external_class_weights=external_class_weights,
        )
        sample_weight = np.concatenate([
            np.ones((int(np.sum(train_mask)),), dtype=np.float64),
            external_sample_weight,
        ])
        classifier = make_targetp_feature_classifier(
            model_kind=model_kind,
            n_estimators=int(n_estimators),
            random_state=int(random_state) + int(fold_i),
            class_weight=class_weight,
            max_features=max_features,
            min_samples_leaf=int(min_samples_leaf),
        )
        classifier.fit(features, labels, sample_weight=sample_weight)
        prob_matrix[valid_mask, :] = predict_feature_classifier_prob_matrix(
            classifier=classifier,
            feature_matrix=target_features[valid_mask, :],
            class_names=class_names,
        )
        fold_report.append({
            'fold_id': str(fold_id),
            'n_target_train': int(np.sum(train_mask)),
            'n_target_valid': int(np.sum(valid_mask)),
            'n_external_train': int(len(external_idx)),
        })

    argmax_metrics = _metrics_from_prediction_indices(
        pred_idx=np.argmax(prob_matrix, axis=1).astype(np.int64),
        true_idx=true_idx,
        class_names=class_names,
    )
    if threshold_grid is None:
        threshold_grid = np.arange(0.05, 2.05, 0.05, dtype=np.float64)
    threshold_metrics = evaluate_foldwise_thresholds(
        prob_matrix=prob_matrix,
        true_idx=true_idx,
        fold_ids=fold_ids,
        class_names=class_names,
        threshold_grid=threshold_grid,
    )
    return {
        'prob_matrix': prob_matrix,
        'true_idx': true_idx,
        'class_names': class_names,
        'fold_ids': fold_ids,
        'argmax': argmax_metrics,
        'foldwise_threshold': threshold_metrics,
        'external_report': external_report,
        'folds': fold_report,
        'profile': {
            'training_tsv': str(training_tsv),
            'uniprot_tsv': str(uniprot_tsv),
            'extra_uniprot_tsvs': [
                str(path) for path in (extra_uniprot_tsvs or [])
            ],
            'exclusion_tsvs': [
                str(path) for path in (exclusion_tsvs or [])
            ],
            'deeploc_dir': str(deeploc_dir),
            'include_deeploc': bool(include_deeploc),
            'max_external_per_class': int(max_external_per_class),
            'external_weight': float(external_weight),
            'external_class_weights': dict(external_class_weights or {}),
            'seed': int(seed),
            'use_mmseqs': bool(use_mmseqs),
            'exclusion_mmseqs': None if exclusion_mmseqs is None else bool(exclusion_mmseqs),
            'mmseqs_min_seq_id': float(mmseqs_min_seq_id),
            'mmseqs_min_coverage': float(mmseqs_min_coverage),
            'threads': int(threads),
            'model_kind': str(model_kind),
            'n_estimators': int(n_estimators),
            'random_state': int(random_state),
            'class_weight': str(class_weight),
            'max_features': str(max_features),
            'min_samples_leaf': int(min_samples_leaf),
        },
    }


def fit_external_augmented_feature_runtime_model(
    training_tsv,
    uniprot_tsv,
    extra_uniprot_tsvs=None,
    exclusion_tsvs=None,
    deeploc_dir='',
    include_deeploc=True,
    max_external_per_class=5000,
    external_weight=0.25,
    external_class_weights=None,
    calibration_fraction=0.0,
    calibration_seed=None,
    calibration_threshold_grid=None,
    seed=101,
    use_mmseqs=False,
    exclusion_mmseqs=None,
    mmseqs_min_seq_id=0.30,
    mmseqs_min_coverage=0.80,
    threads=1,
    class_thresholds=None,
    model_kind='extra_trees',
    n_estimators=200,
    random_state=2100,
    class_weight='balanced',
    max_features='sqrt',
    min_samples_leaf=1,
    ltp_specialist=False,
    ltp_specialist_source_classes=None,
    ltp_specialist_negative_classes=None,
    ltp_specialist_score_grid=None,
    ltp_specialist_mass_threshold=0.0,
):
    class_names = list(LOCALIZATION_CLASSES)
    target_rows = read_tsv(training_tsv)
    external_rows, external_report = build_external_augmented_training_rows(
        targetp_tsv=training_tsv,
        uniprot_tsv=uniprot_tsv,
        extra_uniprot_tsvs=extra_uniprot_tsvs,
        exclusion_tsvs=exclusion_tsvs,
        deeploc_dir=deeploc_dir,
        include_deeploc=include_deeploc,
        max_per_class=int(max_external_per_class),
        seed=int(seed),
        use_mmseqs=bool(use_mmseqs),
        exclusion_mmseqs=exclusion_mmseqs,
        mmseqs_min_seq_id=float(mmseqs_min_seq_id),
        mmseqs_min_coverage=float(mmseqs_min_coverage),
        threads=int(threads),
    )
    if len(external_rows) == 0:
        raise ValueError('No external rows were available after filtering.')
    calibration_seed = int(seed) + 791 if calibration_seed is None else int(calibration_seed)
    external_train_rows, external_calibration_rows, calibration_report = split_external_train_calibration_rows(
        rows=external_rows,
        calibration_fraction=float(calibration_fraction),
        seed=calibration_seed,
    )
    if len(external_train_rows) == 0:
        raise ValueError('No external training rows were available after calibration split.')

    target_features = build_targetp_feature_matrix(rows=target_rows).astype(np.float32)
    external_features = build_targetp_feature_matrix(rows=external_train_rows).astype(np.float32)
    target_idx = _true_idx_from_rows(rows=target_rows, class_names=class_names)
    external_idx = _true_idx_from_rows(rows=external_train_rows, class_names=class_names)
    features = np.vstack([target_features, external_features])
    labels = np.concatenate([target_idx, external_idx])
    external_sample_weight = _external_sample_weight_vector(
        rows=external_train_rows,
        external_weight=float(external_weight),
        external_class_weights=external_class_weights,
    )
    sample_weight = np.concatenate([
        np.ones((len(target_idx),), dtype=np.float64),
        external_sample_weight,
    ])
    classifier = make_targetp_feature_classifier(
        model_kind=model_kind,
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        class_weight=class_weight,
        max_features=max_features,
        min_samples_leaf=int(min_samples_leaf),
    )
    classifier.fit(features, labels, sample_weight=sample_weight)

    broad_features, _, _, perox_labels, skipped, _ = build_training_matrix(
        rows=target_rows,
        seq_col='sequence',
        seqtype='protein',
        codontable=1,
        label_mode='explicit',
        localization_col='localization',
        perox_col='peroxisome',
        skip_ambiguous=True,
        cv_fold_col='',
    )
    if int(skipped) != 0:
        raise ValueError('External-augmented feature runtime export requires no skipped target rows.')
    if class_thresholds is None:
        class_thresholds = {class_name: 1.0 for class_name in class_names}
        calibration_idx = (
            _true_idx_from_rows(rows=external_calibration_rows, class_names=class_names)
            if len(external_calibration_rows) > 0 else np.zeros((0,), dtype=np.int64)
        )
        if set(calibration_idx.tolist()) == set(range(len(class_names))):
            calibration_features = build_targetp_feature_matrix(rows=external_calibration_rows).astype(np.float32)
            calibration_prob = predict_feature_classifier_prob_matrix(
                classifier=classifier,
                feature_matrix=calibration_features,
                class_names=class_names,
            )
            threshold_grid = (
                np.arange(0.05, 2.05, 0.05, dtype=np.float64)
                if calibration_threshold_grid is None
                else np.asarray([float(value) for value in calibration_threshold_grid], dtype=np.float64)
            )
            threshold_vec, threshold_metrics = optimize_class_thresholds(
                prob_matrix=calibration_prob,
                true_idx=calibration_idx,
                class_names=class_names,
                grid=threshold_grid,
            )
            class_thresholds = {
                class_names[class_i]: float(threshold_vec[class_i])
                for class_i in range(len(class_names))
            }
            calibration_report['threshold_tuning'] = {
                'enabled': True,
                'macro_f1': float(threshold_metrics['macro_f1']),
                'overall_accuracy': float(threshold_metrics['overall_accuracy']),
                'class_thresholds': dict(class_thresholds),
            }
        elif len(external_calibration_rows) > 0:
            calibration_report['threshold_tuning'] = {
                'enabled': False,
                'reason': 'calibration split does not contain every class',
                'observed_classes': [
                    class_names[class_i] for class_i in sorted(set(calibration_idx.tolist()))
                ],
            }
        else:
            calibration_report['threshold_tuning'] = {
                'enabled': False,
                'reason': 'no calibration rows',
            }
    ltp_specialist_payload = None
    ltp_specialist_report = {'enabled': False}
    if bool(ltp_specialist):
        training_rows = list(target_rows) + list(external_train_rows)
        ltp_specialist_payload, ltp_specialist_report = _train_feature_ltp_specialist(
            classifier=classifier,
            training_rows=training_rows,
            training_features=features,
            training_labels=labels,
            training_sample_weight=sample_weight,
            calibration_rows=external_calibration_rows,
            class_thresholds=class_thresholds,
            class_names=class_names,
            source_classes=ltp_specialist_source_classes,
            negative_classes=ltp_specialist_negative_classes,
            score_grid=ltp_specialist_score_grid,
            model_kind=model_kind,
            n_estimators=int(n_estimators),
            random_state=int(random_state) + 919,
            class_weight=class_weight,
            max_features=max_features,
            min_samples_leaf=int(min_samples_leaf),
            mass_threshold=float(ltp_specialist_mass_threshold),
        )
    model = {
        'model_type': 'targetp_feature_ensemble_v1',
        'feature_names': list(FEATURE_NAMES),
        'localization_model': {
            'mode': 'targetp_feature_ensemble',
            'class_order': class_names,
            'classifier': classifier,
            'binary_classifiers': None,
            'class_thresholds': dict(class_thresholds),
            'feature_dim': int(features.shape[1]),
            'feature_profile': 'targetp_feature_ensemble_v1',
            'classifier_profile': {
                'model_kind': str(model_kind),
                'n_estimators': int(n_estimators),
                'random_state': int(random_state),
                'class_weight': str(class_weight),
                'max_features': str(max_features),
                'min_samples_leaf': int(min_samples_leaf),
                'external_augmented': True,
                'external_weight': float(external_weight),
                'external_class_weights': dict(external_class_weights or {}),
            },
        },
        'perox_model': fit_perox_binary_classifier(
            features=broad_features,
            labels=perox_labels,
        ),
        'metadata': {
            'training_tsv': str(training_tsv),
            'uniprot_tsv': str(uniprot_tsv),
            'extra_uniprot_tsvs': [str(path) for path in (extra_uniprot_tsvs or [])],
            'exclusion_tsvs': [str(path) for path in (exclusion_tsvs or [])],
            'use_mmseqs': bool(use_mmseqs),
            'exclusion_mmseqs': None if exclusion_mmseqs is None else bool(exclusion_mmseqs),
            'external_weight': float(external_weight),
            'external_class_weights': dict(external_class_weights or {}),
            'num_target_rows': int(len(target_rows)),
            'num_external_rows': int(len(external_rows)),
            'num_external_train_rows': int(len(external_train_rows)),
            'num_external_calibration_rows': int(len(external_calibration_rows)),
            'external_calibration': calibration_report,
            'ltp_specialist': ltp_specialist_report,
            'model_arch': 'targetp_feature_ensemble_v1_external_augmented',
            'external_report': external_report,
        },
    }
    if ltp_specialist_payload is not None:
        model['localization_model']['targetp_feature_ltp_specialist'] = ltp_specialist_payload
    return model


def write_external_augmented_feature_oof_npz(path, result):
    out_dir = os.path.dirname(path)
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        path,
        prob_matrix=np.asarray(result['prob_matrix'], dtype=np.float64),
        true_idx=np.asarray(result['true_idx'], dtype=np.int64),
        class_names=np.asarray(result['class_names']),
        fold_ids=np.asarray(result['fold_ids']),
    )


def write_external_augmented_feature_report(path, result, external_tsv=''):
    out_dir = os.path.dirname(path)
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)
    report = {
        'profile': result['profile'],
        'argmax': result['argmax'],
        'foldwise_threshold': result['foldwise_threshold'],
        'external_report': result['external_report'],
        'folds': result['folds'],
    }
    if str(external_tsv).strip() != '':
        report['external_tsv'] = str(external_tsv)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Build a fair TargetP OOF from target folds plus strict non-overlapping external weak labels.',
    )
    parser.add_argument('--training_tsv', default='data/localize_bench/targetp2_benchmark.tsv', type=str)
    parser.add_argument('--uniprot_tsv', default='data/localize_bench/eukaryota_full_with_lineage.tsv', type=str)
    parser.add_argument('--extra_uniprot_tsvs', default='', type=str)
    parser.add_argument('--exclusion_tsvs', default='', type=str)
    parser.add_argument('--deeploc_dir', default='data/localize_bench/deeploc21', type=str)
    parser.add_argument('--include_deeploc', default='yes', choices=['yes', 'no'], type=str)
    parser.add_argument('--max_external_per_class', default=TARGETP_EXTERNAL_AUG_DEFAULTS['max_external_per_class'], type=int)
    parser.add_argument('--external_weight', default=TARGETP_EXTERNAL_AUG_DEFAULTS['external_weight'], type=float)
    parser.add_argument('--external_class_weights', default='', type=str)
    parser.add_argument('--calibration_fraction', default=0.0, type=float)
    parser.add_argument('--calibration_seed', default='', type=str)
    parser.add_argument('--calibration_threshold_grid', default='0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00,1.05,1.10,1.15,1.20,1.25,1.30,1.35,1.40,1.45,1.50,1.55,1.60,1.65,1.70,1.75,1.80,1.85,1.90,1.95,2.00,3.00,5.00', type=str)
    parser.add_argument('--seed', default=TARGETP_EXTERNAL_AUG_DEFAULTS['seed'], type=int)
    parser.add_argument('--mmseqs', default='no', choices=['yes', 'no'], type=str)
    parser.add_argument('--exclusion_mmseqs', default='auto', choices=['auto', 'yes', 'no'], type=str)
    parser.add_argument('--mmseqs_min_seq_id', default=0.30, type=float)
    parser.add_argument('--mmseqs_min_coverage', default=0.80, type=float)
    parser.add_argument('--threads', default=1, type=int)
    parser.add_argument('--model_kind', default=TARGETP_EXTERNAL_AUG_DEFAULTS['model_kind'], choices=['extra_trees', 'random_forest'], type=str)
    parser.add_argument('--n_estimators', default=TARGETP_EXTERNAL_AUG_DEFAULTS['n_estimators'], type=int)
    parser.add_argument('--random_state', default=TARGETP_EXTERNAL_AUG_DEFAULTS['random_state'], type=int)
    parser.add_argument('--class_weight', default=TARGETP_EXTERNAL_AUG_DEFAULTS['class_weight'], type=str)
    parser.add_argument('--max_features', default=TARGETP_EXTERNAL_AUG_DEFAULTS['max_features'], type=str)
    parser.add_argument('--min_samples_leaf', default=TARGETP_EXTERNAL_AUG_DEFAULTS['min_samples_leaf'], type=int)
    parser.add_argument('--threshold_grid', default='0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00,1.05,1.10,1.15,1.20,1.25,1.30,1.35,1.40,1.45,1.50,1.55,1.60,1.65,1.70,1.75,1.80,1.85,1.90,1.95,2.00', type=str)
    parser.add_argument('--class_thresholds', default='', type=str)
    parser.add_argument('--ltp_specialist', default='no', choices=['yes', 'no'], type=str)
    parser.add_argument('--ltp_specialist_source_classes', default='cTP', type=str)
    parser.add_argument('--ltp_specialist_negative_classes', default='cTP', type=str)
    parser.add_argument('--ltp_specialist_score_grid', default='0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99', type=str)
    parser.add_argument('--ltp_specialist_mass_threshold', default=0.0, type=float)
    parser.add_argument('--out_npz', required=True, type=str)
    parser.add_argument('--out_json', required=True, type=str)
    parser.add_argument('--external_tsv_out', default='', type=str)
    parser.add_argument('--model_out', default='', type=str)
    return parser


def _yes_no(value):
    return str(value).strip().lower() in ['yes', 'y', 'true', '1']


def _parse_grid(text):
    return sorted(set([
        float(value.strip()) for value in str(text).split(',')
        if value.strip() != ''
    ]))


def _parse_paths(text):
    return [
        value.strip() for value in str(text or '').split(',')
        if value.strip() != ''
    ]


def _parse_class_thresholds(text):
    if str(text or '').strip() == '':
        return None
    thresholds = {class_name: 1.0 for class_name in LOCALIZATION_CLASSES}
    for part in str(text).split(','):
        part = part.strip()
        if part == '':
            continue
        if '=' not in part:
            raise ValueError('--class_thresholds entries should be CLASS=VALUE.')
        class_name, value = part.split('=', 1)
        class_name = class_name.strip()
        if class_name not in LOCALIZATION_CLASSES:
            raise ValueError('Unknown class in --class_thresholds: {}'.format(class_name))
        thresholds[class_name] = float(value.strip())
    return thresholds


def _parse_external_class_weights(text):
    if str(text or '').strip() == '':
        return None
    weights = dict()
    for part in str(text).split(','):
        part = part.strip()
        if part == '':
            continue
        if '=' not in part:
            raise ValueError('--external_class_weights entries should be CLASS=VALUE.')
        class_name, value = part.split('=', 1)
        class_name = class_name.strip()
        if class_name not in LOCALIZATION_CLASSES:
            raise ValueError('Unknown class in --external_class_weights: {}'.format(class_name))
        weight = float(value.strip())
        if (not np.isfinite(weight)) or weight < 0.0:
            raise ValueError('--external_class_weights values should be non-negative finite numbers.')
        weights[class_name] = float(weight)
    return weights


def _parse_exclusion_mmseqs(value):
    value = str(value or 'auto').strip().lower()
    if value == 'auto':
        return None
    return _yes_no(value)


def main():
    args = build_parser().parse_args()
    extra_uniprot_tsvs = _parse_paths(args.extra_uniprot_tsvs)
    exclusion_tsvs = _parse_paths(args.exclusion_tsvs)
    class_thresholds = _parse_class_thresholds(args.class_thresholds)
    external_class_weights = _parse_external_class_weights(args.external_class_weights)
    exclusion_mmseqs = _parse_exclusion_mmseqs(args.exclusion_mmseqs)
    result = run_external_augmented_feature_oof(
        training_tsv=args.training_tsv,
        uniprot_tsv=args.uniprot_tsv,
        extra_uniprot_tsvs=extra_uniprot_tsvs,
        exclusion_tsvs=exclusion_tsvs,
        deeploc_dir=args.deeploc_dir,
        include_deeploc=_yes_no(args.include_deeploc),
        max_external_per_class=int(args.max_external_per_class),
        external_weight=float(args.external_weight),
        external_class_weights=external_class_weights,
        seed=int(args.seed),
        use_mmseqs=_yes_no(args.mmseqs),
        exclusion_mmseqs=exclusion_mmseqs,
        mmseqs_min_seq_id=float(args.mmseqs_min_seq_id),
        mmseqs_min_coverage=float(args.mmseqs_min_coverage),
        threads=int(args.threads),
        model_kind=args.model_kind,
        n_estimators=int(args.n_estimators),
        random_state=int(args.random_state),
        class_weight=args.class_weight,
        max_features=args.max_features,
        min_samples_leaf=int(args.min_samples_leaf),
        threshold_grid=_parse_grid(args.threshold_grid),
    )
    write_external_augmented_feature_oof_npz(path=args.out_npz, result=result)
    if str(args.external_tsv_out).strip() != '':
        external_rows, _ = build_external_augmented_training_rows(
            targetp_tsv=args.training_tsv,
            uniprot_tsv=args.uniprot_tsv,
            extra_uniprot_tsvs=extra_uniprot_tsvs,
            exclusion_tsvs=exclusion_tsvs,
            deeploc_dir=args.deeploc_dir,
            include_deeploc=_yes_no(args.include_deeploc),
            max_per_class=int(args.max_external_per_class),
            seed=int(args.seed),
            use_mmseqs=_yes_no(args.mmseqs),
            exclusion_mmseqs=exclusion_mmseqs,
            mmseqs_min_seq_id=float(args.mmseqs_min_seq_id),
            mmseqs_min_coverage=float(args.mmseqs_min_coverage),
            threads=int(args.threads),
        )
        write_tsv(
            path=args.external_tsv_out,
            rows=external_rows,
            fieldnames=['source', 'accession', 'organism_group', 'localization', 'peroxisome', 'sequence'],
        )
    write_external_augmented_feature_report(
        path=args.out_json,
        result=result,
        external_tsv=args.external_tsv_out,
    )
    if str(args.model_out).strip() != '':
        model = fit_external_augmented_feature_runtime_model(
            training_tsv=args.training_tsv,
            uniprot_tsv=args.uniprot_tsv,
            extra_uniprot_tsvs=extra_uniprot_tsvs,
            exclusion_tsvs=exclusion_tsvs,
            deeploc_dir=args.deeploc_dir,
            include_deeploc=_yes_no(args.include_deeploc),
            max_external_per_class=int(args.max_external_per_class),
            external_weight=float(args.external_weight),
            external_class_weights=external_class_weights,
            calibration_fraction=float(args.calibration_fraction),
            calibration_seed=(
                None if str(args.calibration_seed).strip() == ''
                else int(args.calibration_seed)
            ),
            calibration_threshold_grid=_parse_grid(args.calibration_threshold_grid),
            class_thresholds=class_thresholds,
            seed=int(args.seed),
            use_mmseqs=_yes_no(args.mmseqs),
            exclusion_mmseqs=exclusion_mmseqs,
            mmseqs_min_seq_id=float(args.mmseqs_min_seq_id),
            mmseqs_min_coverage=float(args.mmseqs_min_coverage),
            threads=int(args.threads),
            model_kind=args.model_kind,
            n_estimators=int(args.n_estimators),
            random_state=int(args.random_state),
            class_weight=args.class_weight,
            max_features=args.max_features,
            min_samples_leaf=int(args.min_samples_leaf),
            ltp_specialist=_yes_no(args.ltp_specialist),
            ltp_specialist_source_classes=_parse_class_list(
                args.ltp_specialist_source_classes,
                default=['cTP'],
            ),
            ltp_specialist_negative_classes=_parse_class_list(
                args.ltp_specialist_negative_classes,
                default=['cTP'],
            ),
            ltp_specialist_score_grid=_parse_grid(args.ltp_specialist_score_grid),
            ltp_specialist_mass_threshold=float(args.ltp_specialist_mass_threshold),
        )
        save_localize_model(model=model, path=str(args.model_out))
    print('external_rows={} argmax_macro_f1={:.6f} foldwise_threshold_macro_f1={:.6f}'.format(
        int(result['external_report']['sampled_rows']),
        float(result['argmax']['macro_f1']),
        float(result['foldwise_threshold']['metrics']['macro_f1']),
    ))


if __name__ == '__main__':
    main()

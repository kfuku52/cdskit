import argparse
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

from cdskit.localize_learn import LOCALIZATION_CLASSES
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
    build_targetp_feature_matrix,
    evaluate_foldwise_thresholds,
    make_targetp_feature_classifier,
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


def strict_uniprot_targetp_label(location_text, organism_group):
    txt = str(location_text or '').lower()
    organism_group = str(organism_group or '').strip().lower()
    if txt.strip() == '':
        return None, 'missing_location_text'

    has_sp = ('secreted' in txt) or ('signal peptide' in txt)
    has_mtp = 'mitochond' in txt
    has_ctp = ('chloroplast' in txt) or ('plastid' in txt)
    has_thylakoid = 'thylakoid' in txt
    has_lumen = ('lumen' in txt) or ('lumenal' in txt) or ('luminal' in txt)

    if organism_group != 'plant' and (has_ctp or has_thylakoid):
        return None, 'nonplant_plastid'

    plastid_signal = organism_group == 'plant' and has_ctp
    ltp_signal = organism_group == 'plant' and has_thylakoid and has_lumen
    if sum(1 for value in [has_sp, has_mtp, plastid_signal] if value) > 1:
        return None, 'ambiguous'
    if ltp_signal and (has_sp or has_mtp):
        return None, 'ambiguous'

    if ltp_signal:
        if 'membrane' in txt and 'thylakoid lumen' not in txt:
            return None, 'ltp_membrane_noise'
        return 'lTP', ''
    if plastid_signal:
        if has_thylakoid:
            return None, 'thylakoid_not_lumen'
        return 'cTP', ''
    if has_mtp:
        return 'mTP', ''
    if has_sp:
        return 'SP', ''
    return 'noTP', ''


def _external_training_row(source, accession, sequence, organism_group, localization):
    return {
        'source': source,
        'accession': accession,
        'sequence': sequence,
        'organism_group': organism_group,
        'localization': localization,
        'peroxisome': 'no',
    }


def build_strict_uniprot_external_rows(path, targetp_keys, exclude_exact=True):
    out = list()
    skipped = Counter()
    for row in read_tsv(path):
        if exclude_exact and is_exact_targetp_overlap(row=row, targetp_keys=targetp_keys):
            skipped['targetp_exact_overlap'] += 1
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


def build_deeploc_localization_external_rows(path, targetp_keys, exclude_exact=True):
    out = list()
    skipped = Counter()
    for row in read_tsv(path):
        if exclude_exact and is_exact_targetp_overlap(row=row, targetp_keys=targetp_keys):
            skipped['targetp_exact_overlap'] += 1
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


def build_external_augmented_training_rows(
    targetp_tsv,
    uniprot_tsv,
    extra_uniprot_tsvs=None,
    deeploc_dir='',
    include_deeploc=True,
    max_per_class=5000,
    seed=101,
    use_mmseqs=False,
    mmseqs_min_seq_id=0.30,
    mmseqs_min_coverage=0.80,
    threads=1,
):
    targetp_keys = load_targetp_exclusion_keys(targetp_tsv=targetp_tsv)
    targetp_rows = list(targetp_keys['rows'])
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
        )
        rows.extend(localization_rows)
        skipped.update({'deeploc_localization_{}'.format(key): value for key, value in localization_skipped.items()})
        sorting_rows, sorting_skipped = build_deeploc_sorting_rows(
            path=os.path.join(deeploc_dir, 'deeploc21_sorting_signals.tsv'),
            targetp_keys=targetp_keys,
            exclude_exact=True,
        )
        for row in sorting_rows:
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
    filtered, mmseqs_report = filter_rows_by_mmseqs_similarity(
        rows=deduped,
        targetp_rows=targetp_rows,
        min_seq_id=float(mmseqs_min_seq_id),
        min_coverage=float(mmseqs_min_coverage),
        threads=int(threads),
        enabled=bool(use_mmseqs),
    )
    sampled, sample_report = _sample_external_rows(
        rows=filtered,
        max_per_class=max_per_class,
        seed=seed,
    )
    return sampled, {
        'candidate_rows': int(len(rows)),
        'deduplicated_rows': int(len(deduped)),
        'filtered_rows': int(len(filtered)),
        'sampled_rows': int(len(sampled)),
        'sample_report': sample_report,
        'mmseqs_similarity_filter': mmseqs_report,
        'skipped': dict(skipped),
        'sampled_counts': dict(Counter(row.get('localization', '') for row in sampled)),
    }


def _true_idx_from_rows(rows, class_names):
    class_to_idx = {class_name: class_i for class_i, class_name in enumerate(class_names)}
    return np.asarray([
        int(class_to_idx[row.get('localization', '')])
        for row in rows
    ], dtype=np.int64)


def _fold_ids_from_rows(rows):
    return np.asarray([str(row.get('fold_id', '')) for row in rows])


def run_external_augmented_feature_oof(
    training_tsv,
    uniprot_tsv,
    extra_uniprot_tsvs=None,
    deeploc_dir='',
    include_deeploc=True,
    max_external_per_class=5000,
    external_weight=0.25,
    seed=101,
    use_mmseqs=False,
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
        deeploc_dir=deeploc_dir,
        include_deeploc=include_deeploc,
        max_per_class=int(max_external_per_class),
        seed=int(seed),
        use_mmseqs=bool(use_mmseqs),
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
        sample_weight = np.concatenate([
            np.ones((int(np.sum(train_mask)),), dtype=np.float64),
            np.full((len(external_idx),), float(external_weight), dtype=np.float64),
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
            'deeploc_dir': str(deeploc_dir),
            'include_deeploc': bool(include_deeploc),
            'max_external_per_class': int(max_external_per_class),
            'external_weight': float(external_weight),
            'seed': int(seed),
            'use_mmseqs': bool(use_mmseqs),
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
    parser.add_argument('--deeploc_dir', default='data/localize_bench/deeploc21', type=str)
    parser.add_argument('--include_deeploc', default='yes', choices=['yes', 'no'], type=str)
    parser.add_argument('--max_external_per_class', default=TARGETP_EXTERNAL_AUG_DEFAULTS['max_external_per_class'], type=int)
    parser.add_argument('--external_weight', default=TARGETP_EXTERNAL_AUG_DEFAULTS['external_weight'], type=float)
    parser.add_argument('--seed', default=TARGETP_EXTERNAL_AUG_DEFAULTS['seed'], type=int)
    parser.add_argument('--mmseqs', default='no', choices=['yes', 'no'], type=str)
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
    parser.add_argument('--out_npz', required=True, type=str)
    parser.add_argument('--out_json', required=True, type=str)
    parser.add_argument('--external_tsv_out', default='', type=str)
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


def main():
    args = build_parser().parse_args()
    extra_uniprot_tsvs = _parse_paths(args.extra_uniprot_tsvs)
    result = run_external_augmented_feature_oof(
        training_tsv=args.training_tsv,
        uniprot_tsv=args.uniprot_tsv,
        extra_uniprot_tsvs=extra_uniprot_tsvs,
        deeploc_dir=args.deeploc_dir,
        include_deeploc=_yes_no(args.include_deeploc),
        max_external_per_class=int(args.max_external_per_class),
        external_weight=float(args.external_weight),
        seed=int(args.seed),
        use_mmseqs=_yes_no(args.mmseqs),
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
            deeploc_dir=args.deeploc_dir,
            include_deeploc=_yes_no(args.include_deeploc),
            max_per_class=int(args.max_external_per_class),
            seed=int(args.seed),
            use_mmseqs=_yes_no(args.mmseqs),
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
    print('external_rows={} argmax_macro_f1={:.6f} foldwise_threshold_macro_f1={:.6f}'.format(
        int(result['external_report']['sampled_rows']),
        float(result['argmax']['macro_f1']),
        float(result['foldwise_threshold']['metrics']['macro_f1']),
    ))


if __name__ == '__main__':
    main()

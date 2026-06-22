import csv

import numpy as np
import pytest

from cdskit.localize_model import (
    FEATURE_NAMES,
    LOCALIZATION_CLASSES,
    PEROX_FEATURE_NAMES,
    extract_perox_features,
    load_localize_model,
    predict_localization_and_peroxisome,
    save_localize_model,
)
from cdskit.perox_benchmark import (
    assign_cluster_folds,
    binary_metrics,
    build_arg_parser,
    exclude_rows_overlapping_eval,
    leakage_report,
    maybe_homology_report,
    read_perox_rows,
    read_uniprot_exp_cc_perox_rows,
    run_cluster_oof_perox_benchmark,
    run_deeploc21_perox_benchmark,
    sequence_label_conflict_report,
    split_train_valid,
    tune_threshold,
)


class _FixedBinaryClassifier:
    classes_ = np.asarray([0, 1], dtype=np.int64)

    def __init__(self, yes_probability):
        self.yes_probability = float(yes_probability)

    def predict_proba(self, features):
        n_rows = int(features.shape[0])
        row = np.asarray([1.0 - self.yes_probability, self.yes_probability], dtype=np.float64)
        return np.tile(row.reshape((1, 2)), (n_rows, 1))


def _write_perox_fixture(path):
    rows = [
        {
            'source': 'fixture',
            'accession': 'train_pos',
            'kingdom': 'Metazoa',
            'partition': 'train',
            'sequence': 'MAAAAAAAAAAAAAAAAAAASKL',
            'localization_labels': 'peroxisome',
            'peroxisome': '1',
        },
        {
            'source': 'fixture',
            'accession': 'train_neg',
            'kingdom': 'Metazoa',
            'partition': 'train',
            'sequence': 'MAAAAAAAAAAAAAAAAAAAAAA',
            'localization_labels': 'cytoplasm',
            'peroxisome': '0',
        },
        {
            'source': 'fixture',
            'accession': 'valid_pos',
            'kingdom': 'Fungi',
            'partition': 'valid',
            'sequence': 'MSSSSSSSSSSSSSSSSSSAKL',
            'localization_labels': 'peroxisome',
            'peroxisome': '1',
        },
        {
            'source': 'fixture',
            'accession': 'valid_neg',
            'kingdom': 'Fungi',
            'partition': 'valid',
            'sequence': 'MSSSSSSSSSSSSSSSSSSAAA',
            'localization_labels': 'cytoplasm',
            'peroxisome': '0',
        },
    ]
    with open(path, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                'source',
                'accession',
                'kingdom',
                'partition',
                'sequence',
                'localization_labels',
                'peroxisome',
            ],
            delimiter='\t',
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def _write_uniprot_cc_fixture(path):
    rows = [
        {
            'accession': 'P_EXP_PEROX',
            'sequence': 'MQQQQQQQQQQQQQQQQQQSKL',
            'cc_subcellular_location': 'SUBCELLULAR LOCATION: Peroxisome {ECO:0000269|PubMed:1}.',
            'organism_id': '9606',
            'lineage_ids': '131567 (no rank), 2759 (domain), 33208 (kingdom)',
        },
        {
            'accession': 'P_EXP_NEG',
            'sequence': 'MSSSSSSSSSSSSSSSSSSAAA',
            'cc_subcellular_location': 'SUBCELLULAR LOCATION: Cytoplasm {ECO:0000269|PubMed:2}.',
            'organism_id': '559292',
            'lineage_ids': '131567 (no rank), 2759 (domain), 4751 (kingdom)',
        },
        {
            'accession': 'P_SIM_PEROX',
            'sequence': 'MNNNNNNNNNNNNNNNNNNSKL',
            'cc_subcellular_location': 'SUBCELLULAR LOCATION: Peroxisome {ECO:0000250}.',
            'organism_id': '3702',
            'lineage_ids': '131567 (no rank), 2759 (domain), 33090 (kingdom)',
        },
    ]
    with open(path, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                'accession',
                'sequence',
                'cc_subcellular_location',
                'organism_id',
                'lineage_ids',
            ],
            delimiter='\t',
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def _write_torch_base_localize_model(path):
    model = {
        'model_type': 'targetp_torch_v1',
        'feature_names': list(FEATURE_NAMES),
        'localization_model': {
            'mode': 'targetp_torch',
            'class_order': list(LOCALIZATION_CLASSES),
            'config': {},
            'state_dict': {},
            'class_thresholds': {class_name: 1.0 for class_name in LOCALIZATION_CLASSES},
        },
        'perox_model': {
            'mode': 'constant',
            'yes_probability': 0.0,
        },
        'metadata': {
            'fixture': 'base_model',
        },
    }
    save_localize_model(model=model, path=str(path))
    return model


def test_extract_perox_features_has_stable_tail_onehot():
    features, signals = extract_perox_features('MAAAAAAAAAAAAAAASKF', kingdom='Metazoa')

    assert features.shape[0] == len(PEROX_FEATURE_NAMES)
    assert signals['signal_type'] == 'PTS1'
    assert float(np.sum(features[-(12 * 20):])) == 12.0


def test_perox_benchmark_default_model_kind_matches_release_candidate():
    args = build_arg_parser().parse_args([])

    assert args.model_kind == 'extra_trees'


def test_predict_localization_uses_sklearn_binary_perox_head():
    model = {
        'model_type': 'nearest_centroid_v1',
        'feature_names': list(FEATURE_NAMES),
        'localization_model': {
            'mode': 'constant',
            'class_label': 'noTP',
            'class_order': list(LOCALIZATION_CLASSES),
        },
        'perox_model': {
            'mode': 'sklearn_binary',
            'classifier': _FixedBinaryClassifier(0.8),
            'positive_class': 1,
            'threshold': 0.7,
            'feature_profile': 'perox_sequence_v1',
        },
    }

    pred = predict_localization_and_peroxisome(
        aa_seq='MAAAAAAAAAAAAAAASKF',
        model=model,
        organism_group='Metazoa',
    )

    assert pred['predicted_class'] == 'noTP'
    assert pred['perox_probability_yes'] == pytest.approx(0.8)
    assert pred['perox_signal_type'] == 'PTS1'


def test_binary_metrics_and_threshold_tuning():
    y_true = np.asarray([0, 0, 1, 1], dtype=np.int64)
    y_prob = np.asarray([0.1, 0.2, 0.7, 0.9], dtype=np.float64)

    threshold = tune_threshold(y_true=y_true, y_prob=y_prob, objective='f1')
    metrics = binary_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)

    assert 0.0 < threshold < 1.0
    assert metrics['f1'] == pytest.approx(1.0)
    assert metrics['tp'] == 2
    assert metrics['tn'] == 2


def test_read_rows_and_leakage_report(temp_dir):
    path = temp_dir / 'perox.tsv'
    _write_perox_fixture(path)

    rows = read_perox_rows(str(path))
    report = leakage_report(rows[:2], rows[2:])

    assert len(rows) == 4
    assert report['accession_overlap_count'] == 0
    assert report['exact_sequence_overlap_count'] == 0


def test_exclude_rows_overlapping_eval_removes_accession_and_sequence_overlap():
    train_rows = [
        {'accession': 'same_acc', 'sequence': 'MAAAAA', 'peroxisome': 0},
        {'accession': 'different_acc_same_seq', 'sequence': 'MSSSSS', 'peroxisome': 0},
        {'accession': 'keep', 'sequence': 'MQQQQQ', 'peroxisome': 1},
    ]
    eval_rows = [
        {'accession': 'same_acc', 'sequence': 'MNNNNN', 'peroxisome': 0},
        {'accession': 'eval_seq', 'sequence': 'MSSSSS', 'peroxisome': 0},
    ]

    kept, report = exclude_rows_overlapping_eval(train_rows, eval_rows)

    assert [row['accession'] for row in kept] == ['keep']
    assert report['removed_rows'] == 2
    assert report['accession_overlap_rows'] == 1
    assert report['exact_sequence_overlap_rows'] == 1


def test_sequence_label_conflict_report_flags_exact_duplicate_label_conflicts():
    rows = [
        {'accession': 'pos_a', 'sequence': 'MAAASKL', 'peroxisome': 1},
        {'accession': 'neg_same_seq', 'sequence': 'MAAASKL', 'peroxisome': 0},
        {'accession': 'neg_other', 'sequence': 'MSSSAAA', 'peroxisome': 0},
    ]

    report = sequence_label_conflict_report(rows)

    assert report['rows'] == 3
    assert report['unique_sequences'] == 2
    assert report['duplicate_sequence_count'] == 1
    assert report['duplicate_sequence_rows'] == 2
    assert report['duplicate_sequence_excess_rows'] == 1
    assert report['conflicting_sequence_count'] == 1
    assert report['conflicting_row_count'] == 2
    assert report['examples'][0]['labels'] == [0, 1]


def test_homology_report_can_be_skipped_or_unavailable(monkeypatch):
    rows = [
        {'accession': 'a', 'sequence': 'MAAAAAAAAAAAAAAASKL', 'peroxisome': 1},
        {'accession': 'b', 'sequence': 'MAAAAAAAAAAAAAAAAAA', 'peroxisome': 0},
    ]

    skipped = maybe_homology_report(rows, rows, enabled=False)
    assert skipped['status'] == 'skipped'

    monkeypatch.setattr('cdskit.perox_benchmark.shutil.which', lambda name: None)
    unavailable = maybe_homology_report(rows, rows, enabled=True)
    assert unavailable['status'] == 'unavailable'


def test_cluster_fold_assignment_balances_positive_clusters():
    rows = [
        {'accession': 'p1', 'sequence': 'MAASKL', 'peroxisome': 1},
        {'accession': 'p2', 'sequence': 'MAAKKL', 'peroxisome': 1},
        {'accession': 'n1', 'sequence': 'MAAAAA', 'peroxisome': 0},
        {'accession': 'n2', 'sequence': 'MSSSSS', 'peroxisome': 0},
    ]
    folds, report = assign_cluster_folds(
        rows=rows,
        cluster_ids=['c1', 'c2', 'c3', 'c4'],
        n_folds=2,
    )

    assert sorted(set(folds)) == [0, 1]
    assert report['folds'][0]['positive'] == 1
    assert report['folds'][1]['positive'] == 1


def test_hash_stratified_split_supports_unpartitioned_rows():
    rows = []
    for i in range(10):
        rows.append({
            'accession': 'p{}'.format(i),
            'kingdom': 'Metazoa',
            'partition': 'external_uniprot_exp',
            'sequence': 'MAAAAAAAAAAAAAAA{}SKL'.format('A' * i),
            'peroxisome': 1,
        })
        rows.append({
            'accession': 'n{}'.format(i),
            'kingdom': 'Metazoa',
            'partition': 'external_uniprot_exp',
            'sequence': 'MSSSSSSSSSSSSSSS{}AAA'.format('S' * i),
            'peroxisome': 0,
        })

    train_rows, valid_rows = split_train_valid(
        rows=rows,
        validation_partition='hash_stratified',
        validation_fraction=0.2,
    )

    assert len(train_rows) == 16
    assert len(valid_rows) == 4
    assert sum(int(row['peroxisome']) for row in valid_rows) == 2


def test_hash_stratified_split_keeps_exact_duplicate_sequences_together():
    rows = []
    for i in range(8):
        rows.append({
            'accession': 'p{}'.format(i),
            'kingdom': 'Metazoa',
            'partition': 'same',
            'sequence': 'MAAAAAAAAAAAAAAA{}SKL'.format('A' * i),
            'peroxisome': 1,
        })
        rows.append({
            'accession': 'n{}'.format(i),
            'kingdom': 'Metazoa',
            'partition': 'same',
            'sequence': 'MSSSSSSSSSSSSSSS{}AAA'.format('S' * i),
            'peroxisome': 0,
        })
    rows.append({
        'accession': 'duplicate_a',
        'kingdom': 'Metazoa',
        'partition': 'same',
        'sequence': 'MAAAAAAAAAAAAAAAASKL',
        'peroxisome': 1,
    })
    rows.append({
        'accession': 'duplicate_b',
        'kingdom': 'Metazoa',
        'partition': 'same',
        'sequence': 'MAAAAAAAAAAAAAAAASKL',
        'peroxisome': 1,
    })

    train_rows, valid_rows = split_train_valid(
        rows=rows,
        validation_partition='hash_stratified',
        validation_fraction=0.25,
    )
    report = leakage_report(train_rows, valid_rows)

    assert report['exact_sequence_overlap_count'] == 0


def test_cluster_oof_perox_benchmark_singleton(temp_dir):
    pytest.importorskip('sklearn')
    path = temp_dir / 'perox.tsv'
    _write_perox_fixture(path)
    rows = read_perox_rows(str(path))

    report = run_cluster_oof_perox_benchmark(
        rows=rows,
        n_folds=2,
        cluster_method='singleton',
        model_kind='extra_trees',
        n_estimators=5,
        min_samples_leaf=1,
    )

    assert report['status'] == 'ok'
    assert report['rows'] == 4
    assert report['cluster_report']['status'] == 'singleton'
    assert report['metrics']['rows'] == 4


def test_read_uniprot_exp_cc_rows_filters_similarity_and_overlap(temp_dir):
    path = temp_dir / 'uniprot.tsv'
    _write_uniprot_cc_fixture(path)

    rows, skipped = read_uniprot_exp_cc_perox_rows(
        path=str(path),
        exclude_rows=[{'accession': 'P_EXP_NEG', 'sequence': 'MSSSSSSSSSSSSSSSSSSAAA'}],
    )

    assert len(rows) == 1
    assert rows[0]['accession'] == 'P_EXP_PEROX'
    assert rows[0]['peroxisome'] == 1
    assert skipped['accession_overlap'] == 1
    assert skipped['non_experimental'] == 1


def test_deeploc21_perox_benchmark_fixture(temp_dir):
    pytest.importorskip('sklearn')

    train_path = temp_dir / 'train.tsv'
    external_path = temp_dir / 'external.tsv'
    _write_perox_fixture(train_path)
    _write_perox_fixture(external_path)
    report_json = temp_dir / 'report.json'
    report_md = temp_dir / 'report.md'
    predictions_prefix = temp_dir / 'predictions'

    report = run_deeploc21_perox_benchmark(
        train_tsv=str(train_path),
        external_test_tsv=str(external_path),
        exclude_external_from_train='no',
        validation_partition='valid',
        model_kind='extra_trees',
        n_estimators=5,
        min_samples_leaf=1,
        report_json=str(report_json),
        report_md=str(report_md),
        predictions_prefix=str(predictions_prefix),
        cluster_oof='yes',
        cluster_oof_method='singleton',
        cluster_oof_folds=2,
    )

    assert report['dataset']['validation_rows'] == 2
    assert report['metrics']['validation_model']['rows'] == 2
    assert report['metrics']['external_final_model']['rows'] == 4
    assert report['label_quality']['all_train_validation']['conflicting_sequence_count'] == 0
    assert report['cluster_oof']['status'] == 'ok'
    assert report['cluster_oof']['metrics']['rows'] == 4
    assert report['leakage_checks']['homology_threshold_train_vs_validation']['status'] == 'skipped'
    assert report_json.exists()
    assert report_md.exists()
    assert (temp_dir / 'predictions.validation.tsv').exists()
    assert (temp_dir / 'predictions.external.tsv').exists()


def test_deeploc21_perox_benchmark_writes_attached_model(temp_dir):
    pytest.importorskip('sklearn')
    pytest.importorskip('torch')

    train_path = temp_dir / 'train.tsv'
    external_path = temp_dir / 'external.tsv'
    base_model_path = temp_dir / 'base.pt'
    model_out = temp_dir / 'with_perox.pt'
    _write_perox_fixture(train_path)
    _write_perox_fixture(external_path)
    _write_torch_base_localize_model(base_model_path)

    report = run_deeploc21_perox_benchmark(
        train_tsv=str(train_path),
        external_test_tsv=str(external_path),
        exclude_external_from_train='no',
        validation_partition='valid',
        model_kind='extra_trees',
        n_estimators=5,
        min_samples_leaf=1,
        base_model=str(base_model_path),
        model_out=str(model_out),
    )
    loaded = load_localize_model(str(model_out))

    assert report['model_out'] == str(model_out)
    assert loaded['model_type'] == 'targetp_torch_v1'
    assert loaded['perox_model']['mode'] == 'sklearn_binary'
    assert loaded['perox_model']['feature_profile'] == 'perox_sequence_v1'
    assert loaded['perox_model']['prediction_scope'].startswith('peroxisome sequence-label')
    assert loaded['metadata']['fixture'] == 'base_model'
    assert loaded['metadata']['perox_model_source_tsv'] == str(train_path)
    assert loaded['metadata']['perox_model_exclude_external_from_train'] is False
    assert model_out.exists()


def test_deeploc21_perox_benchmark_uniprot_external_format(temp_dir):
    pytest.importorskip('sklearn')

    train_path = temp_dir / 'train.tsv'
    external_path = temp_dir / 'uniprot.tsv'
    _write_perox_fixture(train_path)
    _write_uniprot_cc_fixture(external_path)

    report = run_deeploc21_perox_benchmark(
        train_tsv=str(train_path),
        external_test_tsv=str(external_path),
        external_format='uniprot_exp_cc',
        validation_partition='valid',
        model_kind='extra_trees',
        n_estimators=5,
        min_samples_leaf=1,
    )

    assert report['dataset']['external_format'] == 'uniprot_exp_cc'
    assert report['dataset']['external_rows'] == 1
    assert report['metrics']['external_final_model']['rows'] == 1


def test_deeploc21_perox_benchmark_hash_stratified_validation(temp_dir):
    pytest.importorskip('sklearn')

    train_path = temp_dir / 'train.tsv'
    external_path = temp_dir / 'external.tsv'
    _write_perox_fixture(train_path)
    _write_perox_fixture(external_path)

    report = run_deeploc21_perox_benchmark(
        train_tsv=str(train_path),
        external_test_tsv=str(external_path),
        exclude_external_from_train='no',
        validation_partition='hash_stratified',
        validation_fraction=0.5,
        model_kind='extra_trees',
        n_estimators=5,
        min_samples_leaf=1,
    )

    assert report['dataset']['validation_partition'] == 'hash_stratified'
    assert report['dataset']['validation_fraction'] == pytest.approx(0.5)
    assert report['dataset']['validation_rows'] == 2
    assert 'hash-stratified' in report['threshold_selection']['tuned_on']


def test_deeploc21_perox_benchmark_can_exclude_external_from_train(temp_dir):
    pytest.importorskip('sklearn')

    train_path = temp_dir / 'train.tsv'
    external_path = temp_dir / 'external.tsv'
    train_rows = _write_perox_fixture(train_path)
    with open(external_path, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                'source',
                'accession',
                'kingdom',
                'partition',
                'sequence',
                'localization_labels',
                'peroxisome',
            ],
            delimiter='\t',
        )
        writer.writeheader()
        writer.writerow(train_rows[0])
        writer.writerow({
            'source': 'fixture',
            'accession': 'external_only',
            'kingdom': 'Metazoa',
            'partition': 'external',
            'sequence': 'MCCCCCCCCCCCCCCCCCCCCC',
            'localization_labels': 'cytoplasm',
            'peroxisome': '0',
        })

    report = run_deeploc21_perox_benchmark(
        train_tsv=str(train_path),
        external_test_tsv=str(external_path),
        exclude_external_from_train='yes',
        validation_partition='hash_stratified',
        validation_fraction=0.5,
        model_kind='extra_trees',
        n_estimators=5,
        min_samples_leaf=1,
    )

    assert report['dataset']['train_exclusion_report']['removed_rows'] == 1
    assert report['dataset']['all_train_validation_rows'] == 3

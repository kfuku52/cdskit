import csv
import json
import sys
from types import SimpleNamespace

import pytest

from cdskit.deeploc_benchmark import (
    DEEPLOC_LOCALIZATION_LABELS,
    DEEPLOC_MEMBRANE_LABELS,
    compute_multilabel_metrics,
    evaluate_deeploc21_task_cv,
    fit_deeploc_multilabel_model,
    prepare_all_deeploc21,
    prepare_deeploc21_hpa_tsv,
    prepare_deeploc21_localization_tsv,
    prepare_deeploc21_membrane_tsv,
    prepare_deeploc21_sorting_signal_tsv,
    run_deeploc21_benchmark,
    main,
)
from cdskit.localize import localize_main


def _read_tsv(path):
    with open(path, 'r', encoding='utf-8', newline='') as inp:
        return list(csv.DictReader(inp, delimiter='\t'))


def _write_prepared_localization_tsv(path, rows):
    fieldnames = [
        'source',
        'accession',
        'kingdom',
        'partition',
        'sequence',
        'localization_labels',
    ] + list(DEEPLOC_LOCALIZATION_LABELS)
    with open(path, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(
            out,
            fieldnames=fieldnames,
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writeheader()
        for row in rows:
            active = set(row['localization_labels'].split(';'))
            out_row = {key: row.get(key, '') for key in fieldnames}
            for label in DEEPLOC_LOCALIZATION_LABELS:
                out_row[label] = 1 if label in active else 0
            writer.writerow(out_row)


def test_prepare_deeploc21_localization_maps_plastid_to_chloroplast(temp_dir):
    csv_path = temp_dir / 'Swissprot_Train_Validation_dataset.csv'
    out_path = temp_dir / 'localization.tsv'
    csv_path.write_text(
        (
            ',ACC,Kingdom,Partition,Membrane,Cytoplasm,Nucleus,Extracellular,'
            'Cell membrane,Mitochondrion,Plastid,Endoplasmic reticulum,'
            'Lysosome/Vacuole,Golgi apparatus,Peroxisome,Sequence\n'
            '0,P1,Viridiplantae,2,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,MASSS\n'
        ),
        encoding='utf-8',
    )

    report = prepare_deeploc21_localization_tsv(
        csv_path=str(csv_path),
        out_tsv_path=str(out_path),
    )
    rows = _read_tsv(path=out_path)

    assert report['n_rows'] == 1
    assert report['n_multi_label_rows'] == 1
    assert report['label_counts']['chloroplast'] == 1
    assert report['label_counts']['peroxisome'] == 1
    assert rows[0]['chloroplast'] == '1'
    assert rows[0]['peroxisome'] == '1'
    assert rows[0]['localization_labels'] == 'chloroplast;peroxisome'
    assert set(DEEPLOC_LOCALIZATION_LABELS).issuperset(rows[0]['localization_labels'].split(';'))


def test_prepare_deeploc21_hpa_uses_fasta_column(temp_dir):
    csv_path = temp_dir / 'hpa_testset.csv'
    out_path = temp_dir / 'hpa.tsv'
    csv_path.write_text(
        (
            'sid,Cell membrane,Cytoplasm,Endoplasmic reticulum,Golgi apparatus,'
            'Lysosome/Vacuole,Mitochondrion,Nucleus,Peroxisome,Lengths,fasta\n'
            'ENSP1,0,1,0,0,0,0,1,0,6,MAAAAA\n'
        ),
        encoding='utf-8',
    )

    report = prepare_deeploc21_hpa_tsv(
        csv_path=str(csv_path),
        out_tsv_path=str(out_path),
    )
    rows = _read_tsv(path=out_path)

    assert report['label_counts']['cytoplasm'] == 1
    assert report['label_counts']['nucleus'] == 1
    assert rows[0]['source'] == 'hpa'
    assert rows[0]['sequence'] == 'MAAAAA'
    assert rows[0]['partition'] == 'test'


def test_prepare_deeploc21_membrane_and_sorting_signal_tsv(temp_dir):
    membrane_csv = temp_dir / 'Swissprot_Membrane_Train_Validation_dataset.csv'
    membrane_out = temp_dir / 'membrane.tsv'
    membrane_csv.write_text(
        (
            ',ACC,Kingdom,Partition,Peripheral,Transmembrane,LipidAnchor,Soluble,Sequence\n'
            '0,P2,Metazoa,1,1,0,1,0,MVVVV\n'
        ),
        encoding='utf-8',
    )
    membrane_report = prepare_deeploc21_membrane_tsv(
        csv_path=str(membrane_csv),
        out_tsv_path=str(membrane_out),
    )
    membrane_rows = _read_tsv(path=membrane_out)

    assert membrane_report['n_multi_label_rows'] == 1
    assert membrane_report['label_counts']['peripheral'] == 1
    assert membrane_report['label_counts']['lipid_anchor'] == 1
    assert set(DEEPLOC_MEMBRANE_LABELS).issuperset(membrane_rows[0]['membrane_labels'].split(';'))

    sorting_csv = temp_dir / 'SortingSignalsSwissprot.csv'
    sorting_out = temp_dir / 'sorting.tsv'
    sorting_csv.write_text(
        'ACC,AnnotEncoded,Kingdom,Sequence,Types\nP3,111000,Metazoa,MKKLLL,SP_GPI\n',
        encoding='utf-8',
    )
    sorting_report = prepare_deeploc21_sorting_signal_tsv(
        csv_path=str(sorting_csv),
        out_tsv_path=str(sorting_out),
    )
    sorting_rows = _read_tsv(path=sorting_out)

    assert sorting_report['label_counts']['SP'] == 1
    assert sorting_report['label_counts']['GPI'] == 1
    assert sorting_rows[0]['sorting_signal_labels'] == 'SP;GPI'


def test_prepare_all_deeploc21_and_main_without_download(temp_dir, monkeypatch):
    data_dir = temp_dir / 'raw'
    out_dir = temp_dir / 'prepared'
    data_dir.mkdir()
    (data_dir / 'Swissprot_Train_Validation_dataset.csv').write_text(
        (
            ',ACC,Kingdom,Partition,Membrane,Cytoplasm,Nucleus,Extracellular,'
            'Cell membrane,Mitochondrion,Plastid,Endoplasmic reticulum,'
            'Lysosome/Vacuole,Golgi apparatus,Peroxisome,Sequence\n'
            '0,P1,Metazoa,0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,MAAAA\n'
        ),
        encoding='utf-8',
    )
    (data_dir / 'Swissprot_Membrane_Train_Validation_dataset.csv').write_text(
        ',ACC,Kingdom,Partition,Peripheral,Transmembrane,LipidAnchor,Soluble,Sequence\n'
        '0,P2,Metazoa,0,0,1,0,0,MVVVV\n',
        encoding='utf-8',
    )
    (data_dir / 'hpa_testset.csv').write_text(
        (
            'sid,Cell membrane,Cytoplasm,Endoplasmic reticulum,Golgi apparatus,'
            'Lysosome/Vacuole,Mitochondrion,Nucleus,Peroxisome,Lengths,fasta\n'
            'ENSP1,0,0,0,0,0,0,1,0,5,MAAAA\n'
        ),
        encoding='utf-8',
    )
    (data_dir / 'SortingSignalsSwissprot.csv').write_text(
        'ACC,AnnotEncoded,Kingdom,Sequence,Types\nP3,111000,Metazoa,MKKLLL,SP\n',
        encoding='utf-8',
    )

    report = prepare_all_deeploc21(data_dir=str(data_dir), out_dir=str(out_dir))
    assert report['localization_train_validation']['label_counts']['cytoplasm'] == 1
    assert (out_dir / 'deeploc21_hpa_test.tsv').exists()

    report_json = out_dir / 'report.json'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'deeploc_benchmark.py',
            '--download',
            'no',
            '--data_dir',
            str(data_dir),
            '--out_dir',
            str(out_dir),
            '--report_json',
            str(report_json),
        ],
    )
    main()
    with open(report_json, 'r', encoding='utf-8') as inp:
        cli_report = json.load(inp)
    assert cli_report['prepare']['hpa_test']['label_counts']['nucleus'] == 1


def test_multilabel_metrics_and_deeploc_cv(temp_dir):
    y_true = [
        [1, 0],
        [1, 1],
        [0, 1],
    ]
    y_pred = [
        [1, 0],
        [1, 0],
        [0, 1],
    ]
    metrics = compute_multilabel_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=['nucleus', 'cytoplasm'],
    )
    assert metrics['micro_f1'] > 0.80
    assert metrics['by_label']['cytoplasm']['fn'] == 1

    tsv_path = temp_dir / 'deeploc21_localization_train_validation.tsv'
    _write_prepared_localization_tsv(
        path=tsv_path,
        rows=[
            {
                'source': 'swissprot',
                'accession': 'P1',
                'kingdom': 'Metazoa',
                'partition': '0',
                'sequence': 'MKKKRKAAAGGG',
                'localization_labels': 'nucleus',
            },
            {
                'source': 'swissprot',
                'accession': 'P2',
                'kingdom': 'Metazoa',
                'partition': '0',
                'sequence': 'MLLLLLLLLLLAAA',
                'localization_labels': 'extracellular',
            },
            {
                'source': 'swissprot',
                'accession': 'P3',
                'kingdom': 'Metazoa',
                'partition': '1',
                'sequence': 'MKKRKRKGGGGG',
                'localization_labels': 'nucleus',
            },
            {
                'source': 'swissprot',
                'accession': 'P4',
                'kingdom': 'Metazoa',
                'partition': '1',
                'sequence': 'MFFFFFFFFFFAAA',
                'localization_labels': 'extracellular',
            },
        ],
    )

    cv = evaluate_deeploc21_task_cv(
        tsv_path=str(tsv_path),
        labels=DEEPLOC_LOCALIZATION_LABELS,
        label_col='localization_labels',
        task_name='localization',
    )
    assert cv['n_rows'] == 4
    assert len(cv['folds']) == 2
    assert 'macro_f1' in cv


def test_deeploc_rare_label_threshold_objective_metadata():
    rows = [
        {
            'source': 'swissprot',
            'accession': 'P1',
            'kingdom': 'Metazoa',
            'partition': '0',
            'sequence': 'MKKKRKAAAGGG',
            'sorting_signal_labels': 'SP',
        },
        {
            'source': 'swissprot',
            'accession': 'P2',
            'kingdom': 'Metazoa',
            'partition': '0',
            'sequence': 'MLLLLLLLLLLAAA',
            'sorting_signal_labels': 'SP',
        },
        {
            'source': 'swissprot',
            'accession': 'P3',
            'kingdom': 'Metazoa',
            'partition': '1',
            'sequence': 'MSSSSSSSSRRLLLLL',
            'sorting_signal_labels': 'TH',
        },
    ]
    model = fit_deeploc_multilabel_model(
        rows=rows,
        labels=['SP', 'TH'],
        label_col='sorting_signal_labels',
        task_name='sorting_signals',
        model_arch='centroid',
        dl_params={
            'threshold_objective': 'f1',
            'rare_label_threshold_objective': 'f2',
            'rare_label_max_count': 1,
        },
    )
    by_class = model['metadata']['rare_label_threshold_objective_by_class']
    assert by_class == {'TH': 'f2'}
    assert model['metadata']['threshold_params']['rare_label_max_count'] == 1


def test_run_deeploc_benchmark_writes_model_and_localize_predicts(temp_dir):
    prepared_dir = temp_dir / 'prepared'
    prepared_dir.mkdir()
    train_tsv = prepared_dir / 'deeploc21_localization_train_validation.tsv'
    hpa_tsv = prepared_dir / 'deeploc21_hpa_test.tsv'
    model_out = prepared_dir / 'deeploc_model.json'
    comparison_json = prepared_dir / 'comparison.json'
    comparison_md = prepared_dir / 'comparison.md'
    report_tsv = prepared_dir / 'predictions.tsv'
    fasta_path = prepared_dir / 'input.faa'

    train_rows = [
        {
            'source': 'swissprot',
            'accession': 'P1',
            'kingdom': 'Metazoa',
            'partition': '0',
            'sequence': 'MKKKRKAAAGGG',
            'localization_labels': 'nucleus',
        },
        {
            'source': 'swissprot',
            'accession': 'P2',
            'kingdom': 'Metazoa',
            'partition': '0',
            'sequence': 'MLLLLLLLLLLAAA',
            'localization_labels': 'extracellular',
        },
        {
            'source': 'swissprot',
            'accession': 'P3',
            'kingdom': 'Metazoa',
            'partition': '1',
            'sequence': 'MKKRKRKGGGGG',
            'localization_labels': 'nucleus',
        },
        {
            'source': 'swissprot',
            'accession': 'P4',
            'kingdom': 'Metazoa',
            'partition': '1',
            'sequence': 'MFFFFFFFFFFAAA',
            'localization_labels': 'extracellular',
        },
    ]
    _write_prepared_localization_tsv(path=train_tsv, rows=train_rows)
    _write_prepared_localization_tsv(path=hpa_tsv, rows=train_rows[:2])

    result = run_deeploc21_benchmark(
        prepared_dir=str(prepared_dir),
        task_name='localization',
        comparison_json=str(comparison_json),
        comparison_md=str(comparison_md),
        model_out=str(model_out),
    )
    assert result['model_out'] == str(model_out)
    assert comparison_json.exists()
    assert 'Published reference' in comparison_md.read_text(encoding='utf-8')

    fasta_path.write_text('>q1\nMKKKRKAAAGGG\n', encoding='utf-8')
    localize_main(
        SimpleNamespace(
            seqfile=str(fasta_path),
            inseqformat='fasta',
            codontable=999,
            model=str(model_out),
            report=str(report_tsv),
            include_features=False,
            seqtype='protein',
            threads=1,
            organism_group='metazoa',
        )
    )
    rows = _read_tsv(path=report_tsv)
    assert rows[0]['seq_id'] == 'q1'
    assert rows[0]['predicted_labels'] != ''
    assert 'p_nucleus' in rows[0]
    assert 'p_peroxisome' in rows[0]


def test_run_deeploc_benchmark_writes_cnn_model_and_localize_predicts(temp_dir):
    pytest.importorskip('torch')
    prepared_dir = temp_dir / 'prepared'
    prepared_dir.mkdir()
    train_tsv = prepared_dir / 'deeploc21_localization_train_validation.tsv'
    hpa_tsv = prepared_dir / 'deeploc21_hpa_test.tsv'
    model_out = prepared_dir / 'deeploc_cnn_model.pt'
    comparison_json = prepared_dir / 'cnn_comparison.json'
    comparison_md = prepared_dir / 'cnn_comparison.md'
    report_tsv = prepared_dir / 'cnn_predictions.tsv'
    fasta_path = prepared_dir / 'input.faa'

    train_rows = [
        {
            'source': 'swissprot',
            'accession': 'P1',
            'kingdom': 'Metazoa',
            'partition': '0',
            'sequence': 'MKKKRKAAAGGG',
            'localization_labels': 'nucleus',
        },
        {
            'source': 'swissprot',
            'accession': 'P2',
            'kingdom': 'Metazoa',
            'partition': '0',
            'sequence': 'MLLLLLLLLLLAAA',
            'localization_labels': 'extracellular',
        },
        {
            'source': 'swissprot',
            'accession': 'P3',
            'kingdom': 'Metazoa',
            'partition': '1',
            'sequence': 'MKKRKRKGGGGG',
            'localization_labels': 'nucleus',
        },
        {
            'source': 'swissprot',
            'accession': 'P4',
            'kingdom': 'Metazoa',
            'partition': '1',
            'sequence': 'MFFFFFFFFFFAAA',
            'localization_labels': 'extracellular',
        },
    ]
    _write_prepared_localization_tsv(path=train_tsv, rows=train_rows)
    _write_prepared_localization_tsv(path=hpa_tsv, rows=train_rows[:2])

    result = run_deeploc21_benchmark(
        prepared_dir=str(prepared_dir),
        task_name='localization',
        comparison_json=str(comparison_json),
        comparison_md=str(comparison_md),
        model_out=str(model_out),
        model_arch='cnn',
        dl_params={
            'seq_len': 32,
            'embed_dim': 8,
            'num_filters': 4,
            'kernel_sizes': '3,5',
            'dropout': 0.1,
            'epochs': 1,
            'batch_size': 2,
            'learning_rate': 1.0e-3,
            'weight_decay': 0.0,
            'class_weight': 'no',
            'feature_fusion': 'no',
            'seed': 1,
            'device': 'cpu',
        },
    )
    assert result['model'] == 'multilabel_cnn_v1'
    assert result['model_out'] == str(model_out)
    assert comparison_json.exists()
    assert model_out.exists()

    fasta_path.write_text('>q1\nMKKKRKAAAGGG\n', encoding='utf-8')
    localize_main(
        SimpleNamespace(
            seqfile=str(fasta_path),
            inseqformat='fasta',
            codontable=999,
            model=str(model_out),
            report=str(report_tsv),
            include_features=False,
            seqtype='protein',
            threads=1,
            organism_group='metazoa',
        )
    )
    rows = _read_tsv(path=report_tsv)
    assert rows[0]['seq_id'] == 'q1'
    assert rows[0]['predicted_labels'] != ''
    assert 'p_nucleus' in rows[0]
    assert 'p_peroxisome' in rows[0]

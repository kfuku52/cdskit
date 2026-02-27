import csv

import numpy as np

from cdskit.targetp_benchmark import (
    TARGETP_LABEL_TO_LOCALIZATION,
    TARGETP_TABLE1_REFERENCE,
    TARGETP_YTYPE_TO_LABEL,
    build_targetp_comparison_table,
    compute_prf_by_class,
    prepare_targetp_benchmark_tsv,
    render_markdown_table,
)


def _write_text(path, text):
    with open(path, 'w', encoding='utf-8') as out:
        out.write(text)


def test_prepare_targetp_benchmark_tsv_from_small_fixture(temp_dir):
    fasta_path = temp_dir / 'targetp.fasta'
    tab_path = temp_dir / 'swissprot_annotated_proteins.tab'
    npz_path = temp_dir / 'targetp_data.npz'
    out_tsv = temp_dir / 'targetp2_benchmark.tsv'
    report_json = temp_dir / 'targetp2_prepare_report.json'

    _write_text(
        fasta_path,
        (
            '>A0A0001\n'
            'MAAA\n'
            '>A0A0002\n'
            'MBBB\n'
            '>A0A0003\n'
            'MCCC\n'
        ),
    )
    _write_text(
        tab_path,
        (
            'A0A0001\tSP\t21\n'
            'A0A0002\tOther\t0\n'
            'A0A0003\tMT\t34\n'
        ),
    )
    np.savez(
        npz_path,
        ids=np.asarray(['A0A0001', 'A0A0002', 'A0A0003']),
        fold=np.asarray([0, 1, 0], dtype=np.int32),
        org=np.asarray([1, 0, 1], dtype=np.int32),
        y_type=np.asarray([1, 0, 2], dtype=np.int32),
    )

    report = prepare_targetp_benchmark_tsv(
        fasta_path=str(fasta_path),
        annotation_tab_path=str(tab_path),
        npz_path=str(npz_path),
        out_tsv_path=str(out_tsv),
        report_json_path=str(report_json),
    )

    assert report['n_rows'] == 3
    assert report['class_counts']['SP'] == 1
    assert report['class_counts']['noTP'] == 1
    assert report['class_counts']['mTP'] == 1
    assert report['fold_counts']['fold1'] == 2
    assert report['fold_counts']['fold2'] == 1
    assert report['y_type_mismatch_count'] == 0

    with open(out_tsv, 'r', encoding='utf-8', newline='') as inp:
        rows = list(csv.DictReader(inp, delimiter='\t'))
    assert len(rows) == 3

    first = rows[0]
    assert first['accession'] == 'A0A0001'
    assert first['sequence'] == 'MAAA'
    assert first['localization'] == 'SP'
    assert first['peroxisome'] == 'no'
    assert first['fold_id'] == 'fold1'
    assert first['organism_group'] == 'plant'


def test_compute_prf_by_class_and_comparison_table():
    classes = ['noTP', 'SP', 'mTP']
    true_labels = ['noTP', 'SP', 'SP', 'mTP', 'mTP']
    pred_labels = ['noTP', 'noTP', 'SP', 'mTP', 'SP']

    by_class = compute_prf_by_class(
        true_classes=true_labels,
        pred_classes=pred_labels,
        class_names=classes,
    )
    assert by_class['noTP']['tp'] == 1
    assert by_class['SP']['tp'] == 1
    assert by_class['SP']['fp'] == 1
    assert by_class['mTP']['fn'] == 1

    fake_oof_by_class = dict()
    for class_name in TARGETP_TABLE1_REFERENCE.keys():
        fake_oof_by_class[class_name] = {
            'precision': 0.50,
            'recall': 0.50,
            'f1': 0.50,
            'tp': 1,
            'fp': 1,
            'fn': 1,
            'support': 2,
        }
    comparison = build_targetp_comparison_table(
        cdskit_result={
            'oof_by_class': fake_oof_by_class,
            'oof_macro_f1': 0.50,
        }
    )
    assert len(comparison['rows']) == 5
    assert comparison['targetp_macro_f1'] > 0.5
    assert comparison['delta_macro_f1_cdskit_minus_targetp'] < 0.0

    md = render_markdown_table(comparison=comparison)
    assert '| Class | TargetP P | TargetP R | TargetP F1 |' in md
    assert '| Macro F1 (5-class) |' in md


def test_targetp_label_mapping_constants_are_consistent():
    for y_type, label in TARGETP_YTYPE_TO_LABEL.items():
        assert label in TARGETP_LABEL_TO_LOCALIZATION
        loc = TARGETP_LABEL_TO_LOCALIZATION[label]
        assert loc in TARGETP_TABLE1_REFERENCE

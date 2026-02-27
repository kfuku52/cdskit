import argparse
import csv
import json
import os

import numpy as np

from cdskit.localize_learn import (
    LOCALIZATION_CLASSES,
    build_training_matrix,
    evaluate_cross_validation,
    write_rows_tsv,
)

TARGETP_LABEL_TO_LOCALIZATION = {
    'Other': 'noTP',
    'SP': 'SP',
    'MT': 'mTP',
    'CH': 'cTP',
    'TH': 'lTP',
}

TARGETP_YTYPE_TO_LABEL = {
    0: 'Other',
    1: 'SP',
    2: 'MT',
    3: 'CH',
    4: 'TH',
}

TARGETP_TABLE1_REFERENCE = {
    'noTP': {'precision': 0.98, 'recall': 0.98, 'f1': 0.98},
    'SP': {'precision': 0.97, 'recall': 0.98, 'f1': 0.98},
    'mTP': {'precision': 0.87, 'recall': 0.85, 'f1': 0.86},
    'cTP': {'precision': 0.90, 'recall': 0.86, 'f1': 0.88},
    'lTP': {'precision': 0.75, 'recall': 0.75, 'f1': 0.75},
}


def _parse_targetp_fasta(path):
    seq_by_accession = dict()
    accession = None
    seq_chunks = list()
    with open(path, 'r', encoding='utf-8') as inp:
        for raw in inp:
            line = raw.strip()
            if line == '':
                continue
            if line.startswith('>'):
                if accession is not None:
                    seq_by_accession[accession] = ''.join(seq_chunks)
                accession = line[1:].split()[0]
                seq_chunks = list()
            else:
                seq_chunks.append(line)
    if accession is not None:
        seq_by_accession[accession] = ''.join(seq_chunks)
    return seq_by_accession


def _read_targetp_tab_rows(path):
    rows = list()
    with open(path, 'r', encoding='utf-8', newline='') as inp:
        reader = csv.reader(inp, delimiter='\t')
        for raw in reader:
            if len(raw) == 0:
                continue
            if len(raw) < 3:
                raise ValueError('Invalid TargetP annotation row: {}'.format(raw))
            rows.append({
                'accession': str(raw[0]).strip(),
                'targetp_label': str(raw[1]).strip(),
                'cleavage_site': str(raw[2]).strip(),
            })
    return rows


def _read_targetp_npz(path):
    npz = np.load(path, allow_pickle=True)
    required = ['ids', 'fold', 'org', 'y_type']
    for key in required:
        if key not in npz.files:
            raise ValueError('TargetP npz is missing key: {}'.format(key))
    ids = [str(v) for v in npz['ids'].tolist()]
    folds = [int(v) for v in npz['fold'].tolist()]
    orgs = [int(v) for v in npz['org'].tolist()]
    y_types = [int(v) for v in npz['y_type'].tolist()]
    if not (len(ids) == len(folds) == len(orgs) == len(y_types)):
        raise ValueError('Length mismatch among ids/fold/org/y_type in TargetP npz.')
    return ids, folds, orgs, y_types


def _map_targetp_label_to_localization(targetp_label):
    if targetp_label not in TARGETP_LABEL_TO_LOCALIZATION:
        raise ValueError('Unsupported TargetP label: {}'.format(targetp_label))
    return TARGETP_LABEL_TO_LOCALIZATION[targetp_label]


def _to_fold_id(fold_value):
    return 'fold{}'.format(int(fold_value) + 1)


def prepare_targetp_benchmark_tsv(
    fasta_path,
    annotation_tab_path,
    npz_path,
    out_tsv_path,
    report_json_path='',
):
    seq_by_accession = _parse_targetp_fasta(path=fasta_path)
    ann_rows = _read_targetp_tab_rows(path=annotation_tab_path)
    ids, folds, orgs, y_types = _read_targetp_npz(path=npz_path)

    fold_by_accession = dict()
    org_by_accession = dict()
    ytype_by_accession = dict()
    for i, acc in enumerate(ids):
        fold_by_accession[acc] = int(folds[i])
        org_by_accession[acc] = int(orgs[i])
        ytype_by_accession[acc] = int(y_types[i])

    ann_by_accession = dict()
    for row in ann_rows:
        acc = row['accession']
        if acc in ann_by_accession:
            raise ValueError('Duplicate accession in annotation tab: {}'.format(acc))
        ann_by_accession[acc] = row

    ids_ann = set(ann_by_accession.keys())
    ids_fasta = set(seq_by_accession.keys())
    ids_npz = set(ids)
    if (ids_ann != ids_fasta) or (ids_ann != ids_npz):
        txt = (
            'TargetP sources are inconsistent. '
            'ann_only={}, fasta_only={}, npz_only={}'
        )
        raise ValueError(
            txt.format(
                sorted(list(ids_ann - ids_fasta))[:5] + sorted(list(ids_ann - ids_npz))[:5],
                sorted(list(ids_fasta - ids_ann))[:5],
                sorted(list(ids_npz - ids_ann))[:5],
            )
        )

    out_rows = list()
    class_counts = {class_name: 0 for class_name in LOCALIZATION_CLASSES}
    fold_counts = dict()
    organism_counts = {'plant': 0, 'non_plant': 0}
    y_type_mismatch = 0
    for acc in ids:
        ann = ann_by_accession[acc]
        targetp_label = ann['targetp_label']
        localization = _map_targetp_label_to_localization(targetp_label=targetp_label)
        y_type_label = TARGETP_YTYPE_TO_LABEL.get(ytype_by_accession[acc], '')
        if y_type_label != targetp_label:
            y_type_mismatch += 1
        org_group = 'plant' if int(org_by_accession[acc]) == 1 else 'non_plant'
        fold_id = _to_fold_id(fold_by_accession[acc])
        out_rows.append({
            'accession': acc,
            'sequence': seq_by_accession[acc],
            'localization': localization,
            'peroxisome': 'no',
            'fold_id': fold_id,
            'targetp_label': targetp_label,
            'targetp_fold': int(fold_by_accession[acc]),
            'organism_group': org_group,
            'cleavage_site': ann['cleavage_site'],
        })
        class_counts[localization] = class_counts.get(localization, 0) + 1
        fold_counts[fold_id] = fold_counts.get(fold_id, 0) + 1
        organism_counts[org_group] = organism_counts.get(org_group, 0) + 1

    out_dir = os.path.dirname(out_tsv_path)
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)
    output_fields = [
        'accession',
        'sequence',
        'localization',
        'peroxisome',
        'fold_id',
        'targetp_label',
        'targetp_fold',
        'organism_group',
        'cleavage_site',
    ]
    write_rows_tsv(
        rows=out_rows,
        output_path=out_tsv_path,
        fieldnames=output_fields,
    )

    report = {
        'fasta_path': fasta_path,
        'annotation_tab_path': annotation_tab_path,
        'npz_path': npz_path,
        'out_tsv_path': out_tsv_path,
        'n_rows': int(len(out_rows)),
        'class_counts': class_counts,
        'fold_counts': fold_counts,
        'organism_counts': organism_counts,
        'y_type_mismatch_count': int(y_type_mismatch),
        'fields': output_fields,
    }
    if report_json_path != '':
        report_dir = os.path.dirname(report_json_path)
        if report_dir != '':
            os.makedirs(report_dir, exist_ok=True)
        with open(report_json_path, 'w', encoding='utf-8') as out:
            json.dump(report, out, indent=2)
    return report


def _predict_class_from_prob_dict(prob_dict):
    best_name = LOCALIZATION_CLASSES[0]
    best_prob = float(prob_dict.get(best_name, 0.0))
    for class_name in LOCALIZATION_CLASSES[1:]:
        prob = float(prob_dict.get(class_name, 0.0))
        if prob > best_prob:
            best_prob = prob
            best_name = class_name
    return best_name


def compute_prf_by_class(true_classes, pred_classes, class_names):
    out = dict()
    for class_name in class_names:
        tp = 0
        fp = 0
        fn = 0
        support = 0
        for i in range(len(true_classes)):
            true_name = true_classes[i]
            pred_name = pred_classes[i]
            if true_name == class_name:
                support += 1
            if (true_name == class_name) and (pred_name == class_name):
                tp += 1
            elif (true_name != class_name) and (pred_name == class_name):
                fp += 1
            elif (true_name == class_name) and (pred_name != class_name):
                fn += 1
        precision = 0.0
        if (tp + fp) > 0:
            precision = float(tp) / float(tp + fp)
        recall = 0.0
        if (tp + fn) > 0:
            recall = float(tp) / float(tp + fn)
        f1 = 0.0
        if (precision + recall) > 0:
            f1 = (2.0 * precision * recall) / (precision + recall)
        out[class_name] = {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'support': int(support),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }
    return out


def _macro_mean(metric_by_class, class_names, metric_name):
    vals = [float(metric_by_class[name][metric_name]) for name in class_names]
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def run_cdskit_cv_on_targetp(
    training_tsv,
    model_arch='bilstm_attention',
    localize_strategy='single_stage',
    dl_params=None,
    cv_seed=1,
):
    if dl_params is None:
        dl_params = {
            'seq_len': 200,
            'embed_dim': 32,
            'hidden_dim': 64,
            'num_layers': 1,
            'dropout': 0.2,
            'epochs': 15,
            'batch_size': 128,
            'learning_rate': 1.0e-3,
            'weight_decay': 1.0e-4,
            'use_class_weight': True,
            'loss_name': 'ce',
            'balanced_batch': False,
            'aux_tp_weight': 0.0,
            'aux_ctp_ltp_weight': 0.0,
            'seed': 1,
            'device': 'cpu',
            'esm_model_name': 'facebook/esm2_t6_8M_UR50D',
            'esm_model_local_dir': '',
            'esm_pooling': 'cls',
            'esm_max_len': 200,
        }
    rows = list()
    with open(training_tsv, 'r', encoding='utf-8', newline='') as inp:
        rows = list(csv.DictReader(inp, delimiter='\t'))
    x, aa_sequences, class_labels, perox_labels, skipped, fold_ids = build_training_matrix(
        rows=rows,
        seq_col='sequence',
        seqtype='protein',
        codontable=1,
        label_mode='explicit',
        localization_col='localization',
        perox_col='peroxisome',
        skip_ambiguous=True,
        cv_fold_col='fold_id',
    )
    cv_metrics = evaluate_cross_validation(
        x=x,
        aa_sequences=aa_sequences,
        class_labels=class_labels,
        perox_labels=perox_labels,
        n_folds=5,
        seed=int(cv_seed),
        model_arch=model_arch,
        dl_train_params=dl_params,
        dl_device=str(dl_params.get('device', 'cpu')),
        localize_strategy=localize_strategy,
        fold_ids=fold_ids,
        verbose=True,
    )
    oof_rows = sorted(cv_metrics['oof_rows'], key=lambda r: int(r['index']))
    true_classes = [row['true_class'] for row in oof_rows]
    pred_classes = [
        _predict_class_from_prob_dict(row.get('class_probabilities', {}))
        for row in oof_rows
    ]
    by_class = compute_prf_by_class(
        true_classes=true_classes,
        pred_classes=pred_classes,
        class_names=list(LOCALIZATION_CLASSES),
    )
    overall = 0.0
    if len(true_classes) > 0:
        n_ok = 0
        for i in range(len(true_classes)):
            if true_classes[i] == pred_classes[i]:
                n_ok += 1
        overall = float(n_ok) / float(len(true_classes))
    return {
        'training_tsv': training_tsv,
        'n_rows_total': int(len(rows)),
        'n_rows_used': int(len(true_classes)),
        'n_rows_skipped': int(skipped),
        'cv_class_accuracy_mean': float(cv_metrics['class_accuracy_mean']),
        'cv_class_accuracy_std': float(cv_metrics['class_accuracy_std']),
        'cv_class_accuracy_by_class': dict(cv_metrics['class_accuracy_by_class']),
        'oof_overall_accuracy': float(overall),
        'oof_by_class': by_class,
        'oof_macro_f1': _macro_mean(by_class, list(LOCALIZATION_CLASSES), 'f1'),
        'oof_macro_precision': _macro_mean(by_class, list(LOCALIZATION_CLASSES), 'precision'),
        'oof_macro_recall': _macro_mean(by_class, list(LOCALIZATION_CLASSES), 'recall'),
    }


def build_targetp_comparison_table(cdskit_result):
    table_rows = list()
    for class_name in LOCALIZATION_CLASSES:
        ref = TARGETP_TABLE1_REFERENCE[class_name]
        ours = cdskit_result['oof_by_class'][class_name]
        table_rows.append({
            'class': class_name,
            'targetp_precision': float(ref['precision']),
            'targetp_recall': float(ref['recall']),
            'targetp_f1': float(ref['f1']),
            'cdskit_precision': float(ours['precision']),
            'cdskit_recall': float(ours['recall']),
            'cdskit_f1': float(ours['f1']),
            'delta_f1_cdskit_minus_targetp': float(ours['f1'] - ref['f1']),
        })
    targetp_macro_f1 = float(np.mean(np.asarray(
        [TARGETP_TABLE1_REFERENCE[c]['f1'] for c in LOCALIZATION_CLASSES],
        dtype=np.float64,
    )))
    return {
        'rows': table_rows,
        'targetp_macro_f1': targetp_macro_f1,
        'cdskit_macro_f1': float(cdskit_result['oof_macro_f1']),
        'delta_macro_f1_cdskit_minus_targetp': float(cdskit_result['oof_macro_f1'] - targetp_macro_f1),
        'targetp_source': (
            'TargetP-2.0 paper Table 1 '
            '(https://pmc.ncbi.nlm.nih.gov/articles/PMC7723994/)'
        ),
    }


def render_markdown_table(comparison):
    out = list()
    out.append('| Class | TargetP P | TargetP R | TargetP F1 | cdskit P | cdskit R | cdskit F1 | ΔF1 (cdskit-TargetP) |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for row in comparison['rows']:
        out.append(
            '| {name} | {tp_p:.3f} | {tp_r:.3f} | {tp_f1:.3f} | {our_p:.3f} | {our_r:.3f} | {our_f1:.3f} | {d:.3f} |'.format(
                name=row['class'],
                tp_p=row['targetp_precision'],
                tp_r=row['targetp_recall'],
                tp_f1=row['targetp_f1'],
                our_p=row['cdskit_precision'],
                our_r=row['cdskit_recall'],
                our_f1=row['cdskit_f1'],
                d=row['delta_f1_cdskit_minus_targetp'],
            )
        )
    out.append('')
    out.append('| Metric | TargetP | cdskit | Δ (cdskit-TargetP) |')
    out.append('|---|---:|---:|---:|')
    out.append(
        '| Macro F1 (5-class) | {tp:.3f} | {our:.3f} | {d:.3f} |'.format(
            tp=float(comparison['targetp_macro_f1']),
            our=float(comparison['cdskit_macro_f1']),
            d=float(comparison['delta_macro_f1_cdskit_minus_targetp']),
        )
    )
    return '\n'.join(out)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Prepare TargetP-2.0 benchmark TSV and run fair fold-fixed cdskit comparison.',
    )
    parser.add_argument('--targetp_fasta', default='data/targetp_raw/targetp.fasta', type=str)
    parser.add_argument('--targetp_tab', default='data/targetp_raw/swissprot_annotated_proteins.tab', type=str)
    parser.add_argument('--targetp_npz', default='data/targetp_raw/targetp_data.npz', type=str)
    parser.add_argument('--download', default='no', choices=['yes', 'no'], type=str)
    parser.add_argument('--prepared_tsv', default='data/localize_bench/targetp2_benchmark.tsv', type=str)
    parser.add_argument('--prepare_report_json', default='data/localize_bench/targetp2_prepare_report.json', type=str)
    parser.add_argument('--run_cdskit_cv', default='yes', choices=['yes', 'no'], type=str)
    parser.add_argument('--comparison_json', default='data/localize_bench/targetp2_cdskit_comparison.json', type=str)
    parser.add_argument('--comparison_md', default='data/localize_bench/targetp2_cdskit_comparison.md', type=str)
    parser.add_argument('--model_arch', default='bilstm_attention', choices=['nearest_centroid', 'bilstm_attention', 'esm_head'], type=str)
    parser.add_argument('--localize_strategy', default='single_stage', choices=['single_stage', 'two_stage', 'two_stage_ctp_ltp'], type=str)
    parser.add_argument('--dl_seq_len', default=200, type=int)
    parser.add_argument('--dl_embed_dim', default=32, type=int)
    parser.add_argument('--dl_hidden_dim', default=64, type=int)
    parser.add_argument('--dl_num_layers', default=1, type=int)
    parser.add_argument('--dl_dropout', default=0.2, type=float)
    parser.add_argument('--dl_epochs', default=15, type=int)
    parser.add_argument('--dl_batch_size', default=128, type=int)
    parser.add_argument('--dl_lr', default=1.0e-3, type=float)
    parser.add_argument('--dl_weight_decay', default=1.0e-4, type=float)
    parser.add_argument('--dl_class_weight', default='yes', choices=['yes', 'no'], type=str)
    parser.add_argument('--dl_loss', default='ce', choices=['ce', 'focal'], type=str)
    parser.add_argument('--dl_balanced_batch', default='no', choices=['yes', 'no'], type=str)
    parser.add_argument('--dl_aux_tp_weight', default=0.0, type=float)
    parser.add_argument('--dl_aux_ctp_ltp_weight', default=0.0, type=float)
    parser.add_argument('--dl_seed', default=1, type=int)
    parser.add_argument('--dl_device', default='cpu', choices=['cpu', 'cuda', 'auto'], type=str)
    parser.add_argument('--esm_model_name', default='facebook/esm2_t6_8M_UR50D', type=str)
    parser.add_argument('--esm_model_local_dir', default='', type=str)
    parser.add_argument('--esm_pooling', default='cls', choices=['cls', 'mean'], type=str)
    parser.add_argument('--esm_max_len', default=200, type=int)
    parser.add_argument('--cv_seed', default=1, type=int)
    return parser


def _to_bool_yes_no(text):
    return str(text).strip().lower() in ['yes', 'y', 'true', '1']


def _download_if_requested(args):
    if not _to_bool_yes_no(args.download):
        return
    from urllib import request as urllib_request

    os.makedirs(os.path.dirname(args.targetp_fasta), exist_ok=True)
    url_map = [
        ('https://services.healthtech.dtu.dk/services/TargetP-2.0/targetp.fasta', args.targetp_fasta),
        ('https://services.healthtech.dtu.dk/services/TargetP-2.0/swissprot_annotated_proteins.tab', args.targetp_tab),
        ('https://raw.githubusercontent.com/JJAlmagro/TargetP-2.0/master/data/targetp_data.npz', args.targetp_npz),
    ]
    for url, path in url_map:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib_request.urlopen(url, timeout=120) as resp:
            body = resp.read()
        with open(path, 'wb') as out:
            out.write(body)


def main():
    parser = build_parser()
    args = parser.parse_args()
    _download_if_requested(args=args)

    prep_report = prepare_targetp_benchmark_tsv(
        fasta_path=args.targetp_fasta,
        annotation_tab_path=args.targetp_tab,
        npz_path=args.targetp_npz,
        out_tsv_path=args.prepared_tsv,
        report_json_path=args.prepare_report_json,
    )
    out = {'prepare_report': prep_report}

    if _to_bool_yes_no(args.run_cdskit_cv):
        dl_params = {
            'seq_len': int(args.dl_seq_len),
            'embed_dim': int(args.dl_embed_dim),
            'hidden_dim': int(args.dl_hidden_dim),
            'num_layers': int(args.dl_num_layers),
            'dropout': float(args.dl_dropout),
            'epochs': int(args.dl_epochs),
            'batch_size': int(args.dl_batch_size),
            'learning_rate': float(args.dl_lr),
            'weight_decay': float(args.dl_weight_decay),
            'use_class_weight': _to_bool_yes_no(args.dl_class_weight),
            'loss_name': str(args.dl_loss),
            'balanced_batch': _to_bool_yes_no(args.dl_balanced_batch),
            'aux_tp_weight': float(args.dl_aux_tp_weight),
            'aux_ctp_ltp_weight': float(args.dl_aux_ctp_ltp_weight),
            'seed': int(args.dl_seed),
            'device': str(args.dl_device),
            'esm_model_name': str(args.esm_model_name),
            'esm_model_local_dir': str(args.esm_model_local_dir),
            'esm_pooling': str(args.esm_pooling),
            'esm_max_len': int(args.esm_max_len),
        }
        cdskit_result = run_cdskit_cv_on_targetp(
            training_tsv=args.prepared_tsv,
            model_arch=str(args.model_arch),
            localize_strategy=str(args.localize_strategy),
            dl_params=dl_params,
            cv_seed=int(args.cv_seed),
        )
        comparison = build_targetp_comparison_table(cdskit_result=cdskit_result)
        comparison_md = render_markdown_table(comparison=comparison)
        out['cdskit_cv'] = cdskit_result
        out['comparison'] = comparison
        out['comparison_markdown'] = comparison_md

        if args.comparison_md != '':
            md_dir = os.path.dirname(args.comparison_md)
            if md_dir != '':
                os.makedirs(md_dir, exist_ok=True)
            with open(args.comparison_md, 'w', encoding='utf-8') as out_md:
                out_md.write(comparison_md + '\n')

    if args.comparison_json != '':
        json_dir = os.path.dirname(args.comparison_json)
        if json_dir != '':
            os.makedirs(json_dir, exist_ok=True)
        with open(args.comparison_json, 'w', encoding='utf-8') as out_json:
            json.dump(out, out_json, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

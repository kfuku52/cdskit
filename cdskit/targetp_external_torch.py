import argparse
import json
import os
from collections import Counter

from cdskit.localize_model import LOCALIZATION_CLASSES, save_localize_model
from cdskit.targetp_external_aug import (
    build_external_augmented_training_rows,
    split_external_train_calibration_rows,
)
from cdskit.targetp_external_eval import read_tsv, write_tsv
from cdskit.targetp_torch import (
    TARGETP_CLASS_THRESHOLD_GRID,
    export_targetp2_torch_localize_model,
    fit_targetp2_torch_model,
    save_torch_payload,
    targetp_rows_to_torch_arrays,
)


TARGETP_EXTERNAL_TORCH_DEFAULTS = {
    'calibration_fraction': 0.20,
    'seed': 3101,
    'external_seed': 101,
    'seq_len': 200,
    'hidden_rnn': 128,
    'n_filters': 32,
    'hidden_fc': 128,
    'n_attention': 13,
    'attention_size': 96,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 6,
    'patience_epochs': 4,
    'max_lr_reductions': 1,
    'type_class_weight': 'none',
    'cleavage_loss_weight': 0.0,
    'selection_metric': 'val_macro_f1',
    'balanced_batch': 'yes',
    'initializer': 'targetp_tf',
    'grad_clip_norm': 1.0,
    'rnn_impl': 'torch_lstm',
}


def _class_counts(rows):
    counts = Counter(row.get('localization', '') for row in rows)
    return {
        class_name: int(counts.get(class_name, 0))
        for class_name in LOCALIZATION_CLASSES
    }


def _require_complete_calibration(rows):
    counts = _class_counts(rows)
    missing = [
        class_name for class_name in LOCALIZATION_CLASSES
        if int(counts.get(class_name, 0)) <= 0
    ]
    if len(missing) > 0:
        raise ValueError(
            'External Torch calibration split is missing classes: {}'.format(
                ','.join(missing)
            )
        )


def _torch_train_kwargs(seq_len, train_kwargs):
    out = dict(TARGETP_EXTERNAL_TORCH_DEFAULTS)
    out.update({key: value for key, value in dict(train_kwargs or {}).items() if value is not None})
    out['seq_len'] = int(seq_len)
    return out


def fit_external_augmented_torch_runtime_model(
    training_tsv,
    uniprot_tsv,
    extra_uniprot_tsvs=None,
    exclusion_tsvs=None,
    deeploc_dir='data/localize_bench/deeploc21',
    include_deeploc=True,
    max_external_per_class=5000,
    calibration_fraction=TARGETP_EXTERNAL_TORCH_DEFAULTS['calibration_fraction'],
    calibration_seed=None,
    external_seed=TARGETP_EXTERNAL_TORCH_DEFAULTS['external_seed'],
    seed=TARGETP_EXTERNAL_TORCH_DEFAULTS['seed'],
    use_mmseqs=False,
    exclusion_mmseqs=None,
    mmseqs_min_seq_id=0.30,
    mmseqs_min_coverage=0.80,
    threads=1,
    device='auto',
    verbose=False,
    epoch_checkpoint_path='',
    torch_payload_out='',
    external_tsv_out='',
    class_thresholds=None,
    **train_kwargs
):
    seq_len = int(train_kwargs.get('seq_len', TARGETP_EXTERNAL_TORCH_DEFAULTS['seq_len']))
    target_rows = read_tsv(training_tsv)
    external_rows, external_report = build_external_augmented_training_rows(
        targetp_tsv=training_tsv,
        uniprot_tsv=uniprot_tsv,
        extra_uniprot_tsvs=extra_uniprot_tsvs,
        exclusion_tsvs=exclusion_tsvs,
        deeploc_dir=deeploc_dir,
        include_deeploc=include_deeploc,
        max_per_class=int(max_external_per_class),
        seed=int(external_seed),
        use_mmseqs=bool(use_mmseqs),
        exclusion_mmseqs=exclusion_mmseqs,
        mmseqs_min_seq_id=float(mmseqs_min_seq_id),
        mmseqs_min_coverage=float(mmseqs_min_coverage),
        threads=int(threads),
    )
    if len(external_rows) == 0:
        raise ValueError('No external rows were available after filtering.')
    calibration_seed = int(external_seed) + 791 if calibration_seed is None else int(calibration_seed)
    external_train_rows, external_calibration_rows, calibration_report = split_external_train_calibration_rows(
        rows=external_rows,
        calibration_fraction=float(calibration_fraction),
        seed=calibration_seed,
    )
    if len(external_train_rows) == 0 or len(external_calibration_rows) == 0:
        raise ValueError(
            'External Torch training requires a non-empty stratified calibration split.'
        )
    _require_complete_calibration(external_calibration_rows)
    if str(external_tsv_out or '').strip() != '':
        write_tsv(
            path=str(external_tsv_out),
            rows=external_rows,
            fieldnames=['source', 'accession', 'organism_group', 'localization', 'peroxisome', 'sequence'],
        )

    train_rows = list(target_rows) + list(external_train_rows)
    train_arrays = targetp_rows_to_torch_arrays(rows=train_rows, seq_len=seq_len)
    val_arrays = targetp_rows_to_torch_arrays(rows=external_calibration_rows, seq_len=seq_len)
    fit_kwargs = _torch_train_kwargs(seq_len=seq_len, train_kwargs=train_kwargs)
    fit_verbose = bool(verbose)
    if 'verbose' in fit_kwargs:
        fit_verbose = bool(fit_kwargs.pop('verbose'))
    fit_kwargs.pop('seed', None)
    fit_kwargs.pop('external_seed', None)
    fit_kwargs.pop('calibration_fraction', None)
    payload = fit_targetp2_torch_model(
        x_train=train_arrays['x'],
        y_type_train=train_arrays['y_type'],
        y_cs_train=train_arrays['y_cs'],
        len_train=train_arrays['len_seq'],
        org_train=train_arrays['org'],
        x_val=val_arrays['x'],
        y_type_val=val_arrays['y_type'],
        y_cs_val=val_arrays['y_cs'],
        len_val=val_arrays['len_seq'],
        org_val=val_arrays['org'],
        seed=int(seed),
        device=device,
        verbose=fit_verbose,
        epoch_checkpoint_path=epoch_checkpoint_path,
        **fit_kwargs
    )
    if str(torch_payload_out or '').strip() != '':
        save_torch_payload(path=str(torch_payload_out), payload=payload)
    model = export_targetp2_torch_localize_model(
        model_payload=payload,
        training_tsv=training_tsv,
        class_thresholds=class_thresholds,
    )
    report = {
        'training_tsv': str(training_tsv),
        'uniprot_tsv': str(uniprot_tsv),
        'extra_uniprot_tsvs': [str(path) for path in (extra_uniprot_tsvs or [])],
        'exclusion_tsvs': [str(path) for path in (exclusion_tsvs or [])],
        'external_tsv_out': str(external_tsv_out or ''),
        'use_mmseqs': bool(use_mmseqs),
        'exclusion_mmseqs': None if exclusion_mmseqs is None else bool(exclusion_mmseqs),
        'num_target_rows': int(len(target_rows)),
        'num_external_rows': int(len(external_rows)),
        'num_external_train_rows': int(len(external_train_rows)),
        'num_external_calibration_rows': int(len(external_calibration_rows)),
        'train_counts': _class_counts(train_rows),
        'external_train_counts': _class_counts(external_train_rows),
        'external_calibration_counts': _class_counts(external_calibration_rows),
        'external_report': external_report,
        'external_calibration': calibration_report,
        'torch_config': dict(payload.get('config', {})),
        'best_metrics': payload.get('best_metrics', {}),
        'final_val_metrics': payload.get('final_val_metrics', {}),
    }
    model['metadata'].update({
        'uniprot_tsv': str(uniprot_tsv),
        'extra_uniprot_tsvs': [str(path) for path in (extra_uniprot_tsvs or [])],
        'exclusion_tsvs': [str(path) for path in (exclusion_tsvs or [])],
        'external_tsv': str(external_tsv_out or ''),
        'num_target_rows': int(len(target_rows)),
        'num_external_rows': int(len(external_rows)),
        'num_external_train_rows': int(len(external_train_rows)),
        'num_external_calibration_rows': int(len(external_calibration_rows)),
        'external_split_report': calibration_report,
        'external_report': external_report,
        'model_arch': 'targetp_torch_v1_external_augmented',
    })
    return {
        'model': model,
        'payload': payload,
        'report': report,
    }


def write_external_augmented_torch_report(path, report):
    out_dir = os.path.dirname(path)
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def _yes_no(value):
    return str(value).strip().lower() in ['yes', 'y', 'true', '1']


def _parse_paths(text):
    return [
        value.strip() for value in str(text or '').split(',')
        if value.strip() != ''
    ]


def _parse_exclusion_mmseqs(value):
    value = str(value or 'auto').strip().lower()
    if value == 'auto':
        return None
    return _yes_no(value)


def _parse_grid(text):
    values = [
        float(part.strip()) for part in str(text or '').split(',')
        if part.strip() != ''
    ]
    if len(values) == 0:
        raise ValueError('threshold grid should contain at least one value.')
    return sorted(set(values))


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


def build_parser():
    parser = argparse.ArgumentParser(
        description='Train a CPU-inference TargetP-style Torch model with strict non-overlapping external weak labels.',
    )
    parser.add_argument('--training_tsv', default='data/localize_bench/targetp2_benchmark.tsv', type=str)
    parser.add_argument(
        '--uniprot_tsv',
        default='data/localize_bench/eukaryota_full_with_lineage_plus_thylakoid_lumen_20260530.tsv',
        type=str,
    )
    parser.add_argument('--extra_uniprot_tsvs', default='', type=str)
    parser.add_argument('--exclusion_tsvs', default='', type=str)
    parser.add_argument('--deeploc_dir', default='data/localize_bench/deeploc21', type=str)
    parser.add_argument('--include_deeploc', default='yes', choices=['yes', 'no'], type=str)
    parser.add_argument('--max_external_per_class', default=5000, type=int)
    parser.add_argument('--calibration_fraction', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['calibration_fraction'], type=float)
    parser.add_argument('--calibration_seed', default='', type=str)
    parser.add_argument('--external_seed', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['external_seed'], type=int)
    parser.add_argument('--seed', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['seed'], type=int)
    parser.add_argument('--mmseqs', default='no', choices=['yes', 'no'], type=str)
    parser.add_argument('--exclusion_mmseqs', default='yes', choices=['auto', 'yes', 'no'], type=str)
    parser.add_argument('--mmseqs_min_seq_id', default=0.30, type=float)
    parser.add_argument('--mmseqs_min_coverage', default=0.80, type=float)
    parser.add_argument('--threads', default=1, type=int)
    parser.add_argument('--device', default='auto', type=str)
    parser.add_argument('--seq_len', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['seq_len'], type=int)
    parser.add_argument('--hidden_rnn', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['hidden_rnn'], type=int)
    parser.add_argument('--n_filters', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['n_filters'], type=int)
    parser.add_argument('--hidden_fc', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['hidden_fc'], type=int)
    parser.add_argument('--n_attention', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['n_attention'], type=int)
    parser.add_argument('--attention_size', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['attention_size'], type=int)
    parser.add_argument('--learning_rate', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['learning_rate'], type=float)
    parser.add_argument('--batch_size', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['batch_size'], type=int)
    parser.add_argument('--epochs', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['epochs'], type=int)
    parser.add_argument('--patience_epochs', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['patience_epochs'], type=int)
    parser.add_argument('--max_lr_reductions', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['max_lr_reductions'], type=int)
    parser.add_argument('--type_class_weight', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['type_class_weight'], choices=['none', 'balanced', 'sqrt_balanced', 'log_balanced'], type=str)
    parser.add_argument('--cleavage_loss_weight', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['cleavage_loss_weight'], type=float)
    parser.add_argument('--selection_metric', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['selection_metric'], choices=['val_loss', 'val_macro_f1', 'val_threshold_macro_f1'], type=str)
    parser.add_argument('--balanced_batch', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['balanced_batch'], choices=['yes', 'no'], type=str)
    parser.add_argument('--initializer', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['initializer'], choices=['targetp_tf', 'pytorch'], type=str)
    parser.add_argument('--grad_clip_norm', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['grad_clip_norm'], type=float)
    parser.add_argument('--rnn_impl', default=TARGETP_EXTERNAL_TORCH_DEFAULTS['rnn_impl'], choices=['torch_lstm', 'targetp_tf_cell'], type=str)
    parser.add_argument('--threshold_grid', default=','.join(str(value) for value in TARGETP_CLASS_THRESHOLD_GRID), type=str)
    parser.add_argument('--class_thresholds', default='', type=str)
    parser.add_argument('--epoch_checkpoint_path', default='', type=str)
    parser.add_argument('--torch_payload_out', default='', type=str)
    parser.add_argument('--external_tsv_out', default='', type=str)
    parser.add_argument('--model_out', required=True, type=str)
    parser.add_argument('--out_json', required=True, type=str)
    parser.add_argument('--verbose', default='no', choices=['yes', 'no'], type=str)
    return parser


def main():
    args = build_parser().parse_args()
    result = fit_external_augmented_torch_runtime_model(
        training_tsv=args.training_tsv,
        uniprot_tsv=args.uniprot_tsv,
        extra_uniprot_tsvs=_parse_paths(args.extra_uniprot_tsvs),
        exclusion_tsvs=_parse_paths(args.exclusion_tsvs),
        deeploc_dir=args.deeploc_dir,
        include_deeploc=_yes_no(args.include_deeploc),
        max_external_per_class=int(args.max_external_per_class),
        calibration_fraction=float(args.calibration_fraction),
        calibration_seed=(
            None if str(args.calibration_seed).strip() == ''
            else int(args.calibration_seed)
        ),
        external_seed=int(args.external_seed),
        seed=int(args.seed),
        use_mmseqs=_yes_no(args.mmseqs),
        exclusion_mmseqs=_parse_exclusion_mmseqs(args.exclusion_mmseqs),
        mmseqs_min_seq_id=float(args.mmseqs_min_seq_id),
        mmseqs_min_coverage=float(args.mmseqs_min_coverage),
        threads=int(args.threads),
        device=args.device,
        verbose=_yes_no(args.verbose),
        epoch_checkpoint_path=args.epoch_checkpoint_path,
        torch_payload_out=args.torch_payload_out,
        external_tsv_out=args.external_tsv_out,
        class_thresholds=_parse_class_thresholds(args.class_thresholds),
        seq_len=int(args.seq_len),
        hidden_rnn=int(args.hidden_rnn),
        n_filters=int(args.n_filters),
        hidden_fc=int(args.hidden_fc),
        n_attention=int(args.n_attention),
        attention_size=int(args.attention_size),
        learning_rate=float(args.learning_rate),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        patience_epochs=int(args.patience_epochs),
        max_lr_reductions=int(args.max_lr_reductions),
        type_class_weight=args.type_class_weight,
        cleavage_loss_weight=float(args.cleavage_loss_weight),
        selection_metric=args.selection_metric,
        selection_threshold_grid=_parse_grid(args.threshold_grid),
        balanced_batch=args.balanced_batch,
        initializer=args.initializer,
        grad_clip_norm=float(args.grad_clip_norm),
        rnn_impl=args.rnn_impl,
    )
    save_localize_model(model=result['model'], path=str(args.model_out))
    write_external_augmented_torch_report(path=args.out_json, report=result['report'])
    best = result['report'].get('best_metrics', {})
    print('target_rows={} external_train_rows={} external_calibration_rows={} best_val_macro_f1={:.6f}'.format(
        int(result['report']['num_target_rows']),
        int(result['report']['num_external_train_rows']),
        int(result['report']['num_external_calibration_rows']),
        float(best.get('val_macro_f1', 0.0)),
    ))


if __name__ == '__main__':
    main()

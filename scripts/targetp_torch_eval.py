#!/usr/bin/env python
import argparse

from cdskit.targetp_torch import (
    TARGETP_CLASS_THRESHOLD_GRID,
    TARGETP_TORCH_DEFAULTS,
    run_targetp2_torch_nested_oof,
    run_targetp2_torch_paired_oof,
    write_targetp2_torch_oof_npz,
    write_targetp2_torch_report,
)


def _parse_threshold_grid(text):
    values = [
        float(part.strip()) for part in str(text).split(',')
        if part.strip() != ''
    ]
    if len(values) == 0:
        raise ValueError('--threshold_grid should contain at least one value.')
    return sorted(set(values))


def build_parser():
    parser = argparse.ArgumentParser(
        description='Train/evaluate a PyTorch TargetP2-style model with fold-fixed nested OOF.',
    )
    parser.add_argument('--targetp_npz', default='data/targetp_raw/targetp_data.npz', type=str)
    parser.add_argument('--model_dir', default='data/localize_bench/targetp2_torch_models', type=str)
    parser.add_argument('--out_npz', default='data/localize_bench/targetp2_oof_targetp_torch.npz', type=str)
    parser.add_argument('--out_json', default='data/localize_bench/targetp2_torch_eval.json', type=str)
    parser.add_argument('--outer_folds', default='all', type=str)
    parser.add_argument('--val_folds', default='all', type=str)
    parser.add_argument(
        '--fold_pairs',
        default='',
        type=str,
        help='Optional comma-separated outer:val pairs, for example 0:1,1:2,2:3.',
    )
    parser.add_argument('--max_models', default=0, type=int)
    parser.add_argument('--reuse_cache', default='yes', choices=['yes', 'no'], type=str)
    parser.add_argument('--device', default='auto', type=str)
    parser.add_argument('--seed_offset', default=0, type=int)
    parser.add_argument('--epochs', default=TARGETP_TORCH_DEFAULTS['epochs'], type=int)
    parser.add_argument('--batch_size', default=TARGETP_TORCH_DEFAULTS['batch_size'], type=int)
    parser.add_argument('--learning_rate', default=TARGETP_TORCH_DEFAULTS['learning_rate'], type=float)
    parser.add_argument('--hidden_rnn', default=TARGETP_TORCH_DEFAULTS['hidden_rnn'], type=int)
    parser.add_argument('--n_filters', default=TARGETP_TORCH_DEFAULTS['n_filters'], type=int)
    parser.add_argument('--hidden_fc', default=TARGETP_TORCH_DEFAULTS['hidden_fc'], type=int)
    parser.add_argument('--n_attention', default=TARGETP_TORCH_DEFAULTS['n_attention'], type=int)
    parser.add_argument('--attention_size', default=TARGETP_TORCH_DEFAULTS['attention_size'], type=int)
    parser.add_argument('--input_keep_prob', default=TARGETP_TORCH_DEFAULTS['input_keep_prob'], type=float)
    parser.add_argument('--encoder_keep_prob', default=TARGETP_TORCH_DEFAULTS['encoder_keep_prob'], type=float)
    parser.add_argument('--rnn_keep_prob', default=TARGETP_TORCH_DEFAULTS['rnn_keep_prob'], type=float)
    parser.add_argument('--patience_epochs', default=TARGETP_TORCH_DEFAULTS['patience_epochs'], type=int)
    parser.add_argument('--max_lr_reductions', default=TARGETP_TORCH_DEFAULTS['max_lr_reductions'], type=int)
    parser.add_argument(
        '--type_class_weight',
        default=TARGETP_TORCH_DEFAULTS['type_class_weight'],
        choices=['none', 'balanced', 'sqrt_balanced', 'log_balanced'],
        type=str,
    )
    parser.add_argument('--cleavage_loss_weight', default=TARGETP_TORCH_DEFAULTS['cleavage_loss_weight'], type=float)
    parser.add_argument(
        '--selection_metric',
        default=TARGETP_TORCH_DEFAULTS['selection_metric'],
        choices=['val_loss', 'val_macro_f1', 'val_threshold_macro_f1'],
        type=str,
    )
    parser.add_argument('--balanced_batch', default=TARGETP_TORCH_DEFAULTS['balanced_batch'], choices=['yes', 'no'], type=str)
    parser.add_argument('--initializer', default=TARGETP_TORCH_DEFAULTS['initializer'], choices=['targetp_tf', 'pytorch'], type=str)
    parser.add_argument('--grad_clip_norm', default=TARGETP_TORCH_DEFAULTS['grad_clip_norm'], type=float)
    parser.add_argument('--rnn_impl', default=TARGETP_TORCH_DEFAULTS['rnn_impl'], choices=['torch_lstm', 'targetp_tf_cell'], type=str)
    parser.add_argument('--val_threshold_eval', default='no', choices=['yes', 'no'], type=str)
    parser.add_argument(
        '--threshold_grid',
        default=','.join(str(value) for value in TARGETP_CLASS_THRESHOLD_GRID),
        type=str,
    )
    parser.add_argument('--verbose', default='no', choices=['yes', 'no'], type=str)
    return parser


def main():
    args = build_parser().parse_args()
    threshold_grid = _parse_threshold_grid(args.threshold_grid)
    train_kwargs = {
        'epochs': int(args.epochs),
        'batch_size': int(args.batch_size),
        'learning_rate': float(args.learning_rate),
        'hidden_rnn': int(args.hidden_rnn),
        'n_filters': int(args.n_filters),
        'hidden_fc': int(args.hidden_fc),
        'n_attention': int(args.n_attention),
        'attention_size': int(args.attention_size),
        'input_keep_prob': float(args.input_keep_prob),
        'encoder_keep_prob': float(args.encoder_keep_prob),
        'rnn_keep_prob': float(args.rnn_keep_prob),
        'patience_epochs': int(args.patience_epochs),
        'max_lr_reductions': int(args.max_lr_reductions),
        'type_class_weight': args.type_class_weight,
        'cleavage_loss_weight': float(args.cleavage_loss_weight),
        'selection_metric': args.selection_metric,
        'selection_threshold_grid': threshold_grid,
        'balanced_batch': args.balanced_batch,
        'initializer': args.initializer,
        'grad_clip_norm': float(args.grad_clip_norm),
        'rnn_impl': args.rnn_impl,
        'verbose': str(args.verbose).strip().lower() == 'yes',
    }
    if str(args.fold_pairs).strip() != '':
        result = run_targetp2_torch_paired_oof(
            targetp_npz=args.targetp_npz,
            model_dir=args.model_dir,
            fold_pairs=args.fold_pairs,
            reuse_cache=str(args.reuse_cache).strip().lower() == 'yes',
            seed_offset=int(args.seed_offset),
            device=args.device,
            val_threshold_eval=str(args.val_threshold_eval).strip().lower() == 'yes',
            threshold_grid=threshold_grid,
            **train_kwargs
        )
    else:
        result = run_targetp2_torch_nested_oof(
            targetp_npz=args.targetp_npz,
            model_dir=args.model_dir,
            outer_folds=args.outer_folds,
            val_folds=args.val_folds,
            reuse_cache=str(args.reuse_cache).strip().lower() == 'yes',
            max_models=int(args.max_models),
            seed_offset=int(args.seed_offset),
            device=args.device,
            val_threshold_eval=str(args.val_threshold_eval).strip().lower() == 'yes',
            threshold_grid=threshold_grid,
            **train_kwargs
        )
    profile = {
        'targetp_npz': args.targetp_npz,
        'model_dir': args.model_dir,
        'outer_folds': args.outer_folds,
        'val_folds': args.val_folds,
        'fold_pairs': args.fold_pairs,
        'max_models': int(args.max_models),
        'reuse_cache': args.reuse_cache,
        'device': args.device,
        'seed_offset': int(args.seed_offset),
        'val_threshold_eval': args.val_threshold_eval,
        'threshold_grid': threshold_grid,
        'train_kwargs': train_kwargs,
    }
    write_targetp2_torch_oof_npz(path=args.out_npz, result=result)
    write_targetp2_torch_report(path=args.out_json, result=result, profile=profile)
    print('covered_rows={}/{} macro_f1={:.6f} accuracy={:.6f}'.format(
        int(result['metrics']['covered_rows']),
        int(result['metrics']['total_rows']),
        float(result['metrics']['macro_f1']),
        float(result['metrics']['overall_accuracy']),
    ))
    if 'val_threshold_metrics' in result:
        threshold_metrics = result['val_threshold_metrics']
        print('val_threshold_rows={}/{} macro_f1={:.6f} accuracy={:.6f}'.format(
            int(threshold_metrics['covered_rows']),
            int(threshold_metrics['total_rows']),
            float(threshold_metrics['macro_f1']),
            float(threshold_metrics['overall_accuracy']),
        ))


if __name__ == '__main__':
    main()

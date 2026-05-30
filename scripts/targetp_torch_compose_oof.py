#!/usr/bin/env python
import argparse

from cdskit.targetp_torch import (
    compose_targetp_torch_oof_replacements,
    write_targetp2_torch_compose_report,
    write_targetp2_torch_oof_npz,
)


def _parse_paths(text):
    paths = [
        part.strip() for part in str(text).split(',')
        if part.strip() != ''
    ]
    if len(paths) == 0:
        raise ValueError('--replacement_oof_npzs should contain at least one path.')
    return paths


def build_parser():
    parser = argparse.ArgumentParser(
        description='Compose a TargetP torch OOF cache by replacing full folds from partial OOF caches.',
    )
    parser.add_argument('--base_oof_npz', required=True, type=str)
    parser.add_argument('--replacement_oof_npzs', required=True, type=str)
    parser.add_argument(
        '--source',
        default='val_threshold',
        choices=['val_threshold', 'prob_matrix'],
        type=str,
        help='Which replacement scores to use. val_threshold scores are row-normalized before export.',
    )
    parser.add_argument('--out_npz', required=True, type=str)
    parser.add_argument('--out_json', required=True, type=str)
    return parser


def main():
    args = build_parser().parse_args()
    replacements = _parse_paths(args.replacement_oof_npzs)
    result = compose_targetp_torch_oof_replacements(
        base_oof_npz=args.base_oof_npz,
        replacement_oof_npzs=replacements,
        source=args.source,
    )
    write_targetp2_torch_oof_npz(path=args.out_npz, result=result)
    write_targetp2_torch_compose_report(
        path=args.out_json,
        result=result,
        profile={
            'base_oof_npz': args.base_oof_npz,
            'replacement_oof_npzs': replacements,
            'source': args.source,
        },
    )
    metrics = result['metrics']
    print('rows={}/{} macro_f1={:.6f} accuracy={:.6f}'.format(
        int(metrics['covered_rows']),
        int(metrics['total_rows']),
        float(metrics['macro_f1']),
        float(metrics['overall_accuracy']),
    ))


if __name__ == '__main__':
    main()

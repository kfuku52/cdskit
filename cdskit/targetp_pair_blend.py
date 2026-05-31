import argparse
import json

from cdskit.localize_model import LOCALIZATION_CLASSES, load_localize_model, save_localize_model
from cdskit.targetp_blend import build_targetp_pair_blend_runtime_model


def _parse_class_values(text, default):
    text = str(text or '').strip()
    if text == '':
        return {class_name: float(default) for class_name in LOCALIZATION_CLASSES}
    if '=' not in text:
        value = float(text)
        return {class_name: value for class_name in LOCALIZATION_CLASSES}
    out = {class_name: float(default) for class_name in LOCALIZATION_CLASSES}
    for part in text.split(','):
        part = part.strip()
        if part == '':
            continue
        if '=' not in part:
            raise ValueError('Class-specific values should use CLASS=VALUE entries.')
        class_name, value = part.split('=', 1)
        class_name = class_name.strip()
        if class_name not in LOCALIZATION_CLASSES:
            raise ValueError('Unknown class: {}'.format(class_name))
        out[class_name] = float(value.strip())
    return out


def build_parser():
    parser = argparse.ArgumentParser(
        description='Build a CPU-runtime TargetP pair blend from two trained cdskit localize models.',
    )
    parser.add_argument('--model_a', required=True, type=str)
    parser.add_argument('--model_b', required=True, type=str)
    parser.add_argument('--alpha', default='0.5', type=str)
    parser.add_argument('--class_thresholds', default='', type=str)
    parser.add_argument('--perox_source', default='a', choices=['a', 'b'], type=str)
    parser.add_argument('--metadata_json', default='', type=str)
    parser.add_argument('--model_out', required=True, type=str)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    model_a = load_localize_model(args.model_a)
    model_b = load_localize_model(args.model_b)
    metadata = {}
    if str(args.metadata_json).strip() != '':
        metadata = json.loads(str(args.metadata_json))
        if not isinstance(metadata, dict):
            raise ValueError('--metadata_json should decode to an object.')
    metadata.update({
        'base_model_a': str(args.model_a),
        'base_model_b': str(args.model_b),
        'alpha_by_class': _parse_class_values(args.alpha, default=0.5),
        'class_thresholds': _parse_class_values(args.class_thresholds, default=1.0),
    })
    model = build_targetp_pair_blend_runtime_model(
        base_model_a=model_a,
        base_model_b=model_b,
        alpha_by_class=metadata['alpha_by_class'],
        class_thresholds=metadata['class_thresholds'],
        perox_source=args.perox_source,
        metadata=metadata,
    )
    save_localize_model(model=model, path=str(args.model_out))
    print(json.dumps({
        'model_out': str(args.model_out),
        'model_type': model['model_type'],
        'alpha_by_class': metadata['alpha_by_class'],
        'class_thresholds': metadata['class_thresholds'],
    }, indent=2, sort_keys=True))
    return model


if __name__ == '__main__':
    main()

from functools import partial

from cdskit.localize_model import (
    BROAD_FEATURE_NAMES,
    FEATURE_NAMES,
    LOCALIZATION_CLASSES,
    load_localize_model,
    predict_localization_and_peroxisome,
    predict_multilabel_localization,
    to_canonical_aa_sequence,
    translate_inframe_cds_to_aa,
    write_rows_json,
    write_rows_tsv,
)
from cdskit.util import (
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
    stop_if_invalid_codontable,
    stop_if_not_dna,
    stop_if_not_multiple_of_three,
    stop_if_not_protein,
)


MULTILABEL_MODEL_TYPES = {'multilabel_centroid_v1', 'multilabel_cnn_v1'}


def _record_to_aa_sequence(record, codontable, seqtype):
    seqtype = str(seqtype or 'dna').strip().lower()
    if seqtype == 'protein':
        return to_canonical_aa_sequence(aa_seq=str(record.seq))
    if seqtype == 'dna':
        return translate_inframe_cds_to_aa(
            cds_seq=str(record.seq),
            codontable=codontable,
            seq_id=record.id,
        )
    raise ValueError('--seqtype should be dna or protein.')


def _predict_single_record(record, codontable, seqtype, model, include_features, organism_group=''):
    aa_seq = _record_to_aa_sequence(
        record=record,
        codontable=codontable,
        seqtype=seqtype,
    )
    if str(model.get('model_type', '')) in MULTILABEL_MODEL_TYPES:
        pred = predict_multilabel_localization(
            aa_seq=aa_seq,
            model=model,
            kingdom=organism_group,
        )
        class_order = list(model['localization_model']['class_order'])
        row = {
            'seq_id': record.id,
            'predicted_labels': ';'.join(pred['predicted_labels']),
        }
        for class_name in class_order:
            row['p_{}'.format(class_name)] = float(
                pred['class_probabilities'].get(class_name, 0.0)
            )
        if 'peroxisome' in class_order:
            row['perox_signal_type'] = pred['perox_signal_type']
        if include_features:
            for name, value in zip(BROAD_FEATURE_NAMES, pred['feature_values']):
                row[name] = float(value)
        return row

    pred = predict_localization_and_peroxisome(
        aa_seq=aa_seq,
        model=model,
        organism_group=organism_group,
    )
    row = {
        'seq_id': record.id,
        'predicted_class': pred['predicted_class'],
        'p_noTP': float(pred['class_probabilities'].get('noTP', 0.0)),
        'p_SP': float(pred['class_probabilities'].get('SP', 0.0)),
        'p_mTP': float(pred['class_probabilities'].get('mTP', 0.0)),
        'p_cTP': float(pred['class_probabilities'].get('cTP', 0.0)),
        'p_lTP': float(pred['class_probabilities'].get('lTP', 0.0)),
        'p_peroxisome': float(pred['perox_probability_yes']),
        'perox_signal_type': pred['perox_signal_type'],
    }
    if include_features:
        for name, value in zip(FEATURE_NAMES, pred['feature_values']):
            row[name] = float(value)
    return row


def _resolve_output_fields(include_features, model=None):
    if isinstance(model, dict) and str(model.get('model_type', '')) in MULTILABEL_MODEL_TYPES:
        class_order = list(model['localization_model']['class_order'])
        fields = ['seq_id', 'predicted_labels']
        fields.extend(['p_{}'.format(class_name) for class_name in class_order])
        if 'peroxisome' in class_order:
            fields.append('perox_signal_type')
        if include_features:
            fields.extend(BROAD_FEATURE_NAMES)
        return fields

    fields = [
        'seq_id',
        'predicted_class',
        'p_noTP',
        'p_SP',
        'p_mTP',
        'p_cTP',
        'p_lTP',
        'p_peroxisome',
        'perox_signal_type',
    ]
    if include_features:
        fields.extend(FEATURE_NAMES)
    return fields


def localize_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    seqtype = str(getattr(args, 'seqtype', 'dna') or 'dna').strip().lower()
    if seqtype == 'protein':
        stop_if_not_protein(records=records, label='--seqfile')
    elif seqtype == 'dna':
        stop_if_not_dna(records=records, label='--seqfile')
        stop_if_not_multiple_of_three(records=records)
        stop_if_invalid_codontable(codontable=args.codontable, label='--codontable')
    else:
        raise ValueError('--seqtype should be dna or protein.')

    model = load_localize_model(path=args.model)
    if str(model.get('model_type', '')) not in MULTILABEL_MODEL_TYPES:
        model_classes = tuple(model['localization_model']['class_order'])
        if model_classes != LOCALIZATION_CLASSES:
            txt = 'Model class order mismatch: expected {}, got {}. Exiting.'
            raise Exception(txt.format(','.join(LOCALIZATION_CLASSES), ','.join(model_classes)))

    threads = resolve_threads(getattr(args, 'threads', 1))
    worker = partial(
        _predict_single_record,
        codontable=args.codontable,
        seqtype=seqtype,
        model=model,
        include_features=args.include_features,
        organism_group=getattr(args, 'organism_group', ''),
    )
    rows = parallel_map_ordered(items=records, worker=worker, threads=threads)

    report_path = args.report
    if report_path == '':
        report_path = '-'
    if report_path.endswith('.json'):
        write_rows_json(rows=rows, output_path=report_path)
    else:
        write_rows_tsv(
            rows=rows,
            output_path=report_path,
            fieldnames=_resolve_output_fields(
                include_features=args.include_features,
                model=model,
            ),
        )

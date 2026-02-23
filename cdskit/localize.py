from functools import partial

from cdskit.localize_model import (
    FEATURE_NAMES,
    LOCALIZATION_CLASSES,
    load_localize_model,
    predict_localization_and_peroxisome,
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
)


def _predict_single_record(record, codontable, model, include_features):
    aa_seq = translate_inframe_cds_to_aa(
        cds_seq=str(record.seq),
        codontable=codontable,
        seq_id=record.id,
    )
    pred = predict_localization_and_peroxisome(aa_seq=aa_seq, model=model)
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


def _resolve_output_fields(include_features):
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
    stop_if_not_dna(records=records, label='--seqfile')
    stop_if_not_multiple_of_three(records=records)
    stop_if_invalid_codontable(codontable=args.codontable, label='--codontable')

    model = load_localize_model(path=args.model)
    model_classes = tuple(model['localization_model']['class_order'])
    if model_classes != LOCALIZATION_CLASSES:
        txt = 'Model class order mismatch: expected {}, got {}. Exiting.'
        raise Exception(txt.format(','.join(LOCALIZATION_CLASSES), ','.join(model_classes)))

    threads = resolve_threads(getattr(args, 'threads', 1))
    worker = partial(
        _predict_single_record,
        codontable=args.codontable,
        model=model,
        include_features=args.include_features,
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
            fieldnames=_resolve_output_fields(include_features=args.include_features),
        )


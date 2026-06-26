from functools import partial

import numpy as np

from cdskit.localize_models import resolve_localize_model_path
from cdskit.localize_model import (
    BROAD_FEATURE_NAMES,
    FEATURE_NAMES,
    LOCALIZATION_CLASSES,
    apply_organism_group_constraints,
    extract_broad_localize_features,
    extract_localize_features,
    load_localize_model,
    postprocess_localization_probabilities,
    predict_perox,
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
SINGLE_LABEL_BATCH_MODEL_TYPES = {
    'bilstm_attention_v1',
    'esm_head_v1',
    'targetp_torch_v1',
}
DEFAULT_LOCALIZE_BATCH_SIZE = 512
DEFAULT_ESM_BATCH_SIZE = 128


def _is_true_arg(value):
    if isinstance(value, bool):
        return value
    return str(value or '').strip().lower() in {'1', 'true', 't', 'yes', 'y', 'on'}


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


def _row_from_single_label_prediction(
    record_id,
    pred_class,
    class_probs,
    perox_probs,
    perox_signals,
    feature_vec,
    include_features,
):
    row = {
        'seq_id': record_id,
        'predicted_class': pred_class,
        'p_noTP': float(class_probs.get('noTP', 0.0)),
        'p_SP': float(class_probs.get('SP', 0.0)),
        'p_mTP': float(class_probs.get('mTP', 0.0)),
        'p_cTP': float(class_probs.get('cTP', 0.0)),
        'p_lTP': float(class_probs.get('lTP', 0.0)),
        'p_peroxisome': float(perox_probs.get('yes', 0.0)),
        'perox_signal_type': perox_signals['signal_type'],
    }
    if include_features:
        for name, value in zip(FEATURE_NAMES, feature_vec):
            row[name] = float(value)
    return row


def _predict_single_label_records_batched(
    records,
    aa_sequences,
    model,
    include_features,
    organism_group='',
):
    model_type = str(model.get('model_type', ''))
    localization_model = model['localization_model']
    feature_rows = [
        extract_localize_features(aa_seq=aa_seq)
        for aa_seq in aa_sequences
    ]
    feature_matrix = np.asarray(
        [feature_vec for feature_vec, _ in feature_rows],
        dtype=np.float32,
    )
    if model_type == 'bilstm_attention_v1':
        from cdskit.localize_bilstm import predict_bilstm_attention_batch
        prob_matrix = predict_bilstm_attention_batch(
            aa_sequences=aa_sequences,
            localization_model=localization_model,
            device='cpu',
            batch_size=DEFAULT_LOCALIZE_BATCH_SIZE,
            feature_matrix=feature_matrix,
        )
        class_order = list(localization_model['class_order'])
    elif model_type == 'esm_head_v1':
        from cdskit.localize_esm_head import predict_esm_head_batch
        prob_matrix = predict_esm_head_batch(
            aa_sequences=aa_sequences,
            localization_model=localization_model,
            device='cpu',
            batch_size=DEFAULT_ESM_BATCH_SIZE,
        )
        class_order = list(localization_model['class_order'])
    elif model_type == 'targetp_torch_v1':
        from cdskit.targetp_torch import predict_targetp2_torch_batch
        prob_matrix = predict_targetp2_torch_batch(
            aa_sequences=aa_sequences,
            organism_groups=[organism_group] * len(aa_sequences),
            localization_model=localization_model,
            device='cpu',
            batch_size=DEFAULT_LOCALIZE_BATCH_SIZE,
        )
        class_order = list(LOCALIZATION_CLASSES)
    else:
        raise ValueError('Unsupported batched localize model_type: {}'.format(model_type))

    rows = list()
    for i, record in enumerate(records):
        feature_vec, perox_signals = feature_rows[i]
        class_probs = {
            class_order[class_i]: float(prob_matrix[i, class_i])
            for class_i in range(len(class_order))
        }
        class_probs = apply_organism_group_constraints(
            class_probs=class_probs,
            organism_group=organism_group,
        )
        pred_class, class_probs = postprocess_localization_probabilities(
            class_probs=class_probs,
            localization_model=localization_model,
        )
        _, perox_probs = predict_perox(
            feature_vec=feature_vec,
            perox_model=model['perox_model'],
            aa_seq=aa_sequences[i],
            organism_group=organism_group,
        )
        rows.append(_row_from_single_label_prediction(
            record_id=record.id,
            pred_class=pred_class,
            class_probs=class_probs,
            perox_probs=perox_probs,
            perox_signals=perox_signals,
            feature_vec=feature_vec,
            include_features=include_features,
        ))
    return rows


def _predict_multilabel_cnn_records_batched(
    records,
    aa_sequences,
    model,
    include_features,
    organism_group='',
):
    from cdskit.localize_multilabel_cnn import predict_multilabel_cnn_batch

    localization_model = model['localization_model']
    feature_rows = [
        extract_broad_localize_features(
            aa_seq=aa_seq,
            kingdom=organism_group,
        )
        for aa_seq in aa_sequences
    ]
    feature_matrix = None
    if int(localization_model.get('feature_dim', 0)) > 0:
        feature_matrix = np.asarray(
            [feature_vec for feature_vec, _ in feature_rows],
            dtype=np.float32,
        )
    pred = predict_multilabel_cnn_batch(
        aa_sequences=aa_sequences,
        localization_model=localization_model,
        device='cpu',
        batch_size=DEFAULT_LOCALIZE_BATCH_SIZE,
        feature_matrix=feature_matrix,
        apply_thresholds=True,
    )
    class_order = list(localization_model['class_order'])
    prob_matrix = pred['prob_matrix']
    pred_matrix = pred['prediction_matrix']
    rows = list()
    for i, record in enumerate(records):
        feature_vec, perox_signals = feature_rows[i]
        labels = [
            class_order[class_i]
            for class_i in range(len(class_order))
            if int(pred_matrix[i, class_i]) == 1
        ]
        row = {
            'seq_id': record.id,
            'predicted_labels': ';'.join(labels),
        }
        for class_i, class_name in enumerate(class_order):
            row['p_{}'.format(class_name)] = float(prob_matrix[i, class_i])
        if 'peroxisome' in class_order:
            row['perox_signal_type'] = perox_signals['signal_type']
        if include_features:
            for name, value in zip(BROAD_FEATURE_NAMES, feature_vec):
                row[name] = float(value)
        rows.append(row)
    return rows


def _predict_records_batched_if_supported(
    records,
    codontable,
    seqtype,
    model,
    include_features,
    organism_group='',
):
    model_type = str(model.get('model_type', ''))
    if model_type not in SINGLE_LABEL_BATCH_MODEL_TYPES and model_type != 'multilabel_cnn_v1':
        return None
    if model_type in SINGLE_LABEL_BATCH_MODEL_TYPES:
        localization_strategy = str(
            model['localization_model'].get('strategy', 'single_stage')
        ).strip().lower()
        if localization_strategy != 'single_stage':
            return None
    if len(records) == 0:
        return []
    aa_sequences = [
        _record_to_aa_sequence(
            record=record,
            codontable=codontable,
            seqtype=seqtype,
        )
        for record in records
    ]
    if model_type == 'multilabel_cnn_v1':
        return _predict_multilabel_cnn_records_batched(
            records=records,
            aa_sequences=aa_sequences,
            model=model,
            include_features=include_features,
            organism_group=organism_group,
        )
    return _predict_single_label_records_batched(
        records=records,
        aa_sequences=aa_sequences,
        model=model,
        include_features=include_features,
        organism_group=organism_group,
    )


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

    model_path = resolve_localize_model_path(
        model=args.model,
        allow_download=not _is_true_arg(getattr(args, 'no_model_download', False)),
    )
    model = load_localize_model(path=model_path)
    if str(model.get('model_type', '')) not in MULTILABEL_MODEL_TYPES:
        model_classes = tuple(model['localization_model']['class_order'])
        if model_classes != LOCALIZATION_CLASSES:
            txt = 'Model class order mismatch: expected {}, got {}. Exiting.'
            raise Exception(txt.format(','.join(LOCALIZATION_CLASSES), ','.join(model_classes)))

    threads = resolve_threads(getattr(args, 'threads', 1))
    rows = _predict_records_batched_if_supported(
        records=records,
        codontable=args.codontable,
        seqtype=seqtype,
        model=model,
        include_features=args.include_features,
        organism_group=getattr(args, 'organism_group', ''),
    )
    if rows is None:
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

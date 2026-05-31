import copy

import numpy as np
import pytest

from cdskit.localize_model import (
    LOCALIZATION_CLASSES,
    load_localize_model,
    predict_localization_and_peroxisome,
    save_localize_model,
)
from cdskit.targetp_torch import (
    TARGETP_BLOSUM62_ORDER,
    TARGETP_BLOSUM62_PROBABILITIES,
    compose_targetp_torch_oof_replacements,
    encode_targetp_blosum_sequences,
    export_targetp2_torch_localize_model,
    fit_targetp2_torch_model,
    initialize_targetp2_torch_module,
    _build_targetp2_torch_module,
    _can_resume_optimizer_state,
    _normalize_targetp_torch_state_dict,
    _optimize_targetp_class_thresholds,
    _payload_rnn_impl_from_state,
    _targetp_prediction_index_from_prob_vector,
    _targetp_prediction_indices_with_thresholds,
    _targetp2_loss,
    _targetp_type_class_weight_vector,
    load_torch_payload,
    organism_group_to_targetp_org,
    parse_targetp_fold_pairs,
    targetp_blosum62_probability_table,
)


def test_targetp_blosum_encoder_matches_official_probability_rows():
    table = targetp_blosum62_probability_table()
    x, lengths = encode_targetp_blosum_sequences(['AUX'], seq_len=5, blosum_table=table)

    assert lengths.tolist() == [3]
    np.testing.assert_allclose(
        x[0, 0, :],
        TARGETP_BLOSUM62_PROBABILITIES[TARGETP_BLOSUM62_ORDER.index('A')],
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(x[0, 1, :], table['C'], rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(x[0, 2, :], np.full((20,), 0.05), rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(x[0, 3, :], np.full((20,), 0.05), rtol=1.0e-6, atol=1.0e-7)


def _tiny_targetp_arrays(seq_len=12):
    sequences = [
        'MAAAAAAAA',
        'MKKLLLLAA',
        'MARRRRSSS',
        'MASTSTSSS',
        'MRRSTSTSS',
        'MGGGGGGGG',
        'MALLLLAAA',
        'MRRRRAAAA',
        'MSSSSRRRR',
        'MRRSTAAAA',
    ]
    x, lengths = encode_targetp_blosum_sequences(sequences, seq_len=seq_len)
    y_type = np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int64)
    y_cs = np.zeros((len(sequences), seq_len), dtype=np.int64)
    for i, class_idx in enumerate(y_type.tolist()):
        if class_idx > 0:
            y_cs[i, min(seq_len - 1, 2 + class_idx)] = 1
    org = np.asarray([0, 0, 0, 1, 1, 0, 0, 0, 1, 1], dtype=np.int64)
    return x, y_type, y_cs, lengths, org


def test_targetp_torch_payload_can_predict_and_roundtrip(temp_dir):
    pytest.importorskip('torch')
    x, y_type, y_cs, lengths, org = _tiny_targetp_arrays()
    payload = fit_targetp2_torch_model(
        x_train=x[:6],
        y_type_train=y_type[:6],
        y_cs_train=y_cs[:6],
        len_train=lengths[:6],
        org_train=org[:6],
        x_val=x[6:],
        y_type_val=y_type[6:],
        y_cs_val=y_cs[6:],
        len_val=lengths[6:],
        org_val=org[6:],
        seed=2,
        device='cpu',
        seq_len=12,
        hidden_rnn=4,
        n_filters=3,
        hidden_fc=5,
        n_attention=4,
        attention_size=4,
        input_keep_prob=1.0,
        encoder_keep_prob=1.0,
        rnn_keep_prob=1.0,
        epochs=1,
        batch_size=3,
        learning_rate=0.001,
        grad_clip_norm=0.0,
    )
    assert payload['config']['grad_clip_norm'] == 0.0
    assert payload['config']['rnn_impl'] == 'targetp_tf_cell'
    model = export_targetp2_torch_localize_model(model_payload=payload)
    out_path = temp_dir / 'targetp_torch.pt'
    save_localize_model(model=model, path=str(out_path))
    loaded = load_localize_model(path=str(out_path))

    result = predict_localization_and_peroxisome(
        aa_seq='MKKLLLLAA',
        model=loaded,
        organism_group='non_plant',
    )

    assert loaded['model_type'] == 'targetp_torch_v1'
    assert result['predicted_class'] in LOCALIZATION_CLASSES
    assert set(result['class_probabilities']) == set(LOCALIZATION_CLASSES)
    assert organism_group_to_targetp_org('plant') == 1
    assert organism_group_to_targetp_org('non_plant') == 0


def test_targetp_tf_initializer_sets_lstm_forget_bias():
    torch = pytest.importorskip('torch')
    import torch.nn as nn

    module = _build_targetp2_torch_module(
        torch=torch,
        nn=nn,
        seq_len=12,
        hidden_rnn=4,
        n_filters=3,
        hidden_fc=5,
        n_attention=4,
        attention_size=4,
    )
    initialize_targetp2_torch_module(
        torch=torch,
        nn=nn,
        module=module,
        initializer='targetp_tf',
    )

    for name, param in module.encoder.named_parameters():
        if 'bias' in name:
            hidden = int(param.shape[0] // 4)
            expected = np.ones((hidden,), dtype=np.float32) if 'bias_ih' in name else np.zeros((hidden,), dtype=np.float32)
            np.testing.assert_allclose(param.detach().cpu().numpy()[hidden:2 * hidden], expected)


def test_targetp_tf_cell_impl_has_single_kernel_bias_and_outputs():
    torch = pytest.importorskip('torch')
    import torch.nn as nn

    module = _build_targetp2_torch_module(
        torch=torch,
        nn=nn,
        seq_len=12,
        hidden_rnn=4,
        n_filters=3,
        hidden_fc=5,
        n_attention=4,
        attention_size=4,
        rnn_impl='targetp_tf_cell',
    )
    initialize_targetp2_torch_module(
        torch=torch,
        nn=nn,
        module=module,
        initializer='targetp_tf',
    )
    assert module.rnn_impl == 'targetp_tf_cell'
    assert hasattr(module, 'fw_cell')
    assert hasattr(module, 'bw_cell')
    for cell in [module.fw_cell, module.bw_cell]:
        assert tuple(cell.kernel.shape) == (3 + 4, 4 * 4)
        assert tuple(cell.bias.shape) == (4 * 4,)
        np.testing.assert_allclose(cell.bias.detach().cpu().numpy(), np.zeros((4 * 4,), dtype=np.float32))
        assert cell.forget_bias == 1.0
    x, _, _, lengths, org = _tiny_targetp_arrays(seq_len=12)
    module.eval()
    with torch.no_grad():
        out = module(
            x=torch.as_tensor(x[:2], dtype=torch.float32),
            lengths=torch.as_tensor(lengths[:2], dtype=torch.long),
            organism=torch.as_tensor(org[:2], dtype=torch.long),
            rnn_keep_prob=0.5,
        )
    assert tuple(out['type_logits'].shape) == (2, len(LOCALIZATION_CLASSES))
    assert tuple(out['attention_logits'].shape) == (2, 12, 4)


def test_targetp_torch_detects_rnn_impl_from_state_dict():
    torch = pytest.importorskip('torch')
    import torch.nn as nn

    legacy = _build_targetp2_torch_module(
        torch=torch,
        nn=nn,
        seq_len=12,
        hidden_rnn=4,
        n_filters=3,
        hidden_fc=5,
        n_attention=4,
        attention_size=4,
        rnn_impl='torch_lstm',
    )
    tf_cell = _build_targetp2_torch_module(
        torch=torch,
        nn=nn,
        seq_len=12,
        hidden_rnn=4,
        n_filters=3,
        hidden_fc=5,
        n_attention=4,
        attention_size=4,
        rnn_impl='targetp_tf_cell',
    )

    assert _payload_rnn_impl_from_state({'state_dict': legacy.state_dict()}) == 'torch_lstm'
    assert _payload_rnn_impl_from_state({'state_dict': tf_cell.state_dict()}) == 'targetp_tf_cell'


def test_targetp_tf_cell_converts_legacy_torch_lstmcell_state():
    torch = pytest.importorskip('torch')
    import torch.nn as nn

    old_cell = nn.LSTMCell(input_size=3, hidden_size=4)
    torch.manual_seed(11)
    for param in old_cell.parameters():
        param.data.normal_(mean=0.0, std=0.2)
    old_state = {
        'fw_cell.weight_ih': old_cell.weight_ih.detach().clone(),
        'fw_cell.weight_hh': old_cell.weight_hh.detach().clone(),
        'fw_cell.bias_ih': old_cell.bias_ih.detach().clone(),
        'fw_cell.bias_hh': old_cell.bias_hh.detach().clone(),
    }
    converted = _normalize_targetp_torch_state_dict(
        torch=torch,
        state=old_state,
        config={'rnn_impl': 'targetp_tf_cell'},
    )
    module = _build_targetp2_torch_module(
        torch=torch,
        nn=nn,
        seq_len=12,
        hidden_rnn=4,
        n_filters=3,
        hidden_fc=5,
        n_attention=4,
        attention_size=4,
        rnn_impl='targetp_tf_cell',
    )
    module.fw_cell.load_state_dict({
        'kernel': converted['fw_cell.kernel'],
        'bias': converted['fw_cell.bias'],
    })

    x = torch.randn(2, 3)
    h = torch.randn(2, 4)
    c = torch.randn(2, 4)
    old_h, old_c = old_cell(x, (h, c))
    new_h, new_c = module.fw_cell(x, (h, c))
    torch.testing.assert_close(new_h, old_h, rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(new_c, old_c, rtol=1.0e-6, atol=1.0e-6)


def test_targetp_type_class_weight_modes_are_ordered_and_normalized():
    y = np.asarray([0] * 20 + [1] * 10 + [2] * 5 + [3] * 2 + [4], dtype=np.int64)
    balanced = _targetp_type_class_weight_vector(y, mode='balanced', num_class=5)
    sqrt_balanced = _targetp_type_class_weight_vector(y, mode='sqrt_balanced', num_class=5)
    log_balanced = _targetp_type_class_weight_vector(y, mode='log_balanced', num_class=5)

    assert _targetp_type_class_weight_vector(y, mode='none', num_class=5) is None
    for weights in [balanced, sqrt_balanced, log_balanced]:
        assert tuple(weights.shape) == (5,)
        np.testing.assert_allclose(np.mean(weights), 1.0, rtol=1.0e-6, atol=1.0e-6)
        assert weights[4] > weights[3] > weights[2] > weights[1] > weights[0]
    assert (balanced[4] / balanced[0]) > (sqrt_balanced[4] / sqrt_balanced[0])
    assert (sqrt_balanced[4] / sqrt_balanced[0]) > (log_balanced[4] / log_balanced[0])


def test_targetp_loss_can_disable_cleavage_auxiliary_term():
    torch = pytest.importorskip('torch')

    outputs = {
        'type_logits': torch.zeros((2, len(LOCALIZATION_CLASSES)), dtype=torch.float32),
        'attention_logits': torch.zeros((2, 12, 4), dtype=torch.float32),
    }
    y_type = torch.asarray([0, 1], dtype=torch.long)
    y_cs = torch.zeros((2, 12), dtype=torch.long)
    y_cs[1, 4] = 1

    type_only = _targetp2_loss(
        torch=torch,
        outputs=outputs,
        y_type=y_type,
        y_cs=y_cs,
        cleavage_loss_weight=0.0,
    )
    with_cleavage = _targetp2_loss(
        torch=torch,
        outputs=outputs,
        y_type=y_type,
        y_cs=y_cs,
        cleavage_loss_weight=1.0,
    )

    assert float(with_cleavage.detach().cpu().item()) > float(type_only.detach().cpu().item())


def test_targetp_validation_threshold_optimizer_uses_classwise_grid():
    class_names = list(LOCALIZATION_CLASSES)
    prob = np.asarray([
        [0.49, 0.51, 0.00, 0.00, 0.00],
        [0.70, 0.30, 0.00, 0.00, 0.00],
        [0.10, 0.90, 0.00, 0.00, 0.00],
        [0.05, 0.05, 0.90, 0.00, 0.00],
        [0.05, 0.05, 0.00, 0.90, 0.00],
        [0.05, 0.05, 0.00, 0.00, 0.90],
    ], dtype=np.float64)
    true_idx = np.asarray([0, 0, 1, 2, 3, 4], dtype=np.int64)

    base_pred = _targetp_prediction_indices_with_thresholds(
        prob_matrix=prob,
        thresholds=np.ones((len(class_names),), dtype=np.float64),
    )
    thresholds, metrics = _optimize_targetp_class_thresholds(
        prob_matrix=prob,
        true_idx=true_idx,
        class_names=class_names,
        grid=[1.0, 2.0],
    )
    tuned_pred = _targetp_prediction_indices_with_thresholds(
        prob_matrix=prob,
        thresholds=thresholds,
    )

    assert base_pred.tolist() == [1, 0, 1, 2, 3, 4]
    assert tuned_pred.tolist() == true_idx.tolist()
    assert float(thresholds[1]) == 2.0
    assert metrics['macro_f1'] == 1.0


def test_targetp_prediction_index_from_prob_vector_applies_thresholds():
    prob = np.asarray([0.48, 0.52, 0.0, 0.0, 0.0], dtype=np.float64)

    assert _targetp_prediction_index_from_prob_vector(prob) == 1
    assert _targetp_prediction_index_from_prob_vector(
        prob,
        class_thresholds={'SP': 2.0},
    ) == 0


def test_targetp_fold_pair_parser_validates_pairs():
    assert parse_targetp_fold_pairs('0:1, 2:3', available=[0, 1, 2, 3]) == [
        (0, 1),
        (2, 3),
    ]

    with pytest.raises(ValueError):
        parse_targetp_fold_pairs('1:1', available=[0, 1])
    with pytest.raises(ValueError):
        parse_targetp_fold_pairs('1-2', available=[0, 1, 2])
    with pytest.raises(ValueError):
        parse_targetp_fold_pairs('1:5', available=[0, 1, 2])


def test_targetp_torch_oof_compose_replaces_only_covered_rows(temp_dir):
    base_path = temp_dir / 'base.npz'
    replacement_path = temp_dir / 'replacement.npz'
    class_names = np.asarray(LOCALIZATION_CLASSES)
    true_idx = np.asarray([0, 1, 2, 3], dtype=np.int64)
    fold_ids = np.asarray([0, 0, 1, 1], dtype=np.int64)
    base_prob = np.asarray([
        [0.9, 0.1, 0.0, 0.0, 0.0],
        [0.1, 0.9, 0.0, 0.0, 0.0],
        [0.9, 0.0, 0.1, 0.0, 0.0],
        [0.9, 0.0, 0.0, 0.1, 0.0],
    ], dtype=np.float64)
    replacement_scores = np.zeros_like(base_prob)
    replacement_scores[2, :] = [0.0, 0.0, 8.0, 0.0, 0.0]
    replacement_scores[3, :] = [0.0, 3.0, 0.0, 1.0, 0.0]
    replacement_count = np.asarray([0, 0, 2, 2], dtype=np.int64)

    np.savez_compressed(
        base_path,
        prob_matrix=base_prob,
        true_idx=true_idx,
        class_names=class_names,
        fold_ids=fold_ids,
    )
    np.savez_compressed(
        replacement_path,
        prob_matrix=base_prob,
        true_idx=true_idx,
        class_names=class_names,
        fold_ids=fold_ids,
        val_threshold_score_matrix=replacement_scores,
        val_threshold_count=replacement_count,
    )

    result = compose_targetp_torch_oof_replacements(
        base_oof_npz=str(base_path),
        replacement_oof_npzs=[str(replacement_path)],
        source='val_threshold',
    )

    np.testing.assert_allclose(result['prob_matrix'][:2], base_prob[:2])
    np.testing.assert_allclose(result['prob_matrix'][2], [0.0, 0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(result['prob_matrix'][3], [0.0, 0.75, 0.0, 0.25, 0.0])
    assert result['replacements'] == [{
        'path': str(replacement_path),
        'source': 'val_threshold',
        'rows': 2,
        'folds': [1],
    }]
    assert result['metrics']['covered_rows'] == 4
    assert result['metrics']['by_class']['mTP']['f1'] == 1.0


def test_targetp_torch_training_can_select_validation_threshold_macro(temp_dir):
    pytest.importorskip('torch')
    x, y_type, y_cs, lengths, org = _tiny_targetp_arrays()
    payload = fit_targetp2_torch_model(
        x_train=x[:6],
        y_type_train=y_type[:6],
        y_cs_train=y_cs[:6],
        len_train=lengths[:6],
        org_train=org[:6],
        x_val=x[6:],
        y_type_val=y_type[6:],
        y_cs_val=y_cs[6:],
        len_val=lengths[6:],
        org_val=org[6:],
        seed=7,
        device='cpu',
        seq_len=12,
        hidden_rnn=4,
        n_filters=3,
        hidden_fc=5,
        attention_size=4,
        n_attention=5,
        batch_size=3,
        epochs=2,
        learning_rate=0.01,
        selection_metric='val_threshold_macro_f1',
        selection_threshold_grid=[1.0, 2.0],
        cleavage_loss_weight=0.0,
        rnn_impl='targetp_tf_cell',
    )

    assert 'val_threshold_macro_f1' in payload['best_metrics']
    assert isinstance(payload['best_metrics']['val_thresholds'], dict)


def test_targetp_torch_training_can_resume_epoch_checkpoint(temp_dir):
    pytest.importorskip('torch')
    x, y_type, y_cs, lengths, org = _tiny_targetp_arrays()
    checkpoint = temp_dir / 'epoch.pt'
    first = fit_targetp2_torch_model(
        x_train=x[:6],
        y_type_train=y_type[:6],
        y_cs_train=y_cs[:6],
        len_train=lengths[:6],
        org_train=org[:6],
        x_val=x[6:],
        y_type_val=y_type[6:],
        y_cs_val=y_cs[6:],
        len_val=lengths[6:],
        org_val=org[6:],
        seed=3,
        device='cpu',
        seq_len=12,
        hidden_rnn=4,
        n_filters=3,
        hidden_fc=5,
        n_attention=4,
        attention_size=4,
        input_keep_prob=1.0,
        encoder_keep_prob=1.0,
        rnn_keep_prob=1.0,
        epochs=1,
        batch_size=3,
        learning_rate=0.001,
        epoch_checkpoint_path=str(checkpoint),
    )
    saved = load_torch_payload(str(checkpoint))
    resumed = fit_targetp2_torch_model(
        x_train=x[:6],
        y_type_train=y_type[:6],
        y_cs_train=y_cs[:6],
        len_train=lengths[:6],
        org_train=org[:6],
        x_val=x[6:],
        y_type_val=y_type[6:],
        y_cs_val=y_cs[6:],
        len_val=lengths[6:],
        org_val=org[6:],
        seed=3,
        device='cpu',
        seq_len=12,
        hidden_rnn=4,
        n_filters=3,
        hidden_fc=5,
        n_attention=4,
        attention_size=4,
        input_keep_prob=1.0,
        encoder_keep_prob=1.0,
        rnn_keep_prob=1.0,
        epochs=2,
        batch_size=3,
        learning_rate=0.001,
        resume_payload=saved,
        epoch_checkpoint_path=str(checkpoint),
    )

    assert first['training_complete'] is True
    assert saved['training_complete'] is True
    assert resumed['training_complete'] is True
    assert len(first['history']) == 1
    assert len(resumed['history']) == 2
    assert resumed['latest_epoch'] == 2
    assert _can_resume_optimizer_state(resumed) is True


def test_targetp_torch_migrates_legacy_lr_epoch_without_relabeling_best(temp_dir):
    pytest.importorskip('torch')
    x, y_type, y_cs, lengths, org = _tiny_targetp_arrays()
    first = fit_targetp2_torch_model(
        x_train=x[:6],
        y_type_train=y_type[:6],
        y_cs_train=y_cs[:6],
        len_train=lengths[:6],
        org_train=org[:6],
        x_val=x[6:],
        y_type_val=y_type[6:],
        y_cs_val=y_cs[6:],
        len_val=lengths[6:],
        org_val=org[6:],
        seed=5,
        device='cpu',
        seq_len=12,
        hidden_rnn=4,
        n_filters=3,
        hidden_fc=5,
        n_attention=4,
        attention_size=4,
        input_keep_prob=1.0,
        encoder_keep_prob=1.0,
        rnn_keep_prob=1.0,
        epochs=1,
        batch_size=3,
        learning_rate=0.001,
    )
    legacy = copy.deepcopy(first)
    legacy['history'] = list(legacy['history']) + [dict(legacy['history'][-1], epoch=2)]
    legacy['best_epoch'] = 2
    legacy.pop('lr_reference_epoch', None)

    resumed = fit_targetp2_torch_model(
        x_train=x[:6],
        y_type_train=y_type[:6],
        y_cs_train=y_cs[:6],
        len_train=lengths[:6],
        org_train=org[:6],
        x_val=x[6:],
        y_type_val=y_type[6:],
        y_cs_val=y_cs[6:],
        len_val=lengths[6:],
        org_val=org[6:],
        seed=5,
        device='cpu',
        seq_len=12,
        hidden_rnn=4,
        n_filters=3,
        hidden_fc=5,
        n_attention=4,
        attention_size=4,
        input_keep_prob=1.0,
        encoder_keep_prob=1.0,
        rnn_keep_prob=1.0,
        epochs=2,
        batch_size=3,
        learning_rate=0.001,
        resume_payload=legacy,
    )

    assert resumed['best_epoch'] == first['best_metrics']['epoch']
    assert resumed['lr_reference_epoch'] == 2


def test_targetp_torch_can_resume_from_best_with_reset_lr(temp_dir):
    pytest.importorskip('torch')
    x, y_type, y_cs, lengths, org = _tiny_targetp_arrays()
    checkpoint = temp_dir / 'targetp_torch_best_resume.pt'
    first = fit_targetp2_torch_model(
        x_train=x[:6],
        y_type_train=y_type[:6],
        y_cs_train=y_cs[:6],
        len_train=lengths[:6],
        org_train=org[:6],
        x_val=x[6:],
        y_type_val=y_type[6:],
        y_cs_val=y_cs[6:],
        len_val=lengths[6:],
        org_val=org[6:],
        seed=4,
        device='cpu',
        seq_len=12,
        hidden_rnn=4,
        n_filters=3,
        hidden_fc=5,
        n_attention=4,
        attention_size=4,
        input_keep_prob=1.0,
        encoder_keep_prob=1.0,
        rnn_keep_prob=1.0,
        epochs=1,
        batch_size=3,
        learning_rate=0.001,
        epoch_checkpoint_path=str(checkpoint),
    )
    resumed = fit_targetp2_torch_model(
        x_train=x[:6],
        y_type_train=y_type[:6],
        y_cs_train=y_cs[:6],
        len_train=lengths[:6],
        org_train=org[:6],
        x_val=x[6:],
        y_type_val=y_type[6:],
        y_cs_val=y_cs[6:],
        len_val=lengths[6:],
        org_val=org[6:],
        seed=4,
        device='cpu',
        seq_len=12,
        hidden_rnn=4,
        n_filters=3,
        hidden_fc=5,
        n_attention=4,
        attention_size=4,
        input_keep_prob=1.0,
        encoder_keep_prob=1.0,
        rnn_keep_prob=1.0,
        epochs=2,
        batch_size=3,
        learning_rate=0.001,
        resume_payload=first,
        resume_state='best',
        resume_learning_rate=0.0005,
        resume_reset_scheduler='yes',
        epoch_checkpoint_path=str(checkpoint),
    )

    assert len(resumed['history']) == 2
    assert resumed['history'][1]['learning_rate'] == pytest.approx(0.0005)
    assert resumed['current_lr'] == pytest.approx(0.0005)
    assert resumed['lr_reductions'] == 0


def test_targetp_torch_skips_legacy_completed_optimizer_state():
    payload = {
        'training_complete': True,
        'history': [{'epoch': 1}],
        'optimizer_state': {'state': {}, 'param_groups': []},
    }
    assert _can_resume_optimizer_state(payload) is False
    payload['training_complete'] = False
    assert _can_resume_optimizer_state(payload) is True
    payload['training_complete'] = True
    payload['latest_epoch'] = 1
    assert _can_resume_optimizer_state(payload) is True
    payload['latest_epoch'] = 0
    assert _can_resume_optimizer_state(payload) is False

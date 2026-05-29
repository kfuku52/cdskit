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
    encode_targetp_blosum_sequences,
    export_targetp2_torch_localize_model,
    fit_targetp2_torch_model,
    initialize_targetp2_torch_module,
    _build_targetp2_torch_module,
    organism_group_to_targetp_org,
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

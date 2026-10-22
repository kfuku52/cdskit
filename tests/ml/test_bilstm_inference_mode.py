"""Pure prediction must leave cached and serialized BiLSTMs usable by autograd."""

from copy import deepcopy

import numpy as np
import pytest

from cdskit import localize_bilstm as bilstm
from cdskit.localize_batch import predict_localization_batch
from cdskit.localize_model import (
    FEATURE_NAMES,
    load_localize_model,
    save_localize_model,
)
from cdskit.localize_runtime import PredictionRuntime, prediction_runtime
from cdskit.localize_schema import CURRENT_FEATURE_SCHEMA


pytestmark = pytest.mark.ml


def _head(feature_dim=0):
    torch = pytest.importorskip("torch")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        module = bilstm._build_bilstm_attention_module(
            torch,
            torch.nn,
            len(bilstm.DEFAULT_AA_TO_IDX),
            embed_dim=4,
            hidden_dim=4,
            num_layers=2,
            dropout=0.2,
            num_class=5,
            feature_dim=feature_dim,
        )
    return {
        "class_order": ["noTP", "SP", "mTP", "cTP", "lTP"],
        "aa_to_idx": dict(bilstm.DEFAULT_AA_TO_IDX),
        "seq_len": 12,
        "embed_dim": 4,
        "hidden_dim": 4,
        "num_layers": 2,
        "dropout": 0.2,
        "feature_dim": feature_dim,
        "use_feature_fusion": feature_dim > 0,
        "feature_mean": [0.5] * feature_dim,
        "feature_scale": [2.0] * feature_dim,
        "state_dict": deepcopy(module.state_dict()),
        "feature_schema": CURRENT_FEATURE_SCHEMA,
    }


@pytest.mark.parametrize("feature_dim", [0, 3])
def test_prediction_keeps_cached_module_trainable(feature_dim):
    torch = pytest.importorskip("torch")
    head = _head(feature_dim)
    sequences = ["MACDEF", "MKK"]
    features = np.ones((2, feature_dim), dtype=np.float32) if feature_dim else None
    kwargs = {
        "aa_sequences": sequences,
        "localization_model": head,
        "device": "cpu",
        "batch_size": 1,
        "feature_matrix": features,
    }
    expected = bilstm.predict_bilstm_attention_batch(**kwargs)
    runtime = head["_runtime_model_cache"]["cpu"]
    assert not runtime.training
    assert all(
        not torch.is_inference(tensor)
        for tensor in [*runtime.parameters(), *runtime.buffers()]
    )
    repeated = bilstm.predict_bilstm_attention_batch(**kwargs)
    assert head["_runtime_model_cache"]["cpu"] is runtime
    assert repeated.tobytes() == expected.tobytes()
    repeated[:] = -1
    assert (
        bilstm.predict_bilstm_attention_batch(**kwargs).tobytes() == expected.tobytes()
    )

    runtime.train()
    tokens = torch.as_tensor(bilstm.encode_aa_sequences(sequences, head["seq_len"]))
    training_features = torch.as_tensor(features) if features is not None else None
    logits = runtime(tokens, tokens.ne(bilstm.PAD_INDEX), training_features)
    logits.square().mean().backward()
    gradients = [parameter.grad for parameter in runtime.parameters()]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in gradients)

    # A public NumPy result must remain usable as another model's training input.
    downstream = torch.nn.Linear(5, 1, dtype=torch.float64)
    downstream(torch.from_numpy(expected)).sum().backward()
    assert torch.isfinite(downstream.weight.grad).all()


def test_saved_roundtrip_preserves_scores_and_safe_input_gates(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    head = _head()
    head["decision_policy"] = "safe-v1"
    model = {
        "model_type": "bilstm_attention_v1",
        "localization_model": head,
        "perox_model": {"mode": "constant", "yes_probability": 0.0},
        "feature_names": list(FEATURE_NAMES),
        "feature_schema": CURRENT_FEATURE_SCHEMA,
    }
    sequences = ["", "MAAAA", "XX", "M", "MKKK"]
    calls = []
    predict = bilstm.predict_bilstm_attention_batch

    def record_valid_rows(*args, **kwargs):
        calls.extend(kwargs["aa_sequences"])
        return predict(*args, **kwargs)

    monkeypatch.setattr(bilstm, "predict_bilstm_attention_batch", record_valid_rows)
    with prediction_runtime(PredictionRuntime(device="cpu", batch_size=2)):
        before = predict_localization_batch(sequences, model)
        assert calls == ["MAAAA", "MKKK"]
        assert [row["score_available"] for row in before] == [
            False,
            True,
            False,
            False,
            True,
        ]
        assert before[0]["decision_status"] == "invalid_input"
        assert (
            before[2]["decision_status"] == before[3]["decision_status"] == "abstained"
        )
        assert predict_localization_batch([], model) == []
        assert predict([], head).shape == (0, 5)

        checkpoint = tmp_path / "bilstm.pt"
        save_localize_model(model, checkpoint)
        restored = load_localize_model(checkpoint)
        restored_head = restored["localization_model"]
        assert "_runtime_model_cache" not in restored_head
        assert restored["feature_schema"] == model["feature_schema"]
        assert restored_head["decision_policy"] == head["decision_policy"]
        assert restored_head["class_order"] == head["class_order"]
        for key, value in head["state_dict"].items():
            assert torch.equal(value, restored_head["state_dict"][key])
        after = predict_localization_batch(sequences, restored)
    for restored_row, original_row in zip(after, before, strict=True):
        np.testing.assert_array_equal(
            restored_row.pop("feature_values"), original_row.pop("feature_values")
        )
        assert restored_row == original_row

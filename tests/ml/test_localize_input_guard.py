"""Exercise real inference plumbing with fixed logits, without retraining weights."""

import copy

import numpy as np
import pytest

from cdskit.localize_bilstm import DEFAULT_AA_TO_IDX
from cdskit.localize_multilabel_cnn import predict_multilabel_cnn_batch
from cdskit.localize_multilabel_plm import predict_multilabel_plm
from cdskit.localize_runtime import PredictionRuntime, prediction_runtime
from cdskit.localize_schema import CURRENT_FEATURE_SCHEMA, LEGACY_FEATURE_SCHEMA
from cdskit.localize_specialists import apply_specialists
from cdskit.localize_model import BROAD_FEATURE_NAMES


pytestmark = pytest.mark.ml


@pytest.fixture
def fixed_backends(monkeypatch):
    import torch
    from cdskit import localize_multilabel_cnn as cnn, localize_multilabel_plm as plm

    calls = []

    class Fixed(torch.nn.Module):
        def forward(self, tokens, feature_vec=None):
            calls.append(len(tokens))
            return torch.tensor([0.0, 2.0]).repeat(len(tokens), 1)

    monkeypatch.setattr(
        cnn, "_get_runtime_cnn_model", lambda **kwargs: (Fixed(), "cpu")
    )
    monkeypatch.setattr(plm, "resolve_torch_device", lambda device: "cpu")
    monkeypatch.setattr(
        plm,
        "_batch",
        lambda encoder, seqs, torch, device: (torch.zeros((len(seqs), 3)), None),
    )
    return calls, Fixed()


def heads(fixed):
    common = {
        "class_order": ["nucleus", "chloroplast"],
        "class_thresholds": {"nucleus": 0.9, "chloroplast": 0.95},
        "ensure_one_label": True,
        "decision_policy": "safe-v1",
    }
    cnn = dict(common, seq_len=16, aa_to_idx=DEFAULT_AA_TO_IDX, feature_dim=0)
    plm = dict(common, _runtime_model_cache={"cpu": (None, fixed)})
    return [(predict_multilabel_cnn_batch, cnn), (predict_multilabel_plm, plm)]


def test_mixed_rows_preserve_order_and_only_valid_rows_reach_backend(fixed_backends):
    calls, fixed = fixed_backends
    seqs = ["", "MAAA", "X" * 100, "M", "MKKK"]
    for predict, model in heads(fixed):
        calls.clear()
        result = predict(seqs, model, batch_size=2)
        assert calls == [2]
        assert result["score_available"].tolist() == [False, True, False, False, True]
        assert result["decision_status"] == [
            "invalid_input",
            "forced_label",
            "abstained",
            "abstained",
            "forced_label",
        ]
        assert not result["prediction_matrix"][[0, 2, 3]].any()
        raw = predict(seqs, model, apply_thresholds=False)
        assert "prediction_matrix" not in raw
        np.testing.assert_array_equal(raw["prob_matrix"], result["prob_matrix"])
        for i, seq in enumerate(seqs):
            single = predict([seq], model)
            np.testing.assert_array_equal(
                single["prediction_matrix"][0], result["prediction_matrix"][i]
            )
        assert predict([], model)["prob_matrix"].shape == (0, 2)
        with pytest.raises(ValueError, match="batch_size"):
            predict([""], model, batch_size=0)


def test_taxonomy_then_forced_choice_and_safe_gate(fixed_backends):
    _, fixed = fixed_backends
    for predict, model in heads(fixed):
        with prediction_runtime(PredictionRuntime(taxonomy_id="9606")):
            result = predict(["MAAA", "M"], model)
        assert result["prediction_matrix"].tolist() == [[1, 0], [0, 0]]
        assert result["forced_label"].tolist() == [True, False]
        assert result["prob_matrix"][0, 1] > result["prob_matrix"][0, 0]
        # The legacy policy can still return labels on low-information inputs.
        with prediction_runtime(PredictionRuntime(decision_policy="legacy")):
            assert predict(["M"], model)["score_available"].tolist() == [True]


def test_corrected_specialist_can_be_attached_to_legacy_sequence_only_cnn():
    index = list(BROAD_FEATURE_NAMES).index("pts2_match")
    tree = {
        "feature_idx": [index, 0, 0],
        "num_threshold": [0.5, 0, 0],
        "left": [1, 0, 0],
        "right": [2, 0, 0],
        "is_leaf": [0, 1, 1],
        "value": [0, -2.0, 2.0],
        "missing_go_to_left": [0, 0, 0],
    }
    old = {
        "feature_schema": LEGACY_FEATURE_SCHEMA,
        "specialist_head": {
            "mode": "localization_specialists_v1",
            "feature_dim": len(BROAD_FEATURE_NAMES),
            "labels": [{"bias": 0, "trees": [tree]}],
        },
        "specialist_weights": [1.0],
    }
    new = copy.deepcopy(old)
    new["specialist_head"]["feature_schema"] = CURRENT_FEATURE_SCHEMA
    seqs = ["MRLQVVLGHLAAAA", "MRLQVVVHLAAAA"]
    baseline = np.zeros((2, 1))
    legacy = apply_specialists(seqs, old, baseline)
    corrected = apply_specialists(seqs, new, baseline)
    assert legacy[0, 0] < 0.5 < corrected[0, 0]
    assert corrected[1, 0] < 0.5 < legacy[1, 0]


def test_plm_receives_canonical_sequences_and_rejects_invalid_raw_scores(
    fixed_backends, monkeypatch
):
    import torch
    from cdskit import localize_multilabel_plm as plm

    _, fixed = fixed_backends
    model = heads(fixed)[1][1]
    seen = []

    def batch(encoder, sequences, torch, device):
        seen.extend(sequences)
        return torch.zeros((len(sequences), 3)), None

    monkeypatch.setattr(plm, "_batch", batch)
    predict_multilabel_plm([" ma?*"], model)
    assert seen == ["MAX"]

    class Invalid(torch.nn.Module):
        def forward(self, tokens, features=None):
            return torch.full((len(tokens), 2), float("nan"))

    model["_runtime_model_cache"]["cpu"] = (None, Invalid())
    for threshold in (False, True):
        with pytest.raises(ValueError, match="finite"):
            predict_multilabel_plm(["MAA"], model, apply_thresholds=threshold)


@pytest.mark.parametrize("batch_size", [True, 0.5, "1"])
def test_invalid_batch_size_is_rejected_even_when_all_rows_are_skipped(
    batch_size, fixed_backends
):
    _, fixed = fixed_backends
    for predict, model in heads(fixed):
        with pytest.raises(ValueError, match="batch_size"):
            predict([""], model, batch_size=batch_size)

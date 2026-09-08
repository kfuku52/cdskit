import numpy as np
import pytest

from cdskit.localize_specialists import (
    fit_specialists,
    predict_specialists,
    calibrate_blend,
    threshold_predictions,
)


def test_specialists_safe_export_and_blend(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 3))
    y = np.column_stack([x[:, 0] > 0, np.zeros(100)])
    head = fit_specialists(x, y, max_iter=5)
    expected = predict_specialists(x, head)
    path = tmp_path / "head.pt"
    torch.save(head, path)
    loaded = torch.load(path, weights_only=True)
    np.testing.assert_array_equal(expected, predict_specialists(x, loaded))
    base = np.full(y.shape, 0.5)
    weights, thresholds = calibrate_blend(base, expected, y, ["a", "b"])
    assert weights[0] > 0 and weights[1] == 0
    pred = threshold_predictions(
        expected, thresholds, ["a", "b"], ensure_one_label=False
    )
    assert pred.shape == y.shape


def test_distillation_validation_and_zero_weight_equivalence():
    pytest.importorskip("torch")
    from cdskit.localize_multilabel_cnn import fit_multilabel_cnn_classifier

    kwargs = dict(
        aa_sequences=["MAAAA", "MCCCC"],
        label_matrix=[[1, 0], [0, 1]],
        class_order=["a", "b"],
        seq_len=8,
        num_filters=2,
        embed_dim=3,
        kernel_sizes=(3,),
        epochs=1,
        device="cpu",
        seed=1,
    )
    with pytest.raises(ValueError, match="Teacher probabilities"):
        fit_multilabel_cnn_classifier(**kwargs, teacher_probabilities=[[2, 0], [0, 1]])
    control = fit_multilabel_cnn_classifier(**kwargs)
    student = fit_multilabel_cnn_classifier(
        **kwargs, teacher_probabilities=[[0.7, 0.3], [0.1, 0.9]], distillation_weight=0
    )
    for name in control["state_dict"]:
        np.testing.assert_array_equal(
            control["state_dict"][name], student["state_dict"][name]
        )


@pytest.fixture
def integrated_model():
    pytest.importorskip("torch")
    from cdskit.localize_multilabel_cnn import fit_multilabel_cnn_classifier
    from cdskit.localize_model import BROAD_FEATURE_NAMES

    head = fit_multilabel_cnn_classifier(
        ["MAAAA", "MCCCC"],
        [[1, 0], [0, 1]],
        ["a", "b"],
        seq_len=8,
        num_filters=2,
        embed_dim=3,
        kernel_sizes=(3,),
        epochs=1,
        seed=1,
        device="cpu",
    )
    head["feature_names"] = list(BROAD_FEATURE_NAMES)
    head["specialist_head"] = {
        "mode": "localization_specialists_v1",
        "feature_dim": len(BROAD_FEATURE_NAMES),
        "labels": [{"constant": 0.1}, {"constant": 0.9}],
    }
    head["specialist_weights"] = [0.25, 0.75]
    return {
        "model_type": "multilabel_cnn_v1",
        "localization_model": head,
        "feature_names": list(BROAD_FEATURE_NAMES),
        "perox_model": {"mode": "embedded_multilabel"},
        "metadata": {"fold": np.str_("4"), "nested": (np.int64(1), np.float32(0.5))},
    }


def test_integrated_safe_roundtrip_and_input_normalization(integrated_model, tmp_path):
    from cdskit.localize_model import save_localize_model, load_localize_model
    from cdskit.localize_multilabel_cnn import predict_multilabel_cnn_batch

    head = integrated_model["localization_model"]
    sequences = ["MKKLLLLAA", "mkkllllaa*", "MKK LLLLAA"]
    before = predict_multilabel_cnn_batch(sequences, head, batch_size=1)
    np.testing.assert_array_equal(before["prob_matrix"][0], before["prob_matrix"][1])
    np.testing.assert_array_equal(before["prob_matrix"][0], before["prob_matrix"][2])
    path = tmp_path / "integrated.pt"
    save_localize_model(
        integrated_model, str(path)
    )  # Includes populated runtime cache and NumPy metadata.
    loaded = load_localize_model(str(path))
    assert type(loaded["metadata"]["fold"]) is str
    assert type(loaded["metadata"]["nested"][0]) is int
    assert "_runtime_model_cache" not in loaded["localization_model"]
    after = predict_multilabel_cnn_batch(
        sequences, loaded["localization_model"], batch_size=3
    )
    np.testing.assert_allclose(before["prob_matrix"], after["prob_matrix"], atol=1e-7)
    np.testing.assert_array_equal(
        before["prediction_matrix"], after["prediction_matrix"]
    )
    assert predict_multilabel_cnn_batch([], head)["prob_matrix"].shape == (0, 2)
    with pytest.raises(Exception, match="Internal stop"):
        predict_multilabel_cnn_batch(["MA*AA"], head)


@pytest.mark.parametrize("model_type", ["multilabel_cnn_v1", "multilabel_plm_v1"])
def test_multilabel_cli_applies_requested_threads(monkeypatch, model_type):
    torch = pytest.importorskip("torch")
    from cdskit.localize import _configure_ml_threads

    calls = []
    monkeypatch.setattr(torch, "set_num_threads", lambda n: calls.append(n))
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    _configure_ml_threads({"model_type": model_type}, 2)
    assert calls == [2]


@pytest.mark.parametrize("weights", [[0.5], [0.5, np.nan], [-0.1, 0.5], [0.5, 1.1]])
def test_corrupt_integrated_weights_rejected_on_load(
    integrated_model, tmp_path, weights
):
    from cdskit.localize_model import save_localize_model, load_localize_model

    integrated_model["localization_model"]["specialist_weights"] = weights
    path = tmp_path / "bad.pt"
    save_localize_model(integrated_model, str(path))
    with pytest.raises(ValueError, match=r"specialist|Specialist"):
        load_localize_model(str(path))


def test_specialist_tree_cycle_rejected():
    from cdskit.localize_specialists import validate_specialists, predict_specialists

    tree = {
        "value": [0.0],
        "feature_idx": [0],
        "num_threshold": [0.0],
        "missing_go_to_left": [1],
        "left": [0],
        "right": [0],
        "is_leaf": [0],
    }
    head = {
        "mode": "localization_specialists_v1",
        "feature_dim": 1,
        "labels": [{"bias": 0.0, "trees": [tree]}],
    }
    with pytest.raises(ValueError, match="cycle"):
        validate_specialists(head, ["a"])
    with pytest.raises(ValueError, match="traversal"):
        predict_specialists([[0]], head)


def test_specialist_missing_values_match_sklearn():
    from cdskit.localize_specialists import _predict_one

    sklearn = pytest.importorskip("sklearn.ensemble")
    rng = np.random.default_rng(91)
    x = rng.normal(size=(180, 3)).astype(np.float32)
    x[::4, 0] = np.nan
    y = ((x[:, 1] > 0) | np.isnan(x[:, 0])).astype(int)
    exported = fit_specialists(x, y[:, None], max_iter=5)
    ref = sklearn.HistGradientBoostingClassifier(
        max_iter=5,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=1,
    ).fit(x, y)
    unseen = rng.normal(size=(50, 3)).astype(np.float32)
    unseen[::2, 0] = np.nan
    np.testing.assert_allclose(
        _predict_one(unseen, exported["labels"][0]), ref.predict_proba(unseen)[:, 1]
    )


def test_resumed_checkpoint_validation_partition_checked():
    from cdskit.localize_evaluation import assert_model_partitions, dataset_digest

    train = [{"accession": "a", "sequence": "MAAA"}]
    val = [{"accession": "b", "sequence": "MCCC"}]
    model = {
        "metadata": {
            "training_data_sha256": dataset_digest(train),
            "num_training_rows": 1,
            "validation_data_sha256": dataset_digest(val),
            "num_validation_rows": 1,
        }
    }
    assert_model_partitions(model, train, val)
    with pytest.raises(ValueError, match="validation partition"):
        assert_model_partitions(model, train, [{"accession": "c", "sequence": "MDDD"}])

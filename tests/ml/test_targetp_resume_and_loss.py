"""Regressions for interrupted nested OOF training and validation aggregation."""

import numpy as np
import pytest

from cdskit import targetp_torch as tp

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("cleavage_weight", [0.0, 0.7])
def test_validation_loss_is_independent_of_batches_and_order(weighted, cleavage_weight):
    class FixedPredictions(torch.nn.Module):
        def forward(self, x, lengths, organism, rnn_keep_prob):
            return {"type_logits": x[:, 0, :], "attention_logits": x[:, 1:, :4]}

    rng = np.random.default_rng(17)
    x = rng.normal(size=(11, 4, 5)).astype(np.float32)
    labels = np.asarray([0, 1, 0, 3, 1, 0, 4, 0, 2, 0, 0])
    cleavage = np.eye(3, dtype=np.int64)[np.arange(11) % 3]
    weight = torch.tensor([0.2, 2, 1, 3, 4]) if weighted else None
    module = FixedPredictions()
    outputs = module(torch.as_tensor(x), None, None, None)
    # Independent, whole-dataset formula: weighted classification and per-record CS.
    expected = torch.nn.functional.cross_entropy(
        outputs["type_logits"], torch.as_tensor(labels), weight=weight
    )
    for class_idx, head_idx in tp.TARGETP_SIGNAL_CLASS_TO_HEAD.items():
        mask = torch.as_tensor(labels == class_idx)
        expected += (
            cleavage_weight
            * torch.nn.functional.cross_entropy(
                outputs["attention_logits"][mask, :, head_idx],
                torch.as_tensor(cleavage[labels == class_idx].argmax(axis=1)),
                reduction="sum",
            )
            / len(labels)
        )
    for batch_size in [1, 2, 4, 11, 20]:
        order = rng.permutation(len(labels))
        result = tp._evaluate_encoded(
            torch,
            module,
            "cpu",
            x[order],
            labels[order],
            cleavage[order],
            np.full(len(labels), 3),
            np.zeros(len(labels)),
            batch_size,
            type_weight=weight,
            cleavage_loss_weight=cleavage_weight,
        )
        assert result["loss"] == pytest.approx(float(expected), rel=1e-6)


def test_nested_oof_resumes_an_interrupted_epoch_and_rejects_changed_data(
    tmp_path, monkeypatch
):
    rows = [
        {
            "sequence": "MAAAAAAA",
            "localization": "noTP",
            "organism_group": "animal",
            "fold_id": str(i % 3),
            "accession": f"seq{i}",
        }
        for i in range(12)
    ]
    arrays = tp.targetp_rows_to_torch_arrays(rows, seq_len=12)
    source = tmp_path / "training.npz"
    np.savez(source, **arrays)
    options = dict(
        targetp_npz=str(source),
        outer_folds="0",
        val_folds="1",
        device="cpu",
        seq_len=12,
        hidden_rnn=4,
        n_filters=3,
        hidden_fc=5,
        n_attention=4,
        attention_size=4,
        epochs=2,
        batch_size=3,
        learning_rate=0.001,
    )
    model_dir = tmp_path / "interrupted"
    original_save = tp.save_torch_payload

    class Interrupted(Exception):
        pass

    def save_then_interrupt(path, payload):
        original_save(path, payload)
        if not payload["training_complete"]:
            raise Interrupted()

    with monkeypatch.context() as patch:
        patch.setattr(tp, "save_torch_payload", save_then_interrupt)
        with pytest.raises(Interrupted):
            tp.run_targetp2_torch_nested_oof(model_dir=str(model_dir), **options)
    checkpoint = next(model_dir.glob("*.pt"))
    interrupted = tp.load_torch_payload(checkpoint)
    assert interrupted["latest_epoch"] == 1
    assert interrupted["training_fingerprint"]
    assert not interrupted["training_complete"]

    tp.run_targetp2_torch_nested_oof(model_dir=str(model_dir), **options)
    resumed = tp.load_torch_payload(checkpoint)
    continuous_dir = tmp_path / "continuous"
    tp.run_targetp2_torch_nested_oof(model_dir=str(continuous_dir), **options)
    continuous = tp.load_torch_payload(next(continuous_dir.glob("*.pt")))
    assert resumed["latest_epoch"] == 2
    for name, value in continuous["latest_state_dict"].items():
        torch.testing.assert_close(
            value, resumed["latest_state_dict"][name], rtol=0, atol=0
        )
    assert resumed["final_val_metrics"] == continuous["final_val_metrics"]

    # Increasing the epoch budget retains provenance while changing the data does not.
    tp.run_targetp2_torch_nested_oof(
        model_dir=str(model_dir), **{**options, "epochs": 3}
    )
    assert tp.load_torch_payload(checkpoint)["latest_epoch"] == 3
    arrays["x"][0, 0, 0] += 0.1
    np.savez(source, **arrays)
    with pytest.raises(ValueError, match="Checkpoint provenance"):
        tp.run_targetp2_torch_nested_oof(model_dir=str(model_dir), **options)

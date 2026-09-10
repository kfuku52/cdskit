import numpy as np
import pytest

from cdskit.localize_labels import masked_bce


def test_masked_bce_has_zero_unknown_gradient():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.1, 100.0], [-0.3, -100.0]], requires_grad=True)
    target = torch.tensor([[1.0, float("nan")], [0.0, float("nan")]])
    loss = masked_bce(logits, target)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, 0], target[:, 0]
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(logits.grad[:, 1], torch.zeros(2))
    all_unknown = masked_bce(logits, torch.full_like(target, float("nan")))
    assert all_unknown.item() == 0


def test_weights_count_observations_only():
    from cdskit.localize_multilabel_cnn import _class_pos_weight

    np.testing.assert_array_equal(
        _class_pos_weight([[1, 0], [0, 1], [np.nan, 0]]), [1, 2]
    )


def test_cnn_unknown_teacher_predictions_do_not_change_student():
    from cdskit.localize_multilabel_cnn import fit_multilabel_cnn_classifier

    options = dict(
        seq_len=8,
        embed_dim=2,
        num_filters=2,
        kernel_sizes=[3],
        dropout=0,
        epochs=1,
        batch_size=2,
        device="cpu",
        seed=5,
    )
    y = np.array([[1, np.nan], [0, 1]], dtype=float)

    def run(unknown):
        return fit_multilabel_cnn_classifier(
            ["MAAA", "MCCC"],
            y,
            ["a", "b"],
            teacher_probabilities=np.array([[0.8, unknown], [0.2, 0.8]]),
            **options,
        )

    first, second = run(0.0), run(1.0)
    for key in first["state_dict"]:
        np.testing.assert_array_equal(
            first["state_dict"][key], second["state_dict"][key]
        )
    assert first["observed_counts"] == [2, 1]
    assert first["distillation_scope"] == "observed_cells"


def test_cnn_rejects_unobserved_validation():
    from cdskit.localize_multilabel_cnn import fit_multilabel_cnn_classifier

    with pytest.raises(ValueError, match="no observed"):
        fit_multilabel_cnn_classifier(
            ["MAAA"],
            [[1]],
            ["a"],
            validation_sequences=["MCCC"],
            validation_labels=[[np.nan]],
            epochs=1,
            device="cpu",
        )

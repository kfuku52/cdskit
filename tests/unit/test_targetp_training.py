import pytest

from cdskit.targetp_training import (
    merge_targetp_training_config,
    parse_targetp_training_options,
    should_reduce_learning_rate,
    targetp_metric_is_better,
)


def test_resume_architecture_overrides_explicit_option_when_state_requires_it():
    config = merge_targetp_training_config(
        defaults={"rnn_impl": "default", "epochs": 2},
        overrides={"rnn_impl": "requested", "epochs": None},
        resume_payload={"config": {"rnn_impl": "saved"}},
        resume_state_impl="state-detected",
    )

    assert config == {"rnn_impl": "state-detected", "epochs": 2}


def test_saved_architecture_is_used_without_explicit_override():
    config = merge_targetp_training_config(
        defaults={"rnn_impl": "default"},
        overrides={},
        resume_payload={"config": {"rnn_impl": "saved"}},
        resume_state_impl="",
    )

    assert config["rnn_impl"] == "saved"


def test_training_options_are_validated_and_normalized():
    options = parse_targetp_training_options(
        {
            "batch_size": "8",
            "epochs": 3,
            "balanced_batch": "YES",
            "selection_metric": "VAL_MACRO_F1",
            "selection_threshold_grid": [0.2, "0.5"],
            "resume_state": "BEST",
            "resume_learning_rate": "0.001",
            "resume_reset_optimizer": "true",
        },
        default_threshold_grid=[1.0],
    )

    assert options.batch_size == 8
    assert options.balanced_batch is True
    assert options.selection_metric == "val_macro_f1"
    assert options.selection_threshold_grid == [0.2, 0.5]
    assert options.resume_state == "best"
    assert options.resume_learning_rate == 0.001
    assert options.resume_reset_optimizer is True


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {"batch_size": 1, "epochs": 1, "selection_metric": "unknown"},
            "selection_metric",
        ),
        (
            {
                "batch_size": 1,
                "epochs": 1,
                "selection_metric": "val_loss",
                "resume_state": "unknown",
            },
            "resume_state",
        ),
    ],
)
def test_invalid_training_options_raise_value_error(config, message):
    with pytest.raises(ValueError, match=message):
        parse_targetp_training_options(config, default_threshold_grid=[1.0])


def test_metric_selection_and_scheduler_boundaries():
    assert targetp_metric_is_better(
        row={"val_loss": 0.4},
        best_metrics={"val_loss": 0.5},
        selection_metric="val_loss",
    )
    assert not targetp_metric_is_better(
        row={"val_macro_f1": 0.4},
        best_metrics={"val_macro_f1": 0.5},
        selection_metric="val_macro_f1",
    )
    assert should_reduce_learning_rate(
        epoch=12, lr_reference_epoch=8, patience_epochs=4
    )
    assert not should_reduce_learning_rate(
        epoch=10, lr_reference_epoch=6, patience_epochs=4
    )

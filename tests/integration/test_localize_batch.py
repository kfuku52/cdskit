import numpy as np
import pytest

from cdskit.localize_batch import predict_localization_batch
from cdskit.localize_learn import fit_localization_model
from cdskit.localize_model import (
    LOCALIZATION_CLASSES,
    extract_localize_features,
    predict_localization_and_peroxisome,
)
from cdskit.localize_runtime import (
    PredictionRuntime,
    prediction_runtime,
    current_prediction_runtime,
)


@pytest.mark.parametrize("strategy", ["single_stage", "two_stage", "two_stage_ctp_ltp"])
def test_batch_predictions_preserve_organism_rules_and_stage_details(strategy):
    rng = np.random.default_rng(27)
    sequences = [
        "M" + "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 80)) for _ in range(20)
    ]
    features = np.asarray([extract_localize_features(seq)[0] for seq in sequences])
    model = {
        "model_type": "nearest_centroid_v1",
        "localization_model": fit_localization_model(
            features,
            sequences,
            list(LOCALIZATION_CLASSES) * 4,
            "nearest_centroid",
            {},
            localize_strategy=strategy,
        ),
        "perox_model": {"mode": "constant", "yes_probability": 0.25},
    }
    groups = ["plant", "animal", "fungi", ""] * 5
    expected = [
        predict_localization_and_peroxisome(seq, model, group)
        for seq, group in zip(sequences, groups, strict=True)
    ]
    with prediction_runtime(PredictionRuntime(batch_size=7)):
        actual = predict_localization_batch(sequences, model, groups, features)
    assert current_prediction_runtime().batch_size == 512
    for single, batch in zip(expected, actual, strict=True):
        assert batch["predicted_class"] == single["predicted_class"]
        assert batch["class_probabilities"] == pytest.approx(
            single["class_probabilities"], abs=1e-12
        )
        assert batch["perox_probability_yes"] == single["perox_probability_yes"]
        if strategy == "two_stage_ctp_ltp":
            single_details = single["two_stage_ctp_ltp_details"]
            for key, value in batch["two_stage_ctp_ltp_details"].items():
                assert value == pytest.approx(single_details[key])


def test_prediction_runtime_is_restored_after_failure():
    previous = current_prediction_runtime()
    with pytest.raises(ValueError, match="counts must match"):
        with prediction_runtime(PredictionRuntime(device="mps", offline=True)):
            predict_localization_batch(["MAAA"], {}, [])
    assert current_prediction_runtime() is previous

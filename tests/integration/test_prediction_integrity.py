"""A numerical failure must not become a successful biological prediction."""

import json

import numpy as np
import pytest

from cdskit.cli import main
from cdskit.localize_batch import predict_localization_batch
from cdskit.localize_model import (
    FEATURE_NAMES,
    LOCALIZATION_CLASSES,
    load_localize_model,
    predict_localization_and_peroxisome,
    predict_perox,
    predict_perox_batch,
)


def overflow_model():
    width = len(FEATURE_NAMES)
    return {
        "model_type": "nearest_centroid_v1",
        "feature_names": list(FEATURE_NAMES),
        "localization_model": {
            "mode": "centroid",
            "decision_policy": "safe-v1",
            "class_order": list(LOCALIZATION_CLASSES),
            "mean": [0.0] * width,
            "std": [1.0] * width,
            "centroids": [[1e200] * width for _ in LOCALIZATION_CLASSES],
            "log_priors": [0.0] * len(LOCALIZATION_CLASSES),
        },
        "perox_model": {"mode": "constant", "yes_probability": 0.0},
    }


def test_overflow_fails_scalar_batch_and_cli_without_replacing_report(tmp_path, capsys):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(overflow_model(), allow_nan=False))
    model = load_localize_model(path)
    sequence = "MALWMRLLPLL"
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite"):
            predict_localization_and_peroxisome(sequence, model)
        with pytest.raises(ValueError, match="finite"):
            predict_localization_batch([sequence], model)
        source, output = tmp_path / "input.fa", tmp_path / "report.tsv"
        source.write_text(">s\n" + sequence + "\n")
        output.write_text("original report")
        assert (
            main(
                [
                    "localize",
                    "--seq_file",
                    str(source),
                    "--seq_type",
                    "protein",
                    "--model",
                    str(path),
                    "--report",
                    str(output),
                ]
            )
            == 1
        )
    assert "finite" in capsys.readouterr().err
    assert output.read_text() == "original report"


class InvalidBinaryClassifier:
    classes_ = np.asarray([0, 1])

    def __init__(self, value):
        self.value = value

    def predict_proba(self, features):
        return np.tile([0.5, self.value], (len(features), 1))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
@pytest.mark.parametrize("mode", ["constant", "sklearn_binary"])
def test_perox_rejects_invalid_scores_in_scalar_and_batch(value, mode):
    head = {
        "mode": mode,
        "yes_probability": value,
        "classifier": InvalidBinaryClassifier(value),
    }
    features = np.zeros(len(FEATURE_NAMES))
    with pytest.raises(ValueError, match="finite"):
        predict_perox(features, head)
    with pytest.raises(ValueError, match="finite"):
        predict_perox_batch(features[None, :], head)

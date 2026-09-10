import numpy as np
import pytest

from cdskit.cli import psr
from cdskit.localize_learn import localize_learn_main, fit_localization_model
from cdskit.localize_model import (
    BROAD_FEATURE_NAMES,
    FEATURE_NAMES,
    fit_multilabel_centroid_classifier,
)
from cdskit.deeploc_benchmark import _predict_model_on_rows


@pytest.mark.parametrize("classes", [["SP", "noTP"], ["SP", "SP"]])
def test_incomplete_final_model_is_rejected_before_writing(tmp_path, classes):
    training = tmp_path / "training.tsv"
    training.write_text(
        "sequence\tlocalization\tperoxisome\n"
        + "\n".join(
            f"{seq}\t{label}\tno"
            for seq, label in zip(["MAAAA", "MCCCC"], classes, strict=True)
        )
        + "\n"
    )
    model, report = tmp_path / "model.json", tmp_path / "report.tsv"
    model.write_text("preserve existing model")
    report.write_text("preserve existing report")
    args = psr.parse_args(
        [
            "localize-learn",
            "--training_tsv",
            str(training),
            "--seq_type",
            "protein",
            "--label_mode",
            "explicit",
            "--model_out",
            str(model),
            "--report",
            str(report),
        ]
    )
    with pytest.raises(ValueError, match="missing training classes"):
        localize_learn_main(args)
    assert model.read_text() == "preserve existing model"
    assert report.read_text() == "preserve existing report"
    # Partial classes still work inside cross validation.
    fitted = fit_localization_model(
        np.zeros((2, len(FEATURE_NAMES))),
        ["MAAAA", "MCCCC"],
        classes,
        "nearest_centroid",
        {},
    )
    assert set(fitted["class_order"]) == set(classes)


def test_centroid_benchmark_uses_sequence_abstention_contract():
    head = fit_multilabel_centroid_classifier(
        np.zeros((2, len(BROAD_FEATURE_NAMES))),
        [[1], [0]],
        ["nucleus"],
        tune_thresholds=False,
    )
    head["decision_policy"] = "safe-v1"
    result = _predict_model_on_rows(
        {"model_type": "multilabel_centroid_v1", "localization_model": head},
        [{"sequence": s} for s in ["MAAAA", "", "X", "M"]],
    )
    assert result["score_available"].tolist() == [True, False, False, False]
    assert result["decision_status"][1:] == ["invalid_input", "abstained", "abstained"]
    np.testing.assert_array_equal(result["prob_matrix"][1:], np.zeros((3, 1)))

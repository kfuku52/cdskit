"""Synthetic review checks; no network, model downloads, or production data writes.

Run from the checkout with .venvs/full-3.12-cpu/bin/python.
Homology search alone is stubbed: these tests exercise evaluation bookkeeping,
not the biological independence of synthetic sequences.
"""

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from cdskit import localize_frozen_evaluation as frozen
from cdskit.localize_model import save_localize_model
from cdskit.localize_multilabel_cnn import (
    fit_multilabel_cnn_classifier,
    predict_multilabel_cnn_batch,
)


def training_round_trip(root):
    from cdskit.cli import psr
    from cdskit.localize import localize_main
    from cdskit.localize_learn import localize_learn_main
    from cdskit.localize_model import load_localize_model

    training = root / "targeting.tsv"
    training.write_text(
        "sequence\tlocalization\tperoxisome\nMAAAA\tSP\tno\nMCCCC\tnoTP\tno\n"
    )
    model = root / "targeting.json"
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
            str(root / "training-report.tsv"),
        ]
    )
    try:
        localize_learn_main(args)
    except ValueError as error:
        assert "missing training classes" in str(error)
        assert not model.exists()
        print("incomplete final model rejected before writing:", str(error))
        return
    print(
        "successful training class order:",
        load_localize_model(str(model))["localization_model"]["class_order"],
    )
    query = root / "query.fa"
    query.write_text(">q\nMAAAA\n")
    args = psr.parse_args(
        [
            "localize",
            "--seq_file",
            str(query),
            "--seq_type",
            "protein",
            "--model",
            str(model),
            "--report",
            str(root / "prediction.tsv"),
        ]
    )
    try:
        localize_main(args)
    except ValueError as error:
        print("trained model rejected:", str(error))


def main():
    torch.set_num_threads(1)
    head = fit_multilabel_cnn_classifier(
        ["AAAA", "CCCC"],
        [[1], [0]],
        ["nucleus"],
        seq_len=4,
        embed_dim=1,
        num_filters=1,
        kernel_sizes=[1],
        epochs=1,
        device="cpu",
        sequence_layout="windows",
        mask_padding=False,
    )
    # Valid exported weights chosen to expose bias from empty batch windows.
    state = head["state_dict"]
    state["embedding.weight"].fill_(-1)
    state["embedding.weight"][0].zero_()
    state["convs.0.weight"].fill_(1)
    state["convs.0.bias"].fill_(1)
    state["classifier.weight"].fill_(1)
    state["classifier.bias"].zero_()
    head["class_thresholds"] = {"nucleus": 0.6}
    head["ensure_one_label"] = False
    for sequences in (["AAAA"], ["AAAA", "CCCCCCCC"]):
        result = predict_multilabel_cnn_batch(sequences, head)
        print(
            "batch dependence:",
            sequences,
            result["prob_matrix"][0].tolist(),
            result["prediction_matrix"][0].tolist(),
        )

    head["decision_policy"] = "safe-v1"
    with tempfile.TemporaryDirectory(prefix="cdskit-review-") as temporary:
        root = Path(temporary)
        training_round_trip(root)
        rows = []
        for i, (split, sequence, positive) in enumerate(
            [
                ("train", "AAAA", True),
                ("validation", "CCCC", False),
                ("test", "X", True),
                ("test", "M", False),
            ]
        ):
            rows.append(
                dict(
                    accession=str(i),
                    sequence=sequence,
                    split=split,
                    cluster_id=str(i),
                    localization_labels="nucleus" if positive else "",
                    negative_labels="" if positive else "nucleus",
                    label_evidence=json.dumps(
                        [
                            dict(
                                label="nucleus",
                                state="positive" if positive else "negative",
                                evidence_type="experimental",
                                source="synthetic",
                                source_version="1",
                                reference="synthetic",
                            )
                        ]
                    ),
                )
            )
        data = root / "data.tsv"
        with data.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        config = root / "config.json"
        config.write_text(
            json.dumps(
                dict(
                    schema_version=2,
                    data=dict(path=str(data)),
                    labels=["nucleus"],
                )
            )
        )
        protocol = root / "protocol.json"
        protocol.write_text(
            json.dumps(
                dict(
                    population="synthetic",
                    annotation_protocol="synthetic",
                    selection_history="synthetic",
                    primary_metric="macro_f1",
                    test_used_for_selection=False,
                    comparison=dict(reference="control", candidate="student"),
                )
            )
        )
        model = root / "model.pt"
        save_localize_model(
            dict(
                model_type="multilabel_cnn_v1",
                localization_model=head,
                feature_names=[],
                perox_model={"mode": "embedded_multilabel"},
            ),
            str(model),
        )
        models = dict(control=model, student=model)
        direct = predict_multilabel_cnn_batch(["X", "M"], head)
        print("actual score availability:", direct["score_available"].tolist())
        with patch.object(frozen, "audit_homology_partitions", return_value=[]):
            manifest = frozen.freeze_evaluation(
                config, models, protocol, root / "scored"
            )
            interrupted = frozen.freeze_evaluation(
                config, models, protocol, root / "interrupted"
            )
        report = json.loads(frozen.evaluate_frozen(manifest).read_text())
        scores = report["models"]["student"]
        print(
            "probability metrics after excluding unscored rows:",
            {
                key: scores[key]
                for key in ["brier_score", "micro_average_precision", "observed_count"]
            },
        )
        with np.load(root / "scored" / "student.npz") as saved:
            print("saved availability:", "score_available" in saved.files)

        original = np.savez_compressed
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("synthetic output failure")
            return original(*args, **kwargs)

        with patch.object(frozen.np, "savez_compressed", side_effect=fail_second):
            try:
                frozen.evaluate_frozen(interrupted)
            except OSError as error:
                print("injected failure:", str(error))
        print("partial output:", sorted(p.name for p in interrupted.parent.iterdir()))
        print("retry succeeded:", frozen.evaluate_frozen(interrupted).is_file())


if __name__ == "__main__":
    main()

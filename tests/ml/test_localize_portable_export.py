"""Portable export must change runtime policy without changing predictions."""

from copy import deepcopy

import pytest

from scripts.export_localize_plm import portable_model


def checkpoint():
    return {
        "model_type": "multilabel_plm_v1",
        "localization_model": {
            "encoder": {
                "model_name": "owner/encoder",
                "revision": "a" * 40,
                "cache_dir": "/training/cache",
                "local_files_only": True,
            },
            "encoder_identity": {
                "model": "owner/encoder",
                "revision": "a" * 40,
                "window": 1000,
                "overlap": 128,
                "format": "esm_residues_v1",
            },
            "state_dict": {"weight": [1.0, 2.0]},
            "class_thresholds": {"nucleus": 0.4},
        },
    }


def test_export_preserves_all_but_runtime_settings():
    original = checkpoint()
    before = deepcopy(original)
    exported = portable_model(original)
    assert original == before
    expected = deepcopy(before)
    expected["localization_model"]["encoder"].update(
        cache_dir="", local_files_only=False
    )
    assert exported == expected


@pytest.mark.parametrize("name", ["/missing/encoder", "../encoder", "encoder"])
def test_export_rejects_nonportable_encoder(name):
    model = checkpoint()
    model["localization_model"]["encoder"]["model_name"] = name
    with pytest.raises(ValueError, match="remote owner/model"):
        portable_model(model)


def test_export_rejects_changed_revision():
    model = checkpoint()
    model["localization_model"]["encoder"]["revision"] = "b" * 40
    with pytest.raises(ValueError, match="identity differs"):
        portable_model(model)

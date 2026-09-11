"""Portability of trusted sklearn models without changing global imports."""

import io
import json
from pathlib import Path
import pickle
import sys

import pytest

from cdskit import localize_pickle


def test_short_cython_loss_module_is_resolved_without_global_alias():
    loss = pytest.importorskip("sklearn._loss._loss")
    before = sys.modules.get("_loss")
    result = localize_pickle.Unpickler(
        io.BytesIO(b"c_loss\nCyHalfMultinomialLoss\n.")
    ).load()
    assert result is loss.CyHalfMultinomialLoss
    assert sys.modules.get("_loss") is before


def test_unknown_short_loss_global_is_not_remapped():
    with pytest.raises((ModuleNotFoundError, AttributeError)):
        localize_pickle.Unpickler(io.BytesIO(b"c_loss\nUnknownClass\n.")).load()


def test_trusted_torch_loader_accepts_reader(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "model.pt"
    torch.save({"weights": torch.tensor([1.0, 2.0])}, path)
    result = torch.load(path, weights_only=False, pickle_module=localize_pickle)
    assert torch.equal(result["weights"], torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize("module", ["_loss", "sklearn._loss._loss"])
def test_legacy_binomial_reducer_without_generated_function(monkeypatch, module):
    loss = pytest.importorskip("sklearn._loss._loss")
    monkeypatch.delattr(loss, "__pyx_unpickle_CyHalfBinomialLoss", raising=False)
    before = set(vars(loss))
    payload = (
        f"c{module}\n__pyx_unpickle_CyHalfBinomialLoss\n"
        f"(c{module}\nCyHalfBinomialLoss\nI238750788\n)tR."
    ).encode()
    restored = localize_pickle.Unpickler(io.BytesIO(payload)).load()
    assert type(restored) is loss.CyHalfBinomialLoss
    assert set(vars(loss)) == before


@pytest.mark.parametrize("checksum,state", [(0, ()), (238750788, (1,))])
def test_legacy_binomial_reducer_rejects_unknown_state(checksum, state):
    loss = pytest.importorskip("sklearn._loss._loss")
    with pytest.raises(pickle.UnpicklingError, match="Unsupported legacy"):
        localize_pickle._restore_legacy_binomial_loss(
            loss.CyHalfBinomialLoss, checksum, state
        )


def test_legacy_binomial_reducer_rejects_other_classes():
    loss = pytest.importorskip("sklearn._loss._loss")
    with pytest.raises(pickle.UnpicklingError, match="Unsupported legacy"):
        localize_pickle._restore_legacy_binomial_loss(
            loss.CyHalfMultinomialLoss, 238750788, ()
        )


def test_existing_binomial_reconstructor_is_preserved(monkeypatch):
    loss = pytest.importorskip("sklearn._loss._loss")
    original = object()
    monkeypatch.setattr(
        loss, "__pyx_unpickle_CyHalfBinomialLoss", original, raising=False
    )
    assert (
        localize_pickle.Unpickler(io.BytesIO()).find_class(
            "_loss", "__pyx_unpickle_CyHalfBinomialLoss"
        )
        is original
    )


def test_sklearn152_classifier_retains_reference_probabilities():
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    fixture = Path(__file__).parents[1] / "fixtures" / "localize_pickle"
    reference = json.loads((fixture / "sklearn152-binomial.json").read_text())
    with (fixture / "sklearn152-binomial.pkl").open("rb") as handle:
        model = localize_pickle.Unpickler(handle).load()
    actual = model.predict_proba(np.array(reference["inputs"], dtype=float))
    np.testing.assert_allclose(actual, reference["probabilities"], rtol=0, atol=1e-14)

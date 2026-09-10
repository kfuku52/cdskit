"""Portability of trusted sklearn models without changing global imports."""

import io
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

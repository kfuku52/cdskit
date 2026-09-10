import numpy as np
import pytest
import torch

from cdskit.localize_bilstm import DEFAULT_AA_TO_IDX, PAD_INDEX
from cdskit.localize_multilabel_cnn import _build_multilabel_cnn_module, _encode_layout


@pytest.mark.parametrize("mask_padding", [False, True])
def test_empty_batch_windows_cannot_change_prediction_or_gradient(mask_padding):
    net = _build_multilabel_cnn_module(
        torch,
        torch.nn,
        len(DEFAULT_AA_TO_IDX),
        1,
        1,
        (1,),
        0.0,
        1,
        mask_padding=mask_padding,
    )
    with torch.no_grad():
        net.embedding.weight.fill_(-1)
        net.embedding.weight[PAD_INDEX].zero_()
        net.convs[0].weight.fill_(1)
        net.convs[0].bias.fill_(1)
        net.classifier.weight.fill_(1)
        net.classifier.bias.zero_()
    outputs, gradients = [], []
    for sequences in (["AAAA"], ["AAAA", "CCCCCCCC"], ["CCCCCCCC", "AAAA"]):
        tokens = torch.as_tensor(
            _encode_layout(sequences, 4, DEFAULT_AA_TO_IDX, "windows")
        )
        net.zero_grad(set_to_none=True)
        value = net(tokens)[sequences.index("AAAA")]
        outputs.append(value.detach().numpy())
        value.sum().backward()
        gradients.append({name: p.grad.clone() for name, p in net.named_parameters()})
    for value, gradient in zip(outputs[1:], gradients[1:], strict=True):
        np.testing.assert_array_equal(value, outputs[0])
        for name in gradient:
            torch.testing.assert_close(gradient[name], gradients[0][name])
    # An entirely empty input remains finite, including the legacy policy path.
    assert torch.isfinite(net(torch.zeros((2, 3, 4), dtype=torch.long))).all()

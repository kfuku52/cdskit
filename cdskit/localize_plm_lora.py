"""Query/value LoRA for ESM residue heads, without caching fitted embeddings."""

import math
from typing import Any

from cdskit.localize_bilstm import require_torch
from cdskit.localize_multilabel_plm import ResidueEncoder


def _install_lora(model, rank, alpha):
    torch, nn = require_torch()
    # A private CPU stream preserves head/dropout RNGs on CPU, CUDA and MPS.
    adapter_rng = torch.Generator().manual_seed(torch.initial_seed() ^ 0x4C4F5241)

    class LoRALinear(nn.Module):  # type: ignore[name-defined]
        def __init__(self, base):
            super().__init__()
            self.base = base
            initial = base.weight.new_empty((rank, base.in_features), device="cpu")
            nn.init.kaiming_uniform_(initial, a=math.sqrt(5), generator=adapter_rng)
            self.lora_a = nn.Parameter(initial.to(base.weight.device))
            self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))

        def forward(self, value):
            update = nn.functional.linear(
                nn.functional.linear(value, self.lora_a), self.lora_b
            )
            return self.base(value) + update * (alpha / rank)

    names = [
        name
        for name, module in model.named_modules()
        if name.endswith(("attention.self.query", "attention.self.value"))
        and isinstance(module, nn.Linear)
    ]
    if not names:
        raise ValueError("ESM encoder has no query/value projections for LoRA.")
    for name in names:
        parent, child = name.rsplit(".", 1)
        module = model.get_submodule(parent)
        setattr(module, child, LoRALinear(getattr(module, child)))


class LoRAResidueEncoder(ResidueEncoder):
    def __init__(self, config, device="cpu", adapter_state=None):
        super().__init__(config, device)
        rank = config.get("lora_rank")
        alpha = config.get("lora_alpha", 8.0)
        if type(rank) is not int or rank < 1:
            raise ValueError("lora_rank must be a positive integer.")
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("lora_alpha must be finite and positive.")
        if config.get("cache_dir"):
            raise ValueError("LoRA embeddings cannot use a shared residue cache.")
        if type(config.get("lora_checkpointing", True)) is not bool:
            raise ValueError("lora_checkpointing must be boolean.")
        self.adapter_state = adapter_state

    def _load(self):
        if self.model is not None:
            return
        try:
            super()._load()
            assert self.model is not None
            _install_lora(
                self.model, self.config["lora_rank"], self.config.get("lora_alpha", 8)
            )
            if self.adapter_state is not None:
                self.load_adapter(self.adapter_state)
            self.model.eval()
        except BaseException:
            self.model = self.tokenizer = None
            raise

    def adapter_parameters(self):
        self._load()
        return {
            name: value
            for name, value in self.model.named_parameters()
            if name.endswith((".lora_a", ".lora_b"))
        }

    def adapter_weights(self):
        return {
            name: value.detach().cpu().clone()
            for name, value in self.adapter_parameters().items()
        }

    def load_adapter(self, state):
        torch, _ = require_torch()
        parameters = self.adapter_parameters()
        if not isinstance(state, dict) or set(state) != set(parameters):
            raise ValueError("LoRA adapter parameters do not match the encoder.")
        prepared = {}
        for name, parameter in parameters.items():
            value = state[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameter.shape
                or not value.is_floating_point()
            ):
                raise ValueError("Invalid LoRA adapter tensor: " + name)
            value = value.detach().to(device="cpu", dtype=parameter.dtype)
            if not torch.isfinite(value).all():
                raise ValueError("Invalid LoRA adapter tensor: " + name)
            prepared[name] = value
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(prepared[name])

    def _hidden(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

    def _window_tensor(self, part):
        torch, _ = require_torch()
        tokens = self.tokenizer(
            part, return_tensors="pt", return_special_tokens_mask=True
        )
        special = tokens.pop("special_tokens_mask")[0].bool().to(self.device)
        if int((~special).sum()) != len(part):
            raise ValueError("Tokenizer must produce exactly one token per amino acid.")
        tokens = {key: tensor.to(self.device) for key, tensor in tokens.items()}
        if torch.is_grad_enabled() and self.config.get("lora_checkpointing", True):
            from torch.utils.checkpoint import checkpoint

            hidden = checkpoint(self._hidden, **tokens, use_reentrant=False)
        else:
            hidden = self._hidden(**tokens)
        return hidden[0, ~special]

    def _sequence_tensor(self, sequence):
        if len(sequence) <= self.window:
            return self._window_tensor(sequence)
        torch, nn = require_torch()
        stride = self.window - self.overlap
        if len(sequence) <= self.window + stride:
            left = self._window_tensor(sequence[: self.window])
            right = self._window_tensor(sequence[stride:])
            return torch.cat(
                (
                    left[:stride],
                    (left[stride:] + right[: self.overlap]) / 2,
                    right[self.overlap :],
                )
            )
        pending = torch.zeros((0, self.model.config.hidden_size), device=self.device)
        counts = torch.zeros((0, 1), device=self.device)
        pieces = []
        for start in range(0, len(sequence), stride):
            part = sequence[start : start + self.window]
            hidden = self._window_tensor(part)
            # Only the overlapping tail can receive more contributions. Finalize
            # the prefix without padding each window to the entire protein.
            extra = len(part) - len(pending)
            pending = nn.functional.pad(pending, (0, 0, 0, extra)) + hidden
            counts = nn.functional.pad(counts, (0, 0, 0, extra)) + 1
            last = start + len(part) == len(sequence)
            size = len(part) if last else stride
            pieces.append(pending[:size] / counts[:size])
            if last:
                break
            pending, counts = pending[size:], counts[size:]
        return torch.cat(pieces)

    def tensor_batch(self, sequences):
        self._load()
        torch, nn = require_torch()
        sequences = [str(sequence).upper() for sequence in sequences]
        if not sequences or any(not sequence for sequence in sequences):
            raise ValueError("LoRA requires nonempty protein sequences.")
        values = [self._sequence_tensor(sequence) for sequence in sequences]
        length = max(map(len, sequences))
        embeddings = torch.stack(
            [
                nn.functional.pad(value, (0, 0, 0, length - len(value)))
                for value in values
            ]
        )
        lengths = torch.tensor(
            [len(sequence) for sequence in sequences], device=self.device
        )
        mask = torch.arange(length, device=self.device)[None, :] < lengths[:, None]
        return embeddings, mask

    def encode_many(self, sequences, batch_size=8):
        torch, _ = require_torch()
        if batch_size < 1:
            raise ValueError("PLM embedding batch_size must be positive.")
        values: list[Any] = []
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                batch = sequences[start : start + batch_size]
                embeddings, _ = self.tensor_batch(batch)
                values.extend(
                    embeddings[i, : len(sequence)].cpu().numpy().copy()
                    for i, sequence in enumerate(batch)
                )
        return values

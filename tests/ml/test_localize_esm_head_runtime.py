from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from cdskit import localize_esm_head as esm_head

torch = pytest.importorskip("torch")


class _FakeTokenizer:
    def __call__(
        self,
        sequences,
        return_tensors,
        padding,
        truncation,
        max_length,
    ):
        assert return_tensors == "pt"
        assert padding is True
        assert truncation is True
        width = min(int(max_length), max(2, max(map(len, sequences), default=0) + 2))
        input_ids = torch.zeros((len(sequences), width), dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row_index, sequence in enumerate(sequences):
            encoded = [1]
            encoded.extend((ord(char) % 17) + 2 for char in sequence)
            encoded.append(1)
            encoded = encoded[:width]
            input_ids[row_index, : len(encoded)] = torch.as_tensor(encoded)
            attention_mask[row_index, : len(encoded)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class _FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)

    def forward(self, input_ids, attention_mask):
        values = input_ids.to(torch.float32)
        mask = attention_mask.to(torch.float32)
        hidden = torch.stack(
            [
                values / 20.0,
                mask,
                (values % 3.0) / 3.0,
                torch.ones_like(values),
            ],
            dim=-1,
        )
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeAutoTokenizer:
    calls: ClassVar[list] = []

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.calls.append((source, kwargs))
        return _FakeTokenizer()


class _FakeAutoModel:
    calls: ClassVar[list] = []

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.calls.append((source, kwargs))
        return _FakeEncoder()


@pytest.fixture
def fake_transformers(monkeypatch):
    _FakeAutoTokenizer.calls.clear()
    _FakeAutoModel.calls.clear()
    monkeypatch.setattr(
        esm_head,
        "require_transformers",
        lambda: (torch, torch.nn, _FakeAutoTokenizer, _FakeAutoModel),
    )
    return _FakeAutoTokenizer, _FakeAutoModel


def test_fit_predict_and_cache_esm_head_without_network(fake_transformers):
    tokenizer_factory, model_factory = fake_transformers
    model = esm_head.fit_esm_head_classifier(
        aa_sequences=["MAAA", "MCCC", "MDDD", "MEEE"],
        labels=["cytoplasm", "nucleus", "cytoplasm", "nucleus"],
        class_order=["cytoplasm", "nucleus"],
        model_name="example/pinned-esm",
        model_local_dir="",
        max_len=12,
        pooling="mean",
        epochs=2,
        batch_size=2,
        learning_rate=0.01,
        weight_decay=0.0,
        seed=7,
        use_class_weight=True,
        device="cpu",
        model_revision="0123456789abcdef",
    )

    assert model["model_revision"] == "0123456789abcdef"
    assert model["model_source_type"] == "huggingface"
    assert model["head_in_dim"] == 4
    assert set(model["head_state_dict"]) == {"weight", "bias"}
    assert tokenizer_factory.calls[0] == (
        "example/pinned-esm",
        {
            "local_files_only": False,
            "trust_remote_code": False,
            "revision": "0123456789abcdef",
        },
    )
    assert model_factory.calls[0][1]["use_safetensors"] is True
    assert model_factory.calls[0][1]["revision"] == "0123456789abcdef"

    model["_runtime_offline"] = True
    probabilities = esm_head.predict_esm_head_batch(
        aa_sequences=["MAAA", "MEEE"],
        localization_model=model,
        device="cpu",
        batch_size=1,
    )
    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), atol=1e-6)
    assert tokenizer_factory.calls[-1][1]["local_files_only"] is True
    assert model_factory.calls[-1][1]["trust_remote_code"] is False
    assert model_factory.calls[-1][1]["use_safetensors"] is True

    runtime_call_count = len(model_factory.calls)
    predicted_class, class_probs = esm_head.predict_esm_head(
        aa_seq="MAAA",
        localization_model=model,
        device="cpu",
    )
    assert predicted_class in model["class_order"]
    assert set(class_probs) == set(model["class_order"])
    assert sum(class_probs.values()) == pytest.approx(1.0)
    assert len(model_factory.calls) == runtime_call_count

    empty = esm_head.predict_esm_head_batch(
        aa_sequences=[],
        localization_model=model,
        device="cpu",
    )
    assert empty.shape == (0, 2)


def test_esm_head_rejects_unpinned_remote_artifact(fake_transformers):
    model = {
        "class_order": ["cytoplasm", "nucleus"],
        "model_name": "example/unpinned-esm",
        "model_revision": "",
        "model_local_dir": "",
        "head_in_dim": 4,
        "head_state_dict": {},
        "max_len": 12,
    }

    with pytest.raises(ValueError, match="revision is missing"):
        esm_head.predict_esm_head_batch(
            aa_sequences=["MAAA"],
            localization_model=model,
        )


def test_esm_head_pooling_and_device_validation(fake_transformers, monkeypatch):
    hidden = torch.arange(24, dtype=torch.float32).reshape((2, 3, 4))
    mask = torch.as_tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)

    cls = esm_head._pool_last_hidden(hidden, mask, "cls", torch)
    mean = esm_head._pool_last_hidden(hidden, mask, "mean", torch)
    torch.testing.assert_close(cls, hidden[:, 0, :])
    torch.testing.assert_close(mean[0], hidden[0, :2, :].mean(dim=0))
    torch.testing.assert_close(mean[1], hidden[1, 0, :])
    with pytest.raises(ValueError, match="Unsupported --esm_pooling"):
        esm_head._pool_last_hidden(hidden, mask, "median", torch)

    assert esm_head.resolve_torch_device("cpu") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA device was requested"):
        esm_head.resolve_torch_device("cuda")
    with pytest.raises(ValueError, match="Unsupported --dl_device"):
        esm_head.resolve_torch_device("quantum")


def test_fit_esm_head_accepts_local_model_without_revision(fake_transformers):
    tokenizer_factory, model_factory = fake_transformers
    model = esm_head.fit_esm_head_classifier(
        aa_sequences=["MAAA", "MCCC"],
        labels=["cytoplasm", "nucleus"],
        class_order=["cytoplasm", "nucleus"],
        model_name="ignored/remote-name",
        model_local_dir="/models/local-esm",
        max_len=8,
        pooling="cls",
        epochs=1,
        batch_size=2,
        learning_rate=0.01,
        weight_decay=0.0,
        seed=3,
        use_class_weight=False,
        device="cpu",
        model_revision="",
    )

    assert model["model_source_type"] == "local"
    assert model["model_revision"] == ""
    assert tokenizer_factory.calls[0][0] == "/models/local-esm"
    assert tokenizer_factory.calls[0][1]["local_files_only"] is True
    assert "revision" not in tokenizer_factory.calls[0][1]
    assert model_factory.calls[0][1]["use_safetensors"] is True


def _fit_fake_model(**overrides):
    options = dict(
        aa_sequences=["MAAA", "MCCC", "MDDD", "MEEE"],
        labels=["noTP", "SP", "noTP", "SP"],
        class_order=["noTP", "SP", "mTP", "cTP", "lTP"],
        model_name="example/pinned-esm",
        model_local_dir="",
        max_len=12,
        pooling="mean",
        epochs=4,
        batch_size=2,
        learning_rate=0.01,
        weight_decay=0,
        seed=7,
        use_class_weight=True,
        device="cpu",
        model_revision="0123456789abcdef",
    )
    return esm_head.fit_esm_head_classifier(**{**options, **overrides})


def test_frozen_embeddings_are_reused_without_sharing_heads(
    fake_transformers, monkeypatch
):
    from cdskit.esm_embeddings import ESMEmbeddingCache

    calls = []
    forward = _FakeEncoder.forward

    def counted(self, input_ids, attention_mask):
        calls.append(len(input_ids))
        return forward(self, input_ids, attention_mask)

    monkeypatch.setattr(_FakeEncoder, "forward", counted)
    cache = ESMEmbeddingCache()
    cold = _fit_fake_model(embedding_cache=cache)
    assert sum(calls) == 4  # Not 4 sequences * 4 epochs.
    warm = _fit_fake_model(embedding_cache=cache)
    assert sum(calls) == 4
    for key in cold["head_state_dict"]:
        torch.testing.assert_close(
            cold["head_state_dict"][key], warm["head_state_dict"][key], rtol=0, atol=0
        )
    changed_labels = _fit_fake_model(
        embedding_cache=cache, labels=["SP", "noTP", "SP", "noTP"]
    )
    assert sum(calls) == 4
    assert not torch.equal(
        cold["head_state_dict"]["weight"], changed_labels["head_state_dict"]["weight"]
    )
    _fit_fake_model(embedding_cache=cache, max_len=8)
    assert sum(calls) == 8
    _fit_fake_model(embedding_cache=cache, pooling="cls")
    assert sum(calls) == 12
    _fit_fake_model(embedding_cache=cache, model_revision="different")
    assert sum(calls) == 16


@pytest.mark.parametrize(
    "nested", ["single", "blend", "two_stage", "two_stage_ctp_ltp"]
)
def test_cli_offline_applies_to_every_nested_encoder(
    fake_transformers, monkeypatch, tmp_path, nested
):
    import copy
    from cdskit import localize
    from cdskit.localize_runtime import offline_requested

    local_model = _fit_fake_model()
    if nested in {"two_stage", "two_stage_ctp_ltp"}:
        local_model = {
            "class_order": local_model["class_order"],
            "strategy": nested,
            "stage1_model": _fit_fake_model(
                class_order=["noTP", "TP"], labels=["noTP", "TP"] * 2
            ),
            "stage2_model": _fit_fake_model(
                class_order=["SP", "mTP", "cTP", "lTP"], labels=["SP", "mTP"] * 2
            ),
            "stage3_model": _fit_fake_model(
                class_order=["cTP", "lTP"], labels=["cTP", "lTP"] * 2
            ),
        }
    model = {
        "model_type": "esm_head_v1",
        "localization_model": local_model,
        "perox_model": {"mode": "constant", "yes_probability": 0.0},
    }
    if nested == "blend":
        model = {
            "model_type": "targetp_blend_v1",
            "localization_model": {
                "class_order": local_model["class_order"],
                "base_models": [copy.deepcopy(model), copy.deepcopy(model)],
            },
            "perox_model": model["perox_model"],
        }
    tokenizer, encoder = fake_transformers
    tokenizer.calls.clear()
    encoder.calls.clear()
    source = tmp_path / "in.fa"
    source.write_text(">test\nMAAA\n")
    model_path = tmp_path / "model.pt"
    model_path.touch()
    monkeypatch.setattr(
        localize, "resolve_localize_model_path", lambda **kwargs: str(model_path)
    )
    monkeypatch.setattr(localize, "load_localize_model", lambda **kwargs: model)
    localize.localize_main(
        SimpleNamespace(
            seqfile=str(source),
            inseqformat="fasta",
            seqtype="protein",
            codontable=1,
            model=str(model_path),
            model_download=False,
            allow_unsafe_model=False,
            include_features=False,
            organism_group="plant",
            threads=1,
            report=str(tmp_path / "report.tsv"),
        )
    )
    expected_calls = {"single": 1, "blend": 2, "two_stage": 2, "two_stage_ctp_ltp": 3}[
        nested
    ]
    assert len(encoder.calls) == expected_calls
    assert all(
        kwargs["local_files_only"] for _, kwargs in tokenizer.calls + encoder.calls
    )
    assert not offline_requested()


def test_cross_validation_batches_predictions_on_requested_device(
    fake_transformers, monkeypatch
):
    from cdskit.localize_learn import evaluate_cross_validation
    from cdskit.localize_model import extract_localize_features

    sequences = ["MAAA", "MCCC", "MDDD", "MEEE"] * 2
    features = np.asarray([extract_localize_features(seq)[0] for seq in sequences])
    calls = []

    def predict(aa_sequences, localization_model, device, batch_size):
        calls.append((len(aa_sequences), device))
        probabilities = np.zeros(
            (len(aa_sequences), len(localization_model["class_order"]))
        )
        probabilities[:, 0] = 1.0
        return probabilities

    monkeypatch.setattr(esm_head, "predict_esm_head_batch", predict)
    metrics = evaluate_cross_validation(
        features,
        sequences,
        ["noTP", "SP"] * 4,
        ["no"] * 8,
        2,
        7,
        "esm_head",
        dict(
            esm_model_name="fake/esm",
            esm_model_local_dir="",
            esm_model_revision="fixed",
            esm_max_len=12,
            esm_pooling="mean",
            epochs=2,
            batch_size=4,
            learning_rate=0.01,
            weight_decay=0,
            seed=1,
            use_class_weight=True,
            device="cpu",
        ),
        "mps",
        organism_groups=["plant", "animal"] * 4,
    )
    assert len(metrics["oof_rows"]) == 8
    assert sum(count for count, _ in calls) == 8
    assert len(calls) < 8
    assert all(device == "mps" for _, device in calls)

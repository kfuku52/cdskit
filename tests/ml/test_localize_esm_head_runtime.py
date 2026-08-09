from types import SimpleNamespace

import numpy as np
import pytest

from cdskit import localize_esm_head as esm_head

torch = pytest.importorskip('torch')


class _FakeTokenizer:
    def __call__(
        self,
        sequences,
        return_tensors,
        padding,
        truncation,
        max_length,
    ):
        assert return_tensors == 'pt'
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
            input_ids[row_index, :len(encoded)] = torch.as_tensor(encoded)
            attention_mask[row_index, :len(encoded)] = 1
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
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
    calls = []

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.calls.append((source, kwargs))
        return _FakeTokenizer()


class _FakeAutoModel:
    calls = []

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
        'require_transformers',
        lambda: (torch, torch.nn, _FakeAutoTokenizer, _FakeAutoModel),
    )
    return _FakeAutoTokenizer, _FakeAutoModel


def test_fit_predict_and_cache_esm_head_without_network(fake_transformers):
    tokenizer_factory, model_factory = fake_transformers
    model = esm_head.fit_esm_head_classifier(
        aa_sequences=['MAAA', 'MCCC', 'MDDD', 'MEEE'],
        labels=['cytoplasm', 'nucleus', 'cytoplasm', 'nucleus'],
        class_order=['cytoplasm', 'nucleus'],
        model_name='example/pinned-esm',
        model_local_dir='',
        max_len=12,
        pooling='mean',
        epochs=2,
        batch_size=2,
        learning_rate=0.01,
        weight_decay=0.0,
        seed=7,
        use_class_weight=True,
        device='cpu',
        model_revision='0123456789abcdef',
    )

    assert model['model_revision'] == '0123456789abcdef'
    assert model['model_source_type'] == 'huggingface'
    assert model['head_in_dim'] == 4
    assert set(model['head_state_dict']) == {'weight', 'bias'}
    assert tokenizer_factory.calls[0] == (
        'example/pinned-esm',
        {
            'local_files_only': False,
            'trust_remote_code': False,
            'revision': '0123456789abcdef',
        },
    )
    assert model_factory.calls[0][1]['use_safetensors'] is True
    assert model_factory.calls[0][1]['revision'] == '0123456789abcdef'

    model['_runtime_offline'] = True
    probabilities = esm_head.predict_esm_head_batch(
        aa_sequences=['MAAA', 'MEEE'],
        localization_model=model,
        device='cpu',
        batch_size=1,
    )
    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), atol=1e-6)
    assert tokenizer_factory.calls[-1][1]['local_files_only'] is True
    assert model_factory.calls[-1][1]['trust_remote_code'] is False
    assert model_factory.calls[-1][1]['use_safetensors'] is True

    runtime_call_count = len(model_factory.calls)
    predicted_class, class_probs = esm_head.predict_esm_head(
        aa_seq='MAAA',
        localization_model=model,
        device='cpu',
    )
    assert predicted_class in model['class_order']
    assert set(class_probs) == set(model['class_order'])
    assert sum(class_probs.values()) == pytest.approx(1.0)
    assert len(model_factory.calls) == runtime_call_count

    empty = esm_head.predict_esm_head_batch(
        aa_sequences=[],
        localization_model=model,
        device='cpu',
    )
    assert empty.shape == (0, 2)


def test_esm_head_rejects_unpinned_remote_artifact(fake_transformers):
    model = {
        'class_order': ['cytoplasm', 'nucleus'],
        'model_name': 'example/unpinned-esm',
        'model_revision': '',
        'model_local_dir': '',
        'head_in_dim': 4,
        'head_state_dict': {},
        'max_len': 12,
    }

    with pytest.raises(ValueError, match='revision is missing'):
        esm_head.predict_esm_head_batch(
            aa_sequences=['MAAA'],
            localization_model=model,
        )


def test_esm_head_pooling_and_device_validation(fake_transformers, monkeypatch):
    hidden = torch.arange(24, dtype=torch.float32).reshape((2, 3, 4))
    mask = torch.as_tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)

    cls = esm_head._pool_last_hidden(hidden, mask, 'cls', torch)
    mean = esm_head._pool_last_hidden(hidden, mask, 'mean', torch)
    torch.testing.assert_close(cls, hidden[:, 0, :])
    torch.testing.assert_close(mean[0], hidden[0, :2, :].mean(dim=0))
    torch.testing.assert_close(mean[1], hidden[1, 0, :])
    with pytest.raises(ValueError, match='Unsupported --esm_pooling'):
        esm_head._pool_last_hidden(hidden, mask, 'median', torch)

    assert esm_head.resolve_torch_device('cpu') == 'cpu'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='CUDA device was requested'):
        esm_head.resolve_torch_device('cuda')
    with pytest.raises(ValueError, match='Unsupported --dl_device'):
        esm_head.resolve_torch_device('quantum')


def test_fit_esm_head_accepts_local_model_without_revision(fake_transformers):
    tokenizer_factory, model_factory = fake_transformers
    model = esm_head.fit_esm_head_classifier(
        aa_sequences=['MAAA', 'MCCC'],
        labels=['cytoplasm', 'nucleus'],
        class_order=['cytoplasm', 'nucleus'],
        model_name='ignored/remote-name',
        model_local_dir='/models/local-esm',
        max_len=8,
        pooling='cls',
        epochs=1,
        batch_size=2,
        learning_rate=0.01,
        weight_decay=0.0,
        seed=3,
        use_class_weight=False,
        device='cpu',
        model_revision='',
    )

    assert model['model_source_type'] == 'local'
    assert model['model_revision'] == ''
    assert tokenizer_factory.calls[0][0] == '/models/local-esm'
    assert tokenizer_factory.calls[0][1]['local_files_only'] is True
    assert 'revision' not in tokenizer_factory.calls[0][1]
    assert model_factory.calls[0][1]['use_safetensors'] is True

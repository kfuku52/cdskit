import copy

import numpy as np
import pytest

from cdskit.localize_evaluation import (
    assert_disjoint,
    average_precision,
    cluster_bootstrap,
    grouped_folds,
)
from cdskit.deeploc_benchmark import (
    _fold_ids_from_rows,
    compute_multilabel_metrics,
    fit_deeploc_multilabel_model,
)


def test_partition_overlap_and_missing_folds_fail():
    rows = [
        {"accession": "a", "sequence": "MAAA", "fold_id": "0"},
        {"accession": "b", "sequence": "MAAA", "fold_id": "1"},
    ]
    with pytest.raises(ValueError, match="overlap"):
        _fold_ids_from_rows(rows)
    with pytest.raises(ValueError, match="Complete fold"):
        _fold_ids_from_rows([{"sequence": "MAAA"}])
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(
            [dict(rows[0], cluster_id="c")],
            [dict(rows[1], sequence="MBBB", cluster_id="c")],
        )


def test_grouped_folds_and_bootstrap():
    rows = [{}, {}, {}, {}, {}]
    groups = ["a", "a", "b", "c", "d"]
    folds = grouped_folds(rows, groups, n_folds=3)
    assert folds[0] == folds[1]
    np.testing.assert_array_equal(folds, grouped_folds(rows, groups, n_folds=3))
    y = np.array([[1], [1], [0], [0], [1]])
    result = cluster_bootstrap(
        y, y, groups, ["x"], compute_multilabel_metrics, iterations=20
    )
    assert result["cluster_count"] == 4
    assert result["percentile_95"]["micro_f1"][1] == 1.0


def test_average_precision_handles_ties_and_unsupported_labels():
    from sklearn.metrics import average_precision_score

    y, p = [1, 0, 1, 0], [0.9, 0.9, 0.1, 0.1]
    assert average_precision(y, p) == pytest.approx(average_precision_score(y, p))
    assert average_precision([0, 0], [0.1, 0.2]) is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_plm_batch_preserves_padding_dtype_and_independent_storage(device):
    torch = pytest.importorskip("torch")
    from cdskit.localize_multilabel_plm import _batch

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    # Noncontiguous source embeddings exercise the copy, not just allocation.
    source = np.arange(24, dtype=np.float32).reshape(6, 4)
    values = {"short": source[:2, ::2], "long": source[:, ::2]}

    class Encoder:
        def encode(self, sequence):
            return values[sequence]

    batch, mask = _batch(Encoder(), ["short", "long"], torch, device)
    expected = np.zeros((2, 6, 2), dtype=np.float32)
    expected[0, :2] = values["short"]
    expected[1] = values["long"]
    np.testing.assert_array_equal(batch.cpu().numpy(), expected)
    assert batch.dtype == torch.float32
    assert batch.device.type == device
    assert batch.is_contiguous()
    assert mask.dtype == torch.bool
    assert mask.tolist() == [[True, True, False, False, False, False], [True] * 6]

    # A later batch or an encoder-cache mutation must not overwrite this batch.
    next_batch, _ = _batch(Encoder(), ["long", "short"], torch, device)
    next_batch.fill_(-1)
    source.fill(-2)
    np.testing.assert_array_equal(batch.cpu().numpy(), expected)


def test_calibration_partition_excluded_from_fitting(monkeypatch):
    import cdskit.deeploc_benchmark as benchmark

    seen = []
    original = benchmark.fit_multilabel_centroid_classifier

    def capture(**kwargs):
        seen.append(kwargs["label_matrix"].shape[0])
        return original(**kwargs)

    monkeypatch.setattr(benchmark, "fit_multilabel_centroid_classifier", capture)
    rows = [
        dict(
            accession=str(i),
            sequence="M" + aa * 5,
            fold_id=str(i // 2),
            localization_labels="nucleus" if i % 2 else "cytoplasm",
        )
        for i, aa in enumerate("ACDEFG")
    ]
    model = fit_deeploc_multilabel_model(
        rows, ["nucleus", "cytoplasm"], "localization_labels", "localization"
    )
    assert seen == [4]
    assert model["metadata"]["num_validation_rows"] == 2
    assert model["metadata"]["validation_fold"] == "2"
    assert model["metadata"]["threshold_source"] == "validation_partition"


def test_no_validation_uses_fixed_thresholds():
    rows = [dict(sequence="MAAAA", localization_labels="nucleus")]
    model = fit_deeploc_multilabel_model(
        rows, ["nucleus"], "localization_labels", "localization"
    )
    assert model["localization_model"]["class_thresholds"] == {"nucleus": 0.5}


def test_cnn_padding_invariance_and_no_synthetic_join():
    torch = pytest.importorskip("torch")
    from cdskit.localize_multilabel_cnn import (
        _build_multilabel_cnn_module,
        _encode_layout,
    )
    from cdskit.localize_bilstm import DEFAULT_AA_TO_IDX

    net = _build_multilabel_cnn_module(
        torch, torch.nn, len(DEFAULT_AA_TO_IDX), 4, 3, (3, 4), 0.0, 2, mask_padding=True
    ).eval()
    seq = torch.tensor([[1, 2, 3, 4]])
    padded = torch.nn.functional.pad(seq, (0, 12))
    with torch.no_grad():
        torch.testing.assert_close(net(seq), net(padded))
        assert torch.isfinite(net(torch.zeros_like(padded))).all()
    tokens = _encode_layout(["AAAAACCCCC"], 8, DEFAULT_AA_TO_IDX, "separate_termini")
    assert tokens.shape == (1, 2, 4)
    assert np.all(tokens[0, 0] == DEFAULT_AA_TO_IDX["A"])
    assert np.all(tokens[0, 1] == DEFAULT_AA_TO_IDX["C"])
    windows = _encode_layout(["AAAAAKCCCCCAAAAA"], 8, DEFAULT_AA_TO_IDX, "windows")
    assert DEFAULT_AA_TO_IDX["K"] in windows


def test_cnn_legacy_load_and_new_roundtrip():
    pytest.importorskip("torch")
    from cdskit.localize_multilabel_cnn import (
        fit_multilabel_cnn_classifier,
        predict_multilabel_cnn_batch,
    )

    model = fit_multilabel_cnn_classifier(
        ["MAAAA", "MCCCC"],
        [[1, 0], [0, 1]],
        ["a", "b"],
        seq_len=8,
        epochs=1,
        embed_dim=4,
        num_filters=3,
        kernel_sizes=(3,),
        device="cpu",
        sequence_layout="legacy",
        mask_padding=False,
    )
    expected = predict_multilabel_cnn_batch(["MAAAA"], model)["prob_matrix"]
    legacy = copy.deepcopy(model)
    legacy.pop("_runtime_model_cache", None)
    legacy.pop("sequence_layout")
    legacy.pop("mask_padding")
    np.testing.assert_array_equal(
        expected, predict_multilabel_cnn_batch(["MAAAA"], legacy)["prob_matrix"]
    )


@pytest.mark.parametrize("pooling", ["mean", "terminal_attention"])
def test_plm_pooling_padding_invariance(pooling):
    torch = pytest.importorskip("torch")
    from cdskit.localize_multilabel_plm import _build_head

    head = _build_head(torch.nn, 4, 2, pooling, hidden=8, dropout=0).eval()
    x = torch.randn(1, 7, 4)
    mask = torch.ones(1, 7, dtype=torch.bool)
    with torch.no_grad():
        expected = head(x, mask)
        padded = torch.cat([x, torch.randn(1, 11, 4)], dim=1)
        padded_mask = torch.cat([mask, torch.zeros(1, 11, dtype=torch.bool)], dim=1)
        torch.testing.assert_close(expected, head(padded, padded_mask))


@pytest.fixture
def tiny_esm(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    path = tmp_path / "encoder"
    path.mkdir()
    vocab = ["<cls>", "<pad>", "<eos>", "<unk>", *"LAGVSERTIDPKQNFYMHWCXBUZO", "<mask>"]
    (path / "vocab.txt").write_text("\n".join(vocab))
    tokenizer = transformers.EsmTokenizer(str(path / "vocab.txt"))
    tokenizer.save_pretrained(path)
    config = transformers.EsmConfig(
        vocab_size=len(vocab),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=64,
        pad_token_id=1,
        mask_token_id=len(vocab) - 1,
    )
    transformers.EsmModel(config).save_pretrained(path, safe_serialization=True)
    return path


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_lora_preserves_initial_embeddings_and_only_updates_adapters(
    tiny_esm, checkpointing, device
):
    import torch
    from cdskit.localize_multilabel_plm import ResidueEncoder, _batch
    from cdskit.localize_plm_lora import LoRAResidueEncoder

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = {"model_name": str(tiny_esm), "window": 8, "overlap": 3}
    sequences = ["MACDEFGHIKLMNPQ", "MCCCC"]
    frozen = ResidueEncoder(config, device)
    expected, expected_mask = _batch(frozen, sequences, torch, device)
    encoder = LoRAResidueEncoder(
        dict(config, lora_rank=2, lora_alpha=4, lora_checkpointing=checkpointing),
        device,
    )
    rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    actual, mask = _batch(encoder, sequences, torch, device)
    assert torch.equal(rng, torch.random.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(mask, expected_mask)
    base = {
        name: value.detach().clone()
        for name, value in encoder.model.named_parameters()
        if not value.requires_grad
    }
    before = encoder.adapter_weights()
    optimizer = torch.optim.AdamW(encoder.adapter_parameters().values(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        batch, _ = encoder.tensor_batch(sequences)
        batch[:, :, 0].square().sum().backward()
        optimizer.step()
    after = encoder.adapter_weights()
    assert any(
        not torch.equal(before[key], after[key])
        for key in before
        if key.endswith("lora_a")
    )
    assert any(
        not torch.equal(before[key], after[key])
        for key in before
        if key.endswith("lora_b")
    )
    for name, value in encoder.model.named_parameters():
        if name in base:
            assert value.grad is None
            torch.testing.assert_close(value, base[name], rtol=0, atol=0)
    assert not encoder._memory_cache._entries


def test_lora_fitted_checkpoint_roundtrip_and_adapter_validation(tiny_esm, tmp_path):
    from cdskit.localize_multilabel_plm import (
        fit_multilabel_plm,
        predict_multilabel_plm,
    )
    from cdskit.localize_model import load_localize_model, save_localize_model
    from cdskit.localize_plm_lora import LoRAResidueEncoder

    config = {"model_name": str(tiny_esm), "window": 8, "overlap": 2, "lora_rank": 2}
    head = fit_multilabel_plm(
        ["MAAAA", "MCCCC", "MDDD", "MEEEE"],
        [[1, 0], [0, 1], [1, np.nan], [np.nan, 1]],
        ["nucleus", "cytoplasm"],
        config,
        validation_sequences=["MAACC", "MCCDD"],
        validation_y=[[1, 0], [0, 1]],
        epochs=2,
        batch_size=2,
        device="cpu",
    )
    head["ensure_one_label"] = False
    assert any(
        value.count_nonzero()
        for key, value in head["lora_state_dict"].items()
        if key.endswith("lora_b")
    )
    model = {
        "model_type": "multilabel_plm_lora_v1",
        "feature_names": [],
        "perox_model": {"mode": "embedded_multilabel"},
        "localization_model": head,
    }
    expected = predict_multilabel_plm(["MAACC"], head)["prob_matrix"]
    path = tmp_path / "lora.pt"
    save_localize_model(model, str(path))
    with pytest.raises(ValueError, match="require model_type"):
        save_localize_model(
            dict(model, model_type="multilabel_plm_v1"), str(tmp_path / "wrong-type.pt")
        )
    missing = {
        **model,
        "localization_model": {
            key: value for key, value in head.items() if key != "lora_state_dict"
        },
    }
    with pytest.raises(ValueError, match="configuration and weights"):
        save_localize_model(missing, str(tmp_path / "missing-adapter.pt"))
    restored_model = load_localize_model(str(path))
    restored = restored_model["localization_model"]
    np.testing.assert_array_equal(
        expected, predict_multilabel_plm(["MAACC"], restored)["prob_matrix"]
    )
    from cdskit.localize_model import predict_multilabel_localization
    from cdskit.localize import _predict_records_batched_if_supported
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    single = predict_multilabel_localization("MAACC", restored_model)
    assert single["class_probabilities"]["nucleus"] == expected[0, 0]
    rows = _predict_records_batched_if_supported(
        [SeqRecord(Seq("MAACC"), id="a")], 1, "protein", restored_model, False
    )
    assert rows[0]["p_nucleus"] == expected[0, 0]
    restored.pop("_runtime_model_cache")
    restored.pop("lora_state_dict")
    with pytest.raises(ValueError, match="missing its adapter"):
        predict_multilabel_plm(["MAACC"], restored)
    encoder = LoRAResidueEncoder(config)
    with pytest.raises(ValueError, match="do not match"):
        encoder.load_adapter({})
    with pytest.raises(ValueError, match="shared residue cache"):
        LoRAResidueEncoder(dict(config, cache_dir=str(tmp_path)))


def test_lora_mps_initialization_preserves_head_dropout_rng(tiny_esm):
    import torch
    from cdskit.localize_plm_lora import LoRAResidueEncoder

    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    encoder = LoRAResidueEncoder(
        {"model_name": str(tiny_esm), "lora_rank": 2, "window": 8, "overlap": 2}, "mps"
    )
    before = torch.mps.get_rng_state().clone()
    encoder._load()
    assert torch.equal(before, torch.mps.get_rng_state())


def test_lora_rejects_bad_checkpoint_on_every_prediction_attempt(tiny_esm):
    import torch
    from cdskit.localize_multilabel_plm import _build_head, predict_multilabel_plm
    from cdskit.localize_plm_lora import LoRAResidueEncoder

    config = {"model_name": str(tiny_esm), "window": 8, "overlap": 2, "lora_rank": 2}
    encoder = LoRAResidueEncoder(config)
    state = encoder.adapter_weights()
    state[next(reversed(state))] = torch.empty(0)
    head = {
        "encoder": config,
        "encoder_identity": encoder.identity,
        "embedding_dim": 8,
        "class_order": ["nucleus"],
        "pooling": "mean",
        "state_dict": _build_head(torch.nn, 8, 1, "mean").state_dict(),
        "lora_state_dict": state,
    }
    for _ in range(2):
        with pytest.raises(ValueError, match="Invalid LoRA adapter tensor"):
            predict_multilabel_plm(["MAAAA"], head)


def test_lora_validates_all_weights_before_changing_parameters(tiny_esm):
    import torch
    from cdskit.localize_plm_lora import LoRAResidueEncoder

    encoder = LoRAResidueEncoder(
        {"model_name": str(tiny_esm), "window": 8, "overlap": 2, "lora_rank": 2}
    )
    before = encoder.adapter_weights()
    invalid = {key: torch.zeros_like(value) for key, value in before.items()}
    invalid[next(reversed(invalid))].fill_(float("nan"))
    with pytest.raises(ValueError, match="Invalid LoRA adapter tensor"):
        encoder.load_adapter(invalid)
    for key, value in encoder.adapter_weights().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("overlap", [0, 3, 7])
@pytest.mark.parametrize("length", [3, 8, 9, 15])
def test_lora_window_prefix_finalization_preserves_outputs_and_gradients(
    overlap, length
):
    import torch
    from scripts.benchmark_localize_lora import fixture, padded_reference

    encoder, scale = fixture(torch, 8)
    encoder.window, encoder.overlap = 8, overlap
    sequence = "M" + "A" * (length - 1)
    expected = padded_reference(encoder, sequence, torch)
    expected.square().mean().backward()
    expected_gradient = scale.grad.clone()
    scale.grad = None
    actual = encoder._sequence_tensor(sequence)
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(scale.grad, expected_gradient, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "pooling,loss,metric",
    [
        ("label_attention", "bce", "bce"),
        ("terminal_attention", "weighted_bce", "macro_ap"),
    ],
)
def test_plm_window_cache_and_model_roundtrip(
    tiny_esm, tmp_path, pooling, loss, metric
):
    from cdskit.localize_multilabel_plm import (
        ResidueEncoder,
        fit_multilabel_plm,
        predict_multilabel_plm,
    )
    from cdskit.localize_model import (
        save_localize_model,
        load_localize_model,
        predict_multilabel_localization,
    )

    config = dict(
        model_name=str(tiny_esm),
        revision="",
        window=8,
        overlap=3,
        cache_dir=str(tmp_path / "cache"),
        pooling=pooling,
    )
    seqs = ["MACKWDEFGHILMNPQRS", "MLLLKAA"]
    encoder = ResidueEncoder(config)
    encoded = encoder.encode(seqs[0])
    assert encoded.shape == (len(seqs[0]), 8)
    encoder.model = None
    np.testing.assert_array_equal(encoder.encode(seqs[0]), encoded)
    assert encoder.model is None  # Cached embeddings bypass model execution.
    head = fit_multilabel_plm(
        seqs,
        [[1, 0], [0, 1]],
        ["nucleus", "cytoplasm"],
        config,
        validation_sequences=["MYYYY"],
        validation_y=[[1, 0]],
        loss=loss,
        selection_metric=metric,
        epochs=2,
        batch_size=2,
        device="cpu",
    )
    assert head["training_loss"] == loss
    assert head["selection_metric"] == metric
    expected = predict_multilabel_plm(seqs, head)["prob_matrix"]
    path = str(tmp_path / "model.pt")
    save_localize_model(
        dict(
            model_type="multilabel_plm_v1",
            localization_model=head,
            feature_names=[],
            perox_model={},
        ),
        path,
    )
    loaded = load_localize_model(path)
    np.testing.assert_array_equal(
        expected,
        predict_multilabel_plm(seqs, loaded["localization_model"])["prob_matrix"],
    )
    assert "class_probabilities" in predict_multilabel_localization(seqs[0], loaded)
    assert len(list((tmp_path / "cache").glob("*.npy"))) == 3
    from types import SimpleNamespace
    from cdskit.localize import localize_main

    fasta = tmp_path / "input.faa"
    report = tmp_path / "predictions.tsv"
    fasta.write_text(">a\n" + seqs[0] + "\n>b\n" + seqs[1] + "\n")
    localize_main(
        SimpleNamespace(
            seqfile=str(fasta),
            inseqformat="fasta",
            codontable=999,
            model=path,
            report=str(report),
            include_features=False,
            seqtype="protein",
            threads=1,
            organism_group="non_plant",
        )
    )
    assert len(report.read_text().splitlines()) == 3
    assert "p_nucleus" in report.read_text()


def test_remote_encoder_rejects_mutable_revision():
    pytest.importorskip("torch")
    from cdskit.localize_multilabel_plm import ResidueEncoder

    with pytest.raises(ValueError, match="immutable"):
        ResidueEncoder(dict(model_name="facebook/esm2_t6_8M_UR50D", revision="main"))


def test_mmseqs_failure_does_not_fall_back_to_random(monkeypatch):
    import cdskit.perox_benchmark as perox
    from cdskit.deeploc_benchmark import _prepare_partitions

    monkeypatch.setattr(
        perox,
        "mmseqs_cluster_assignments",
        lambda *a, **k: (["0", "1"], {"status": "unavailable"}),
    )
    with pytest.raises(ValueError, match="Homology partitioning failed"):
        _prepare_partitions(
            [{"sequence": "MAAA"}, {"sequence": "MCCC"}], {"split_method": "mmseqs"}
        )


def test_development_run_writes_oof_without_external_file(tmp_path):
    import csv
    from cdskit.deeploc_benchmark import run_deeploc21_benchmark

    path = tmp_path / "deeploc21_localization_train_validation.tsv"
    with path.open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["sequence", "accession", "fold_id", "localization_labels"],
            delimiter="\t",
        )
        writer.writeheader()
        for i, aa in enumerate("ACDEFG"):
            writer.writerow(
                dict(
                    sequence="M" + aa * 7,
                    accession=str(i),
                    fold_id=str(i // 2),
                    localization_labels="nucleus" if i % 2 else "cytoplasm",
                )
            )
    output = str(tmp_path / "report.json")
    result = run_deeploc21_benchmark(
        str(tmp_path), comparison_json=output, evaluate_external=False
    )
    assert "independent_test" not in result
    assert result["cross_validation"]["cluster_bootstrap"]["status"] == "not_computed"
    with open(output + ".oof.tsv") as stream:
        predictions = list(csv.DictReader(stream, delimiter="\t"))
    assert len(predictions) == 6
    assert "p_nucleus" in predictions[0] and "true_nucleus" in predictions[0]
    for fold in result["cross_validation"]["folds"]:
        assert fold["n_train"] == 2
        assert fold["n_validation"] == 2
        assert fold["threshold_source"] == "validation_partition"


def test_encoder_loading_preserves_training_rng(tiny_esm):
    import torch
    from transformers import EsmModel
    from cdskit.localize_multilabel_plm import ResidueEncoder

    # Public masked-language-model checkpoints omit the unused sequence pooler.
    EsmModel.from_pretrained(tiny_esm, add_pooling_layer=False).save_pretrained(
        tiny_esm
    )
    encoder = ResidueEncoder(
        {"model_name": str(tiny_esm), "window": 16, "overlap": 4}, "cpu"
    )
    torch.manual_seed(11)
    before = torch.get_rng_state().clone()
    encoder.encode("MAAA")
    assert torch.equal(before, torch.get_rng_state())


def test_teacher_weights_do_not_depend_on_embedding_cache(tiny_esm, tmp_path):
    import torch
    from transformers import EsmModel
    from cdskit.localize_multilabel_plm import fit_multilabel_plm

    EsmModel.from_pretrained(tiny_esm, add_pooling_layer=False).save_pretrained(
        tiny_esm
    )
    config = {
        "model_name": str(tiny_esm),
        "cache_dir": str(tmp_path / "residues"),
        "window": 16,
        "overlap": 4,
        "pooling": "light_attention",
    }
    parameters = dict(
        sequences=["MAAA", "MCCC"],
        y=[[1, 0], [0, 1]],
        labels=["nucleus", "cytoplasm"],
        config=config,
        validation_sequences=["MDDD", "MEEE"],
        validation_y=[[1, 0], [0, 1]],
        epochs=2,
        batch_size=2,
        seed=11,
        device="cpu",
    )
    cold = fit_multilabel_plm(**parameters)
    warm = fit_multilabel_plm(**parameters)
    assert cold["training_history"] == warm["training_history"]
    for key in cold["state_dict"]:
        assert torch.equal(cold["state_dict"][key], warm["state_dict"][key])


@pytest.mark.parametrize("name", ["bce", "weighted_bce", "asl"])
def test_plm_training_losses_are_finite_at_extreme_logits(name):
    torch = pytest.importorskip("torch")
    from cdskit.localize_multilabel_plm import _training_loss

    y = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.float32)
    logits = torch.tensor(
        [[-1000.0, 1000.0, -1000.0], [1000.0, -1000.0, 1000.0]], requires_grad=True
    )
    loss_fn = _training_loss(torch, torch.nn, name, y, "cpu")
    value = loss_fn(logits, torch.tensor(y))
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0] < 0
    assert logits.grad[1, 1] < 0
    if name == "bce":
        torch.testing.assert_close(
            value,
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits, torch.tensor(y)
            ),
        )
    if name == "weighted_bce":
        torch.testing.assert_close(loss_fn.pos_weight, torch.ones(3))


def test_plm_terminal_branch_uses_actual_termini():
    torch = pytest.importorskip("torch")
    from cdskit.localize_multilabel_plm import _build_head

    head = _build_head(torch.nn, 1, 1, "terminal_attention", hidden=2, dropout=0).eval()
    with torch.no_grad():
        head.output.weight.zero_()
        head.output.bias.zero_()
        head.output.weight[0, -2:] = torch.tensor([1.0, 2.0])
    x = torch.zeros(1, 100, 1)
    x[0, 0, 0], x[0, 79, 0] = 3, 5
    x[0, 40, 0], x[0, 80:, 0] = 999, 999
    mask = torch.arange(100)[None, :] < 80
    torch.testing.assert_close(head(x, mask), torch.tensor([[13 / 32]]))
    short = torch.tensor([[[2.0], [4.0], [999.0]]])
    torch.testing.assert_close(
        head(short, torch.tensor([[True, True, False]])), torch.tensor([[9.0]])
    )


def test_plm_weighted_loss_uses_training_frequency_and_cap():
    torch = pytest.importorskip("torch")
    from cdskit.localize_multilabel_plm import _training_loss

    y = np.zeros((401, 3), dtype=np.float32)
    y[0, 0] = 1
    y[:80, 1] = 1
    y[:, 2] = 1
    loss = _training_loss(torch, torch.nn, "weighted_bce", y, "cpu")
    np.testing.assert_allclose(loss.pos_weight.numpy(), [10, np.sqrt(321 / 80), 1])


def test_residue_memory_cache_is_bounded_and_cannot_be_poisoned(tiny_esm):
    from cdskit.localize_multilabel_plm import ResidueEncoder

    encoder = ResidueEncoder(
        {
            "model_name": str(tiny_esm),
            "window": 8,
            "overlap": 3,
            "memory_cache_bytes": 1400,
        }
    )
    expected = encoder.encode("MACDEFGHI")
    expected_copy = expected.copy()
    expected[:] = -99
    np.testing.assert_array_equal(encoder.encode("macdefghi"), expected_copy)
    encoder.encode("MLLLL")
    assert encoder._memory_cache.size_bytes <= 1400
    assert encoder._memory_cache.get(encoder._cache_identity, "MACDEFGHI") is None
    calls = []
    encoder.model.register_forward_hook(lambda *args: calls.append(1))
    encoder.encode("MACDEFGHI")
    assert calls  # Eviction recomputes instead of serving stale/fitted features.


def test_residue_window_batching_matches_scalar_without_new_forward_on_hits(tiny_esm):
    from cdskit.localize_multilabel_plm import ResidueEncoder

    config = {"model_name": str(tiny_esm), "window": 8, "overlap": 3}
    sequences = ["MACDEFGHILMN", "MLLLKAAQQRS", "MACDEFGHILMN", "MYYYY"]
    scalar = ResidueEncoder({**config, "memory_cache_bytes": 0, "window_batch_size": 1})
    scalar._load()
    scalar_calls = []
    scalar.model.register_forward_hook(lambda *args: scalar_calls.append(1))
    expected = [scalar.encode(sequence) for sequence in sequences]
    encoder = ResidueEncoder(config)
    encoder._load()
    calls = []
    encoder.model.register_forward_hook(lambda *args: calls.append(1))
    values = encoder.encode_many(sequences, batch_size=4)
    assert len(calls) < len(scalar_calls)
    for value, reference in zip(values, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=1e-6, atol=1e-6)
    forwards = len(calls)
    for value, reference in zip(encoder.encode_many(sequences), values, strict=True):
        np.testing.assert_array_equal(value, reference)
    assert len(calls) == forwards
    values[0][:] = 0
    assert values[2].any()  # Duplicate inputs own independent result buffers.

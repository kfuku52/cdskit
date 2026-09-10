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


@pytest.mark.parametrize(
    "pooling", ["mean", "light_attention", "label_attention", "terminal_attention"]
)
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


@pytest.mark.parametrize(
    "pooling,loss,metric",
    [
        ("label_attention", "bce", "bce"),
        ("terminal_attention", "weighted_bce", "macro_ap"),
        ("light_attention", "asl", "macro_ap"),
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


def test_plm_honors_scoped_offline_setting(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from cdskit.localize_multilabel_plm import ResidueEncoder
    from cdskit.localize_runtime import PredictionRuntime, prediction_runtime

    calls = []

    def tokenizer_probe(*args, **kwargs):
        calls.append(kwargs)
        raise ValueError("offline probe")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_probe)
    encoder = ResidueEncoder({"model_name": "test/esm", "revision": "a" * 40})
    with prediction_runtime(PredictionRuntime(offline=True)):
        with pytest.raises(ValueError, match="offline probe"):
            encoder._load()
    assert calls[0]["local_files_only"] is True
    assert calls[0]["trust_remote_code"] is False


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

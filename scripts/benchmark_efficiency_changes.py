"""Benchmark the October summary, streaming and frozen-feature optimizations.

Uses the fresh-process snapshot runner from benchmark_recent_changes.py. No
pretrained checkpoint is downloaded: the PLM case creates a tiny local ESM.
Example:
  .venvs/full-3.12-cpu/bin/python scripts/benchmark_efficiency_changes.py \
    --source before=/tmp/before --source after=. --repeats 5 --output /tmp/results.json
"""

import contextlib
import io
from types import SimpleNamespace

import benchmark_recent_changes as runner

CASES = (
    "stats_dna",
    "stats_aa",
    "bootstrap_full",
    "bootstrap_partial",
    "bootstrap_paired",
    "codon_concrete",
    "codon_mixed",
    "codon_dual",
    "probability",
    "rmseq",
    "record_digest",
    "dataset_digest",
    "mask_stream",
    "backtrim_duplicates",
    "plm_epochs",
    "backtrim",
)
_PREVIOUS_BUILD = runner.build_case


def build_case(name, root, model_path):
    import numpy as np
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    rng = np.random.default_rng(55)

    def common(value):
        return value

    if name == "backtrim":
        return _PREVIOUS_BUILD(name, root, model_path)
    if name.startswith("stats_"):
        from cdskit.stats import summarize_alignment, ALPHABETS

        kind = name.removeprefix("stats_")
        records = [
            SeqRecord(
                Seq("".join(rng.choice(list(ALPHABETS[kind]), size=10_000))), id=str(i)
            )
            for i in range(200)
        ]

        def run():
            return summarize_alignment(records, kind)

        workload = {"records": 200, "sites": 10_000, "type": kind}
    elif name.startswith("bootstrap_"):
        from cdskit.localize_evaluation import (
            cluster_bootstrap,
            paired_cluster_bootstrap,
        )
        from cdskit.deeploc_benchmark import compute_multilabel_metrics

        paired = name == "bootstrap_paired"
        rows, clusters, iterations = (2000, 100, 1000) if paired else (10000, 500, 200)
        target = rng.integers(2, size=(rows, 10)).astype(float)
        if name != "bootstrap_full":
            target[rng.random(target.shape) < 0.3] = np.nan
        predictions = [rng.integers(2, size=target.shape) for _ in range(2)]
        groups = [str(i % clusters) for i in range(rows)]
        labels = [str(i) for i in range(10)]
        kwargs = dict(
            groups=groups,
            labels=labels,
            metric_fn=compute_multilabel_metrics,
            iterations=iterations,
            seed=1,
        )
        run = (
            (lambda: paired_cluster_bootstrap(target, *predictions, **kwargs))
            if paired
            else (lambda: cluster_bootstrap(target, predictions[0], **kwargs))
        )
        workload = {
            "rows": rows,
            "labels": 10,
            "clusters": clusters,
            "iterations": iterations,
            "seed": 1,
        }
    elif name.startswith("codon_"):
        import cdskit.trimcodon as module

        alphabet = "ACGT" if name == "codon_concrete" else "ACGTRYSWKMBDHVNX-?."
        code = 28 if name == "codon_dual" else 1
        sequences = [
            "".join(rng.choice(list(alphabet), size=15000)) for _ in range(100)
        ]
        if hasattr(module, "summarize_codon_sites"):

            def run():
                return module.summarize_codon_sites(sequences, code)
        else:

            def run():
                return [
                    module.summarize_codon_site(sequences, site, code)
                    for site in range(5000)
                ]

        workload = {"records": 100, "codons": 5000, "code": code, "alphabet": alphabet}
    elif name == "probability":
        from cdskit.localize_probabilities import (
            normalize_class_probabilities,
            LOCALIZATION_CLASSES,
        )

        rows = [
            dict(zip(LOCALIZATION_CLASSES, row, strict=True))
            for row in rng.random((10000, 5)).tolist()
        ]
        rows.extend([{}, None, {"SP": 0.0}])

        def run():
            return [normalize_class_probabilities(row) for row in rows]

        workload = {"rows": len(rows), "classes": 5}
    elif name == "rmseq":
        from cdskit.rmseq import problematic_rate

        sequences = [Seq("acgtNN?-.A" * 10000) for _ in range(100)]

        def run():
            return [
                problematic_rate(sequence, frozenset("N?-.")) for sequence in sequences
            ]

        workload = {
            "records": 100,
            "residues_per_record": len(sequences[0]),
            "problematic_chars": "N?-.",
        }
    elif name in ("record_digest", "dataset_digest"):
        sequences = [str(i) + "ACGT" * 5000 for i in range(2000)]
        if name == "record_digest":
            from cdskit.backtrim import _record_digest

            records = [
                SeqRecord(Seq(sequence), id=str(i))
                for i, sequence in enumerate(sequences)
            ]

            def run():
                return _record_digest(records)
        else:
            from cdskit.localize_evaluation import dataset_digest

            rows = [
                {"sequence": sequence, "id": str(i), "labels": ["a", "b"]}
                for i, sequence in enumerate(sequences)
            ]

            def run():
                return dataset_digest(rows)

        del sequences
        workload = {"rows": 2000, "residues_per_row": "20000 + identifier length"}
    elif name == "mask_stream":
        from cdskit.mask import mask_main
        from cdskit.localize_pipeline_config import file_digest

        source, destination = root / "input.fa", root / "output.fa"
        sequence = "ATGNNNGCN---TAA" * 200
        with source.open("w") as handle:
            for i in range(10000):
                handle.write(f">row{i}\n{sequence}\n")
        args = SimpleNamespace(
            seqfile=str(source),
            outfile=str(destination),
            inseqformat="fasta",
            outseqformat="fasta",
            codontable=1,
            maskchar="N",
            ambiguouscodon=True,
            stopcodon=True,
            threads=1,
        )

        def run():
            return mask_main(args)

        def common(value):
            return file_digest(destination)

        workload = {
            "records": 10000,
            "residues_per_record": len(sequence),
            "named_output": True,
            "format": "fasta",
        }
    elif name == "backtrim_duplicates":
        from cdskit.backtrim import find_kept_aa_sites

        class CountingWriter(io.TextIOBase):
            def write(self, value):
                return len(value)

        def run():
            with contextlib.redirect_stderr(CountingWriter()):
                return find_kept_aa_sites(["A" * 2000] * 20, ["A" * 1000] * 20)

        workload = {
            "records": 20,
            "source_sites": 2000,
            "target_sites": 1000,
            "pattern": "identical columns",
        }
    elif name == "plm_epochs":
        import torch
        from transformers import EsmModel, EsmConfig, EsmTokenizer
        from cdskit.localize_multilabel_plm import ResidueEncoder

        checkpoint = root / "encoder"
        checkpoint.mkdir()
        vocab = [
            "<cls>",
            "<pad>",
            "<eos>",
            "<unk>",
            *"LAGVSERTIDPKQNFYMHWCXBUZO",
            "<mask>",
        ]
        (checkpoint / "vocab.txt").write_text("\n".join(vocab))
        EsmTokenizer(str(checkpoint / "vocab.txt")).save_pretrained(checkpoint)
        torch.manual_seed(5)
        model = EsmModel(
            EsmConfig(
                vocab_size=len(vocab),
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=64,
                max_position_embeddings=128,
                pad_token_id=1,
                mask_token_id=len(vocab) - 1,
            ),
            add_pooling_layer=False,
        )
        model.save_pretrained(checkpoint, safe_serialization=True)
        config = {
            "model_name": str(checkpoint),
            "window": 64,
            "overlap": 16,
            "window_batch_size": 1,
        }
        sequences = [
            "".join(rng.choice(list("MACDEFGHILKNPQRSTVWY"), size=128))
            for _ in range(12)
        ]

        def run():
            encoder = ResidueEncoder(config, "cpu")
            for _ in range(12):
                values = [encoder.encode(sequence) for sequence in sequences]
            return values

        workload = {
            "epochs": 12,
            "proteins": 12,
            "residues": 128,
            "hidden_size": 32,
            "layers": 2,
            "window": 64,
            "overlap": 16,
            "window_batch_size": 1,
            "disk_cache": False,
            "synthetic_local_checkpoint": True,
        }
    else:
        raise ValueError(name)
    return run, common, workload


if __name__ == "__main__":
    runner.main(
        cases=CASES, build_case_fn=build_case, script_path=__file__, description=__doc__
    )

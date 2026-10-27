#!/usr/bin/env python3
"""Isolate LoRA window assembly, including backward; this is not a PLM speed test."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def padded_reference(encoder, sequence, torch):
    value = torch.zeros((len(sequence), encoder.model.config.hidden_size))
    counts = torch.zeros((len(sequence), 1))
    for start in range(0, len(sequence), encoder.window - encoder.overlap):
        part = sequence[start : start + encoder.window]
        tokens = encoder.tokenizer(
            part, return_tensors="pt", return_special_tokens_mask=True
        )
        special = tokens.pop("special_tokens_mask")[0].bool()
        hidden = encoder._hidden(**tokens)[0, ~special]
        value = value + torch.nn.functional.pad(
            hidden, (0, 0, start, len(sequence) - start - len(part))
        )
        counts[start : start + len(part)] += 1
        if start + len(part) == len(sequence):
            break
    return value / counts


def fixture(torch, hidden_size):
    from cdskit.localize_plm_lora import LoRAResidueEncoder

    encoder = object.__new__(LoRAResidueEncoder)
    encoder.window, encoder.overlap, encoder.device = 1000, 128, "cpu"
    encoder.config = {"lora_checkpointing": False}
    encoder.model = SimpleNamespace(config=SimpleNamespace(hidden_size=hidden_size))
    scale = torch.tensor(0.75, requires_grad=True)
    bases = {}

    def tokenize(sequence, **kwargs):
        ids = torch.arange(len(sequence) + 2)[None, :]
        special = torch.zeros_like(ids)
        special[:, 0] = special[:, -1] = 1
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "special_tokens_mask": special,
        }

    def hidden(input_ids, attention_mask):
        length = input_ids.shape[1]
        if length not in bases:
            bases[length] = torch.sin(
                (input_ids[:, :, None] + torch.arange(hidden_size)[None, None, :]) / 300
            )
        return bases[length] * scale

    encoder.tokenizer, encoder._hidden = tokenize, hidden
    return encoder, scale


def worker(args):
    import torch

    if args.implementation == "paired":
        return paired_worker(args, torch)
    torch.set_num_threads(1)
    encoder, scale = fixture(torch, args.hidden_size)
    sequence = "M" + "A" * (args.length - 1)
    run = (
        (lambda: padded_reference(encoder, sequence, torch))
        if args.implementation == "reference"
        else (lambda: encoder._sequence_tensor(sequence))
    )
    times = []
    digest = None
    gradient = None
    for i in range(args.warmup + args.repeats):
        scale.grad = None
        started = time.perf_counter()
        value = run()
        value.square().mean().backward()
        elapsed = time.perf_counter() - started
        current = hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
        if digest is not None and (current != digest or gradient != float(scale.grad)):
            raise AssertionError("Repeated output or gradient changed.")
        digest, gradient = current, float(scale.grad)
        if i >= args.warmup:
            times.append(elapsed)
        del value
    rss = None
    if sys.platform != "win32":
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            rss *= 1024
    return {
        "median_seconds": statistics.median(times),
        "seconds": times,
        "process_peak_rss_bytes": rss,
        "output_sha256": digest,
        "scale_gradient": gradient,
    }


def paired_worker(args, torch):
    """Alternate order inside one process to reduce load and thermal drift."""
    torch.set_num_threads(1)
    encoder, scale = fixture(torch, args.hidden_size)
    sequence = "M" + "A" * (args.length - 1)
    functions = {
        "reference": lambda: padded_reference(encoder, sequence, torch),
        "current": lambda: encoder._sequence_tensor(sequence),
    }
    samples = {name: [] for name in functions}
    digest = None
    gradients = []
    for i in range(args.warmup + args.repeats):
        for name in (
            ("reference", "current") if i % 2 == 0 else ("current", "reference")
        ):
            scale.grad = None
            started = time.perf_counter()
            value = functions[name]()
            value.square().mean().backward()
            elapsed = time.perf_counter() - started
            current = hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
            if digest is not None and current != digest:
                raise AssertionError("Window assembly changed outputs.")
            digest = current
            gradients.append(float(scale.grad))
            if i >= args.warmup:
                samples[name].append(elapsed)
            del value
    if max(gradients) - min(gradients) > 1e-6:
        raise AssertionError("Window assembly changed gradients.")
    return {
        "median_seconds": {
            name: statistics.median(values) for name, values in samples.items()
        },
        "seconds": samples,
        "median_paired_speedup": statistics.median(
            a / b for a, b in zip(samples["reference"], samples["current"], strict=True)
        ),
        "output_sha256": digest,
        "gradient_max_absolute_difference": max(gradients) - min(gradients),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation", choices=["reference", "current", "paired"], required=True
    )
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[128, 1024, 4096, 16384]
    )
    parser.add_argument("--hidden_size", type=int, default=1280)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--length", type=int)
    args = parser.parse_args()
    if args.length is not None:
        print(json.dumps(worker(args)))
        return
    import torch

    results = {}
    for length in args.lengths:
        command = [
            sys.executable,
            __file__,
            "--implementation",
            args.implementation,
            "--length",
            str(length),
            "--hidden_size",
            str(args.hidden_size),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        results[str(length)] = json.loads(result.stdout)
    print(
        json.dumps(
            {
                "scope": "Window assembly + scalar backward; cached synthetic residue windows, no PLM forward.",
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "threads": 1,
                },
                "configuration": vars(args),
                "command": sys.argv,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

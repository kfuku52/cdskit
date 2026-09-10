#!/usr/bin/env python3
"""Measure PLM batch assembly, retaining the NumPy allocator as a reference.

Run each implementation in a fresh process with identical affinity and inputs.
NUMPY_MADVISE_HUGEPAGE=0/1 can isolate Linux huge-page allocation effects.
This measures batch assembly, not encoder inference or end-to-end training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch

from cdskit.localize_multilabel_plm import _batch


def numpy_reference(encoder, sequences, device):
    values = [encoder.encode(seq) for seq in sequences]
    length = max(len(value) for value in values)
    x = np.zeros((len(values), length, values[0].shape[1]), dtype=np.float32)
    mask = np.zeros((len(values), length), dtype=bool)
    for i, value in enumerate(values):
        x[i, : len(value)] = value
        mask[i, : len(value)] = True
    return torch.as_tensor(x, device=device), torch.as_tensor(mask, device=device)


class FrozenEncoder:
    def __init__(self):
        # Representative ESM2 dimensions and mixed protein lengths; no downloads.
        self.values = {
            str(i): np.full((length, 1280), (i + 1) / 16, dtype=np.float32)
            for i, length in enumerate(
                (
                    400,
                    600,
                    900,
                    1000,
                    1500,
                    1800,
                    2100,
                    2213,
                    2326,
                    2439,
                    2552,
                    2665,
                    2778,
                )
            )
        }

    def encode(self, sequence):
        return self.values[sequence]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("numpy", "tensor"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--iterations", type=int, default=28)
    parser.add_argument("--warmup", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.threads < 1:
        parser.error("Iterations/threads must be positive and warmup nonnegative.")
    torch.set_num_threads(args.threads)
    encoder = FrozenEncoder()
    batches = [["0", "1", "2", "3", "4", "5", "6", str(6 + i)] for i in range(7)]
    assemble = (
        (lambda seqs: numpy_reference(encoder, seqs, args.device))
        if args.implementation == "numpy"
        else (lambda seqs: _batch(encoder, seqs, torch, args.device))
    )

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    for i in range(args.warmup):
        x, mask = assemble(batches[i % len(batches)])
        sync()
        del x, mask
    before = resource.getrusage(resource.RUSAGE_SELF)
    times = []
    for i in range(args.iterations):
        start = time.perf_counter()
        x, mask = assemble(batches[i % len(batches)])
        sync()
        times.append(time.perf_counter() - start)
        del x, mask
    after = resource.getrusage(resource.RUSAGE_SELF)
    digest = hashlib.sha256()
    for seqs in batches:
        x, mask = assemble(seqs)
        digest.update(x.cpu().numpy().tobytes())
        digest.update(mask.cpu().numpy().tobytes())
        del x, mask
    report = {
        "scope": "Frozen residue batch assembly and optional host-to-device copy only",
        "implementation": args.implementation,
        "device": args.device,
        "threads": args.threads,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "seconds": sum(times),
        "batch_seconds": times,
        "median_seconds": float(np.median(times)),
        "p95_seconds": float(np.quantile(times, 0.95)),
        "max_seconds": max(times),
        "user_seconds": after.ru_utime - before.ru_utime,
        "system_seconds": after.ru_stime - before.ru_stime,
        "minor_faults": after.ru_minflt - before.ru_minflt,
        "major_faults": after.ru_majflt - before.ru_majflt,
        "peak_rss": after.ru_maxrss,
        "peak_rss_unit": "bytes" if platform.system() == "Darwin" else "KiB",
        "output_sha256": digest.hexdigest(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "numpy_madvise_hugepage": os.environ.get("NUMPY_MADVISE_HUGEPAGE", "default"),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "batch_seconds"}))


if __name__ == "__main__":
    main()

# CUDA PLM batch allocation stalls

The CUDA path in `localize_multilabel_plm._batch` now allocates its padded tensor
on the GPU and copies each frozen residue array into its row. This removes the
large temporary padded NumPy array on the CPU and avoids transferring its
padding. The CPU and MPS paths retain their existing allocation behavior.
No model, loss, sample order, precision, label, or training-budget settings change.

## Investigation on 2026-09-10

During a slow interval of an existing RTX 6000 Ada training run, a 10-second
sample recorded 9.56 seconds of process system CPU time and only 0.07 seconds of
user CPU time. GPU utilization was 0–2%; disk reads and major faults did not
increase and the process had no swapped pages. The process's local NUMA node
had little free memory and no immediately available 2-MiB blocks, although the
machine had ample memory on other nodes. Kernel stack sampling was unavailable
under the host's ptrace/perf permissions.

A separate allocation probe on CPU 32 reproduced the delay using mixed-size
padded float32 arrays with batch size 8 and embedding width 1280. Changing only
`NUMPY_MADVISE_HUGEPAGE` before importing NumPy produced these results:

| Allocation probe, 80 batches | Run 1 | Run 2 |
| --- | ---: | ---: |
| NumPy, huge-page advice enabled | 127.13 s | 124.57 s |
| NumPy, huge-page advice disabled | 2.25 s | 2.20 s |
| PyTorch host allocation | 4.21 s | 4.15 s |

The slow runs spent 122.38/119.91 seconds in system CPU time. This isolates
huge-page allocation as a reproducible stall mechanism consistent with the live
run; it is not a captured kernel stack from that run. NumPy's default large-array
huge-page advice and Linux's synchronous reclamation/compaction policy explain
why the same operation depends on NUMA memory pressure and fragmentation.
See [NumPy's configuration documentation](https://numpy.org/doc/stable/reference/global_state.html#performance-related-options)
and [Linux's transparent huge-page documentation](https://docs.kernel.org/admin-guide/mm/transhuge.html#sysfs).
The fix does not change either global setting.

## CUDA measurements and equivalence

Three fresh processes per implementation, alternating order, each used seven
warmup batches followed by 28 measured batches. Inputs contain frozen synthetic
residues of length 400–2778, width 1280, and batch size 8. Each timed batch includes
assembly and CPU-to-GPU copies followed by CUDA synchronization. The table uses
the median of the three per-process statistics.

| CUDA batch assembly | Before | After |
| --- | ---: | ---: |
| Median batch latency | 18.04 ms | 5.80 ms |
| 95th-percentile batch latency | 20.60 ms | 7.47 ms |
| Peak host RSS | 685.93 MiB | 629.12 MiB |
| System CPU time per 28 measured batches | 0.1881 s | 0.0029 s |

This is a **3.11× improvement in this batch-assembly benchmark**, not a measured
end-to-end training speedup. The short CUDA benchmark did not reproduce the
pathological allocation delays. The old training process remained active while
bounded diagnostic steps ran; the timings are not isolated production throughput
measurements. Peak RSS includes imports and frozen input arrays and does not
measure device memory.

All six processes produced identical batch-and-mask SHA-256 digests. A separate
CUDA test compared all ten original pooling/loss/selection recipes over three
epochs using frozen synthetic 1280-dimensional inputs. Training histories,
selected epochs, and weights were bit-identical. Deterministic cuDNN was enabled
for that paired diagnostic only; this does not assert bitwise reproducibility
between arbitrary GPU training runs. CPU and CUDA regression tests also verify
noncontiguous input copies, padding, dtype, and independent batch storage.

The initial host-tensor-only approach was discarded because it added overhead
in healthy runs. The final CUDA implementation eliminates the padded host
allocation entirely while preserving the CPU and MPS paths.

## Reproduction

From an installed development checkout, run each implementation in a fresh
process on the same GPU and with the same CPU affinity. Repeat in alternating
order at least three times, using distinct output filenames:

```sh
NUMPY_MADVISE_HUGEPAGE=1 PYTHONPATH=. python scripts/benchmark_plm_batch.py \
  --implementation numpy --device cuda --threads 8 --warmup 7 --iterations 28 \
  --output before.json
NUMPY_MADVISE_HUGEPAGE=1 PYTHONPATH=. python scripts/benchmark_plm_batch.py \
  --implementation tensor --device cuda --threads 8 --warmup 7 --iterations 28 \
  --output after.json
```

The script retains the former NumPy implementation as its reference. Compare
`output_sha256` before interpreting timings. The benchmark does not download an
encoder or change system settings. Setting `NUMPY_MADVISE_HUGEPAGE=0` in a fresh
process is a diagnostic control; its effect depends on host memory conditions.

Machine details and numerical summaries are stored in
[the measurement record](plm-cuda-batch-benchmark-2026-09-10.json).

## Validation and rollout status

The repository's `scripts/check.py all` passed: 1,348 tests, one CUDA-only skip
on the local CPU environment, 78.00% coverage, critical-module coverage floors,
Ruff checks/format/complexity, mypy, Bandit, dependency audit, and installed-wheel
checks. The CPU and CUDA batch regression cases both passed on the remote GPU.
After making the source tensor's CPU device explicit, the ten-recipe CUDA
comparison and both regression cases were rerun successfully, along with
lint/format, typing, and diff checks.

The fix ships in v0.30.1. Applying it to a previously submitted job requires
starting that job with the updated source; an already running Python process
continues to use its loaded implementation. Preserve the version provenance of
reused checkpoints when resuming an experiment across versions. No production-job
completion-time improvement has been measured yet.

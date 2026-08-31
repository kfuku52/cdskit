# Implementation review and follow-up: 0.28

The 2026-08-31 review started from `master` at
`85d1b0305b1261d497274a34d56914ee77578cb2` (0.27.0), after a fast-forward pull.
The existing 740 tests passed, but additional failure-path and numerical checks
reproduced eight implementation defects and one development-environment mismatch.
Version 0.28.0 addresses all nine, together with the measured performance and
development-workflow issues below.

## Correctness and data protection

| Finding | Change | Regression evidence |
|---|---|---|
| F1: interrupted nested TargetP OOF runs could not resume | Every checkpoint records data, training configuration and fold provenance before saving. The epoch budget can increase without bypassing provenance checks. | Interrupt after the first saved epoch, resume, and compare final CPU tensors exactly against an uninterrupted run. Reject altered training data. |
| F2: auxiliary CLI inputs could be overwritten | A shared command declaration covers alignments, accession lists, models and generated output filenames. Check resolved model aliases before loading. | Command-level collisions plus hard-link/symlink aliases preserve the original input. |
| F3: a multi-file transaction could replace a directory | Validate every destination and its parents before staging and again before committing; reject directories, special files and overlapping output paths. | Directory/FIFO rejection leaves existing data and parent directories unchanged. An injected commit failure restores a dangling symlink. |
| F4: sequence output and its report could disagree after failure | Stage `filter` and `trimcodon` file outputs and reports in the same transaction. | A sequence serialization failure preserves both old files; success exercises JSON and TSV reports. |
| F5: CDS extraction retained genomic annotation coordinates | Extract in coding orientation, rebuild the selected CDS location, remap `transl_except`, and remove stale feature/reference locations. | Forward, reverse and joined CDSs, per-letter annotations, translation exceptions and actual GenBank write/read round trips. |
| F6: translation and backalignment disagreed on dual-coding stops | Share codon translation with amino-acid precedence for codes 27, 28 and 31, while handling an optional terminal stop in context. | Translation/backalignment round trips for all 27 available genetic codes, plus explicit/omitted terminal stops and invalid internal stops. |
| F7: the last validation batch was overweighted | Sum classification and cleavage loss components with their own denominators, including class weights. | Validation loss is invariant to batch size and ordering with and without class/cleavage weights. |
| F8: offline mode did not reach nested ESM models | Use scoped prediction settings throughout single, blend, two-stage and three-stage predictors. | All nested encoder/tokenizer loads receive `local_files_only=True`; the tests perform no network download. |
| F9: local and CI mypy checked different environments | Use the same locked full environment and its running Python version for typing. Compile and test the minimum Python version separately. | Full installed NumPy/ML typing passes, as does the Python 3.10 core suite. |

The new shared components have explicit types: command path declarations,
prediction runtime/batching, embedding cache and loss accumulation. Training and
inference now share the optimized specialist feature functions instead of
maintaining duplicate calculations.

## Performance measurements

Before/after measurements used an Apple M1 Pro, macOS arm64, Python 3.12.13,
NumPy 2.5.2, Torch 2.13.0 and Transformers 5.14.1. Each row below uses the median
of three interleaved before/after pairs in separate processes, with Torch set to
one CPU thread. These are small synthetic workloads on a shared workstation,
not production model-quality or GPU benchmarks.

| Workload | Operation seconds before → after | Whole-process wall seconds before → after | Process peak RSS median, MiB before → after |
|---|---:|---:|---:|
| Frozen ESM head training: 128 sequences × 200 aa, 10 epochs | 1.483 → 0.262 (5.7×) | 8.362 → 6.448 | 309.1 → 354.7 |
| Specialist features: 1,280 sequences × 200 aa | 1.109 → 0.458 (2.4×) | 3.372 → 2.266 | 220.7 → 222.7 |
| Two-fold BiLSTM CV: 100 sequences × 200 aa, one epoch | 1.168 → 1.044 | 2.755 → 2.321 | 290.9 → 291.2 |

The ESM measurement used a real, locally initialized two-layer transformer with
hidden size 64, four attention heads and no dropout, plus a deterministic test
tokenizer. It did not download or benchmark pretrained ESM-2 weights. Encoder
calls fell from 80 to 8, and processed sequences from 1,280 to 128. Final head
tensors were exactly equal in all three pairs.

The CV model used sequence length 100, embedding/hidden sizes 8, one BiLSTM
layer, batch size 16, seed 7, and mixed animal/plant records. Timing its prediction
phase separately gave 0.201 → 0.029 seconds (7.0×); top-level prediction calls
fell from 100 single-record calls to two fold-level batch calls, which are
internally grouped by organism. All output structures and labels matched; the
maximum probability difference was `2.6004567e-8`. Feature outputs differed by at
most `7.7715612e-16`.

An earlier three-pair CV series without prediction-phase instrumentation had
overall medians of 1.755 → 1.937 seconds and process wall medians of
4.262 → 4.482 seconds. It does **not** support a consistent whole-CV speedup.
The prediction-phase improvement is the supported result; training/import time
and workstation load dominate this small end-to-end workload.

RSS is the process high-water mark, including imports and the model, not an
allocation count for the timed operation. For ESM it ranged from 291–336 MiB
before and 301–411 MiB after. These measurements do not establish a memory
reduction. The shared LRU embedding cache defaults to 256 MiB, separately from
each fit's `number_of_sequences × hidden_size × 4` byte float32 feature matrix
and transient encoder buffers. Large datasets must budget for those allocations.
The cache contains embeddings only, never labels or fitted heads; its identity
includes encoder source/revision, local-file metadata, length and pooling.

## Development checks and CI

Use [the common check runner](../TESTING.md), for example:

```bash
python scripts/check.py quick
python scripts/check.py all
```

Core, full and build profiles use separate `.venvs/` environments, preserving an
existing `.venv`. The full Linux CI profile uses the official CPU PyTorch
resolution; an exported Python 3.12 Linux environment selects `torch==2.13.0+cpu`
without CUDA/NVIDIA/Triton packages. Normal `ml` installations and dependency
ranges remain unchanged. The weekly security audit still covers the normal ML
resolution as well.

Short quality, ML, coverage, audit and wheel checks now share one CI job.
Core/platform-affecting changes run the existing broad OS/Python coverage;
pure ML/documentation changes retain the Linux Python boundaries and full
validation. Scheduled/manual runs cover all nine core combinations. Branch
protection was not changed.

Benchmark schema 2 records environment metadata, complete output fingerprints,
individual timing samples and process peak RSS. Scheduled CI compares retained
results only under matching conditions and warns about changed outputs or a
median slowdown over 30%; noisy timing alone does not fail the test suite.
Legacy reports are incomparable and establish a new baseline.

Final local validation for 0.28.0:

- `python scripts/check.py all`: 828 tests passed, two expected trusted-legacy
  pickle warnings, 75.49% combined branch/statement coverage, all critical
  module floors passed. Pytest took 48.48 seconds; this is not a controlled
  comparison with the original cold audit run.
- Ruff lint/format/complexity, mypy over 62 modules and high-severity Bandit passed.
- The installed dependency audit found no known vulnerabilities; the editable
  project itself is excluded by the audit command.
- An sdist and wheel were built. A fresh environment imported the installed
  0.28.0 wheel outside the checkout and passed version, padding, SVG and dependency
  consistency checks.
- The minimum-version Python 3.10 core suite passed 743 tests (15 optional ML
  tests deselected), together with compile checks.
- `uv lock --check`, `git diff --check` and actionlint 1.7.12 passed.
- The benchmark CI driver downloaded the previous retained GitHub artifact and
  ran all seven workloads with five repetitions. Its legacy-schema baseline was
  correctly marked incomparable.

The first GitHub run exposed one additional runner defect: direct executable
wrappers used the caller's `PATH`, resolving a Python outside the newly selected
environment. This failed on clean Linux/macOS runners but was hidden by local
global dependencies; Windows explicitly supplies the test interpreter and passed.
Version 0.28.1 activates the selected environment for all subprocesses as well.
The executable-wrapper test is retained unchanged. The uv cache key now also
includes the version source after verifying that a version-only change left
installed editable metadata at 0.28.0 while the code already reported 0.28.1.
The full check passed all 828 tests again with another Python environment placed
first on the caller's `PATH`, and the core check passed 743 tests. The 0.28.1
installed-wheel smoke check and dependency audit passed; a subsequent sync
confirmed that code and installed package metadata both report 0.28.1.

## Compatibility and limits

- Old OOF checkpoints without valid matching provenance are still rejected.
  Start a fresh model directory for 0.28.0 rather than relabelling old checkpoints.
  Resume equivalence was tested on CPU; actual CUDA/MPS execution was not tested
  locally, although requested-device propagation has regression coverage.
- CDS extraction intentionally retains only the selected CDS feature. Other
  genomic features, contig metadata and reference spans cannot keep their old
  coordinates on the extracted sequence. Unsupported/ambiguous translation
  exceptions fail explicitly instead of silently producing invalid annotations.
- Multi-file transactions roll back ordinary write/commit failures. They are not
  crash-atomic across several filenames, do not lock out concurrent external
  writers, and cannot roll back bytes already sent to standard output. Use file
  destinations for sequence/report rollback guarantees.
- The numerical tests and timings establish implementation consistency on the
  stated workloads. They do not measure biological accuracy or guarantee a
  production speedup on every dataset or machine.

# Installation and dependencies

The current cdskit source requires **Python 3.10 or newer**; core CI tests
Python 3.10–3.14 on the supported OS matrix. See
[`pyproject.toml`](https://github.com/kfuku52/cdskit/blob/master/pyproject.toml)
for package requirements and
[TESTING.md](https://github.com/kfuku52/cdskit/blob/master/TESTING.md) for developer
environments. This page describes current `master`; older packages may have
different options.

## GitHub installation

Create and activate a virtual environment, then install from GitHub. CDSKIT is
not published on PyPI, so `pip install cdskit` or `pip install 'cdskit[ml]'`
without a repository URL is not a supported installation command.

```bash
python -m venv .venv
# POSIX shells; on Windows use .venv\Scripts\Activate.ps1 in PowerShell.
source .venv/bin/activate
python -m pip install --upgrade 'cdskit @ git+https://github.com/kfuku52/cdskit.git'
cdskit --version
cdskit --help
```

Using `python -m pip` helps ensure that cdskit is installed into the same Python
environment you intend to run.

## Bioconda

Use [Bioconda's channel order and strict priority](https://bioconda.github.io/index.html#with-conda):

```bash
conda create -n cdskit -c conda-forge -c bioconda --strict-channel-priority \
  cdskit 'python>=3.10' 'biopython>=1.80' 'numpy>=1.23' 'matplotlib-base>=3.6'
conda activate cdskit
cdskit --version
```

As checked on 2026-08-31, the
[Bioconda 0.27.0 recipe](https://github.com/bioconda/bioconda-recipes/blob/master/recipes/cdskit/meta.yaml)
still declares Python >=3.8 and Biopython >=1.77, and omits Matplotlib.
The explicit requirements above are a workaround for that packaging mismatch,
not extra CDSKIT features. They can be removed once the published recipe's
runtime dependencies match upstream metadata. GitHub source installation
already resolves the required base dependencies. Tagged releases also need a
downstream recipe update and successful build before appearing in Bioconda;
publication is not immediate.

Bioconda supports Linux and macOS. For native Windows, use the GitHub pip
installation; the project tests Windows separately from Bioconda packaging.

## Base dependencies

The GitHub pip installation includes:

- [Biopython](https://biopython.org/) >=1.80 for sequence and GenBank I/O;
- [NumPy](https://numpy.org/) >=1.23 for numerical operations;
- [Matplotlib](https://matplotlib.org/) >=3.6 for `cdskit plot`.

These dependencies are installed automatically.

## Optional machine-learning dependencies

Lightweight centroid JSON localization models need only the base installation.
Neural training and prediction use the `ml` extra:

```bash
python -m pip install --upgrade 'cdskit[ml] @ git+https://github.com/kfuku52/cdskit.git'
```

This extra includes **torch >=2.2, scikit-learn >=1.4, and transformers >=4.40**.
Transformers is already included; installing only torch and scikit-learn is not
equivalent to installing the full extra. ESM models also need their encoder
files, downloaded from a pinned revision or supplied locally.

GPU support is optional. Published cdskit localization models run on CPU; CUDA
or Apple MPS mainly helps when retraining neural models.

## Pretrained targeting5 runtime

The published `targeting5` and `targeting5-perox-deeploc21-et-v1` artifacts
contain scikit-learn **1.5.2** estimators. Installing the latest `ml` extra alone
does not reproduce that environment: scikit-learn 1.9.0 fails to load both
artifacts with `No module named '_loss'`. Scikit-learn does not support loading
pickled estimators across versions; see its
[model persistence guidance](https://scikit-learn.org/stable/model_persistence.html).

Use a separate Python 3.12 environment for these existing artifacts:

```bash
python3.12 -m venv .venv-targeting5
source .venv-targeting5/bin/activate
python -m pip install \
  'cdskit[ml] @ git+https://github.com/kfuku52/cdskit.git' 'scikit-learn==1.5.2'
```

On Windows, use `py -3.12 -m venv .venv-targeting5` and activate
`.venv-targeting5\Scripts\Activate.ps1` in PowerShell. The runtime was checked
on macOS ARM64 with Python 3.12.13, torch 2.13.0, NumPy 2.5.2, and scikit-learn
1.5.2; this is a smoke check, not a claim that every platform or dependency
combination has been validated for these artifacts.

The exact scikit-learn pin is specific to the published model files, not a
library-wide upper bound or a recommendation for new training. It can be
retired after replacement artifacts are trained or re-exported and verified
with the newer runtime. Keep this environment separate from current ML
development, and record dependencies when training custom sklearn models.

## Parallel work and resource limits

Only commands that advertise `--threads` support it. `--threads 1` is the
default; positive values request that many workers and `0` detects CPUs up to
the safety limit (64 by default). Explicit values above the limit are rejected.
Set `CDSKIT_MAX_THREADS` to change the ceiling. Small workloads may still run
serially, so this option does not guarantee a particular number of active
threads.

CPU-bound record operations can use processes instead of threads. Their
default crossover is 16,000,000 input residues, configurable with
`CDSKIT_PROCESS_PARALLEL_MIN_RESIDUES`. Vectorized translation runs serially.
`label` and other metadata-only commands have no `--threads` option.

## Troubleshooting

- If `cdskit` is not found, confirm that the environment's scripts directory is
  on `PATH`, or run the executable from the same activated virtual environment.
- Use `cdskit COMMAND --help` to check current option names and defaults.
- Boolean options accept `yes/no`, `true/false`, `on/off`, and `1/0`.
- Legacy model compatibility and offline cache setup are documented in
  [the prediction guide](https://github.com/kfuku52/cdskit/wiki/cdskit-localize#model-safety-and-offline-use).

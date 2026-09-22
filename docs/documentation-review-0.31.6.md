# Documentation/implementation audit — 0.31.6

Reviewed on 2026-09-22 from `master` at
`b92adb78a0f6418c517232f51b8a1262f882c9c4` (0.31.5), with a clean worktree.
The fetched `origin/master` matched that commit. Applicable instructions were
`AGENTS.md`, `TESTING.md`, `docs/documentation.md`, `RELEASING.md`, and the
prepare-github-push / verify-cli-example skills. No nested AGENTS.md was found
in the reviewed documentation paths. Runtime behavior and dependency metadata
were not changed; the required publication bump is 0.31.6.

## Scope and findings

Priority was installation, the README pipeline, complete small wiki examples,
sequence statistics and QC outputs. Reviewed parser definitions, dispatch/path
validation, selected command implementations and their docstrings, and existing
tests, rather than treating help as the specification. Localization review was
limited to installation, runtime/cache/offline controls and staged configuration;
no biological accuracy claims were re-evaluated.

### A — corrected documentation or missing operational detail

| Location | Previous problem and correction | Evidence |
| --- | --- | --- |
| README; Installation-and-dependencies | Conda workaround omitted required filelock; recipe version was stale. Add filelock >=3.12, update current recipe observation and distinguish recipe from binary availability. | `pyproject.toml`; [Bioconda recipe](https://github.com/bioconda/bioconda-recipes/blob/master/recipes/cdskit/meta.yaml), inspected 2026-09-22: version 0.31.0, Python >=3.8, Biopython >=1.77, NumPy, no Matplotlib/filelock; stale BSD license. No conda solve performed. |
| README | Said targeting5 required its original sklearn environment. Link to existing compatibility setup instead. | `localize_pickle.py`, `tests/ml/test_localize_pickle.py`, 0.31.1 changelog and existing installation guide. Retained synthetic 1.5.2 fixture passes on installed 1.9.0. Published weights were not re-tested here. |
| README; installation guide; ESM2 model card | `ml-cpu` described as selecting CPU-only wheels without an installer distinction. Explain uv source configuration versus pip metadata. | `pyproject.toml` extras, Linux-specific `tool.uv.sources`, and existing TESTING.md; no dependency changes. |
| ESM2 model card, offline use | Claimed `--model_download no` did not disable backbone downloads, conflicting with the prediction guide. Correct nested encoder behavior and environment precedence. | `localize_main`, `PredictionRuntime`, `offline_requested`, `localize_esm_head.py`, `localize_models.resolve_localize_model_path`; `test_cli_offline_applies_to_every_nested_encoder`. |
| README; installation output section | Input filename placeholder and overwrite behavior were not explicit. Document user-supplied input, replacement without prompt, parent creation, collision rejection, and shell redirection boundary. | `command_paths.py`, `atomicio.py`, `tests/integration/test_output_safety.py`; small CLI overwrite/parent-creation checks. Link staged resume rules rather than duplicate them. |
| codonstats guide | Usage fraction denominator, excluded resolvable ambiguities, complete-codon meaning, zero GC denominator, units and duplicate IDs were underspecified. Add definitions and actual field names. | `summarize_record`, `gc_percent`, `print_summary_table`, `print_usage_table`; `tests/integration/test_codonstats.py`; small CLI output below. No scientific meaning inferred beyond the reported calculation. |
| filter / trimcodon guides | Missing defaults and threshold range; site-coordinate basis unclear. Add defaults from parser, 0–1 bounds and original one-based codon position. | `cli.py`, `filter.validate_fraction`, `trimcodon.validate_fraction`, `summarize_codon_site`; integration tests and small report output. |
| validate guide | QC issues could be mistaken for a failing exit status; rate denominator and gap-only label were unclear. Explain exit status, rate and all-N/all-X handling. | `validate_main`, `summarize_records`, `GAP_ONLY_CHARS`, CLI dispatch and `tests/integration/test_validate.py`; CLI reproduction. |

### B — unresolved implementation issue

`validate --report -` violates the README single-output convention and the
rectangular report contract. With this input:

```fasta
>a description
ATGGCNTAA
>b
ATG---NNN
```

run:

```bash
cdskit validate --seq_file cds.fasta --report -
```

Exit status is 0, but stdout starts with `Validation summary` and heterogeneous
key/value lines, followed by the five-column TSV header
`schema_version, section, metric, value, ids` (tab separated) and report rows.
It is not a single rectangular TSV. `validate_main` unconditionally calls
`print_validate_summary` before `write_validate_report`; `COMMAND_PATHS` counts
only the explicit report, not the implicit summary. Existing stdout-collision
tests cover sequence/report and sequence/GFF combinations, not this case.

No behavior was changed. The guide now labels this as a known issue and gives
the verified workaround `--report validate.tsv` (or `.json`). Deciding whether
to suppress/redirect the summary or reject this combination requires a separate
implementation change; this audit does not redefine the contract to bless it.

### C — unresolved intent/scientific meaning

No additional in-scope discrepancy required choosing an undocumented scientific
interpretation. Published model accuracy, historical experiments, biological
validity of thresholds and datasets remain outside this audit, not validated
by synthetic tests. Historical release descriptions and metrics were preserved.

## Execution record

Commands ran on macOS ARM64 / Python 3.12.14. CLI examples used the absolute
checkout executable `.venvs/core-3.12/bin/cdskit`, inside temporary directories;
stdout and stderr were captured separately. All command lines below exited 0
unless explicitly described as an audit-script assertion mistake.

- Exact published input, command and complete expected output compared:
  `pad --seq_file input.fasta --out_file output.fasta`,
  `mask --seq_file input.fasta --out_file output.fasta`,
  `aggregate --seq_file input.fasta --out_file output.fasta --expression ":.*" "\|.*"`,
  `stats --seq_file input.fasta`, and
  `split --seq_file input.fasta --prefix output`.
  All matched, including all three `output_{1st,2nd,3rd}_codon_positions.fasta`
  files. Page input blocks were materialized; repository fixtures were untouched.
- README pipeline ran verbatim with supplied small input (`a:1=ATGGCC`,
  `a:2=ATG`), because README does not ship `input.fasta`:
  `cdskit pad --seq_file input.fasta | cdskit mask | cdskit translate | cdskit aggregate --expression ':.*' > output.faa`.
  Output was exactly `>a:1\nMA\n`. This is a substitute-input check.
- With the two-record input above, `translate --seq_file cds.fasta` returned
  `MA*` and `M-X`, retaining the FASTA description. `codonstats --seq_file
  cds.fasta --mode summary` produced the 14 named columns and `gc_all=37.500000`
  for a; `--mode usage` returned ATG count 2/fraction 0.666667 and TAA count
  1/fraction 0.333333. The audit script initially expected 28.571429 by mistake;
  correcting the hand calculation to 3/8 × 100 made the assertion pass. No
  implementation or expected repository fixture was changed to resolve this.
- `filter --seq_file cds.fasta --out_file filter.fasta --report filter.tsv
  --min_clean_codon_fraction 0.3` retained both records; report schema_version
  was 2, with clean fractions 0.333333. `trimcodon --seq_file cds.fasta
  --out_file trimcodon.fasta --report trimcodon.tsv --min_clean_fraction 0.5`
  retained only ATG in both records; site rows used original positions 1,2,3
  and keep=yes,no,no. These are small substitute-input/report checks.
- `validate --seq_file cds.fasta --report validate.json` produced parseable
  JSON with rate 0.4 and two affected IDs, while exiting 0. `--report -`
  reproduced B above. `stats --mode alignment --seq_file cds.fasta` produced
  one TSV row: 2 taxa, 9 positions, 18 cells, missing 38.889%, AT 0.646/GC 0.354.
- `translate --seq_file input.fasta --out_file new/proteins.faa` run twice
  created the directory and replaced the file; exact sequences were checked.
- Help exited 0 for pad, mask, aggregate, stats, codonstats, filter, validate,
  trimcodon, translate, split, localize and localize-learn.
- `python scripts/check.py quick`: 979 passed, 2 PyYAML skips, 21 deselected.
- `python scripts/check.py core`: 984 passed, 2 PyYAML skips, 16 deselected.
  Both emitted three existing UniProt weak-label warnings.
- `.venvs/full-3.12-cpu/bin/python -m pytest -q
  tests/ml/test_localize_esm_head_runtime.py tests/ml/test_localize_pickle.py
  tests/unit/test_localize_pipeline_config.py`: 51 passed, no skips; two expected
  sklearn cross-version warnings. This also covers the two skipped YAML cases.
  It is a focused existing-environment run, not the full ML profile.
- `python scripts/check.py build`: sdist/wheel built; fresh temporary wheel
  environment passed import, version/help, padding, alignment statistics, SVG
  output and dependency compatibility checks. This validates the local wheel,
  not the literal GitHub VCS pip command or a Bioconda installation.
- Local Markdown file/image destinations in README, TESTING, RELEASING and
  top-level wiki/docs pages: 44 checked before edits and 46 after edits, no missing paths. Anchor
  validity and all external links were not exhaustively checked. No dedicated
  documentation build/link-check configuration was found. Final diff whitespace
  and changed-file list were reviewed before publication.

## Configuration, publication and remaining boundaries

Staged configuration was inspected against `localize_pipeline_config.load_config`
and `pipeline_main`: paths resolve relative to the config; non-default legacy
CLI training settings are rejected rather than overriding staged configuration.
The YAML example exists, but `partitions.tsv` and the immutable encoder revision
are explicitly user-supplied placeholders. No staged training was run. Cache-root
precedence and offline behavior were compared to `localize_models.py` and
`localize_runtime.py`; no real model cache was populated or modified.

The separate wiki checkout was clean at `80f9706`. Comparison found two existing
unpublished targeting5 compatibility corrections in the main repository. They
agree with current code/tests; publish these with the changed pages rather than
restore the stale public claims. No newer conflicting wiki edits were found.

Not executed: live NCBI/UniProt retrieval, published model/ESM downloads, long
training/benchmarks, conda solve, Linux/Windows or Python boundary environments,
full ML/quality/coverage/security profiles. Those are not needed to validate the
documentation-only corrections and version bump locally; CI remains separate.
Other command pages and helper docstrings received no exhaustive behavioral
review. No dependency updates, model retuning or historical result changes were made.

<!-- BEGIN KF AGENT POLICY: source=https://github.com/kfuku52/kf-agent-policy; version=12; sha256=d38271eaba87f3f2c5c92b6c4ea821eca662d38f6f0db388fda5270edb5e9afa -->
# Common agent policy

Repository-specific instructions override these defaults.

- Follow the user's task scope within higher-priority instructions and execution
  permissions. Complete implementation through affected verification and a result
  report; a plan or investigation ends with its requested deliverable. Continue
  authorized work without repeated approval; identify actual blocking boundaries.
- Inspect the worktree and preserve unrelated changes. Refresh remote information
  when needed; do not merge, rebase, or switch branches merely to inspect it.
- Prefer the default branch when starting work without an established branch.
  Preserve an existing task branch; follow explicit user branch instructions.
  Never create or switch branches solely for a commit, push, release, or PR.
- Change or recommend branch protection only when explicitly asked. Honor explicit
  repository-specific direct-push exceptions; otherwise report a rejected push
  without bypassing protection or inventing a branch or PR.
- Prefer the simplest design that meets established requirements. Avoid
  complexity for speculative needs; extract helpers when they improve clarity
  or enable reuse.
- Unpublished implementation details may be redesigned; preserve existing public
  APIs, file formats, and saved-data compatibility unless a breaking change is
  authorized. Update affected producers, consumers, tests, examples, and docs.
- Validate untrusted input at the responsible boundary; rely on established
  invariants internally. Revalidate when mutation or a new trust boundary
  invalidates those guarantees.
- Fix verified root causes; do not hide failures with fallbacks or weaker checks.
  Document unavoidable workarounds and their removal conditions.
- Read relevant docs and run the repository's check entrypoint for the change and
  phase. Verify affected behavior; report checks run and omitted. Repeat or broaden
  successful checks only for new changes, failures, or unresolved concerns.
- For library metadata, require demonstrated incompatibility for exact pins or
  upper bounds; keep reproducibility locks separate.
- When editing READMEs, keep them concise with useful visuals inline; put extended
  guides in linked documentation.
- For GitHub push/release work, use `prepare-github-push` in `.agents/skills/`.
  Local-only commits need no version bump; GitHub pushes require one.
- Optimize bottlenecks in RAM usage or CPU time whenever found. For software
  performance work, use `benchmark-performance` in `.agents/skills/`.
  Performance claims require comparable measurements and equivalent output.
- For GitHub Actions edits, use `optimize-github-actions` in `.agents/skills/`.
  Preserve required coverage; never run untrusted PR code on self-hosted runners.
<!-- END KF AGENT POLICY -->

# Working on CDSKIT

Read [README.md](README.md) for the CLI/output contract and
[TESTING.md](TESTING.md) for setup and change-specific checks. There is no
separate CONTRIBUTING guide. Read only the relevant command page in `wiki/`;
for documentation edits use [docs/documentation.md](docs/documentation.md),
and before push use [RELEASING.md](RELEASING.md).

## Start and verify

Run commands from the repository root with Python and uv available:
`python scripts/check.py quick` creates the locked core environment and tests it.
Use `.venvs/core-3.12/bin/cdskit COMMAND --help` to inspect the checkout's CLI
(Windows: `.venvs/core-3.12/Scripts/cdskit.exe`).
`python scripts/check.py quality` runs lint, format, type, complexity and source
security checks; it requires the full ML environment. Select the minimum tests
from [the change map](TESTING.md#choose-checks-by-change), rather than using
`all` for every iteration. `all` includes a network dependency audit.

The parser and dispatch start in `cdskit/cli.py` and `command_runtime.py`;
follow the selected command's lazy import to its implementation. Shared codon
meaning lives in `codonutil.py`; sequence I/O in `util.py`, transactional output
in `atomicio.py`, and TSV contracts in `tsvio.py`. For localization, start with
`localize.py`, `localize_runtime.py` and the specific model backend.

## Preserve scientific and compatibility contracts

- Read [codon semantics](wiki/codon-semantics.md) before changing frame selection,
  ambiguity, missing bases or stops. Genetic codes 27/28/31 distinguish ordinary
  translation from complete-CDS termination; do not replace this with a raw stop
  table lookup. Preserve selected genetic codes and documented tie handling.
- For localization read [the scientific contract](docs/localize-scientific-contract.md)
  and, for artifact loading, [model portability](docs/localize-model-portability.md).
  Feature schemas, legacy decisions, label order and saved model compatibility
  matter; synthetic tests do not validate biological accuracy. Do not retune
  thresholds, splits, seeds, encoder revisions or experiment parameters as cleanup.
- Preserve public imports, deprecated CLI aliases, stdout/stderr separation,
  report columns/schema versions and atomic output behavior. Consult
  [MIGRATION.md](MIGRATION.md) for established compatibility mappings.
- Treat `tests/fixtures/` as immutable inputs during checks. Do not overwrite
  `data/`, model/download caches, experiment run directories, historical results
  under `docs/`, or user environments/settings. Put new test output in `tmp_path`
  or a temporary directory. Change reference artifacts only for an intentional,
  reviewed contract update; do not regenerate them just to make tests pass.
- Keep dependency changes intentional: checks use `uv.lock`; do not upgrade it
  to repair an unrelated environment issue. Pipeline examples include large-model
  training and placeholder revisions, and are not smoke commands.

At completion review `git diff --check` and the file list, report checks and
results (including skips, static-only checks and blockers), and distinguish
local validation from CI. Use the existing push skill and release procedure for
requested publication. CLI example checks can use
[verify-cli-example](.agents/skills/verify-cli-example/SKILL.md).



---
name: verify-cli-example
description: Verify a CDSKIT CLI documentation example against checkout help and actual small-input output. Use for changed CLI options or command examples, not full model evaluation or general test runs.
---

# Verify a CLI example

Inputs: the changed command/page, its complete small input, documented options
and expected output. Work from the repository root. Read
[documentation maintenance](../../../docs/documentation.md) for the shared
review rules and [TESTING.md](../../../TESTING.md) for the environment.

1. Use the existing core profile, or create it with `python scripts/check.py
   quick`. Resolve `.venvs/core-3.12/bin/cdskit` to an absolute executable path
   before changing working directory (Windows: `Scripts/cdskit.exe`). Do not
   substitute a globally installed release or assume `python -m cdskit` exists.
2. Run that executable with `COMMAND --help`. Compare the page's option names,
   choices and defaults with help and `cdskit/cli.py`; check deprecated aliases
   separately from canonical spelling.
3. Copy the documented small input into `tempfile.TemporaryDirectory`, run the
   complete command there with `subprocess.run(..., check=True)`, and capture
   stdout/stderr separately. Use absolute input paths for existing fixtures;
   never write expected output back into `tests/fixtures/` or user data.
4. Compare actual sequence IDs/content and report headers/values with the page's
   expected result. Use exact text for deterministic FASTA/TSV examples, and
   parsed fields for nondeterministic metadata. Exit nonzero on a mismatch.
   Run the command's affected tests using the TESTING.md change map.
5. Report the page, input, executable/profile, command, exit status and output
   comparison, plus discrepancies corrected and checks not run. Temporary
   output need not be committed. Wiki publication is a separate procedure.

A verified starting case is `wiki/cdskit-pad.md`: copy its seven-record
`input.fasta`, run the shown `pad --seq_file input.fasta --out_file output.fasta`,
and compare the entire result with the page's `output.fasta`. This exercises
head/tail padding and preserved complete sequences without external data.

If input or expected output is missing, identify it rather than inventing a
successful reproduction. If a command needs model weights, external services,
or long training, report that boundary and use an explicitly supplied small
local model/input only when it still verifies the stated example. Never silently
change the example's model or parameters. Missing environments and nonzero exits
remain failures/blockers; do not weaken checks or auto-download large artifacts.

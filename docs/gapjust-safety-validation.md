# gapjust safety validation — 2026-09-10

The change rejects CDS-overlapping length edits by default, supports whole-run
skip, and rejects deleted feature endpoints. Phase is preserved, not repaired.
No model definitions, weights, or training data were changed.

## Automated checks

The commit candidate was exported from an isolated Git index based on main
revision `939751038928cddf71fa57f87f2c6dda7fa4fbbb`, including the existing backtrim
changes and excluding unrelated uncommitted main-worktree changes.

- `python scripts/check.py all`: **1,089 passed**, two existing trusted-pickle
  warnings; combined branch/statement coverage **76.9%**, critical floors passed.
  Ruff lint/format, mypy, complexity guard, high-severity Bandit, dependency audit
  (85 dependencies), sdist/wheel build and installed-wheel smoke checks all passed
  in the same run.
- Full core checks on managed Python **3.10.21** and **3.14.7**: **949 passed**
  each, with two optional PyYAML skips per environment. The previous miniforge
  shared-library loading failure required no dependency pin or runtime workaround.
- The suite includes **76 new gapjust safety cases**.
- Two pre-existing Ruff format violations in `cdskit/localize_pickle.py` and
  its test were fixed with formatting-only changes so the full check can finish.
- Windows and Linux were not executed locally.

The final review also fixed unchecked sequence edit plans (including non-N or
stale ranges), colliding stdout outputs, false phase diagnostics across unknown
CDS segments, partial GFF mutation on a late mapping failure, and stale embedded
features/quality values after compensating length edits. All have regression
coverage. A seeded base-identity oracle checks 200 multi-edit coordinate cases.

Tests cover transcription order on both strands, partial initial phase,
multiple/shared Parents, repeated IDs, Parent cycles, unknown strand, CDS SO
accession, overlapping boundaries, complete deletion, in-frame and compensating
edits, skip, threshold exclusions, sequence-region bounds, output preservation,
report collisions, and threaded output equivalence. An independent base-identity
mapping checks coordinate shifts and spliced-CDS preservation.

## Independent extraction

With installed **gffread 0.12.7**, `gffread -E INPUT.gff -g INPUT.fa -x CDS.fa`
was run before and after normalization, without phase-adjustment or filtering
flags. Extraction from the tracked `gapjust_01` fixture retained all three CDS
sequences at target gap lengths 0, 4 and 100.

An additional source outside this repository was fixed to gffread revision
[`05462cdb2bb6025f72a8d93c4797b24155962340`](https://github.com/gpertea/gffread/tree/05462cdb2bb6025f72a8d93c4797b24155962340/examples):

| Input | SHA-256 |
|---|---|
| `examples/genome.fa` | `016a346a6d4ae34b17bdd93d279fcc022e03c640563c75e81a2a1d1dda2afbe2` |
| `examples/annotation.gff` | `0812cb16c718673924c90b42cfd1fc6691c77cfce56d0ca0f37b93d2ea776ad9` |

The unmodified source has no N runs, so its successful round trip alone does
not test coordinate movement. A controlled derivative replaces three bases
starting at 1-based positions **10, 600, and 1,111,560** on `NT_187562.1` with
`NNN`. These intervals were verified not to overlap any CDS or feature endpoint.
The derivative was written with Biopython 1.88 FASTA output; SHA-256:
`86dab14ea402247cdf2d5b5ec209c7c31092710aa1f788fd1c641d4f440c1ff0`.

Rechecked from the isolated commit candidate, for each target length **0, 4 and 100**, the default error policy accepted all
three edits, and gffread extracted the **same three CDS sequences** as from the
unmodified source. Each output used a distinct FASTA filename to avoid reusing
stale `.fai` indexes; reusing an index after replacing its FASTA caused the initial
independent check to fail and was corrected in the validation procedure.

This is a deterministic transformation check on a small fixed external sample,
not an estimate of the frequency of problematic annotations in user datasets,
a general GFF3 compliance certification, or proof of biological splicing or
protein function. Coordinate-valued custom attributes and biological translation
exceptions remain outside the transformation's guarantees.

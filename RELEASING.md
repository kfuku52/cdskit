# Releasing CDSKIT

Keep `cdskit/__init__.py` on the next semantic version for every change merged
to `master`.

The `Run Tests` workflow validates each push. After it succeeds, the release
workflow checks the version from the exact tested commit:

- Versions whose patch component is nonzero (for example, `0.25.2`) remain
  available from `master`, but do not receive a Git tag or GitHub Release.
- Major and minor versions whose patch component is zero (for example,
  `0.26.0` or `1.0.0`) receive an annotated `<version>` tag and a GitHub
  Release automatically.

Bioconda discovers tagged upstream releases. Consequently, its `cdskit`
recipe is updated for major and minor releases only; patch-only versions are
intentionally not autobumped.

Do not create release tags manually unless recovering the automated workflow.
If recovery is necessary, point the annotated tag at the commit that passed
`Run Tests` and preserve the existing tag format without a `v` prefix.

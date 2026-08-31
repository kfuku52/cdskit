# Releasing CDSKIT

Before pushing changes to `master`, bump `__version__` in `cdskit/__init__.py`
using semantic versioning and record the changes in `CHANGELOG.md`. A batch of
local commits can share one bump; documentation-only pushes also need a bump.
Run the appropriate checks from [TESTING.md](TESTING.md) before pushing.

The `Run Tests` workflow validates each push. After it succeeds, the release
workflow checks the version from the exact tested commit, provided that commit
is still the head of `master`:

- Versions whose patch component is nonzero (for example, `0.25.2`) remain
  available from `master`, but do not receive a Git tag or GitHub Release.
- Major and minor versions whose patch component is zero (for example,
  `0.26.0` or `1.0.0`) receive an annotated `<version>` tag and a GitHub
  Release automatically.

These rules apply to numeric package versions. Named model-asset releases such
as `localize-targeting5-v1` are separate and are not created by this workflow.

Bioconda can discover tagged upstream releases, but publication still depends
on a recipe update, review, and successful downstream build. This repository
does not publish conda packages or PyPI distributions. Patch-only source
versions have no new numeric tag for a downstream autobump. Review the
[Bioconda recipe](https://github.com/bioconda/bioconda-recipes/blob/master/recipes/cdskit/meta.yaml)
for dependency metadata as well as version changes.

If a `X.Y.0` push fails CI and is fixed by a `X.Y.1` push, the successful patch
run does not retroactively tag `X.Y.0` or create a `X.Y.1` release. Do not infer
release availability from `CHANGELOG.md`; check the actual tags and releases.

Do not create release tags manually unless recovering the automated workflow.
If recovery is necessary, point the annotated tag at the commit that passed
`Run Tests` and preserve the existing tag format without a `v` prefix.
Manual workflow dispatch is a recovery path; unlike a `workflow_run` event it
does not automatically verify a successful test run for that SHA. Check the
current `master` commit and its completed `Run Tests` result before dispatching.

When pages under `wiki/` change, publish the corresponding wiki update using
the [documentation maintenance procedure](docs/documentation.md). Pushing this
repository alone does not update GitHub's separate wiki repository.

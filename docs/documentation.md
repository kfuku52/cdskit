# Maintaining documentation

The top-level README is a short installation and command index. User guides
and their images live in `wiki/`; developer/release procedures live in
`TESTING.md` and `RELEASING.md`. Keep detailed model experiments in the clearly
labelled historical wiki page, not in the prediction quick start.

The repository's `wiki/` directory is the source for the published GitHub wiki.
GitHub stores that wiki in **a separate Git repository**. A push to `cdskit`
does not publish the wiki, and web edits to the wiki do not update `wiki/`.
Always reconcile newer wiki edits before copying files in either direction.

## Checking changes

- Compare option names, choices, and defaults with `cdskit COMMAND --help` or
  the relevant helper's `python -m cdskit.MODULE --help`.
- Run complete examples in a temporary directory using the documented input;
  compare sequences, headers, and report columns with the shown output.
- Check local links and images in both the repository and wiki views. Link
  between wiki pages with their public wiki URL when necessary; GitHub wiki
  links and repository-relative Markdown links have different resolution rules.
- Check package requirements against `pyproject.toml` and the actual downstream
  recipe. Do not assume a conda recipe automatically mirrors Python metadata.
- Test published model artifacts separately from synthetic ML tests. Preserve
  checksum and safe-loading requirements; record any model-specific runtime
  constraint instead of narrowing unrelated library dependencies.
- Mark illustrative output and historical metrics explicitly. Accuracy claims
  need dataset/split provenance; a smoke prediction is not an accuracy benchmark.
- Run the relevant standard checks from `TESTING.md`. Documentation changes
  do not require new unit tests that only duplicate their text.

## Publishing the wiki

Work from the main repository root. Use an adjacent checkout so the wiki's
`.git` directory is never copied into the main repository:

```bash
git clone https://github.com/kfuku52/cdskit.wiki.git ../cdskit-wiki
git -C ../cdskit-wiki pull --ff-only
git -C ../cdskit-wiki status --short
```

If the checkout already exists, skip `clone`. Preserve any uncommitted work.
Before copying, compare the trees and reconcile changes made directly in the
wiki since the previous publication:

```bash
diff -ru --exclude=.git wiki ../cdskit-wiki
```

Once the intended page set is reviewed, copy pages/images and inspect the diff:

```bash
cp wiki/*.md ../cdskit-wiki/
mkdir -p ../cdskit-wiki/images
cp wiki/images/*.png ../cdskit-wiki/images/
git -C ../cdskit-wiki diff --check
git -C ../cdskit-wiki diff --stat
git -C ../cdskit-wiki status --short
```

These copy commands do not delete pages. Handle intended renames/deletions
explicitly after review; never replace the wiki wholesale with an older
snapshot. Review untracked new pages as well as the tracked diff, then commit
and push the matching changes in both repositories on their default branches.
Follow `RELEASING.md` for the main repository's version bump and validation.
After publication, check representative public pages and verify that the two
page/image trees match. Record both commit IDs in the change report.

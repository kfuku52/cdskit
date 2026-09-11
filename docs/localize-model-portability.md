# Trusted localization model portability

The `targeting5-perox-deeploc21-et-v1` v1 artifact contains Cython loss globals
under the short module name `_loss`. The trusted-pickle reader resolves its
four known binomial/multinomial class and reconstruction names to
`sklearn._loss._loss`, without installing a global module alias. This reader is
used only after the existing unsafe-loading opt-in; safe loading is unchanged.

scikit-learn 1.9 no longer exports `__pyx_unpickle_CyHalfBinomialLoss`. Its
current `CyHalfBinomialLoss` is reconstructed directly with no arguments. When
the old function is absent, the reader translates the published artifact's
known Cython layout (checksum `238750788`, empty tuple state, exact binomial
class) to that constructor. It rejects other states/classes/checksums and uses
the native reconstructor when one exists. No module globals are modified.
This compatibility path is needed while these legacy model artifacts remain
supported; it can be removed when they are retired or replaced by portable
artifacts.

The regression fixture in `tests/fixtures/localize_pickle` was generated with
scikit-learn 1.5.2. It checks an actual fitted classifier's probabilities,
including missing-value inputs. Additional Linux ARM64 validation compared both
published targeting5 models under 1.5.2 and 1.9.0 using 32 synthetic sequences,
DNA and protein input, and all three organism groups. All 384 emitted rows,
including six numeric prediction columns and labels, matched (maximum absolute
numeric difference zero). See `docs/localize-portability-validation/` for the
inputs, comparison script and results.

This is a verified compatibility case for these artifacts, not general support
for arbitrary cross-version sklearn pickles. The sklearn version warnings
remain visible. Safe loading and the existing trusted-loading opt-in are
unchanged; no dependency upper bound or model retraining is introduced.

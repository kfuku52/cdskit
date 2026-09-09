# Trusted localization model portability

The `targeting5-perox-deeploc21-et-v1` v1 artifact contains Cython loss globals
under the short module name `_loss`. The trusted-pickle reader resolves its
four known binomial/multinomial class and reconstruction names to
`sklearn._loss._loss`, without installing a global module alias. This reader is
used only after the existing unsafe-loading opt-in; safe loading is unchanged.

Module-name portability does not make sklearn estimator versions compatible.
This artifact was serialized with scikit-learn 1.5.2. Reproduction with 1.5.2
works; 1.9.0 no longer exports its binomial reconstruction function. Use a
separate environment matching the training version for this artifact until
an updated model is exported and validated for the newer runtime. Do not infer
prediction equivalence across sklearn versions from successful unpickling.

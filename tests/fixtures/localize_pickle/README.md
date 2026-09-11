# Legacy sklearn binomial model fixture

`generate.py` creates this small, deterministic HistGradientBoostingClassifier
and its probability oracle with scikit-learn 1.5.2 (Python 3.12, NumPy 1.26.4).
It uses synthetic numeric inputs only, including missing values. Run it in a
separate 1.5.2 reference environment; it overwrites the two fixture files.

The trusted protocol-2 pickle retains the generated Cython binomial-loss
reconstructor absent from sklearn 1.9. The JSON contains the reference inputs
and probabilities, with JSON null converted to NaN by the regression test.
The fixture checks prediction equivalence after restoring the legacy loss, not
just successful deserialization. It is not a published localization model.

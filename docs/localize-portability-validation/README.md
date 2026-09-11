# Published-model compatibility, 2026-09-11

Compared the unmodified published `targeting5-v1` and
`targeting5-perox-deeploc21-et-v1` artifacts with the updated CDSKIT trusted
loader under scikit-learn 1.5.2 and 1.9.0. The model hashes are respectively
`ddaeab7093533a213ee58117b70ad0f45b0c126cf82c77df32e369eaff2beeb2` and
`d0998df8819d975b4392342ab78dccc0dd95cf301e4d2df8f38c73d0b5aab445`.

Environment: GeneGalleon Linux ARM64 Docker, Python 3.12.14, NumPy 1.26.4,
PyTorch 2.13.0. Only the sklearn import path differs between the two runs.
The reference package was installed with
`python -m pip install --no-deps --target /tmp/sklearn152 scikit-learn==1.5.2`;
the main environment retained sklearn 1.9.0. CDSKIT source was mounted at
`/cdskit`. No model was retrained or altered, and dependency metadata is unpinned.

`compare.py` generates 32 deterministic synthetic proteins and equivalent CDS,
then tests both models and plant/non_plant/unknown settings. It compares every
reported column, using absolute tolerance 1e-14 for numeric values. All 12
comparisons (384 rows; six numeric prediction columns per row) passed, with
maximum observed absolute difference zero. These data establish compatibility
on the fixture, not accuracy of the localization predictions or universal
compatibility with other sklearn versions. Native sklearn version warnings
remain visible.

The TSVs retain both runtimes' outputs. `comparison.json` is the summary;
`proteins.fa` and `cds.fa` are the inputs. `tests.log` records 84 passing loader,
model download/selection and localization integration tests in this runtime.

To repeat the comparison, mount a writable output directory as `/audit`, place
the checksum-verified registered model cache at `/audit/fix-model-cache`, mount
the updated source at `/cdskit`, and install the 1.5.2 reference as above.
Run this script with the container's Python; it writes
`/audit/localize-compatibility`. Model weights are deliberately not duplicated
in this repository.

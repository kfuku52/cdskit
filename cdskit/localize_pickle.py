"""Pickle reader for trusted localization models built with Cython sklearn.

Some sklearn builds serialized loss classes with the short Cython module name
``_loss``. Resolve those exact globals to their canonical module without
installing a process-wide module alias. This is not a safe unpickler; callers
must opt into trusted pickle loading exactly as for the standard pickle module.

scikit-learn 1.9 also replaced the generated binomial-loss pickle reconstructor
with a direct constructor. Read the known, empty legacy state through that
constructor; never discard state from an unrecognized serialized layout.
"""

import pickle

_LOSS_GLOBALS = frozenset(
    {
        "CyHalfBinomialLoss",
        "CyHalfMultinomialLoss",
        "__pyx_unpickle_CyHalfBinomialLoss",
        "__pyx_unpickle_CyHalfMultinomialLoss",
    }
)


def _restore_legacy_binomial_loss(cls, checksum, state):
    from sklearn._loss._loss import CyHalfBinomialLoss

    if cls is not CyHalfBinomialLoss or checksum != 238750788 or state != ():
        raise pickle.UnpicklingError("Unsupported legacy binomial loss state")
    return CyHalfBinomialLoss()


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "_loss" and name in _LOSS_GLOBALS:
            module = "sklearn._loss._loss"
        if (
            module == "sklearn._loss._loss"
            and name == "__pyx_unpickle_CyHalfBinomialLoss"
        ):
            from sklearn._loss import _loss

            if not hasattr(_loss, name):
                return _restore_legacy_binomial_loss
        return super().find_class(module, name)


load = pickle.load
loads = pickle.loads
dump = pickle.dump
dumps = pickle.dumps
Pickler = pickle.Pickler

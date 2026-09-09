"""Pickle reader for trusted localization models built with Cython sklearn.

Some sklearn builds serialized loss classes with the short Cython module name
``_loss``. Resolve those exact globals to their canonical module without
installing a process-wide module alias. This is not a safe unpickler; callers
must opt into trusted pickle loading exactly as for the standard pickle module.
"""
import pickle

_LOSS_GLOBALS = frozenset({
    "CyHalfBinomialLoss", "CyHalfMultinomialLoss",
    "__pyx_unpickle_CyHalfBinomialLoss",
    "__pyx_unpickle_CyHalfMultinomialLoss",
})


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "_loss" and name in _LOSS_GLOBALS:
            module = "sklearn._loss._loss"
        return super().find_class(module, name)


load = pickle.load
loads = pickle.loads
dump = pickle.dump
dumps = pickle.dumps
Pickler = pickle.Pickler

"""Pairwise Goldman-Yang likelihood with corrected empirical codon frequencies."""

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any

import numpy as np
from scipy.linalg import eigh, expm, expm_frechet
from scipy.optimize import minimize, root
from scipy.special import softmax

from cdskit.dnds import _encode_batch, _estimate_encoded, _geometry

_MIN_RATE = 1e-6
_MAX_RATE = 10.0
_LOG_KAPPA_BOUNDS = (float(np.log(0.05)), float(np.log(60.0)))
_BOUNDS = ((_MIN_RATE, _MAX_RATE), (_MIN_RATE, _MAX_RATE), _LOG_KAPPA_BOUNDS)
_LOG_RATE_BOUNDS = (float(np.log(_MIN_RATE)), float(np.log(_MAX_RATE)))
_PENALIZED_BOUNDS = (_LOG_RATE_BOUNDS, _LOG_RATE_BOUNDS, _LOG_KAPPA_BOUNDS)
_LOG_RATIO_CENTER = float(np.log(0.5))
_LOG_RATIO_VARIANCE = 1.8**2
_MODEL = "GY94"
_PENALIZED_MODEL = "GY94_regularized"
_FREQUENCY_TOLERANCE = 1e-10


@lru_cache(maxsize=32)
def _likelihood_geometry(code):
    geometry = _geometry(code)
    sense = np.flatnonzero(geometry["sense"])
    categories = np.where(geometry["synonymous"], 0, 2) + geometry["transition"]
    basis = np.zeros((4, 64, 64), dtype=bool)
    basis[categories, np.arange(64)[:, None], geometry["neighbors"]] = True
    return (
        sense,
        geometry["bases"][sense],
        geometry["position_counts"][sense],
        basis[:, sense[:, None], sense],
    )


def _corrected_frequencies(marginals, bases, positions):
    # Iterative proportional fitting enforces the empirical nucleotide
    # marginals on the sense codons, with stop codons excluded throughout.
    pi = np.prod(marginals[np.arange(3), bases], axis=1)
    pi /= pi.sum()
    for _ in range(32):
        for position in range(3):
            current = pi @ positions[:, position * 4 : position * 4 + 4]
            pi *= (marginals[position] / current)[bases[:, position]]
        if (
            np.max(np.abs((pi @ positions).reshape(3, 4) - marginals))
            <= _FREQUENCY_TOLERANCE
        ):
            return pi
    return _solve_frequency_marginals(marginals, positions, pi)


def _solve_frequency_marginals(marginals, positions, initial):
    # Iterative scaling slows when a stop is excluded from an otherwise
    # abundant combination. Continue in the same log-linear family using
    # the covariance of the nucleotide indicators as the Jacobian.
    reference = np.argmax(marginals, axis=1, keepdims=True)
    free = (np.arange(4) != reference).ravel()
    features = positions[:, free]
    target = marginals.ravel()[free]
    log_initial = np.log(initial)

    def residual_and_jacobian(delta):
        pi = softmax(log_initial + features @ delta)
        expected = pi @ features
        covariance = features.T @ (pi[:, None] * features) - np.outer(
            expected, expected
        )
        return expected - target, covariance

    fit = root(residual_and_jacobian, np.zeros(len(target)), jac=True, method="hybr")
    pi = softmax(log_initial + features @ fit.x)
    # Certify the full set of marginals, including the three reference bases.
    error = np.max(np.abs(pi @ positions - marginals.ravel()))
    if error <= _FREQUENCY_TOLERANCE and np.all(pi > 0):
        return pi
    return None


@dataclass(frozen=True)
class _PairLikelihood:
    matrices: np.ndarray
    neutral_rates: np.ndarray
    coordinates: tuple
    observed: np.ndarray

    def objective(self, parameters, *, direct=False):
        ds, dn, log_kappa = parameters
        kappa = np.exp(log_kappa)
        rates = self.neutral_rates
        # Define each distance relative to the neutral model's site fraction.
        # At dS=dN=1 the expected number of events per codon is three.
        transition_rate = kappa * (rates[1] + rates[3])
        neutral_rate = rates[0] + rates[2] + transition_rate
        scale = 3 / neutral_rate
        syn_ts, non_ts = self.matrices[1] * kappa, self.matrices[3] * kappa
        syn = (self.matrices[0] + syn_ts) * scale
        non = (self.matrices[2] + non_ts) * scale
        generator = syn * ds + non * dn
        log_kappa_derivative = scale * (syn_ts * ds + non_ts * dn) - generator * (
            transition_rate / neutral_rate
        )
        evaluate = _direct_log_likelihood if direct else _log_likelihood
        return evaluate(
            generator,
            (syn, non, log_kappa_derivative),
            self.coordinates,
            self.observed,
        )


def _log_likelihood(generator, derivatives, coordinates, observed):
    # Reversibility makes the generator similar to a symmetric matrix.
    # The similarity factors in the transition probabilities are constant
    # during optimization because equilibrium frequencies are held fixed.
    eigenvalues, vectors = eigh(generator, check_finite=False, driver="evd")
    exponentials = np.exp(eigenvalues)
    transition = (vectors * exponentials) @ vectors.T
    probabilities = transition[coordinates]
    if np.any(probabilities <= 1e-12):
        # Near zero, small off-diagonal probabilities can disappear through
        # cancellation in the eigendecomposition.
        return _direct_log_likelihood(generator, derivatives, coordinates, observed)
    separation = np.abs(eigenvalues[:, None] - eigenvalues[None, :])
    quotient = np.divide(
        -np.expm1(-separation),
        separation,
        out=np.ones_like(separation),
        where=separation > 0,
    )
    divided_difference = (
        np.maximum(exponentials[:, None], exponentials[None, :]) * quotient
    )
    weights = np.zeros_like(transition)
    weights[coordinates] = -observed / probabilities
    adjoint = (
        vectors @ ((vectors.T @ weights @ vectors) * divided_difference) @ vectors.T
    )
    gradient = np.array([np.sum(adjoint * derivative) for derivative in derivatives])
    return -np.dot(observed, np.log(probabilities)), gradient


def _direct_log_likelihood(generator, derivatives, coordinates, observed):
    probabilities = expm(generator)[coordinates]
    if np.any(probabilities <= 0):
        return np.inf, np.zeros(len(derivatives))
    weights = observed / probabilities
    gradient = -np.array(
        [
            np.dot(
                weights,
                expm_frechet(generator, derivative, compute_expm=False)[coordinates],
            )
            for derivative in derivatives
        ]
    )
    return -np.dot(observed, np.log(probabilities)), gradient


def _shrink_codon_frequencies(counts, totals, target):
    sites = counts.sum()
    if sites <= 1:
        return target
    empirical = totals / (2 * sites)
    spread = np.sum((empirical - target) ** 2)
    if spread == 0:
        return target
    # Each independent codon column contributes 0, 1/2 or 1 to a codon's
    # empirical frequency. Use the diagonal counts to retain dependence
    # between the two aligned endpoints in the variance of the sample mean.
    second_moment = (totals + 2 * counts.diagonal()) / (4 * sites)
    variance = np.sum(np.maximum(second_moment - empirical**2, 0)) / (sites - 1)
    # Keep a positive target contribution even for zero sample variance.
    weight = np.clip(variance / spread, 1 / (sites + 1), 1)
    return (1 - weight) * empirical + weight * target


def _prepare_likelihood(counts, code, *, frequency_model="cf3x4"):
    sense, bases, positions, basis = _likelihood_geometry(code)
    counts = counts.reshape(64, 64)[sense[:, None], sense]
    totals = counts.sum(axis=0) + counts.sum(axis=1)
    composition = totals @ positions
    # Positive marginals keep all sense states in the fitted model for
    # short alignments where some bases are unobserved.
    marginals = composition.reshape(3, 4) + 0.5
    marginals /= marginals.sum(axis=1, keepdims=True)
    pi = _corrected_frequencies(marginals, bases, positions)
    if pi is None:
        return None
    if frequency_model == "empirical-shrinkage":
        pi = _shrink_codon_frequencies(counts, totals, pi)
    square_root = np.sqrt(pi)
    matrices = basis * square_root[None, :, None] * square_root[None, None, :]
    outgoing = (basis * pi[None, None, :]).sum(axis=2)
    diagonal = np.arange(len(pi))
    matrices[:, diagonal, diagonal] = -outgoing
    # The two alignment orientations have the same objective. Combine the
    # off-diagonal observations so swapping the input rows is deterministic.
    observed = counts + counts.T
    np.fill_diagonal(observed, counts.diagonal())
    coordinates = np.nonzero(np.triu(observed))
    return _PairLikelihood(matrices, outgoing @ pi, coordinates, observed[coordinates])


def _initial_parameters(row):
    ds = row["dS"] if row["dS"] is not None else 0.4
    dn = row["dN"] if row["dN"] is not None else 0.15
    return np.array(
        [
            np.clip(ds, 0.002, 5.0),
            np.clip(dn, 0.002, 5.0),
            np.log(np.clip(row["kappa"] or 3, 0.3, 12)),
        ]
    )


def _penalized_objective(likelihood, parameters, *, direct=False):
    log_ds, log_dn, log_kappa = parameters
    ds, dn = np.exp(parameters[:2])
    loss, gradient = likelihood.objective(np.array([ds, dn, log_kappa]), direct=direct)
    delta = log_dn - log_ds - _LOG_RATIO_CENTER
    adjust = delta / _LOG_RATIO_VARIANCE
    gradient = gradient * np.array([ds, dn, 1.0]) + np.array([-adjust, adjust, 0.0])
    return loss + delta**2 / (2 * _LOG_RATIO_VARIANCE), gradient


def _fit_likelihood(counts, code, row, *, regularized=False, frequency_model="cf3x4"):
    model = _PENALIZED_MODEL if regularized else _MODEL
    if frequency_model == "empirical-shrinkage":
        model += "_empirical_shrinkage"
    # An identical pair has zero distances and no identifiable kappa.
    if counts.reshape(64, 64).trace() == row["retained_codons"]:
        return dict(dS=0.0, dN=0.0, dS_model=model, dN_model=model, status="identical")

    likelihood = _prepare_likelihood(counts, code, frequency_model=frequency_model)
    if likelihood is None:
        return {"status": "frequency_not_converged"}
    initial = _initial_parameters(row)
    objective, bounds = likelihood.objective, _BOUNDS
    if regularized:
        initial[:2] = np.log(initial[:2])
        objective, bounds = partial(_penalized_objective, likelihood), _PENALIZED_BOUNDS
    minimize_fit = partial(
        minimize,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        # Boundary fits can require more line-search steps than SciPy's default.
        options={"maxiter": 100, "maxls": 60, "ftol": 1e-11},
    )
    fit = minimize_fit(objective, initial)
    if not fit.success and fit.status == 2 and np.all(np.isfinite(fit.x)):
        # Cancellation in small spectral probabilities can stall the line
        # search. Refit the same objective with scaling and squaring throughout.
        fit = minimize_fit(partial(objective, direct=True), fit.x)
    if not fit.success or not np.isfinite(fit.fun) or not np.all(np.isfinite(fit.x)):
        return {"status": "not_converged"}
    ds, dn, log_kappa = (float(value) for value in fit.x)
    if regularized:
        ds, dn = float(np.exp(ds)), float(np.exp(dn))
    kappa = float(np.exp(log_kappa))
    rates = likelihood.neutral_rates
    syn = rates[0] + kappa * rates[1]
    non = rates[2] + kappa * rates[3]
    total_sites = 3 * row["retained_codons"]
    syn_sites = float(total_sites * syn / (syn + non))

    status = "ok"
    estimates: dict[str, Any] = dict(
        kappa=kappa, S=syn_sites, N=total_sites - syn_sites
    )
    if min(abs(log_kappa - endpoint) for endpoint in _LOG_KAPPA_BOUNDS) <= 1e-7:
        status = "kappa_boundary"
    for name, value in (("dS", ds), ("dN", dn)):
        estimates[f"{name}_model"] = model
        if value >= _MAX_RATE - 1e-6:
            estimates[name] = None
            estimates[f"{name}_diagnostic"] = value
            status = "rate_boundary"
        else:
            estimates[name] = estimates[f"{name}_diagnostic"] = (
                0.0 if value <= _MIN_RATE * 1.000001 else value
            )
    return {**estimates, "status": status}


def estimate_ml_batch(
    pairs: list[tuple[str, str]],
    code: int,
    *,
    regularized: bool = False,
    frequency_model: str = "cf3x4",
) -> list[dict[str, Any]]:
    geometry = _geometry(code)
    # Use one orientation even when initializing from YN00's floating-point
    # calculations. Reversing the two input rows then follows the same fit.
    ordered = [
        (first, second) if first <= second else (second, first)
        for first, second in pairs
    ]
    encoded = _encode_batch(ordered, geometry)
    rows = _estimate_encoded(encoded, geometry)
    for row, (counts, _) in zip(rows, encoded, strict=True):
        if counts is None:
            continue
        fit = _fit_likelihood(
            counts, code, row, regularized=regularized, frequency_model=frequency_model
        )
        # YN00 estimates initialize the optimizer only. Avoid returning any
        # of their values as ML estimates if the likelihood fit fails.
        row.update(
            dS=None,
            dN=None,
            dS_diagnostic=None,
            dN_diagnostic=None,
            kappa=None,
            S=None,
            N=None,
            dS_model="not_estimated",
            dN_model="not_estimated",
            kappa_defaulted=False,
        )
        row.update(fit)
    return rows

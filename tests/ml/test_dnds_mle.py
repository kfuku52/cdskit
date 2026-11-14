"""Verify the opt-in codon likelihood and its published distance units."""

import csv
from functools import partial
import os
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cdskit import dnds, dnds_mle
from cdskit.cli import main
from cdskit.codonutil import get_forward_table


def divergent_pair():
    rng = random.Random(81224)
    sense = sorted(get_forward_table(1))
    first = rng.choices(sense, k=350)
    second = [rng.choice(sense) if rng.random() < 0.16 else codon for codon in first]
    return "".join(first), "".join(second)


def test_gy94_equal_codon_frequencies_match_paml_codeml(monkeypatch):
    # PAML 4.10.10, runmode=-2, CodonFreq=0, with kappa and omega free.
    # Fix only the stationary frequencies to compare the same objective.
    monkeypatch.setattr(
        dnds_mle,
        "_corrected_frequencies",
        lambda marginals, bases, positions: np.full(len(bases), 1 / len(bases)),
    )
    row = dnds.estimate_pairs([divergent_pair()], method="gy94")[0]
    assert row["status"] == "ok"
    assert row["dS"] == pytest.approx(0.1370, abs=0.000051)
    assert row["dN"] == pytest.approx(0.1317, abs=0.000051)
    assert row["kappa"] == pytest.approx(0.73591, abs=0.00001)
    assert row["S"] == pytest.approx(257.8, abs=0.051)
    assert row["S"] + row["N"] == pytest.approx(1050)


@pytest.mark.parametrize("regularized", [False, True])
@pytest.mark.parametrize("direct", [False, True])
def test_gy94_analytic_gradient_matches_central_differences(regularized, direct):
    counts, _ = dnds._encode_batch([divergent_pair()], dnds._geometry(1))[0]
    model = dnds_mle._prepare_likelihood(counts, 1)
    assert model is not None
    parameters = np.array([0.23, 0.35, np.log(2.5)])
    objective = partial(model.objective, direct=direct)
    if regularized:
        parameters[:2] = np.log(parameters[:2])
        objective = partial(dnds_mle._penalized_objective, model, direct=direct)
    offsets = np.eye(3) * 1e-6
    gradient = [
        (objective(parameters + step)[0] - objective(parameters - step)[0]) / 2e-6
        for step in offsets
    ]
    assert objective(parameters)[1] == pytest.approx(gradient, abs=3e-6)


@pytest.mark.parametrize("method", ["gy94", "gy94-regularized"])
def test_gy94_refits_the_same_likelihood_after_a_line_search_failure(
    monkeypatch, method
):
    pairs = [divergent_pair()]
    expected = dnds.estimate_pairs(pairs, method=method)[0]
    minimize = dnds_mle.minimize
    starts, results = [], []

    def stop_first_fit(objective, initial, **options):
        starts.append(initial.copy())
        result = minimize(objective, initial, **options)
        results.append(result)
        if len(results) == 1:
            result.success = False
            result.status = 2
        else:
            assert objective.keywords == {"direct": True}
        return result

    monkeypatch.setattr(dnds_mle, "minimize", stop_first_fit)
    actual = dnds.estimate_pairs(pairs, method=method)[0]
    assert len(starts) == 2 and results[1].success
    assert starts[1] == pytest.approx(results[0].x)
    assert actual["status"] == expected["status"] == "ok"
    for field in ("dS", "dN", "kappa"):
        assert actual[field] == pytest.approx(expected[field], abs=2e-6)


@pytest.mark.parametrize(
    ("ds_gradient", "loss", "accepted"),
    [(-0.1, 2.0, True), (0.1, 2.0, False), (-0.1, 4.0, False)],
)
def test_gy94_iteration_retry_checks_projected_gradient_and_loss(
    ds_gradient, loss, accepted
):
    previous = SimpleNamespace(x=np.array([0.2, 0.3, 1.0]))
    fit = SimpleNamespace(
        x=np.array([10.0, 0.2, 1.0]),
        jac=np.array([ds_gradient, 0.0, 0.0]),
        fun=loss,
        success=True,
    )

    def objective(parameters, *, direct=False):
        assert direct
        return 3.0, np.zeros(3)

    def minimize_retry(target, initial, *, options):
        assert initial is previous.x
        assert target(initial)[0] == 3.0
        assert options["maxiter"] > 100
        return fit

    actual = dnds_mle._resume_likelihood_fit(
        minimize_retry, objective, previous, dnds_mle._BOUNDS
    )
    assert actual is (fit if accepted else previous)


def test_gy94_iteration_retry_recovers_a_compositionally_sparse_pair(monkeypatch):
    pairs = [("TCGGGA" * 600, "TCGGGA" * 599 + "GGAGGA")]
    minimize = dnds_mle.minimize
    retries = []

    def exhaust_initial_budget(objective, initial, **kwargs):
        if not retries:
            retries.append(kwargs["options"]["maxiter"])
            # Force the iteration-limit branch independently of BLAS details.
            fit = minimize(objective, initial, **kwargs)
            fit.success = False
            fit.status = 1
            return fit
        retries.append(kwargs["options"]["maxiter"])
        return minimize(objective, initial, **kwargs)

    monkeypatch.setattr(dnds_mle, "minimize", exhaust_initial_budget)
    [row] = dnds.estimate_pairs(
        pairs, 2, method="gy94", frequency_model="empirical-shrinkage"
    )
    assert retries == [100, 300]
    assert row["status"] == "rate_boundary"
    assert row["dS"] is None
    assert row["dN"] == pytest.approx(0.0515, abs=2e-5)


@pytest.mark.parametrize(
    ("method", "frequency", "label"),
    [
        ("gy94", None, "GY94_CF3x4_pairwise"),
        ("gy94-regularized", "cf3x4", "GY94_CF3x4_log_ratio_penalty"),
        ("gy94", "empirical-shrinkage", "GY94_empirical_shrinkage_pairwise"),
        (
            "gy94-regularized",
            "empirical-shrinkage",
            "GY94_empirical_shrinkage_log_ratio_penalty",
        ),
    ],
)
def test_gy94_cli_preserves_pair_order_and_missing_status(
    tmp_path, method, frequency, label
):
    pairs, output = tmp_path / "pairs.tsv", tmp_path / "gy94.tsv"
    first, second = divergent_pair()
    pairs.write_text(
        "pair_id\tsequence_1\tsequence_2\n"
        f"forward\t{first}\t{second}\n"
        f"reverse\t{second}\t{first}\n"
        "terminal\tATGTAA?.A\tATGAAA---\n"
        "internal\tATGTARGCT\tATGAAAGCT\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "dnds",
                "--pairs_file",
                str(pairs),
                "--out_file",
                str(output),
                "--method",
                method,
                "--threads",
                "2",
                *(["--frequency_model", frequency] if frequency else []),
            ]
        )
        == 0
    )
    with output.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["pair_id"] for row in rows] == [
        "forward",
        "reverse",
        "terminal",
        "internal",
    ]
    assert all(row["method"] == label for row in rows)
    assert [row["status"] for row in rows] == ["ok", "ok", "identical", "internal_stop"]
    assert all(rows[0][key] == rows[1][key] for key in ("dN", "dS", "kappa"))
    assert rows[2]["dS"] == "0.0" and rows[2]["retained_codons"] == "1"
    assert rows[3]["dS"] == rows[3]["dN"] == ""


def test_codon_frequency_shrinkage_keeps_unobserved_codons_positive():
    counts = np.array([[0, 8, 0], [8, 0, 0], [0, 0, 0]])
    target = np.array([0.65, 0.25, 0.10])
    frequencies = dnds_mle._shrink_codon_frequencies(
        counts, counts.sum(axis=0) + counts.sum(axis=1), target
    )
    expected = (16 * np.array([0.5, 0.5, 0]) + target) / 17
    assert frequencies == pytest.approx(expected)
    assert frequencies.sum() == pytest.approx(1) and np.all(frequencies > 0)


@pytest.mark.parametrize(
    ("code", "first_codon", "second_codon"),
    [(1, "CAA", "TCA"), (2, "CGG", "ACA"), (22, "ACA", "TCG")],
)
def test_cf3x4_matches_marginals_near_a_structural_zero(
    code, first_codon, second_codon
):
    first = (first_codon + second_codon) * 120
    second = (first_codon + second_codon) * 119 + second_codon * 2
    _, bases, positions, _ = dnds_mle._likelihood_geometry(code)
    expected = np.full((3, 4), 0.5)
    for codon, count in ((first_codon, 239), (second_codon, 241)):
        expected[np.arange(3), [dnds.BASES.index(base) for base in codon]] += count
    expected /= 482

    frequencies = dnds_mle._corrected_frequencies(expected, bases, positions)
    assert frequencies is not None
    assert np.all(frequencies > 0) and frequencies.sum() == pytest.approx(1)
    assert (frequencies @ positions).reshape(3, 4) == pytest.approx(expected, abs=1e-10)

    row = dnds.estimate_pairs([(first, second)], code, method="gy94-regularized")[0]
    assert row["status"] in {"ok", "kappa_boundary"}
    assert row["retained_codons"] == 240
    assert row["dS"] > 0 and row["dN"] > 0


def test_yn00_rejects_gy94_frequency_option():
    with pytest.raises(ValueError, match="requires --method gy94"):
        dnds.estimate_pairs([], frequency_model="empirical-shrinkage")


def test_gy94_regularization_shrinks_an_observed_ratio_toward_half():
    pairs = [divergent_pair()]
    rows = [
        dnds.estimate_pairs(pairs, method=method)[0]
        for method in ("gy94", "gy94-regularized")
    ]
    assert all(row["status"] == "ok" for row in rows)
    likelihood, regularized = (row["dN"] / row["dS"] for row in rows)
    assert 0.5 < regularized < likelihood


@pytest.mark.subprocess
def test_gy94_process_workers_preserve_output_bytes_and_fail_atomically(tmp_path):
    pairs = tmp_path / "pairs.tsv"
    first, second = divergent_pair()
    entries = [f"row_{i}\tATGGCT\tATGGCT\n" for i in range(256)]
    entries[1] = f"forward\t{first}\t{second}\n"
    entries[170] = f"reverse\t{second}\t{first}\n"
    entries[254] = "terminal\tATGTAR?.A\tATGAAA---\n"
    header = "pair_id\tsequence_1\tsequence_2\n"
    pairs.write_text(header + "".join(entries), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "cdskit.cli",
        "dnds",
        "--pairs_file",
        str(pairs),
        "--method",
        "gy94-regularized",
        "--frequency_model",
        "empirical-shrinkage",
    ]
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    serial, parallel = tmp_path / "serial.tsv", tmp_path / "parallel.tsv"
    for workers, output in ((1, serial), (2, parallel)):
        subprocess.run(
            [*command, "--threads", str(workers), "--out_file", str(output)],
            check=True,
            capture_output=True,
            env=env,
            timeout=30,
        )
    expected = serial.read_bytes()
    assert parallel.read_bytes() == expected

    entries[254] = "bad\tATGG\tATGG\n"
    pairs.write_text(header + "".join(entries), encoding="utf-8")
    failed = subprocess.run(
        [*command, "--threads", "2", "--out_file", str(parallel)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert failed.returncode == 1 and "divisible by three" in failed.stderr
    assert parallel.read_bytes() == expected

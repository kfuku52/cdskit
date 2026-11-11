"""Verify the opt-in codon likelihood and its published distance units."""

import csv
import random

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


def test_gy94_analytic_gradient_matches_central_differences():
    counts, _ = dnds._encode_batch([divergent_pair()], dnds._geometry(1))[0]
    model = dnds_mle._prepare_likelihood(counts, 1)
    assert model is not None
    parameters = np.array([0.23, 0.35, np.log(2.5)])
    offsets = np.eye(3) * 1e-6
    gradient = [
        (model.objective(parameters + step)[0] - model.objective(parameters - step)[0])
        / 2e-6
        for step in offsets
    ]
    assert model.objective(parameters)[1] == pytest.approx(gradient, abs=3e-6)


def test_gy94_cli_preserves_pair_order_and_missing_status(tmp_path):
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
                "gy94",
                "--threads",
                "2",
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
    assert all(row["method"] == "GY94_CF3x4_pairwise" for row in rows)
    assert [row["status"] for row in rows] == ["ok", "ok", "identical", "internal_stop"]
    assert all(rows[0][key] == rows[1][key] for key in ("dN", "dS", "kappa"))
    assert rows[2]["dS"] == "0.0" and rows[2]["retained_codons"] == "1"
    assert rows[3]["dS"] == rows[3]["dN"] == ""

"""Guard float64 estimates and decisions when the contraction backend changes."""

import csv
import random
from types import SimpleNamespace

import pytest

from cdskit.codonutil import get_forward_table
from cdskit.dnds import dnds_main, estimate_pairs


@pytest.mark.parametrize(
    "code,expected",
    [
        (
            1,
            [
                (0.0027985272248435313, 0.0144574604365993, 179.04427736165047),
                (1.104664596677364, 2.777784332768069, 199.73489499394196),
            ],
        ),
        (
            2,
            [
                (0.008714938495391233, 0.012060119262745889, 173.26719939290297),
                (1.396244208973498, 2.590156414575106, 196.53351221709107),
            ],
        ),
        (
            27,
            [
                (0.0, 0.004463868068263169, 183.44709820993964),
                (2.146930819442661, 1.6859934779693426, 183.59657409191848),
            ],
        ),
        (
            28,
            [
                (0.0, 0.004463868068263169, 183.44709820993964),
                (2.146930819442661, 1.6859934779693426, 183.59657409191848),
            ],
        ),
        (
            31,
            [
                (0.0, 0.0044048490341946946, 177.44722099483482),
                (1.7955184427029587, 1.6779499661308068, 183.58019503954804),
            ],
        ),
    ],
)
def test_low_and_high_divergence_keep_pre_matmul_estimates(code, expected):
    # Frozen pre-change float64 values supplement the independent PAML tests.
    # The tolerance is the existing contraction-review tolerance, not a change
    # to PAML's separate four-decimal reference precision.
    sense = sorted(get_forward_table(code))
    pairs = []
    for seed, size, change in ((30, 211, 0.01), (33, 232, 0.9)):
        rng = random.Random(seed)
        first = [rng.choice(sense) for _ in range(size)]
        second = [
            rng.choice(sense) if rng.random() < change else codon for codon in first
        ]
        pairs.append(("".join(first), "".join(second)))
    for pair, row, reference in zip(
        pairs, estimate_pairs(pairs, code), expected, strict=True
    ):
        assert row["status"] == "ok"
        assert row["excluded_codons"] == 0
        assert row["retained_codons"] == len(pair[0]) // 3
        assert (row["dS"], row["dN"], row["S"]) == pytest.approx(
            reference, abs=1e-10, rel=1e-12
        )
        assert row["S"] + row["N"] == pytest.approx(len(pair[0]), abs=1e-10, rel=1e-12)


def test_synonymous_correction_boundaries_keep_na_and_report_cells(tmp_path):
    forward = get_forward_table(1)
    codons = sorted(forward)
    alternative = {
        codon: next(
            (
                other
                for other in codons
                if other != codon and forward[other] == forward[codon]
            ),
            codon,
        )
        for codon in codons
    }
    first = [codon for codon in codons for _ in range(8)]
    changed_counts = (32, 33, 218, 219, 224, 225, 255, 256)
    models = ("JC69", "F84", "F84", "K80", "K80", "JC69", "JC69", "saturated")
    pairs = [
        (
            "".join(first),
            "".join(
                alternative[codon] if i < changed else codon
                for i, codon in enumerate(first)
            ),
        )
        for changed in changed_counts
    ]
    rows = estimate_pairs(pairs)
    assert [row["dS_model"] for row in rows] == list(models)
    assert all(row["status"] == "ok" and row["dS"] > 0 for row in rows[:-1])
    assert rows[-1]["status"] == "saturated"
    assert rows[-1]["dS"] is None
    assert rows[-1]["dS_diagnostic"] > 0

    source, output = tmp_path / "pairs.tsv", tmp_path / "result.tsv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("pair_id", "sequence_1", "sequence_2"))
        writer.writerows((str(i), a, b) for i, (a, b) in enumerate(pairs))
    dnds_main(
        SimpleNamespace(
            pairs_file=str(source), outfile=str(output), codontable=1, threads=1
        )
    )
    with output.open(encoding="utf-8", newline="") as handle:
        report = list(csv.DictReader(handle, delimiter="\t"))
    assert report[-1]["dS"] == ""
    for row, cells in zip(rows, report, strict=True):
        assert cells["dS_model"] == row["dS_model"]
        assert cells["dS_diagnostic"] == str(row["dS_diagnostic"])

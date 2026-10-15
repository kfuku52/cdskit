"""Compatibility checks for optimized summaries and bounded diagnostics."""

from contextlib import redirect_stderr
import hashlib
import io
from itertools import product
import json
import re
from types import SimpleNamespace

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
import pytest

from cdskit import backtrim
from cdskit.localize_evaluation import (
    cluster_bootstrap,
    paired_cluster_bootstrap,
    dataset_digest,
)
from cdskit.deeploc_benchmark import compute_multilabel_metrics
from cdskit.localize_probabilities import (
    normalize_class_probabilities,
    LOCALIZATION_CLASSES,
)
from cdskit.rmseq import problematic_rate
from cdskit.trimcodon import summarize_codon_site, summarize_codon_sites

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("code", [1, 2, 27, 28, 31])
def test_bulk_codon_summary_matches_scalar_iupac(code):
    sequence = "".join(
        "".join(codon) for codon in product("ACGTRYSWKMBDHVNX-?.", repeat=3)
    )
    rows = [sequence, sequence.lower()]
    expected = [
        summarize_codon_site(rows, site, code) for site in range(len(sequence) // 3)
    ]
    assert summarize_codon_sites(rows, code) == expected
    # Direct callers retain the scalar semantics outside the aligned DNA domain.
    for rows in (["ATGAA", "ATZAA"], ["ATG", "AT"], ["AZG"], []):
        sites = len(rows[0]) // 3 if rows else 0
        assert summarize_codon_sites(rows, code) == [
            summarize_codon_site(rows, site, code) for site in range(sites)
        ]


@pytest.mark.parametrize("unknown", [0, 0.4, 1])
@pytest.mark.parametrize("labels", [["a", "b", "c"], ["a", "a", "c"], []])
def test_bootstrap_summaries_match_row_resampling(unknown, labels):
    rng = np.random.default_rng(43)
    target = rng.integers(2, size=(37, len(labels))).astype(float)
    target[rng.random(target.shape) < unknown] = np.nan
    a, b = [rng.integers(2, size=target.shape) for _ in range(2)]
    groups = [str(i % 7) for i in range(len(target))]

    def callback(y, p, names):
        return compute_multilabel_metrics(y, p, names)

    for seed in (1, 17):
        arguments = dict(groups=groups, labels=labels, iterations=23, seed=seed)

        def same_result(left, right):
            # Empty label sets retain the legacy NaN fields as well.
            assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)

        same_result(
            cluster_bootstrap(
                target, a, metric_fn=compute_multilabel_metrics, **arguments
            ),
            cluster_bootstrap(target, a, metric_fn=callback, **arguments),
        )
        same_result(
            paired_cluster_bootstrap(
                target, a, b, metric_fn=compute_multilabel_metrics, **arguments
            ),
            paired_cluster_bootstrap(target, a, b, metric_fn=callback, **arguments),
        )


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"z": [1, None, {"x": "é"}], "a": 0.3}],
        [{"id": str(i), "sequence": "ATG" * i} for i in range(17)],
    ],
)
def test_streamed_provenance_matches_established_json(rows):
    expected = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    assert dataset_digest(iter(rows)) == expected
    records = [SeqRecord(Seq("ATG" * i), id="é" + str(i)) for i in range(len(rows))]
    expected = hashlib.sha256(
        json.dumps([[r.id, str(r.seq)] for r in records], ensure_ascii=True).encode()
    ).hexdigest()
    assert backtrim._record_digest(iter(records)) == expected


@pytest.mark.parametrize("sequence", ["", "acgn?.-", "aßN", "ATG" * 200])
@pytest.mark.parametrize(
    "characters", [frozenset(), frozenset("N?-"), frozenset(["NN", None, "N"])]
)
def test_problematic_count_matches_character_membership(sequence, characters):
    expected = (
        sum(ch in characters for ch in sequence.upper()) / len(sequence)
        if sequence
        else 0.0
    )
    assert problematic_rate(sequence, characters) == expected


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -0.1, 1.1, None, "bad", [0.1], [0.1, 0.2]]
)
def test_probability_validation_preserves_scalar_errors(value):
    from cdskit.localize_probabilities import _sanitize_probability

    try:
        _sanitize_probability(value)
    except (TypeError, ValueError, DeprecationWarning) as error:
        with pytest.raises(type(error), match=re.escape(str(error))):
            normalize_class_probabilities({"SP": value})
    else:
        pytest.fail("Expected an invalid scalar score")


def test_probability_normalization_preserves_python_sum():
    from cdskit.localize_probabilities import _sanitize_probability

    rng = np.random.default_rng(1)
    for row in rng.random((100, 5)):
        values = dict(zip(LOCALIZATION_CLASSES, row.tolist(), strict=True))
        expected = {key: _sanitize_probability(value) for key, value in values.items()}
        total = float(sum(expected.values()))
        assert normalize_class_probabilities(values) == {
            key: value / total for key, value in expected.items()
        }


def _reference_mapping(source, target):
    available = list(range(len(source)))
    chosen, multiple = [], set()
    for key in target:
        candidates = [site for site in available if source[site] == key]
        if len(candidates) > 1:
            multiple.update(candidates)
        if candidates:
            site = next(
                (site for site in candidates if not chosen or site > chosen[-1]), None
            )
            if site is not None:
                chosen.append(site)
                available.remove(site)
            elif len(candidates) == 1:
                available.remove(candidates[0])
    return chosen, len(multiple - set(chosen))


def test_duplicate_mapping_matches_legacy_selection_and_bounds_warnings():
    rng = np.random.default_rng(8)
    for _ in range(100):
        source = "".join(rng.choice(list("ABC"), size=70))
        target = "".join(rng.choice(list("ABCZ"), size=40))
        with redirect_stderr(io.StringIO()):
            result = backtrim.find_kept_aa_sites([source], [target])
        assert result == _reference_mapping(source, target)
    warnings = io.StringIO()
    with redirect_stderr(warnings):
        result = backtrim.find_kept_aa_sites(["A" * 2000], ["A" * 1000])
    assert result == (list(range(1000)), 1000)
    assert len(warnings.getvalue()) < 250_000
    assert "2000 candidates" in warnings.getvalue()


def test_backtrim_skips_unrequested_hashes(tmp_path, monkeypatch):
    source, target, output = [
        tmp_path / name for name in ("source.fa", "target.fa", "out.fa")
    ]
    source.write_text(">a\nATGAAATAA\n")
    target.write_text(">a\nMK\n")
    args = SimpleNamespace(
        seqfile=str(source),
        trimmed_aa_aln=str(target),
        outfile=str(output),
        inseqformat="fasta",
        outseqformat="fasta",
        codontable=1,
        threads=1,
    )

    def unexpected(records):
        pytest.fail("Digest requested without a mapping report")

    monkeypatch.setattr(backtrim, "_record_digest", unexpected)
    backtrim.backtrim_main(args)
    assert "ATGAAA" in output.read_text()


def test_file_provenance_hashes_multiple_blocks(tmp_path):
    from cdskit.provenance import file_digest

    content = bytes(range(256)) * 9000
    path = tmp_path / "mapping.txt"
    path.write_bytes(content)
    assert file_digest(path) == hashlib.sha256(content).hexdigest()


def test_bootstrap_preserves_legacy_target_coercion():
    target = np.array([[1 - 1e-8], [1.0], [0.0], [0.0]])
    prediction = np.ones((4, 1))
    arguments = dict(groups=[0, 0, 1, 1], labels=["a"], iterations=12)
    expected = cluster_bootstrap(
        target,
        prediction,
        metric_fn=lambda *args: compute_multilabel_metrics(*args),
        **arguments,
    )
    assert (
        cluster_bootstrap(
            target, prediction, metric_fn=compute_multilabel_metrics, **arguments
        )
        == expected
    )

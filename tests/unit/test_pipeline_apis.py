import io

import pytest

from cdskit.coordinates import pairwise_identity, trimmed_mapping
from cdskit.fasta_contract import SequenceContractError, validate_fasta_stream
from cdskit.gff_attributes import parse_gff_attributes
from cdskit.label_mapping import shorten_labels
from cdskit.translate import coding_protein, translate_complete_cds
from cdskit.util import read_gff


def test_strict_codon_profile_checks_ungapped_frame():
    with pytest.raises(SequenceContractError, match="ungapped"):
        validate_fasta_stream(io.StringIO(">seq\nA--\n"), "codon")
    assert (
        validate_fasta_stream(io.StringIO(">seq\nACG---\n"), "codon").aligned_length
        == 6
    )


def test_mapping_reserves_short_names_and_never_exceeds_limit():
    assert shorten_labels(
        ["long.name+alpha123", "long.nam_1", "long.name+beta456"], 10
    ) == {"long.name+alpha123": "long.nam_2", "long.name+beta456": "long.nam_3"}
    with pytest.raises(ValueError, match="suffix"):
        shorten_labels(["long" + str(i) for i in range(11)], 2)


def test_gff_embedded_fasta_and_quoted_attributes(tmp_path):
    path = tmp_path / "x.gff"
    path.write_text(
        "##gff-version 3\nchr1\ts\tCDS\t1\t3\t.\t+\t0\tID=c;Parent=t\n##FASTA\n>chr1\nATG\n"
    )
    assert read_gff(path) is not None
    assert parse_gff_attributes('gene_id "a;b,c";') == {"gene_id": ("a;b,c",)}


def test_mapping_exposes_ambiguity_and_legacy_policy():
    assert trimmed_mapping(["A"], ["A", "A"])["status"] == "ambiguous"
    assert trimmed_mapping(["A"], ["A", "A"], ambiguity="unresolved")["mapping"] == [
        None
    ]
    assert trimmed_mapping(["A", "04", "C"], ["A", "G", "C"])["mapping"] == [1, None, 3]


def test_pairwise_identity_uses_pair_specific_observed_sites():
    assert pairwise_identity({"a": "MXA", "b": "MCA"}, [("a", "b")]) == [1]
    assert pairwise_identity(
        {"a": "ATGNNNGCT", "b": "ATGTGTGCT"}, [("a", "b")], mode="cds"
    ) == [1]


@pytest.mark.parametrize(
    "table, sequence, ordinary",
    [(27, "ATGTGA", "MW"), (28, "ATGTAA", "MQ"), (31, "ATGTAA", "ME")],
)
def test_coding_protein_never_infers_complete_terminal_context(
    table, sequence, ordinary
):
    assert coding_protein(sequence, table) == ordinary
    assert translate_complete_cds(sequence, table) == "M"


def test_label_mapping_cli_roundtrip_and_alias_rejection(tmp_path):
    from cdskit.cli import psr
    from cdskit.label import label_main

    source, output = tmp_path / "in.fa", tmp_path / "out.fa"
    mapping, restored = tmp_path / "map.tsv", tmp_path / "applied.fa"
    source.write_text(">abcdefgh\nATG\n>other\nATG\n")
    args = psr.parse_args(
        [
            "label",
            "--seq_file",
            str(source),
            "--out_file",
            str(output),
            "--clip_len",
            "5",
            "--mapping-out",
            str(mapping),
        ]
    )
    label_main(args)
    args = psr.parse_args(
        [
            "label",
            "--seq_file",
            str(source),
            "--out_file",
            str(restored),
            "--mapping-in",
            str(mapping),
        ]
    )
    label_main(args)
    assert restored.read_bytes() == output.read_bytes()
    args.mapping_out = str(source)
    with pytest.raises(ValueError, match="different"):
        label_main(args)
    assert source.read_text().startswith(">abcdefgh")


def test_latest_model_filters_explicit_family_schema_without_reselecting_cache(
    tmp_path,
):
    import hashlib
    import json
    from cdskit.latest_model import latest_model, resolve

    releases = [
        dict(
            draft=False,
            prerelease=False,
            tag_name="localize-" + name,
            published_at=date,
            assets=[
                dict(
                    name="cdskit-localize-" + name + ".pt",
                    label=json.dumps(
                        dict(model_family="targeting", feature_schema=schema)
                    ),
                    digest="sha256:" + "a" * 64,
                    browser_download_url="https://github.com/kfuku52/cdskit/releases/download/"
                    + name,
                )
            ],
        )
        for name, date, schema in [
            ("old", "2026-01-01", "v1"),
            ("new", "2026-02-01", "v2"),
        ]
    ]

    def opener(_url):
        return io.BytesIO(json.dumps(releases).encode())

    assert latest_model(opener=opener)["release"] == "localize-new"
    assert (
        latest_model(opener=opener, model_family="targeting", feature_schema="v1")[
            "release"
        ]
        == "localize-old"
    )
    mixed = dict(
        releases[-1],
        tag_name="localize-mixed",
        published_at="2026-03-01",
        assets=[release["assets"][0] for release in releases],
    )
    releases.append(mixed)
    assert (
        latest_model(opener=opener, feature_schema="v1")["release"] == "localize-mixed"
    )
    cache = tmp_path / "latest"
    cache.mkdir()
    (cache / "model.pt").write_bytes(b"fixture")
    spec = dict(
        filename="model.pt",
        sha256=hashlib.sha256(b"fixture").hexdigest(),
        model_family="targeting",
        feature_schema="v1",
    )
    (cache / "selection.json").write_text(json.dumps(spec))
    assert (
        resolve(
            tmp_path, False, model_family="targeting", feature_schema="v1"
        ).read_bytes()
        == b"fixture"
    )
    with pytest.raises(ValueError, match="Frozen"):
        resolve(tmp_path, False, feature_schema="v2")


@pytest.mark.parametrize("alias", ["latest", "LATEST", " latest "])
def test_latest_alias_uses_the_same_normalization_as_named_models(
    alias, tmp_path, monkeypatch
):
    from cdskit import latest_model, localize_models

    def resolve(cache_dir, allow_download):
        assert cache_dir == tmp_path
        assert allow_download is False
        return tmp_path / "selected.pt"

    monkeypatch.setattr(localize_models, "localize_model_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(latest_model, "resolve", resolve)
    assert localize_models.is_pretrained_localize_model_alias(alias)
    assert localize_models.resolve_localize_model_path(alias, False) == str(
        tmp_path / "selected.pt"
    )


def test_mapping_export_rejects_collisions_without_replacing_outputs(tmp_path):
    from cdskit.cli import psr
    from cdskit.label import label_main

    source, output, mapping = (
        tmp_path / name for name in ("in.fa", "out.fa", "map.tsv")
    )
    source.write_text(">alpha\nATG\n>alpine\nATG\n")
    output.write_bytes(b"old sequences")
    mapping.write_bytes(b"old mapping")
    args = psr.parse_args(
        [
            "label",
            "--seq_file",
            str(source),
            "--out_file",
            str(output),
            "--clip_len",
            "3",
            "--mapping-out",
            str(mapping),
        ]
    )
    with pytest.raises(ValueError, match="unique"):
        label_main(args)
    assert output.read_bytes() == b"old sequences"
    assert mapping.read_bytes() == b"old mapping"
    args.unique = True
    label_main(args)
    from cdskit.label_mapping import read_label_mapping

    assert len(set(read_label_mapping(mapping).values())) == 2


def test_partial_mapping_cannot_collide_with_untouched_ids(tmp_path):
    from cdskit.cli import psr
    from cdskit.label import label_main
    from cdskit.label_mapping import write_label_mapping

    source, output, mapping = (
        tmp_path / name for name in ("in.fa", "out.fa", "map.tsv")
    )
    source.write_text(">alpha\nATG\n>beta\nATG\n")
    write_label_mapping({"alpha": "beta"}, mapping)
    args = psr.parse_args(
        [
            "label",
            "--seq_file",
            str(source),
            "--out_file",
            str(output),
            "--mapping-in",
            str(mapping),
        ]
    )
    with pytest.raises(ValueError, match="unique"):
        label_main(args)
    assert not output.exists()


@pytest.mark.parametrize("label", ["beta gamma", "beta\tgamma", "beta\ngamma"])
def test_sequence_mapping_rejects_ids_that_cannot_roundtrip_through_fasta(
    tmp_path, label
):
    from cdskit.cli import psr
    from cdskit.label import label_main
    from cdskit.label_mapping import write_label_mapping

    source, output, mapping = (
        tmp_path / name for name in ("in.fa", "out.fa", "map.tsv")
    )
    source.write_text(">alpha\nATG\n")
    write_label_mapping({"alpha": label}, mapping)
    args = psr.parse_args(
        [
            "label",
            "--seq_file",
            str(source),
            "--out_file",
            str(output),
            "--mapping-in",
            str(mapping),
        ]
    )
    with pytest.raises(ValueError, match="whitespace"):
        label_main(args)
    assert not output.exists()


@pytest.mark.parametrize(
    "profile", ["negative_match", "positive_gaps", "wildcard", "matrix"]
)
def test_identical_proteins_honor_the_callers_alignment_scoring(profile):
    import numpy as np
    from Bio.Align.substitution_matrices import Array
    from cdskit.protein_similarity import protein_aligner, score_protein_junctions

    def factory():
        aligner = protein_aligner(
            match_score=-1, mismatch_score=-1, open_gap_score=0, extend_gap_score=0
        )
        if profile == "positive_gaps":
            aligner.match_score = 2
            aligner.open_gap_score = aligner.extend_gap_score = 2
        elif profile == "wildcard":
            aligner.match_score = 2
            aligner.wildcard = "A"
        elif profile == "matrix":
            aligner.substitution_matrix = Array(
                alphabet="AC", dims=2, data=-np.ones((2, 2))
            )
        return aligner

    reference = factory().align("AA", "AA")[0]
    expected = sum(end - start for start, end in reference.aligned[0])
    actual = score_protein_junctions(
        ("AA", ()), ("AA", ()), max_alignment_cells=100, aligner_factory=factory
    )
    assert actual.aligned_residues == expected


@pytest.mark.parametrize("rows", [[], [""], ["", ""], ["AC-", "A--"], ["NN?", "---"]])
def test_alignment_statistics_preserve_empty_and_gap_contracts(tmp_path, rows):
    from cdskit.alignment_metrics import alignment_length_stats

    path = tmp_path / "alignment.fa"
    path.write_text("".join(f">s{i}\n{row}\n" for i, row in enumerate(rows)))
    lengths = [len(row.replace("-", "")) for row in rows]
    assert alignment_length_stats(path) == dict(
        num_site=len(rows[0]) if rows else 0,
        num_seq=len(rows),
        len_max=max(lengths, default=0),
        len_min=min(lengths, default=0),
    )


def test_alignment_statistics_keep_read_errors_ahead_of_length_errors(tmp_path):
    from cdskit.alignment_metrics import alignment_length_stats

    path = tmp_path / "alignment.fa"
    path.write_bytes(b">a\nA\n>b\nAA\n" + b"\n" * 10000 + b"\xff")
    with pytest.raises(ValueError, match="Failed to read"):
        alignment_length_stats(path)
    path.write_text(">a\nA\n>b\nAA\n")
    with pytest.raises(ValueError, match="same aligned length"):
        alignment_length_stats(path)


@pytest.mark.parametrize("text", [">\n", ">   \nATG\n", ">\n>valid\nATG\n"])
def test_tuple_fasta_parser_never_silently_discards_empty_headers(tmp_path, text):
    from cdskit.sequence_selection import fasta_records

    source = tmp_path / "input.fa"
    source.write_text(text)
    with pytest.raises(ValueError, match="Empty FASTA header"):
        list(fasta_records(source))


def test_coordinate_bridge_protects_requests_and_respects_ambiguity_policy(
    tmp_path, monkeypatch
):
    import json
    from cdskit import coordinates

    source, output = tmp_path / "request.json", tmp_path / "response.json"
    request = dict(
        operation="mapping",
        trimmed={"a": ["A"]},
        untrimmed={"a": ["A", "A"]},
        ambiguity="unresolved",
    )
    source.write_text(json.dumps(request))
    monkeypatch.setattr("sys.argv", ["coordinates", str(source), str(source)])
    with pytest.raises(ValueError, match="different"):
        coordinates.main()
    assert json.loads(source.read_text()) == request
    monkeypatch.setattr("sys.argv", ["coordinates", str(source), str(output)])
    coordinates.main()
    assert json.loads(output.read_text()) == {
        "a": dict(status="ambiguous", mapping=[None])
    }
    request["ambiguity"] = "error"
    source.write_text(json.dumps(request))
    previous = output.read_bytes()
    with pytest.raises(ValueError, match="Ambiguous"):
        coordinates.main()
    assert output.read_bytes() == previous


@pytest.mark.parametrize("ending", ["\n", "\r\n"])
@pytest.mark.parametrize("final_newline", [True, False])
def test_streaming_whole_headers_keep_descriptions_sequence_text_and_line_endings(
    ending, final_newline
):
    from cdskit.label_mapping import iter_fasta_headers, map_fasta_headers

    lines = [
        ">first description" + ending,
        "ACGT" + ending,
        ">second" + ending,
        "A--?" + (ending if final_newline else ""),
    ]
    assert list(iter_fasta_headers(iter(lines))) == ["first description", "second"]
    expected = [">renamed description" + ending, *lines[1:]]
    assert (
        list(
            map_fasta_headers(iter(lines), {"first description": "renamed description"})
        )
        == expected
    )


def test_header_mapping_yields_before_consuming_the_rest_of_the_fasta():
    from cdskit.label_mapping import map_fasta_headers

    def lines():
        yield ">first\n"
        raise AssertionError("The complete input must not be materialized")

    assert next(map_fasta_headers(lines(), {"first": "renamed"})) == ">renamed\n"
    with pytest.raises(ValueError, match="single lines"):
        list(map_fasta_headers([">first\n"], {"first": "injected\n>record"}))

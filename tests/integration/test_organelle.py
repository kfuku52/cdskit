import csv
import gzip
import json

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, SimpleLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from cdskit.cli import main


def test_genbank_copies_and_translation(tmp_path):
    r = SeqRecord(Seq("ATGAAATAAGTGGAATAA"), id="REC.1", name="REC", description="")
    r.annotations["molecule_type"] = "DNA"

    def add(gene, location, **extra):
        q = dict(gene=[gene], transl_table=["11"], **extra)
        r.features.append(SeqFeature(location, type="CDS", qualifiers=q))

    a, b = SimpleLocation(0, 9, 1), SimpleLocation(9, 18, 1)
    r.features.append(
        SeqFeature(
            SimpleLocation(0, 18),
            type="source",
            qualifiers={"db_xref": ["taxon:42"], "isolate": ["I1"]},
        )
    )
    add("x", a, translation=["MK"])
    add("x", a)
    add("y", b, translation=["ME"])
    add("y", a)
    add("z", a, pseudo=[""])
    add("bad", a, translation=["MA"])
    add(
        "minus",
        CompoundLocation([SimpleLocation(12, 15, -1), SimpleLocation(0, 3, -1)]),
    )
    gb = tmp_path / "r.gb.gz"
    with gzip.open(gb, "wt") as handle:
        SeqIO.write([r], handle, "genbank")
    out = tmp_path / "bundle"
    assert main(["parsegb", "-s", str(gb), "--all_cds_dir", str(out)]) == 0
    with (out / "features.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [(d["gene"], d["reason"]) for d in rows] == [
        ("x", ""),
        ("x", "identical_duplicate"),
        ("y", "discordant_copies"),
        ("y", "discordant_copies"),
        ("z", "pseudogene"),
        ("bad", "translation_mismatch"),
        ("minus", ""),
    ]
    assert {d["taxid"] for d in rows} == {"42"}
    assert rows[2]["translation_check"] == "match"
    assert "join" in rows[-1]["location"] and rows[-1]["strand"] == "-1"
    assert [
        str(f.seq) for f in SeqIO.parse(out / "genes" / rows[-1]["gene_file"], "fasta")
    ] == ["TTCCAT"]
    assert len(list((out / "genes").iterdir())) == 2
    assert len(list(SeqIO.parse(out / "copies" / rows[0]["gene_file"], "fasta"))) == 2


def concat_args(tmp_path):
    (tmp_path / "taxa").write_text("taxon\nB\nA\nC\n")
    (tmp_path / "genes").write_text(
        "gene\talignment\tbacktrim_json\ny\ty.fa\tmap\nx\tx.fa\t\n"
    )
    (tmp_path / "map").write_text(
        json.dumps(
            dict(
                index_base=0,
                status="success",
                output_complete=True,
                target_column_count=2,
                source_column_count=5,
                selected_sites=[1, 4],
            )
        )
    )
    (tmp_path / "y.fa").write_text(">C\nGGGCCC\n>A\nATG---\n")
    (tmp_path / "x.fa").write_text(">B\nNNN\n>A\nAAA\n")
    return [
        "concat",
        "--genes_tsv",
        str(tmp_path / "genes"),
        "--taxa_tsv",
        str(tmp_path / "taxa"),
        "--out_dir",
        str(tmp_path / "out"),
    ]


def test_concat_manifest_and_roundtrip(tmp_path):
    args = concat_args(tmp_path)
    assert main(args) == 0
    out = tmp_path / "out"
    assert (
        out / "alignment.fa"
    ).read_text() == ">B\n------NNN\n>A\nATG---AAA\n>C\nGGGCCC---\n"
    assert (out / "partitions.txt").read_text() == "DNA, y = 1-6\nDNA, x = 7-9\n"
    assert (out / "sites.tsv").read_text().splitlines()[1:] == [
        "1\ty\t1\t2",
        "2\ty\t2\t5",
        "3\tx\t1\t1",
    ]
    assert "x\tB\t1\t1\t0" in (out / "coverage.tsv").read_text()
    assert main(args) != 0 and (out / "sites.tsv").exists()


@pytest.mark.parametrize(
    "bad",
    [
        ">C\nATA\n>D\nATA\n",
        ">A\nAAA\n>A\nGGG\n",
        ">A\nA--\n",
        ">C\nAA\n",
        ">C\nAAA\n>A\nAAAAAA\n",
    ],
)
def test_concat_invalid_is_atomic(tmp_path, bad):
    args = concat_args(tmp_path)
    (tmp_path / "x.fa").write_text(bad)
    assert main(args) != 0
    assert not (tmp_path / "out").exists()


def test_parsegb_does_not_assign_the_first_sources_taxid(tmp_path):
    r = SeqRecord(Seq("ATG" * 6), id="R.2", name="R", description="")
    r.annotations["molecule_type"] = "DNA"
    for lo, hi, tid in [(0, 12, 1), (6, 18, 2)]:
        r.features.append(
            SeqFeature(
                SimpleLocation(lo, hi),
                type="source",
                qualifiers={"db_xref": [f"taxon:{tid}"]},
            )
        )
    for i in (0, 6, 12):
        r.features.append(
            SeqFeature(
                SimpleLocation(i, i + 6), type="CDS", qualifiers={"gene": [f"g{i}"]}
            )
        )
    gb = tmp_path / "r.gb"
    SeqIO.write(r, gb, "genbank")
    out = tmp_path / "new"
    args = ["parsegb", "-s", str(gb), "--all_cds_dir", str(out)]
    assert main(args) == 0
    with (out / "features.tsv").open() as stream:
        assert [
            (r["source_status"], r["taxid"])
            for r in csv.DictReader(stream, delimiter="\t")
        ] == [("matched", "1"), ("ambiguous", ""), ("matched", "2")]
    SeqIO.write([r, r], gb, "genbank")
    assert main([*args[:-1], str(tmp_path / "duplicate")]) != 0
    assert not (tmp_path / "duplicate").exists()


def test_concat_rejects_nonmonotone_backtrim_map(tmp_path):
    args = concat_args(tmp_path)
    path = tmp_path / "map"
    data = json.loads(path.read_text())
    data["selected_sites"] = [4, 1]
    path.write_text(json.dumps(data))
    assert main(args) != 0
    assert not (tmp_path / "out").exists()

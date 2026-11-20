import gzip
import json

import pytest
from Bio import SeqIO

from cdskit.cli import main
from cdskit.tsvio import read_tsv


def test_graph_evidence_and_complete_locus_split_feed_aggregate(tmp_path):
    names = [f"NODE_{i + 1}_length_480_cov_2.0_g{i // 3}_i{i % 3}" for i in range(6)]
    fasta = tmp_path / "input.fa.gz"
    with gzip.open(fasta, "wt") as stream:
        for name in names:
            stream.write(f">{name}\n{'ATG' * 160}\n")
    paths = tmp_path / "transcripts.paths"
    paths.write_text(
        "".join(
            f"{name}\n{path}\n{name}'\n1-\n"
            for name, path in zip(
                names, ("1+,3+", "1+,2+", "2+,3+", "4+", "4+", "4+"), strict=True
            )
        )
    )
    fastg = tmp_path / "assembly_graph.fastg"
    fastg.write_text(
        "".join(
            f">EDGE_{i}_length_{length}_cov_9.0;\nA\n"
            for i, length in enumerate((401, 402, 349, 450), 1)
        )
    )
    loci = tmp_path / "loci.tsv"
    loci.write_text(
        "transcript_id\tlocus_id\n"
        + "".join(
            f"{name}\t{locus}\n"
            for name, locus in zip(names[:5], ("a", "a", "b", "c", "d"), strict=True)
        )
    )
    junctions = tmp_path / "reads.tsv"
    junctions.write_text(
        f"transcript_id\tfrom_edge\tto_edge\tread_pairs\n{names[1]}\t1+\t2+\t0\n"
    )
    groups, report = tmp_path / "groups.tsv", tmp_path / "report.tsv"
    args = [
        "regroup",
        "--seq_file",
        str(fasta),
        "--out_file",
        str(groups),
        "--report",
        str(report),
        "--paths",
        str(paths),
        "--fastg",
        str(fastg),
        "--kmer_size",
        "49",
        "--junction_table",
        str(junctions),
    ]
    main(args)
    audit = read_tsv(report)
    assert audit[0]["decision"] == "review_bridge"
    assert json.loads(audit[0]["bridge_edges"]) == [1, 2]
    assert json.loads(audit[0]["bridge_transcripts"]) == [names[1]]
    assert json.loads(audit[0]["bridge_junctions"]) == [
        dict(transcript_id=names[1], from_edge="1+", to_edge="2+", read_pairs=0)
    ]
    assert [row["group_id"] for row in read_tsv(groups)] == ["g0"] * 3 + ["g1"] * 3
    main([*args, "--locus_table", str(loci)])
    assert [row["decision"] for row in read_tsv(report)] == [
        "split_by_locus",
        "partial_multi_locus_evidence",
    ]
    assert [row["group_id"] for row in read_tsv(groups)] == [
        "g0.split1",
        "g0.split1",
        "g0.split2",
        "g1",
        "g1",
        "g1",
    ]
    cds = tmp_path / "cds.fa"
    cds.write_text(
        "".join(f">{name}\nATGCCC\n" for name in names)
        + f">{names[2]}.cdskit-orf-2\nATGCCCCCC\n"
    )
    selected = tmp_path / "selected.fa"
    main(
        [
            "aggregate",
            "--seq_file",
            str(cds),
            "--out_file",
            str(selected),
            "--group_table",
            str(groups),
            "-x",
            r"\.cdskit-orf-[0-9]+$",
        ]
    )
    assert [record.id for record in SeqIO.parse(selected, "fasta")] == [
        names[0],
        names[2] + ".cdskit-orf-2",
        names[3],
    ]
    before = groups.read_bytes(), report.read_bytes()
    with loci.open("a") as stream:
        stream.write(f"{names[0]}\tdifferent\n")
    with pytest.raises(ValueError, match="repeated transcript"):
        from cdskit.regroup import regroup_main
        from types import SimpleNamespace

        regroup_main(
            SimpleNamespace(
                seqfile=fasta,
                paths=paths,
                fastg=fastg,
                kmer_size=49,
                locus_table=loci,
                junction_table=junctions,
                outfile=groups,
                report=report,
            )
        )
    assert before == (groups.read_bytes(), report.read_bytes())

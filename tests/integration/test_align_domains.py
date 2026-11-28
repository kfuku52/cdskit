import csv
import gzip
import io
import json
import re
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from cdskit import align, domain_database, domain_search
from cdskit.cli import main


def put(path, text):
    path.write_text(text)
    return path


def fasta_stub(binary, source, output, seq_type, threads, options):
    rows = align.read_fasta(source, aligned=True)
    if "--addfragments" in options:
        rows.update(align.read_fasta(options[-1]))
    width = max(map(len, rows.values()))
    align.write_fasta(
        output, {key: row.ljust(width, "-") for key, row in rows.items()}, rows
    )


@pytest.mark.parametrize(
    ("seqtype", "rows", "expected"),
    [
        ("aa", "MTR\n>b\nMS", ">a\nMTR\n>b\nMS-\n"),
        ("dna", "ACTCA\n>b\nACN", ">a\nACTCA\n>b\nACN--\n"),
        ("codon", "ATGGCCTAA\n>b\nATGANN", ">a\nATGGCCTAA\n>b\nATGANN---\n"),
    ],
)
def test_align_types_and_gzip(tmp_path, monkeypatch, seqtype, rows, expected):
    monkeypatch.setattr(align, "tool_path", str)
    monkeypatch.setattr(align, "_run_mafft", fasta_stub)
    source = put(tmp_path / "in.fa", ">a\n" + rows + "\n")
    output = tmp_path / "out.fa.gz"
    assert (
        main(["align", "--seq_type", seqtype, "-s", str(source), "-o", str(output)])
        == 0
    )
    assert gzip.decompress(output.read_bytes()).decode() == expected


def test_precomputed_domains_are_reused_including_no_hits(tmp_path, monkeypatch):
    monkeypatch.setattr(align, "_run_mafft", fasta_stub)
    monkeypatch.setattr(align, "tool_path", str)

    def fail(*args):
        raise AssertionError("Complete precomputed domains called the search runtime")

    monkeypatch.setattr(domain_search, "run_search", fail)
    fasta = put(
        tmp_path / "family.fa", ">human\nMDDD\n>mouse\nMDDD\n>fish\nMDDD\n>fly\nMDD\n"
    )
    query = put(tmp_path / "query.fa", ">human\nMDDD\n")
    domains = put(
        tmp_path / "hits.tsv",
        "qacc\tsacc\tqstart\tqend\tsstart\tsend\tbitscore\tqlen\n"
        "human\tpfam\t1\t4\t1\t4\t9\t4\nmouse\tpfam\t1\t2\t1\t2\t5\t4\n"
        "fish\tpfam\t1\t4\t1\t4\t8\t4\nfly\t\t\t\t\t\t\t3\n",
    )
    report, columns, output = (
        tmp_path / name for name in ("report.tsv", "columns.tsv", "out.fa")
    )
    argv = [
        "align",
        "--seq_type",
        "aa",
        "--strategy",
        "query_backbone",
        "-s",
        str(fasta),
        "-o",
        str(output),
        "--query_file",
        str(query),
        "--rpsblast_tsv",
        str(domains),
        "--backbone_size",
        "2",
        "--backbone_report",
        str(report),
        "--column_map",
        str(columns),
        "--no_download",
    ]
    assert main(argv) == 0
    with report.open() as handle:
        selected = {
            r["gene_id"]
            for r in csv.DictReader(handle, delimiter="\t")
            if r["backbone_rank"]
        }
    assert selected == {"human", "fish"}
    assert list(align.read_fasta(output)) == ["human", "mouse", "fish", "fly"]
    previous = output.read_bytes()
    domains.write_text(domains.read_text().replace("fly\t\t\t\t\t\t\t3\n", ""))
    assert main(argv) == 1
    assert output.read_bytes() == previous

    searches = []

    def search(sequences, args, seq_type, directory, expected_database=None):
        searches.append(sequences)
        fields = domain_search.RPS_FIELDS
        no_hit = [dict(qacc=gene, qlen=len(seq)) for gene, seq in sequences.items()]
        return ([row.get(field, "") for field in fields] for row in no_hit), None

    monkeypatch.setattr(domain_search, "run_search", search)
    assert main([*argv, "--search_missing_domains"]) == 0
    assert searches == [{"fly": "MDD"}]


def test_chunked_fasta_index_and_backbone_mapping(tmp_path):
    first = b">a\n" + b"A" * (4 * 1024 * 1024 - 4) + b"\n"
    second = b">b desc\r\nGT\r\n"
    handle = io.BytesIO(first + second)
    assert align.fasta_offsets(handle) == {
        "a": (0, len(first)),
        "b": (len(first), len(second)),
    }
    with pytest.raises(ValueError, match="unique"):
        align.fasta_offsets(io.BytesIO(first + b">a\nGT\n"))
    table = tmp_path / "map.tsv"
    source = {"a": "G-A", "b": "-CA"}
    align.write_column_map(source, {"b": "--CA", "a": "G--A"}, table)
    assert (
        table.read_text()
        == "alignment_column\tbackbone_column\n1\t1\n2\t\n3\t2\n4\t3\n"
    )
    with pytest.raises(ValueError, match="changed the backbone alignment"):
        align.write_column_map(source, {"a": "--GA", "b": "C--A"}, table)


def test_provided_backbone_and_genetic_code(tmp_path, monkeypatch):
    monkeypatch.setattr(align, "_run_mafft", fasta_stub)
    monkeypatch.setattr(align, "tool_path", str)
    source = put(tmp_path / "cds.fa", ">a\nATGTGA\n>b\nATG\n")
    core = put(tmp_path / "core.fa", ">a\nMW\n")
    output = tmp_path / "result.fa"
    assert (
        main(
            [
                "align",
                "--strategy",
                "query_backbone",
                "--seq_type",
                "codon",
                "--codon_table",
                "2",
                "--backbone_file",
                str(core),
                "-s",
                str(source),
                "-o",
                str(output),
            ]
        )
        == 0
    )
    assert output.read_text() == ">a\nATGTGA\n>b\nATG---\n"


def test_search_preserves_ids_and_records_no_hits(tmp_path, monkeypatch):
    def fake(command, **kwargs):
        path = Path(command[command.index("-out") + 1])
        assert command[command.index("-num_threads") + 1] == "2"
        assert command[command.index("-mt_mode") + 1] == "1"
        path.write_text(
            'cdskit1\tCDD:0001\t100\t4\t0\t0\t1\t4\t3\t6\t1e-4\t40\t4\t8\t"Domain name"\n'
        )

    monkeypatch.setattr(domain_search.subprocess, "run", fake)
    monkeypatch.setattr(domain_search, "tool_path", str)
    monkeypatch.setattr(
        domain_search,
        "resolve_database",
        lambda _: domain_database.ProfileDatabase(
            tmp_path / "Pfam", {"archive_sha256": "test"}
        ),
    )
    source = put(tmp_path / "family.fa", ">sp|a\nMFVD\n>sp|ab\nMS\n")
    table = tmp_path / "domains.tsv"
    manifest = tmp_path / "search.json"
    assert (
        main(
            [
                "rpsblast",
                "--seq_type",
                "aa",
                "-s",
                str(source),
                "-o",
                str(table),
                "--manifest",
                str(manifest),
                "--threads",
                "2",
                "--rps_mt_mode",
                "queries",
            ]
        )
        == 0
    )
    domains = domain_search.read_domains(table, "aa", {"sp|a": "MFVD", "sp|ab": "MS"})
    assert domains["sp|a"][0].model == "CDD:0001"
    assert domains["sp|ab"] == []
    assert json.loads(manifest.read_text())["coordinate_unit"] == "aa"
    assert '\t"Domain name"\n' in table.read_text()
    options = SimpleNamespace(evalue=0.01, codontable=1)
    sequences = {"sp|a": "MFVD", "sp|ab": "MS"}
    domain_search.validate_domain_manifest(manifest, table, "aa", sequences, options)
    table.write_text(table.read_text().replace("40\t", "30\t"))
    with pytest.raises(ValueError, match="does not match"):
        domain_search.validate_domain_manifest(
            manifest, table, "aa", sequences, options
        )


def test_dna_frames_are_scored_in_translation_order(tmp_path):
    table = put(
        tmp_path / "dna.tsv",
        "qacc\tsacc\tqstart\tqend\tsstart\tsend\tbitscore\tqframe\n"
        "ref\tA\t45\t30\t1\t5\t30\t-1\nref\tB\t29\t15\t1\t5\t20\t-1\n"
        "ref\tA\t1\t12\t1\t4\t15\t1\n",
    )
    result = domain_search.read_domains(table, "dna", {"ref": "C" * 50})
    assert [hit.model for hit in align.architecture(result["ref"])] == ["A", "B"]


def test_download_is_published_once_and_retried_after_failure(tmp_path, monkeypatch):
    attempts = []

    def fake(url, staging):
        attempts.append(url)
        if len(attempts) == 1:
            raise OSError("incomplete archive")
        put(staging / "Pfam.loo", "reference index")
        put(staging / "Pfam.rps", "reference profiles")
        return {
            "source_url": url,
            "archive_sha256": "test",
            "files": domain_database._describe_database(staging / "Pfam"),
        }

    monkeypatch.setattr(domain_database, "_download", fake)
    monkeypatch.setattr(domain_database, "PFAM_URL", "https://example.invalid/data")
    options = SimpleNamespace(
        db_prefix=None,
        db_preferred=[],
        db_url=None,
        db_cache=tmp_path,
        no_download=False,
    )
    with pytest.raises(OSError, match="incomplete archive"):
        domain_database.resolve_database(options)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: domain_database.resolve_database(options), range(4))
        )
    assert len(attempts) == 2
    assert len({db.prefix for db in results}) == 1
    options.no_download = True
    assert domain_database.resolve_database(options).prefix == results[0].prefix


def test_pfam_database_validates_split_volumes_and_archive_paths(tmp_path):
    put(tmp_path / "Pfam.pal", "DBLIST Pfam.00 Pfam.01\n")
    for name in ("Pfam.00.loo", "Pfam.00.rps", "Pfam.01.loo"):
        put(tmp_path / name, "partial")
    with pytest.raises(FileNotFoundError, match=re.escape("Pfam.01.rps")):
        domain_database.database_files(tmp_path / "Pfam")
    archive = tmp_path / "malformed.tgz"
    with tarfile.open(archive, "w:gz") as writer:
        writer.add(tmp_path / "Pfam.00.loo", arcname="../Pfam.loo")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        domain_database._extract_pfam(archive, tmp_path)

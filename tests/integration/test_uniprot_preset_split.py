import csv

import pytest

import cdskit.uniprot_preset_split as splitter
from cdskit.uniprot_preset_split import (
    classify_lineage_ids,
    parse_taxon_ids,
    split_uniprot_eukaryota_tsv,
)


def test_parse_taxon_ids_and_classify_lineage_ids():
    ids = parse_taxon_ids(
        "131567 (no rank), 2759 (domain), 33090 (clade), 4751 (kingdom)"
    )
    assert "2759" in ids
    assert "33090" in ids
    assert "4751" in ids

    flags = classify_lineage_ids(ids)
    assert flags["eukaryota"] is True
    assert flags["viridiplantae"] is True
    assert flags["non_viridiplantae_euk"] is False
    assert flags["protist_core"] is False


def test_split_uniprot_eukaryota_tsv_outputs_expected_counts(temp_dir):
    input_tsv = temp_dir / "eukaryota_with_lineage.tsv"
    out_dir = temp_dir / "split"
    report_json = temp_dir / "report.json"

    rows = [
        {
            "accession": "VIR1",
            "sequence": "MAAA",
            "cc_subcellular_location": "Chloroplast",
            "lineage_ids": "2759,33090",
        },
        {
            "accession": "MET1",
            "sequence": "MBBB",
            "cc_subcellular_location": "Membrane",
            "lineage_ids": "2759,33208",
        },
        {
            "accession": "FUN1",
            "sequence": "MCCC",
            "cc_subcellular_location": "Cytoplasm",
            "lineage_ids": "2759,4751",
        },
        {
            "accession": "PRO1",
            "sequence": "MDDD",
            "cc_subcellular_location": "Nucleus",
            "lineage_ids": "2759",
        },
        {
            "accession": "BAC1",
            "sequence": "MEEE",
            "cc_subcellular_location": "Cytoplasm",
            "lineage_ids": "2",
        },
        {
            "accession": "MISS1",
            "sequence": "MFFF",
            "cc_subcellular_location": "Unknown",
            "lineage_ids": "",
        },
    ]

    with open(input_tsv, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                "accession",
                "sequence",
                "cc_subcellular_location",
                "lineage_ids",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    report = split_uniprot_eukaryota_tsv(
        input_tsv=str(input_tsv),
        out_dir=str(out_dir),
        out_prefix="demo",
        lineage_col="lineage_ids",
        report_json=str(report_json),
    )

    assert report["total_rows"] == 6
    assert report["rows_missing_lineage"] == 1
    assert report["rows_non_eukaryota"] == 1
    assert report["counts_by_dataset"]["viridiplantae"] == 1
    assert report["counts_by_dataset"]["metazoa"] == 1
    assert report["counts_by_dataset"]["fungi"] == 1
    assert report["counts_by_dataset"]["non_viridiplantae_euk"] == 3
    assert report["counts_by_dataset"]["protist_core"] == 1

    for preset_name in [
        "viridiplantae",
        "metazoa",
        "fungi",
        "non_viridiplantae_euk",
        "protist_core",
    ]:
        out_path = report["output_paths"][preset_name]
        with open(out_path, "r", encoding="utf-8") as inp:
            subset = list(csv.DictReader(inp, delimiter="\t"))
        assert len(subset) == report["counts_by_dataset"][preset_name]


def test_split_rejects_input_output_collision_without_changing_source(tmp_path):
    source = tmp_path / "demo_viridiplantae.tsv"
    content = "accession\tlineage_ids\nVIR1\t2759,33090\nMET1\t2759,33208\n"
    source.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="Input and output paths"):
        split_uniprot_eukaryota_tsv(source, tmp_path, out_prefix="demo")

    assert source.read_text(encoding="utf-8") == content
    assert not (tmp_path / "demo_metazoa.tsv").exists()


def test_split_rolls_back_all_outputs_when_later_write_fails(tmp_path, monkeypatch):
    source = tmp_path / "source.tsv"
    source.write_text(
        "accession\tlineage_ids\nVIR1\t2759,33090\nMET1\t2759,33208\n",
        encoding="utf-8",
    )
    destinations = [
        tmp_path / f"demo_{name}.tsv"
        for name in (
            "viridiplantae",
            "metazoa",
            "fungi",
            "non_viridiplantae_euk",
            "protist_core",
        )
    ]
    report = tmp_path / "report.json"
    for destination in [*destinations, report]:
        destination.write_text("previous result\n", encoding="utf-8")

    original_write = splitter.write_rows_tsv
    calls = 0

    def fail_on_third_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("simulated output failure")
        original_write(*args, **kwargs)

    monkeypatch.setattr(splitter, "write_rows_tsv", fail_on_third_write)
    with pytest.raises(OSError, match="simulated output failure"):
        splitter.split_uniprot_eukaryota_tsv(
            source, tmp_path, out_prefix="demo", report_json=report
        )

    assert all(
        destination.read_text(encoding="utf-8") == "previous result\n"
        for destination in [*destinations, report]
    )

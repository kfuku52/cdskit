#!/usr/bin/env python3
"""Compare fixed localization models on new, evidence-scoped UniProt positives.

Consumes immutable full UniProt JSON pages described by full_manifest.json.
This positive-only panel estimates recall, not precision or population F1.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cdskit.deeploc_benchmark import (
    _read_prepared_tsv,
    _predict_model_on_rows,
    DEEPLOC_LOCALIZATION_LABELS as LABELS,
)
from cdskit.localize_model import load_localize_model, to_canonical_aa_sequence
from cdskit.localize_evaluation import assert_disjoint, dataset_digest
from cdskit.perox_benchmark import mmseqs_homology_report, mmseqs_cluster_assignments
from cdskit.util import atomic_write_json

PEROX_LOCATIONS = {"SL-0204", "SL-0202", "SL-0203"}


def ensembl_protein_ids(entry):
    return {
        prop["value"].split(".")[0]
        for ref in entry.get("uniProtKBCrossReferences", [])
        if ref.get("database") == "Ensembl"
        for prop in ref.get("properties", [])
        if prop.get("key") == "ProteinId"
    }


def experimental_perox_row(entry):
    """Require direct experimental evidence on the peroxisomal location itself."""
    if entry.get("entryType") != "UniProtKB reviewed (Swiss-Prot)":
        return None, "unreviewed"
    if "fragment" in str(entry.get("proteinDescription", {}).get("flag", "")).lower():
        return None, "fragment"
    organism = entry.get("organism", {})
    lineage = organism.get("lineage", [])
    if "Eukaryota" not in lineage:
        return None, "non_eukaryote"
    locations, evidence = set(), set()
    for comment in entry.get("comments", []):
        if comment.get("commentType") != "SUBCELLULAR LOCATION" or comment.get(
            "molecule"
        ):
            continue  # Do not apply an isoform/processed-chain annotation to the canonical sequence.
        for item in comment.get("subcellularLocations", []):
            loc = item.get("location", {})
            direct = [
                e
                for e in loc.get("evidences", [])
                if e.get("evidenceCode") == "ECO:0000269"
            ]
            if loc.get("id") in PEROX_LOCATIONS and direct:
                locations.add(loc["id"])
                evidence.update(
                    e.get("source", "") + ":" + e.get("id", "") for e in direct
                )
    if not locations:
        return None, "no_direct_experimental_perox_location"
    sequence = to_canonical_aa_sequence(entry.get("sequence", {}).get("value", ""))
    if not sequence or not set(sequence) - {"X"}:
        return None, "uninformative_sequence"
    group = next(
        (g for g in ["Metazoa", "Fungi", "Viridiplantae"] if g in lineage), "Other"
    )
    return dict(
        accession=entry["primaryAccession"],
        secondary_accessions=entry.get("secondaryAccessions", []),
        sequence=sequence,
        organism_group=group,
        taxon_id=organism["taxonId"],
        organism=organism["scientificName"],
        locations=sorted(locations),
        evidence=sorted(evidence),
        peroxisome=1,
        localization_labels="peroxisome",
    ), ""


def wilson(k, n):
    if n == 0:
        return None
    z = 1.959963984540054
    center = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, float(center - half)), min(1.0, float(center + half))]


def compare_recall(base, integrated, groups):
    n = len(base)
    if not n:
        return {"n": 0}
    cluster_ids = sorted(set(groups))
    totals = np.asarray(
        [
            [sum(groups == c), sum(base[groups == c]), sum(integrated[groups == c])]
            for c in cluster_ids
        ]
    )
    rng = np.random.default_rng(1)
    ci = None
    if len(totals) >= 2:
        sample = totals[rng.integers(len(totals), size=(10000, len(totals)))].sum(
            axis=1
        )
        ci = np.quantile(
            (sample[:, 2] - sample[:, 1]) / sample[:, 0], [0.025, 0.975]
        ).tolist()
    return {
        "n": n,
        "baseline_tp": int(sum(base)),
        "integrated_tp": int(sum(integrated)),
        "baseline_recall": float(np.mean(base)),
        "integrated_recall": float(np.mean(integrated)),
        "baseline_wilson95": wilson(sum(base), n),
        "integrated_wilson95": wilson(sum(integrated), n),
        "both_detected": int(sum(base & integrated)),
        "baseline_only": int(sum(base & ~integrated)),
        "integrated_only": int(sum(~base & integrated)),
        "neither": int(sum(~base & ~integrated)),
        "cluster_count": len(totals),
        "paired_cluster_bootstrap_recall_delta95": ci,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_dir", default="data/localize_bench/perox_expansion_20260908"
    )
    parser.add_argument(
        "--experiment_dir", default="data/localize_bench/full_localization_20260908"
    )
    parser.add_argument("--prepared_dir", default="data/localize_bench/deeploc21")
    args = parser.parse_args()
    root, experiment, prepared = map(
        Path, (args.source_dir, args.experiment_dir, args.prepared_dir)
    )
    manifest = json.loads((root / "full_manifest.json").read_text())
    entries = []
    releases = set()
    for page in manifest["pages"]:
        path = root / page["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == page["sha256"]
        entries.extend(json.loads(path.read_text())["results"])
        releases.add(page["release"])
    assert len(releases) == 1 and len(entries) == manifest["entries"]
    assert len({e["primaryAccession"] for e in entries}) == len(entries)
    train = _read_prepared_tsv(
        str(prepared / "deeploc21_localization_train_validation.tsv")
    )
    hpa = _read_prepared_tsv(str(prepared / "deeploc21_hpa_test.tsv"))
    known = train + hpa
    accessions = {r["accession"].split("-")[0] for r in known}
    sequences = {to_canonical_aa_sequence(r["sequence"]) for r in known}
    skip, candidates, retained = Counter(), [], []
    seen = set()
    for entry in sorted(entries, key=lambda e: e["primaryAccession"]):
        row, reason = experimental_perox_row(entry)
        if row is None:
            skip[reason] += 1
            continue
        candidates.append(row)
        aliases = set(row["secondary_accessions"] + [row["accession"]])
        if aliases & accessions:
            skip["development_or_hpa_accession_overlap"] += 1
        elif row["sequence"] in sequences:
            skip["development_or_hpa_exact_sequence_overlap"] += 1
        elif ensembl_protein_ids(entry) & accessions:
            skip["development_or_hpa_ensembl_overlap"] += 1
        elif row["sequence"] in seen:
            skip["duplicate_new_sequence"] += 1
        else:
            retained.append(row)
            seen.add(row["sequence"])
    assert_disjoint(known, retained)
    selection = dict(
        retrieved=len(entries),
        experimental_candidates=len(candidates),
        retained=len(retained),
        skipped=dict(skip),
        rows=retained,
        dataset_sha256=dataset_digest(retained),
    )
    frozen = root / "selection_before_scoring.json"
    if frozen.exists():
        assert json.loads(frozen.read_text()) == selection
    else:
        atomic_write_json(str(frozen), selection)
    print(
        "Selection frozen",
        {k: v for k, v in selection.items() if k != "rows"},
        flush=True,
    )
    if not retained:
        raise ValueError("No eligible new positives.")
    audit_path = root / "homology.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        assert audit["dataset_sha256"] == dataset_digest(retained)
        assert audit["training_sha256"] == dataset_digest(train)
    else:
        homology = mmseqs_homology_report(
            [dict(r, peroxisome=0) for r in train],
            retained,
            threads=2,
            include_hit_indices=True,
        )
        if homology["status"] != "ok":
            raise ValueError(str(homology))
        groups, clustering = mmseqs_cluster_assignments(retained, threads=2)
        if clustering["status"] != "ok":
            raise ValueError(str(clustering))
        audit = dict(
            homology=homology,
            groups=list(groups),
            clustering=clustering,
            dataset_sha256=dataset_digest(retained),
            training_sha256=dataset_digest(train),
        )
        atomic_write_json(str(audit_path), audit)
    hits = set(audit["homology"]["_hit_eval_indices"])
    groups = np.asarray(audit["groups"])
    protocol = json.loads((root / "protocol.json").read_text())
    predictions = {}
    import torch

    torch.set_num_threads(1)
    column = list(LABELS).index("peroxisome")
    for name in ["cnn_legacy", "integrated"]:
        path = experiment / "final" / (name + ".pt")
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest() == protocol[name + "_sha256"]
        )
        model = load_localize_model(str(path))
        assert model["localization_model"]["class_order"] == list(LABELS)
        result = _predict_model_on_rows(model, retained)
        predictions[name] = result
        np.savez_compressed(root / (name + "_new_positives.npz"), **result)
    base = predictions["cnn_legacy"]["prediction_matrix"][:, column].astype(bool)
    new = predictions["integrated"]["prediction_matrix"][:, column].astype(bool)
    masks = {
        "all_new": np.ones(len(retained), dtype=bool),
        "no_detected_homology": np.asarray(
            [i not in hits for i in range(len(retained))]
        ),
        "human": np.asarray([r["taxon_id"] == 9606 for r in retained]),
        "nonhuman": np.asarray([r["taxon_id"] != 9606 for r in retained]),
    }
    for name, loc in [("matrix", "SL-0202"), ("membrane", "SL-0203")]:
        masks[name] = np.asarray([loc in r["locations"] for r in retained])
    masks["unspecified"] = np.asarray(
        [not set(r["locations"]) & {"SL-0202", "SL-0203"} for r in retained]
    )
    masks["human_no_detected_homology"] = masks["human"] & masks["no_detected_homology"]
    results = {
        name: compare_recall(base[mask], new[mask], groups[mask])
        for name, mask in masks.items()
    }
    negatives = {}
    for name in ["cnn_legacy", "integrated"]:
        with np.load(experiment / "final" / (name + "_hpa.npz")) as saved:
            mask = saved["target"][:, column] == 0
            fp = int(saved["prediction"][mask, column].sum())
            negatives[name] = dict(
                n=int(mask.sum()),
                false_positives=fp,
                specificity=float(1 - fp / mask.sum()),
            )
    report = dict(
        selection={k: v for k, v in selection.items() if k != "rows"},
        release=next(iter(releases)),
        comparisons=results,
        original_hpa_negative_reference=negatives,
        note="Positive-only extension: recall is estimable; precision/F1 are not. HPA negatives are unchanged and reported separately.",
    )
    atomic_write_json(str(root / "comparison.json"), report)
    with (root / "predictions.tsv").open("w") as stream:
        fields = [
            "accession",
            "organism",
            "taxon_id",
            "locations",
            "evidence",
            "training_homology",
            "baseline_probability",
            "integrated_probability",
            "baseline_detected",
            "integrated_detected",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for i, row in enumerate(retained):
            writer.writerow(
                dict(
                    accession=row["accession"],
                    organism=row["organism"],
                    taxon_id=row["taxon_id"],
                    locations=";".join(row["locations"]),
                    evidence=";".join(row["evidence"]),
                    training_homology=i in hits,
                    baseline_probability=predictions["cnn_legacy"]["prob_matrix"][
                        i, column
                    ],
                    integrated_probability=predictions["integrated"]["prob_matrix"][
                        i, column
                    ],
                    baseline_detected=int(base[i]),
                    integrated_detected=int(new[i]),
                )
            )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

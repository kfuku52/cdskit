#!/usr/bin/env python3
"""Rebuild localization homology groups without changing original fold assignments."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdskit.deeploc_benchmark import _read_prepared_tsv
from cdskit.localize_evaluation import dataset_digest, normalize_partition_id
from cdskit.localize_pipeline_config import file_digest
from cdskit.localize_splits import audit_homology_partitions
from cdskit.util import atomic_write_json


def component_groups(seed_groups, pairs):
    """Keep original groups and join every reported edge, including indirect paths."""
    parents = list(range(len(seed_groups)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    first = {}
    for i, value in enumerate(seed_groups):
        group = normalize_partition_id(value)
        if group is None:
            raise ValueError("Complete original group identifiers are required.")
        union(i, first.setdefault(group, i))
    hit_count = 0
    for a, b in pairs:
        if not 0 <= a < len(parents) or not 0 <= b < len(parents):
            raise ValueError("MMseqs hit identifier is outside the source rows.")
        union(a, b)
        hit_count += 1
    return [str(find(i)) for i in range(len(parents))], hit_count


def read_pairs(path):
    with path.open() as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5 or any(
                not value.startswith("r") or not value[1:].isdigit()
                for value in fields[:2]
            ):
                raise ValueError("Invalid MMseqs all-against-all hit row.")
            yield int(fields[0][1:]), int(fields[1][1:])


def prepare(training_tsv, clusters, output, threads=4):
    rows = _read_prepared_tsv(str(training_tsv))
    source_digest = dataset_digest(rows)
    seed = json.loads(Path(clusters).read_text())
    if (
        not rows
        or seed.get("dataset_sha256") != source_digest
        or len(seed.get("groups", [])) != len(rows)
    ):
        raise ValueError("Original cluster assignments must match the source rows.")
    binary = shutil.which("mmseqs")
    if binary is None:
        raise FileNotFoundError("MMseqs is required to rebuild homology components.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "source_sha256": file_digest(training_tsv),
        "source_dataset_sha256": source_digest,
        "original_clusters_sha256": file_digest(clusters),
        "script_sha256": file_digest(__file__),
        "mmseqs": subprocess.check_output([binary, "version"], text=True).strip(),
        "min_seq_id": 0.3,
        "coverage": 0.8,
        "cov_mode": 0,
        "threads": threads,
        "method": "Union original groups with all-vs-all MMseqs hits; exclude entire connected components crossing official folds. No model scores used.",
    }
    atomic_write_json(str(output / "protocol.json"), protocol)
    fasta = output / "source.faa"
    with fasta.open("w") as stream:
        for i, row in enumerate(rows):
            stream.write(">r{}\n{}\n".format(i, row["sequence"]))
    hits = output / "all_vs_all_hits.tsv"
    subprocess.run(
        [
            binary,
            "easy-search",
            str(fasta),
            str(fasta),
            str(hits),
            str(output / "tmp"),
            "--min-seq-id",
            "0.3",
            "-c",
            "0.8",
            "--cov-mode",
            "0",
            "--format-output",
            "query,target,pident,qcov,tcov",
            "--threads",
            str(threads),
            "-v",
            "1",
        ],
        check=True,
    )
    groups, hit_count = component_groups(seed["groups"], read_pairs(hits))
    folds = defaultdict(set)
    for row, group in zip(rows, groups, strict=True):
        fold = normalize_partition_id(row.get("fold_id"))
        if fold is None:
            raise ValueError("Complete source fold identifiers are required.")
        folds[group].add(fold)
    clean = [
        dict(row, cluster_id=group)
        for row, group in zip(rows, groups, strict=True)
        if len(folds[group]) == 1
    ]
    names = sorted({row["fold_id"] for row in rows})
    partitions = {
        name: [row for row in clean if row["fold_id"] == name] for name in names
    }
    if any(not values for values in partitions.values()):
        raise ValueError("Component exclusion leaves an empty source fold.")
    audit = audit_homology_partitions(partitions, threads=threads)
    report = {
        "status": "ok",
        "tool": "mmseqs",
        "method": protocol["method"],
        "min_seq_id": 0.3,
        "coverage": 0.8,
        "cov_mode": 0,
        "rows": len(rows),
        "cluster_count": len(set(groups)),
        "all_vs_all_hit_rows": hit_count,
        "kept_rows": len(clean),
        "removed_cross_fold_component_rows": len(rows) - len(clean),
        "clean_dataset_sha256": dataset_digest(clean),
        "cross_partition_homology": audit,
        "hit_file_sha256": file_digest(hits),
    }
    atomic_write_json(
        str(output / "clusters.json"),
        {"dataset_sha256": source_digest, "groups": groups, "report": report},
    )
    atomic_write_json(str(output / "report.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_tsv", required=True)
    parser.add_argument("--clusters", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive.")
    print(
        json.dumps(
            prepare(args.training_tsv, args.clusters, args.output, args.threads),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

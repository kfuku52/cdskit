"""Audit rnaSPAdes transcript groups and split groups with complete locus evidence."""

from collections import defaultdict
import gzip
from itertools import pairwise
from pathlib import Path
import re

from cdskit.atomicio import atomic_output_paths, validate_distinct_paths
from cdskit.tsvio import json_cell, read_tsv_columns, write_tsv


TRANSCRIPT_ID = re.compile(r"NODE_\d+_length_\d+_cov_\S+_g(\d+)_i(\d+)")
EDGE_ID = re.compile(r"EDGE_(\d+)_length_(\d+)_cov_[^:';\s]+'?")
SIGNED_EDGE = re.compile(r"\d+[+-]")
GROUP_COLUMNS = ("transcript_id", "original_group_id", "group_id")
REPORT_COLUMNS = (
    "original_group_id",
    "transcripts",
    "mapped_transcripts",
    "locus_ids",
    "decision",
    "bridge_edges",
    "bridge_transcripts",
    "bridge_junctions",
)


def open_text(path):
    if Path(path).suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def transcript_key(identifier):
    match = TRANSCRIPT_ID.fullmatch(identifier)
    if not match:
        raise ValueError(f"Invalid rnaSPAdes transcript ID: {identifier!r}")
    return f"g{match[1]}", f"i{match[2]}"


def transcript_groups(path):
    by_group = defaultdict(list)
    isoforms = set()
    with open_text(path) as stream:
        for line in stream:
            if not line.startswith(">"):
                continue
            identifier, *_ = line[1:].split(maxsplit=1) or [""]
            group, isoform = transcript_key(identifier)
            if (group, isoform) in isoforms:
                raise ValueError(
                    f"Duplicate rnaSPAdes transcript key: {group}/{isoform}"
                )
            isoforms.add((group, isoform))
            by_group[group].append(identifier)
    if not by_group:
        raise ValueError(f"No rnaSPAdes transcripts in {path}")
    return by_group


def edge_lengths(path, kmer_size):
    lengths = {}
    with open_text(path) as stream:
        for line in stream:
            if not line.startswith(">"):
                continue
            token = re.split(r"[:;]", line[1:].strip(), maxsplit=1)[0]
            match = EDGE_ID.fullmatch(token)
            if not match:
                raise ValueError(f"Invalid FASTG edge header: {token!r}")
            edge, length = int(match[1]), int(match[2]) - kmer_size
            if length <= 0 or lengths.setdefault(edge, length) != length:
                raise ValueError(f"Inconsistent FASTG edge length: {token!r}")
    if not lengths:
        raise ValueError(f"No FASTG edges in {path}")
    return lengths


def read_paths(path, needed, retained):
    paths = {}
    seen = set()
    populated = set()
    current = None
    with open_text(path) as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if line.startswith("NODE_"):
                current = line if line in needed else None
                if current is not None:
                    if current in seen:
                        raise ValueError(f"Duplicate transcript path: {current}")
                    seen.add(current)
                    if current in retained:
                        paths[current] = []
                continue
            if current is not None:
                segment = tuple(line.removesuffix(";").split(","))
                if not all(SIGNED_EDGE.fullmatch(edge) for edge in segment):
                    raise ValueError(f"Invalid paths segment for {current}: {line!r}")
                populated.add(current)
                if current in paths:
                    paths[current].append(segment)
    for identifier in needed:
        if identifier not in populated:
            raise ValueError(f"No graph path for {identifier}")
    return paths


def articulation_points(graph):
    """Find cut vertices in a sparse undirected graph without recursive DFS."""
    depth, low = {}, {}
    cuts = set()
    for root in graph:
        if root in depth:
            continue
        depth[root] = low[root] = len(depth)
        stack = [(root, iter(graph[root]))]
        root_children = 0
        while stack:
            node, neighbors = stack[-1]
            neighbor = next(neighbors, None)
            if neighbor is None:
                stack.pop()
                if stack:
                    parent = stack[-1][0]
                    low[parent] = min(low[parent], low[node])
                    if parent != root and low[node] >= depth[parent]:
                        cuts.add(parent)
            elif neighbor not in depth:
                depth[neighbor] = low[neighbor] = len(depth)
                stack.append((neighbor, iter(graph[neighbor])))
                root_children += int(node == root)
            elif len(stack) < 2 or neighbor != stack[-2][0]:
                low[node] = min(low[node], depth[neighbor])
        if root_children > 1:
            cuts.add(root)
    return cuts


def bridges(members, paths, lengths):
    by_edge = defaultdict(set)
    for identifier in members:
        for segment in paths[identifier]:
            for signed in segment:
                edge = int(signed[:-1])
                if edge not in lengths:
                    raise ValueError(f"Unknown graph edge {edge} in path {identifier}")
                if lengths[edge] > 300:
                    by_edge[edge].add(identifier)
    incidence = defaultdict(set)
    for edge, seen in by_edge.items():
        if len(seen) > 1:
            incidence[edge] = seen
            for identifier in seen:
                incidence[identifier].add(edge)
    cuts = articulation_points(incidence)
    return sorted(k for k in cuts if isinstance(k, int)), sorted(
        k for k in cuts if isinstance(k, str)
    )


def read_loci(path, ids):
    result = {}
    for transcript, locus in read_tsv_columns(path, ("transcript_id", "locus_id")):
        if transcript not in ids or transcript in result or not locus.strip():
            raise ValueError(
                f"Unknown or repeated transcript or empty locus: {transcript!r}"
            )
        result[transcript] = locus
    return result


def read_junction_support(path, paths):
    support = {}
    for transcript, left, right, count in read_tsv_columns(
        path, ("transcript_id", "from_edge", "to_edge", "read_pairs")
    ):
        junction = transcript, left, right
        if (
            junction in support
            or not any(
                (left, right) in pairwise(segment)
                for segment in paths.get(transcript, ())
            )
            or not re.fullmatch(r"[0-9]+", count)
        ):
            raise ValueError(f"Invalid or repeated junction support: {junction!r}")
        support[junction] = int(count)
    return support


def report_junctions(bridge_transcripts, paths, support):
    return [
        {
            "transcript_id": identifier,
            "from_edge": left,
            "to_edge": right,
            "read_pairs": support.get((identifier, left, right)),
        }
        for identifier in bridge_transcripts
        for segment in paths[identifier]
        for left, right in pairwise(segment)
    ]


def regroup_main(args):
    if bool(args.paths) != bool(args.fastg) or bool(args.paths) != bool(args.kmer_size):
        raise ValueError("--paths, --fastg and --kmer_size must be supplied together")
    if args.kmer_size < 0 or (args.junction_table and not args.paths):
        raise ValueError(
            "A positive k-mer size and graph inputs are required for junction evidence"
        )
    if not (args.paths or args.locus_table):
        raise ValueError("Specify graph inputs, a locus table, or both")
    inputs = [
        args.seqfile,
        args.paths,
        args.fastg,
        args.locus_table,
        args.junction_table,
    ]
    if args.outfile == "-" or args.report == "-":
        raise ValueError("Group and report outputs require file paths")
    validate_distinct_paths(inputs, [args.outfile, args.report])
    groups = transcript_groups(args.seqfile)
    ids = {identifier for members in groups.values() for identifier in members}
    loci = read_loci(args.locus_table, ids) if args.locus_table else {}
    paths = {}
    if args.paths:
        # Singleton paths still undergo input validation. Only multi-transcript
        # groups participate in the bridge search.
        retained = (
            ids
            if args.junction_table
            else {
                identifier
                for members in groups.values()
                if len(members) > 1
                for identifier in members
            }
        )
        paths = read_paths(args.paths, ids, retained)
    lengths = edge_lengths(args.fastg, args.kmer_size) if args.fastg else {}
    support = (
        read_junction_support(args.junction_table, paths) if args.junction_table else {}
    )
    split_assignments, report_rows = {}, []
    for group, members in groups.items():
        if len(members) < 2:
            continue
        mapped = {
            identifier: loci[identifier] for identifier in members if identifier in loci
        }
        locus_ids = sorted(set(mapped.values()))
        split = len(locus_ids) > 1 and len(mapped) == len(members)
        if split:
            children = {
                locus: f"{group}.split{i}" for i, locus in enumerate(locus_ids, 1)
            }
            split_assignments.update(
                (identifier, children[locus]) for identifier, locus in mapped.items()
            )
        edge_cuts, transcript_cuts = (
            bridges(members, paths, lengths) if paths else ([], [])
        )
        if split:
            decision = "split_by_locus"
        elif len(locus_ids) > 1:
            decision = "partial_multi_locus_evidence"
        elif edge_cuts or transcript_cuts:
            decision = "review_bridge"
        else:
            decision = "retain"
        report_rows.append(
            dict(
                original_group_id=group,
                transcripts=len(members),
                mapped_transcripts=len(mapped),
                locus_ids=json_cell(locus_ids),
                decision=decision,
                bridge_edges=json_cell(edge_cuts),
                bridge_transcripts=json_cell(transcript_cuts),
                bridge_junctions=json_cell(
                    report_junctions(transcript_cuts, paths, support)
                ),
            )
        )
    assignment_rows = (
        dict(
            transcript_id=identifier,
            original_group_id=group,
            group_id=split_assignments.get(identifier, group),
        )
        for group, members in groups.items()
        for identifier in members
    )
    with atomic_output_paths([args.outfile, args.report]) as (assignments, report):
        write_tsv(assignments, assignment_rows, GROUP_COLUMNS)
        write_tsv(report, report_rows, REPORT_COLUMNS)

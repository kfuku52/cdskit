"""Bulk alignment coordinates and pairwise amino-acid identity."""

import argparse
import json
from pathlib import Path
from cdskit.atomicio import atomic_output_path, validate_distinct_paths
from cdskit.column_mapping import analyze_mapping
from cdskit.translate import translate_sequence_string


def r_functions_path():
    return Path(__file__).with_name("data_r") / "coordinates.R"


def trimmed_mapping(trimmed, untrimmed, gap="04", *, ambiguity="leftmost"):
    if ambiguity not in {"leftmost", "unresolved", "error"}:
        raise ValueError("Unknown ambiguity policy")
    trimmed = [trimmed] if isinstance(trimmed, str) else trimmed
    untrimmed = [untrimmed] if isinstance(untrimmed, str) else untrimmed
    positions = [i for i, token in enumerate(trimmed) if token != gap]
    source = [token for token in untrimmed if token != gap]
    target = [trimmed[i] for i in positions]
    analysis = analyze_mapping(source, target)
    result: list[int | None] = [None] * len(trimmed)
    if analysis.status == "ambiguous" and ambiguity == "error":
        raise ValueError("Ambiguous trimmed-to-untrimmed correspondence")
    if analysis.status != "unmatched" and not (
        analysis.status == "ambiguous" and ambiguity == "unresolved"
    ):
        for index, site in zip(positions, analysis.leftmost, strict=True):
            result[index] = site + 1
    return dict(status=analysis.status, mapping=result)


def pairwise_identity(sequences, pairs, *, mode="protein", genetic_code=1):
    if mode not in {"cds", "protein"}:
        raise ValueError("Expected cds or protein")
    if len({len(s) for s in sequences.values()}) > 1:
        raise ValueError("Sequence identity requires an alignment")
    proteins = dict(sequences)
    if mode == "cds":
        for name, sequence in proteins.items():
            if len(sequence) % 3:
                raise ValueError("CDS alignment length is not divisible by three")
            triplets = [sequence[i : i + 3] for i in range(0, len(sequence), 3)]
            masked = "".join(c if set(c) <= set("ACGT") else "NNN" for c in triplets)
            proteins[name] = translate_sequence_string(masked, genetic_code, False)
    alphabet = set("ACDEFGHIKLMNPQRSTVWY")
    values: list[float | None] = []
    for left, right in pairs:
        if left not in proteins or right not in proteins:
            values.append(None)
            continue
        valid = [
            (a, b)
            for a, b in zip(proteins[left], proteins[right], strict=True)
            if a in alphabet and b in alphabet
        ]
        values.append(sum(a == b for a, b in valid) / len(valid) if valid else None)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="Input JSON coordinate request")
    parser.add_argument("response", help="Output JSON response")
    args = parser.parse_args()
    validate_distinct_paths([args.request], [args.response])
    request = json.loads(Path(args.request).read_text(encoding="utf-8-sig"))
    if request["operation"] == "mapping":
        result = {
            key: trimmed_mapping(
                value,
                request["untrimmed"].get(key, []),
                request.get("gap", "04"),
                ambiguity=request.get("ambiguity", "leftmost"),
            )
            for key, value in request["trimmed"].items()
        }
    elif request["operation"] == "identity":
        result = pairwise_identity(
            request["sequences"],
            request["pairs"],
            mode=request["mode"],
            genetic_code=request["genetic_code"],
        )
    else:
        raise ValueError("Unknown coordinate operation")
    with atomic_output_path(args.response) as output:
        Path(output).write_text(json.dumps(result, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Bulk alignment coordinates and pairwise amino-acid identity."""

import argparse
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
from cdskit.atomicio import atomic_output_path, validate_distinct_paths
from cdskit.column_mapping import analyze_mapping
from cdskit.translate import translate_sequence_string

_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_OBSERVED_AA = frozenset(_AMINO_ACIDS)
_AA_BYTES = bytes(_AMINO_ACIDS.find(chr(code)) + 1 for code in range(256))
_CONCRETE_BASES = frozenset("ACGT")
_CONCRETE_CODONS = frozenset("".join(codon) for codon in product("ACGT", repeat=3))
_NONCONCRETE_BYTES = bytes(int(chr(code) not in _CONCRETE_BASES) for code in range(256))


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


def _mask_identity_codons(sequence):
    if (
        type(sequence) is str
        and len(sequence) >= 384
        and len(sequence) % 3 == 0
        and sequence.isascii()
    ):
        import numpy as np

        masked = bytearray(sequence, "ascii")
        flags = np.frombuffer(masked.translate(_NONCONCRETE_BYTES), dtype=np.bool_)
        unusable = flags[::3] | flags[1::3] | flags[2::3]
        if not unusable.any():
            return sequence
        np.frombuffer(masked, dtype=np.uint8).reshape(-1, 3)[unusable] = ord("N")
        return masked.decode("ascii")
    triplets = (sequence[i : i + 3] for i in range(0, len(sequence), 3))
    if type(sequence) is str:
        return "".join(
            codon if codon in _CONCRETE_CODONS else "NNN" for codon in triplets
        )
    return "".join(
        codon if set(codon) <= _CONCRETE_BASES else "NNN" for codon in triplets
    )


def _scalar_identity(first, second):
    observed = identical = 0
    for a, b in zip(first, second, strict=True):
        if a in _OBSERVED_AA and b in _OBSERVED_AA:
            observed += 1
            identical += a == b
    return identical / observed if observed else None


def _pack_residues(sequence):
    if type(sequence) is not str or not sequence.isascii():
        return None
    import numpy as np

    codes = np.frombuffer(sequence.encode("ascii").translate(_AA_BYTES), dtype=np.uint8)

    def pack(values):
        return int.from_bytes(
            np.packbits(values, bitorder="little").tobytes(), "little"
        )

    # Five bits identify each canonical residue. One bitset tracks which
    # positions participate in a comparison at all.
    return pack(codes > 0), tuple(pack(codes & (1 << bit)) for bit in range(5))


def _packed_identity(first, second):
    common = first[0] & second[0]
    if not common:
        return None
    different = 0
    for a, b in zip(first[1], second[1], strict=True):
        different |= a ^ b
    return (common & ~different).bit_count() / common.bit_count()


def pairwise_identity(sequences, pairs, *, mode="protein", genetic_code=1):
    if mode not in {"cds", "protein"}:
        raise ValueError("Expected cds or protein")
    lengths = {len(s) for s in sequences.values()}
    if len(lengths) > 1:
        raise ValueError("Sequence identity requires an alignment")
    length = next(iter(lengths), 0)
    proteins = dict(sequences)
    if mode == "cds":
        for name, sequence in proteins.items():
            if len(sequence) % 3:
                raise ValueError("CDS alignment length is not divisible by three")
            masked = _mask_identity_codons(sequence)
            proteins[name] = translate_sequence_string(masked, genetic_code, False)
        length //= 3
    use_packed = type(pairs) in (list, tuple) and len(pairs) >= 32 and length >= 256
    cache_size = max(2, min(512, (8 * 1024 * 1024) // (length + 256)))

    @lru_cache(maxsize=cache_size)
    def packed(name):
        return _pack_residues(proteins[name])

    values: list[float | None] = []
    for left, right in pairs:
        if left not in proteins or right not in proteins:
            values.append(None)
            continue
        if use_packed:
            first, second = packed(left), packed(right)
            if first is not None and second is not None:
                values.append(_packed_identity(first, second))
                continue
        values.append(_scalar_identity(proteins[left], proteins[right]))
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

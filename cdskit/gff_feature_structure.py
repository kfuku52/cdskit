"""Validate GFF feature order, including explicitly ordered trans-spliced CDS."""

import re
from itertools import pairwise
from urllib.parse import unquote

STRUCTURAL_GFF_ATTRIBUTES = frozenset(
    {"exception", "gg_source_overlap", "pseudo", "ID", "Parent", "number", "part"}
)


def has_trans_splicing_exception(text):
    for field in str(text).split(";"):
        if not field.startswith("exception="):
            continue
        decoded = unquote(field.split("=", 1)[1])
        if any(x.strip() == "trans-splicing" for x in decoded.split(",")):
            return True
    return False


def has_source_overlap_marker(text):
    """Return whether a source-audited CDS block allows coordinate overlap.

    ``gg_source_overlap=confirmed`` is a local provenance marker added only
    after the source CDS and the overlapping GFF blocks have been compared.
    It deliberately has its own mode instead of being treated as a biological
    exception such as ribosomal slippage.
    """
    text = str(text)
    if "gg_source_overlap" not in text:
        return False
    for field in text.split(";"):
        key, separator, value = field.partition("=")
        if (
            separator
            and key.strip() == "gg_source_overlap"
            and value.strip().lower() == "confirmed"
        ):
            return True
    return False


def ordered_annotated_blocks(rows, gene_id):
    """Return (sequence, strand, start, end) blocks and their splice mode.

    Rows contain those four coordinate fields followed by GFF attributes.
    Trans-splicing is accepted only when every row declares the exception and
    every distinct block has an unambiguous, complete 1..N part order.
    """
    rows = list(rows)
    coordinates = [
        (str(seq), str(strand), int(start), int(end))
        for seq, strand, start, end, _attrs in rows
    ]
    attributes = []
    for *_coords, text in rows:
        fields: dict[str, str] = {}
        for field in str(text).split(";"):
            if "=" in field:
                key, value = field.split("=", 1)
                # Only attributes that control block interpretation must be
                # unambiguous. Public GFFs may repeat free-text annotations
                # (for example AltName) within one feature line.
                if (
                    key in STRUCTURAL_GFF_ATTRIBUTES
                    and key in fields
                    and fields[key] != value
                ):
                    raise ValueError(f"Conflicting GFF attribute {key} for {gene_id}")
                fields[key] = value
        attributes.append(fields)
    declared, slippage = [], []
    for attr in attributes:
        exceptions = [
            value.strip() for value in unquote(attr.get("exception", "")).split(",")
        ]
        declared.append("trans-splicing" in exceptions)
        slippage.append("ribosomal slippage" in exceptions)
    source_overlap = [has_source_overlap_marker(text) for *_coords, text in rows]
    pseudogene = [attr.get("pseudo", "").lower() == "true" for attr in attributes]
    if any(pseudogene):
        if not all(pseudogene) or any(declared):
            raise ValueError(f"Incomplete or mixed pseudogene annotation for {gene_id}")
        identities = {(a.get("ID", ""), a.get("Parent", "")) for a in attributes}
        if len(identities) != 1 or not next(iter(identities))[0]:
            raise ValueError(
                f"Pseudogene parts require one explicit CDS ID for {gene_id}"
            )
        return ordered_feature_blocks(
            coordinates, gene_id, allow_overlap=True
        ), "pseudogene"
    if any(slippage):
        # NCBI represents a programmed frameshift as overlapping parts of the
        # SAME CDS feature. A free-text Note alone is not an exception contract.
        if not all(slippage) or any(declared):
            raise ValueError(
                f"Incomplete or mixed ribosomal slippage annotation for {gene_id}"
            )
        identities = {(a.get("ID", ""), a.get("Parent", "")) for a in attributes}
        if len(identities) != 1 or not next(iter(identities))[0]:
            raise ValueError(
                f"Ribosomal slippage requires one explicit CDS ID for {gene_id}"
            )
        return ordered_feature_blocks(
            coordinates, gene_id, allow_overlap=True
        ), "ribosomal-slippage"
    if any(source_overlap):
        if not all(source_overlap) or any(declared) or any(slippage) or any(pseudogene):
            raise ValueError(
                f"Incomplete or mixed source-overlap annotation for {gene_id}"
            )
        identities = {(a.get("ID", ""), a.get("Parent", "")) for a in attributes}
        if len(identities) != 1 or not next(iter(identities))[0]:
            raise ValueError(
                f"Source-overlap parts require one explicit CDS ID for {gene_id}"
            )
        blocks = ordered_feature_blocks(coordinates, gene_id, allow_overlap=True)
        genomic_blocks = sorted(blocks, key=lambda block: (block[2], block[3]))
        if not any(right[2] <= left[3] for left, right in pairwise(genomic_blocks)):
            raise ValueError(
                f"Source-overlap marker requires overlapping blocks for {gene_id}"
            )
        return blocks, "source-overlap"
    if not any(declared):
        if len({block[0] for block in coordinates}) > 1:
            # Fragmented assemblies can explicitly number a transcript's CDS
            # across contigs. Preserve that source order without claiming
            # trans-splicing or inventing genomic introns between contigs.
            parents = {a.get("Parent", "") for a in attributes}
            if (
                len(parents) != 1
                or not next(iter(parents))
                or "," in next(iter(parents))
            ):
                raise ValueError(f"Conflicting GFF coordinate systems for {gene_id}")
            parts: dict[int, tuple[str, str, int, int]] = {}
            for block, attr in zip(coordinates, attributes, strict=True):
                number = attr.get("number", "")
                if not re.fullmatch(r"[1-9][0-9]*", number):
                    raise ValueError(
                        f"Cross-contig CDS requires explicit number order for {gene_id}"
                    )
                index = int(number)
                if index in parts and parts[index] != block:
                    raise ValueError(
                        f"Conflicting cross-contig CDS number for {gene_id}"
                    )
                parts[index] = block
            if set(parts) != set(range(1, len(parts) + 1)) or len(
                set(parts.values())
            ) != len(parts):
                raise ValueError(
                    f"Incomplete cross-contig CDS number order for {gene_id}"
                )
            ordered = [parts[i] for i in sorted(parts)]
            for sequence in {block[0] for block in ordered}:
                local = [block for block in ordered if block[0] == sequence]
                if ordered_feature_blocks(local, gene_id) != local:
                    raise ValueError(
                        f"Conflicting cross-contig CDS coordinate order for {gene_id}"
                    )
            return ordered, "ordered-fragments"
        return ordered_feature_blocks(coordinates, gene_id), "cis"
    if not all(declared):
        raise ValueError(f"Incomplete trans-splicing annotation for {gene_id}")
    parts = {}
    coordinate_parts: dict[tuple[str, str, int, int], int] = {}
    totals = set()
    for coordinate, attr in zip(coordinates, attributes, strict=True):
        match = re.fullmatch(r"([1-9][0-9]*)(?:/([1-9][0-9]*))?", attr.get("part", ""))
        if match is None:
            raise ValueError(
                f"Trans-splicing requires explicit part order for {gene_id}"
            )
        part = int(match[1])
        if match[2]:
            totals.add(int(match[2]))
        if part in parts and parts[part] != coordinate:
            raise ValueError(f"Conflicting trans-splicing part for {gene_id}")
        if coordinate in coordinate_parts and coordinate_parts[coordinate] != part:
            raise ValueError(f"Repeated trans-splicing coordinate for {gene_id}")
        parts[part] = coordinate
        coordinate_parts[coordinate] = part
    if set(parts) != set(range(1, len(parts) + 1)) or (
        totals and totals != {len(parts)}
    ):
        raise ValueError(f"Incomplete trans-splicing part order for {gene_id}")
    # Validate coordinate geometry within each genomic strand, without inventing
    # an order between different strands/contigs or overriding annotated parts.
    systems: dict[tuple[str, str], list[tuple[str, str, int, int]]] = {}
    for block in parts.values():
        systems.setdefault(block[:2], []).append(block)
    for blocks in systems.values():
        ordered_feature_blocks(blocks, gene_id)
    return [parts[i] for i in sorted(parts)], "trans-splicing"


def ordered_feature_blocks(rows, gene_id, allow_overlap=False):
    block_set = {
        (str(sequence), str(strand), int(start), int(end))
        for sequence, strand, start, end in rows
    }
    if len({(sequence, strand) for sequence, strand, _start, _end in block_set}) != 1:
        raise ValueError(f"Conflicting GFF coordinate systems for {gene_id}")
    if any(
        strand not in ("+", "-") or start < 1 or end < start
        for _sequence, strand, start, end in block_set
    ):
        raise ValueError(f"Invalid GFF coordinates or strand for {gene_id}")
    blocks = sorted(block_set, key=lambda block: (block[2], block[3]))
    if not allow_overlap and any(
        right[2] <= left[3] for left, right in pairwise(blocks)
    ):
        raise ValueError(f"Overlapping GFF feature blocks for {gene_id}")
    return blocks[::-1] if blocks[0][1] == "-" else blocks

"""Streaming exact-ID selection without realigning or changing record order."""

from cdskit.util import iter_seq_chunks


def iter_records(source, seqformat="fasta"):
    for chunk in iter_seq_chunks(source, seqformat):
        yield from chunk


def select_records(records, identifiers, *, require_all=True, reject_duplicates=True):
    wanted = set(identifiers)
    seen = set()
    for record in records:
        if record.id in seen and reject_duplicates:
            raise ValueError(f"Duplicate sequence ID: {record.id}")
        seen.add(record.id)
        if record.id in wanted:
            yield record
    if require_all and wanted - seen:
        raise ValueError(f"Missing sequence IDs: {sorted(wanted - seen)}")


def fasta_records(path, opener=None, *, identifiers=None):
    """Stream literal FASTA tuples, optionally materializing only named records.

    Input order and structural validation are the same in both modes. Missing
    requested IDs and duplicate matching records are left to the caller.
    """
    if opener is None:
        from cdskit.sequence_io import open_text

        opener = open_text
    wanted = None if identifiers is None else set(identifiers)
    header = identifier = ""
    selected = False
    sequence: list[str] = []
    with opener(path) as handle:
        for raw_line in handle:
            if raw_line.startswith(">"):
                next_header = raw_line[1:].rstrip("\r\n")
                fields = next_header.split(None, 1)
                if not fields:
                    raise ValueError(f"Empty FASTA header: {path}")
                if selected:
                    yield identifier, header, "".join(sequence)
                identifier = fields[0]
                selected = wanted is None or identifier in wanted
                header = next_header
                sequence = []
            elif selected:
                sequence.append(raw_line.strip())
            elif not header and raw_line.strip():
                raise ValueError(
                    f"FASTA sequence data precedes the first header: {path}"
                )
    if selected:
        yield identifier, header, "".join(sequence)

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


def fasta_records(path, opener=None):
    if opener is None:
        from cdskit.sequence_io import open_text

        opener = open_text
    header = ""
    sequence: list[str] = []
    with opener(path) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line.startswith(">"):
                if not line[1:].strip():
                    raise ValueError(f"Empty FASTA header: {path}")
                if header:
                    identifier = header.split(None, 1)[0]
                    yield identifier, header, "".join(sequence)
                header = line[1:]
                sequence = []
            elif header:
                sequence.append(line.strip())
            elif line.strip():
                raise ValueError(
                    f"FASTA sequence data precedes the first header: {path}"
                )
    if header:
        identifier = header.split(None, 1)[0]
        yield identifier, header, "".join(sequence)

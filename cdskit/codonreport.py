"""Versioned provenance reports for padding and ORF candidate selection."""

from itertools import chain, islice
import json

from cdskit.atomicio import (
    atomic_text_writer,
    atomic_write_json,
    validate_distinct_paths,
    validate_output_paths,
)
from cdskit.codonutil import CODON_SEMANTICS_VERSION
from cdskit.tsvio import write_sectioned_tsv


def validate_codon_output_paths(seqfile, outfile, report):
    """Protect direct callers as well as CLI callers, including stdout streams."""
    if outfile == "-" and report == "-":
        raise ValueError("Sequence output and report cannot both use standard output.")
    outputs = [path for path in (outfile, report) if path not in (None, "", "-")]
    validate_distinct_paths(inputs=[seqfile], outputs=outputs)
    validate_output_paths(outputs)


def _write_streaming_json(path, metadata, records):
    encoder = json.JSONEncoder(indent=2, ensure_ascii=False, allow_nan=False)
    header, tail = encoder.encode({**metadata, "sequences": []}).rsplit("[]", 1)
    with atomic_text_writer(path) as handle:
        handle.write(header + "[")
        separator = "\n    "
        has_rows = False
        for record in records:
            handle.write(separator)
            parts = encoder.iterencode(record)
            while batch := list(islice(parts, 2048)):
                # Newlines in JSON string values are already escaped.
                handle.write("".join(batch).replace("\n", "\n    "))
            separator = ",\n    "
            has_rows = True
        if has_rows:
            handle.write("\n  ")
        handle.write("]" + tail + "\n")


def write_codon_report(path, command, codontable, records, *, stream=False):
    if not path:
        return
    path = str(path)
    metadata = {
        "report_version": "1",
        "codon_semantics_version": CODON_SEMANTICS_VERSION,
        "command": command,
        "codon_table": codontable,
    }
    if path.lower().endswith(".json"):
        if stream:
            _write_streaming_json(path, metadata, records)
        else:
            atomic_write_json(path, {**metadata, "sequences": records}, indent=2)
        return
    metadata_row = {"section": "metadata", "data": metadata}
    rows = chain(
        [metadata_row],
        (
            {
                "section": "sequence",
                "input_order": record["input_order"],
                "seq_id": record["seq_id"],
                "data": record,
            }
            for record in records
        ),
    )
    write_sectioned_tsv(
        path, ["input_order", "seq_id", "data"], rows if stream else list(rows)
    )

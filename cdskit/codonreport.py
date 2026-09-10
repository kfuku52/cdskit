"""Versioned provenance reports for padding and ORF candidate selection."""

from cdskit.atomicio import (
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


def write_codon_report(path, command, codontable, records):
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
        atomic_write_json(path, {**metadata, "sequences": records}, indent=2)
        return
    rows = [{"section": "metadata", "data": metadata}]
    rows.extend(
        {
            "section": "sequence",
            "input_order": record["input_order"],
            "seq_id": record["seq_id"],
            "data": record,
        }
        for record in records
    )
    write_sectioned_tsv(path, ["input_order", "seq_id", "data"], rows)

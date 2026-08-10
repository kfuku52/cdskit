from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

import Bio.Data.CodonTable
import Bio.Seq
import Bio.SeqIO
from Bio.SeqRecord import SeqRecord

from cdskit.cliutil import resolve_threads as resolve_cli_threads
from cdskit.atomicio import (
    atomic_output_path as atomic_output_path,
    atomic_output_paths as atomic_output_paths,
    atomic_text_writer as atomic_text_writer,
    atomic_write_json as atomic_write_json,
    validate_distinct_paths as validate_distinct_paths,
)


T = TypeVar("T")
R = TypeVar("R")

GFF_DTYPE = [
    ("seqid", "U100"),
    ("source", "U100"),
    ("type", "U100"),
    ("start", "i4"),
    ("end", "i4"),
    ("score", "U100"),
    ("strand", "U10"),
    ("phase", "U10"),
    ("attributes", "U500"),
]
GFF_COLUMNS = (
    "seqid",
    "source",
    "type",
    "start",
    "end",
    "score",
    "strand",
    "phase",
    "attributes",
)
DNA_ALLOWED_CHARS = frozenset("ACGTRYSWKMBDHVNXacgtryswkmbdhvnx-?.")
PROTEIN_ALLOWED_CHARS = frozenset(
    "ABCDEFGHIKLMNPQRSTVWXYZJUO*abcdefghiklmnpqrstvwxyzjuo-?."
)
DEFAULT_MAX_SEQUENCE_RECORDS = 1_000_000
DEFAULT_MAX_SEQUENCE_RESIDUES = 1_000_000_000
DEFAULT_MAX_SEQUENCE_ID_LENGTH = 4096
DEFAULT_MAX_REGEX_LENGTH = 512
QUALITY_REQUIRED_FORMATS = frozenset(
    {
        "fastq",
        "fastq-sanger",
        "fastq-illumina",
        "fastq-solexa",
        "qual",
    }
)
DEFAULT_PROCESS_PARALLEL_MIN_RESIDUES = 16_000_000


def _repeated_group_has_ambiguous_content(pattern: str) -> bool:
    stack: list[int] = []
    in_character_class = False
    escaped = False
    for index, character in enumerate(pattern):
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "[":
            in_character_class = True
            continue
        if character == "]" and in_character_class:
            in_character_class = False
            continue
        if in_character_class:
            continue
        if character == "(":
            stack.append(index)
            continue
        if character != ")" or not stack:
            continue
        start = stack.pop()
        following = pattern[index + 1 : index + 2]
        if following not in ("*", "+", "?", "{"):
            continue
        content = pattern[start + 1 : index]
        content = re.sub(
            r"^\?(?::|=|!|<[=!]|P<[^>]+>)",
            "",
            content,
            count=1,
        )
        content_without_escapes = re.sub(r"\\.", "", content)
        if re.search(r"[*+?{|]", content_without_escapes):
            return True
    return False


def compile_safe_regex(
    pattern: Any,
    label: str = "regular expression",
) -> re.Pattern[str]:
    """Compile a bounded regex while rejecting common nested-repeat DoS forms."""
    pattern = str(pattern)
    if len(pattern) > DEFAULT_MAX_REGEX_LENGTH:
        raise ValueError(
            "{} should be at most {} characters.".format(
                label,
                DEFAULT_MAX_REGEX_LENGTH,
            )
        )
    if _repeated_group_has_ambiguous_content(pattern) or re.search(
        r"\\[1-9]|\(\?P=", pattern
    ):
        raise ValueError(
            "{} contains a potentially unsafe repeated group or backreference.".format(
                label
            )
        )
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ValueError("Invalid {}: {}.".format(label, exc)) from exc


def replace_record_sequence(
    record: SeqRecord,
    sequence: Any,
    preserve_features: bool = False,
) -> SeqRecord:
    """Replace sequence data without leaving stale per-letter annotations."""
    sequence = Bio.Seq.Seq(sequence)
    if len(sequence) != len(record):
        record.letter_annotations = {}
        if not preserve_features:
            record.features = []
    record.seq = sequence
    return record


def resolve_threads(threads: Any) -> int:
    try:
        return resolve_cli_threads(threads)
    except ValueError as exc:
        raise ValueError(str(exc) + " Exiting.\n") from exc


def parallel_map_ordered(
    items: Iterable[T],
    worker: Callable[[T], R],
    threads: int,
) -> list[R]:
    items = list(items)
    if (threads <= 1) or (len(items) <= 1):
        return [worker(item) for item in items]

    max_workers = min(threads, len(items))
    if len(items) <= (max_workers * 2):
        return [worker(item) for item in items]

    # Run multiple items per submitted task to reduce scheduler overhead
    # when per-item work is small.
    if len(items) <= (max_workers * 256):
        chunk_size = max(1, (len(items) + max_workers - 1) // max_workers)
    else:
        chunk_size = max(1, len(items) // (max_workers * 8))
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    def process_chunk(chunk: list[T]) -> list[R]:
        return [worker(item) for item in chunk]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunk_results = list(executor.map(process_chunk, chunks))

    out: list[R] = []
    for chunk_result in chunk_results:
        out.extend(chunk_result)
    return out


def should_use_process_pool(
    records: Sequence[SeqRecord],
    threads: int,
    min_total_residues: int | None = None,
) -> bool:
    """Select processes from estimated work instead of a record-count cliff."""
    if threads <= 1 or len(records) <= 1:
        return False
    if min_total_residues is None:
        min_total_residues = int(
            os.environ.get(
                "CDSKIT_PROCESS_PARALLEL_MIN_RESIDUES",
                DEFAULT_PROCESS_PARALLEL_MIN_RESIDUES,
            )
        )
    return sum(len(record) for record in records) >= min_total_residues


def read_seqs(seqfile: Any, seqformat: str) -> list[SeqRecord]:
    parsed = sys.stdin if seqfile == "-" else seqfile
    max_records = int(
        os.environ.get("CDSKIT_MAX_SEQUENCE_RECORDS", DEFAULT_MAX_SEQUENCE_RECORDS)
    )
    max_residues = int(
        os.environ.get("CDSKIT_MAX_SEQUENCE_RESIDUES", DEFAULT_MAX_SEQUENCE_RESIDUES)
    )
    max_id_length = int(
        os.environ.get(
            "CDSKIT_MAX_SEQUENCE_ID_LENGTH",
            DEFAULT_MAX_SEQUENCE_ID_LENGTH,
        )
    )
    records: list[SeqRecord] = []
    residues = 0
    for record in Bio.SeqIO.parse(parsed, seqformat):
        if len(record.id) > max_id_length:
            raise ValueError(
                "Sequence identifier exceeds {} characters. Set "
                "CDSKIT_MAX_SEQUENCE_ID_LENGTH to change the safety limit.".format(
                    max_id_length
                )
            )
        records.append(record)
        residues += len(record)
        if len(records) > max_records:
            raise ValueError(
                "Input exceeds {} sequence records. Set CDSKIT_MAX_SEQUENCE_RECORDS to change the safety limit.".format(
                    max_records
                )
            )
        if residues > max_residues:
            raise ValueError(
                "Input exceeds {:,} residues. Set CDSKIT_MAX_SEQUENCE_RESIDUES to change the safety limit.".format(
                    max_residues
                )
            )
    sys.stderr.write("Number of input sequences: {:,}\n".format(len(records)))
    return records


def iter_seq_chunks(
    seqfile: Any,
    seqformat: str,
    max_chunk_records: int = 10_000,
    max_chunk_residues: int = 32_000_000,
) -> Iterator[list[SeqRecord]]:
    """Yield bounded record batches while enforcing the normal input limits."""
    parsed = sys.stdin if seqfile == "-" else seqfile
    max_records = int(
        os.environ.get("CDSKIT_MAX_SEQUENCE_RECORDS", DEFAULT_MAX_SEQUENCE_RECORDS)
    )
    max_residues = int(
        os.environ.get("CDSKIT_MAX_SEQUENCE_RESIDUES", DEFAULT_MAX_SEQUENCE_RESIDUES)
    )
    max_id_length = int(
        os.environ.get("CDSKIT_MAX_SEQUENCE_ID_LENGTH", DEFAULT_MAX_SEQUENCE_ID_LENGTH)
    )
    chunk: list[SeqRecord] = []
    chunk_residues = 0
    total_records = 0
    total_residues = 0
    for record in Bio.SeqIO.parse(parsed, seqformat):
        if len(record.id) > max_id_length:
            raise ValueError(
                "Sequence identifier exceeds {} characters. Set "
                "CDSKIT_MAX_SEQUENCE_ID_LENGTH to change the safety limit.".format(
                    max_id_length
                )
            )
        record_residues = len(record)
        total_records += 1
        total_residues += record_residues
        if total_records > max_records:
            raise ValueError(
                "Input exceeds {} sequence records. Set CDSKIT_MAX_SEQUENCE_RECORDS to change the safety limit.".format(
                    max_records
                )
            )
        if total_residues > max_residues:
            raise ValueError(
                "Input exceeds {:,} residues. Set CDSKIT_MAX_SEQUENCE_RESIDUES to change the safety limit.".format(
                    max_residues
                )
            )
        if chunk and (
            len(chunk) >= max_chunk_records
            or chunk_residues + record_residues > max_chunk_residues
        ):
            yield chunk
            chunk = []
            chunk_residues = 0
        chunk.append(record)
        chunk_residues += record_residues
    if chunk:
        yield chunk
    sys.stderr.write("Number of input sequences: {:,}\n".format(total_records))


def write_seqs(
    records: Sequence[SeqRecord],
    outfile: Any,
    outseqformat: str,
) -> None:
    sys.stderr.write("Number of output sequences: {:,}\n".format(len(records)))
    if str(outseqformat).lower() in QUALITY_REQUIRED_FORMATS:
        for record in records:
            quality_keys = [
                key
                for key in ("phred_quality", "solexa_quality")
                if key in record.letter_annotations
            ]
            if not quality_keys:
                raise ValueError(
                    "Output format {} requires per-base quality scores, but "
                    "record {} has none. Use FASTA for transformed sequences.".format(
                        outseqformat,
                        record.id,
                    )
                )
            for key in quality_keys:
                if len(record.letter_annotations[key]) != len(record):
                    raise ValueError(
                        "Quality length mismatch for record {} ({}).".format(
                            record.id,
                            key,
                        )
                    )
    if outfile == "-":
        Bio.SeqIO.write(records, sys.stdout, outseqformat)
    else:
        with atomic_output_path(outfile) as temporary:
            Bio.SeqIO.write(records, temporary, outseqformat)


def stop_if_not_multiple_of_three(records: Sequence[SeqRecord]) -> None:
    has_non_triplet_sequence = False
    for record in records:
        if len(record) % 3 != 0:
            txt = "Sequence length is not multiple of three: {}\n".format(record.id)
            sys.stderr.write(txt)
            has_non_triplet_sequence = True
    if has_non_triplet_sequence:
        txt = (
            "Input sequence length should be multiple of three. "
            "Consider applying `cdskit pad` if the input is truncated coding sequences. Exiting.\n"
        )
        raise ValueError(txt)


def stop_if_not_aligned(records: Sequence[SeqRecord]) -> None:
    if len(records) <= 1:
        return
    first_len = len(records[0])
    for record in records[1:]:
        if len(record) != first_len:
            txt = "Sequence lengths were not identical. Please make sure input sequences are correctly aligned. Exiting.\n"
            raise ValueError(txt)


def stop_if_not_dna(
    records: Sequence[SeqRecord],
    label: str = "--seq_file",
) -> None:
    invalid_ids = list()
    invalid_chars = set()
    for record in records:
        seq_str = str(record.seq)
        record_invalid_chars = set(seq_str).difference(DNA_ALLOWED_CHARS)
        if record_invalid_chars:
            invalid_chars.update(record_invalid_chars)
            invalid_ids.append(str(record.id))
    if len(invalid_ids) == 0:
        return
    max_show = 10
    shown = ",".join(invalid_ids[:max_show])
    if len(invalid_ids) > max_show:
        shown += ",..."
    chars = "".join(sorted(invalid_chars))
    txt = (
        "Invalid non-DNA character(s) were detected in {} ({}) [chars: {}]. "
        "DNA-only input is required (use T instead of U). Exiting.\n"
    )
    raise ValueError(txt.format(label, shown, chars))


def stop_if_not_protein(
    records: Sequence[SeqRecord],
    label: str = "--seq_file",
) -> None:
    invalid_ids = list()
    invalid_chars = set()
    for record in records:
        seq_str = str(record.seq)
        record_invalid_chars = set(seq_str).difference(PROTEIN_ALLOWED_CHARS)
        if record_invalid_chars:
            invalid_chars.update(record_invalid_chars)
            invalid_ids.append(str(record.id))
    if len(invalid_ids) == 0:
        return
    max_show = 10
    shown = ",".join(invalid_ids[:max_show])
    if len(invalid_ids) > max_show:
        shown += ",..."
    chars = "".join(sorted(invalid_chars))
    txt = (
        "Invalid non-protein character(s) were detected in {} ({}) [chars: {}]. "
        "Protein-only input is required. Exiting.\n"
    )
    raise ValueError(txt.format(label, shown, chars))


def stop_if_not_seqtype(
    records: Sequence[SeqRecord],
    seqtype: str = "auto",
    label: str = "--seq_file",
) -> None:
    seqtype_value = str(seqtype).lower()
    if seqtype_value == "dna":
        stop_if_not_dna(records=records, label=label)
        return
    if seqtype_value == "protein":
        stop_if_not_protein(records=records, label=label)
        return
    if seqtype_value == "auto":
        allowed_chars = DNA_ALLOWED_CHARS | PROTEIN_ALLOWED_CHARS
        invalid_ids = list()
        invalid_chars = set()
        for record in records:
            seq_str = str(record.seq)
            record_invalid_chars = set(seq_str).difference(allowed_chars)
            if record_invalid_chars:
                invalid_chars.update(record_invalid_chars)
                invalid_ids.append(str(record.id))
        if len(invalid_ids) == 0:
            return
        max_show = 10
        shown = ",".join(invalid_ids[:max_show])
        if len(invalid_ids) > max_show:
            shown += ",..."
        chars = "".join(sorted(invalid_chars))
        txt = (
            "Invalid sequence character(s) were detected in {} ({}) [chars: {}]. "
            "DNA or protein input is required when --seq_type=auto. Exiting.\n"
        )
        raise ValueError(txt.format(label, shown, chars))

    txt = "Invalid --seq_type: {}. Choose from dna, protein, auto. Exiting.\n"
    raise ValueError(txt.format(seqtype))


def stop_if_invalid_codontable(
    codontable: Any,
    label: str = "--codon_table",
) -> None:
    try:
        Bio.Data.CodonTable.unambiguous_dna_by_id[int(codontable)]
        return
    except (KeyError, TypeError, ValueError):
        pass
    try:
        Bio.Data.CodonTable.unambiguous_dna_by_name[str(codontable)]
        return
    except KeyError:
        txt = "Invalid {}: {}. Exiting.\n"
        raise ValueError(txt.format(label, codontable)) from None


def translate_records(
    records: Sequence[SeqRecord],
    codontable: Any,
) -> list[SeqRecord]:
    from cdskit.translate import translate_sequence_string

    return [
        Bio.SeqRecord.SeqRecord(
            seq=Bio.Seq.Seq(
                translate_sequence_string(
                    seq_str=str(record.seq),
                    codontable=codontable,
                    to_stop=False,
                )
            ),
            id=record.id,
        )
        for record in records
    ]


def records2array(records: Sequence[SeqRecord]) -> Any:
    import numpy as np

    return np.array([list(str(record.seq)) for record in records])


def read_item_per_line_file(file: Any) -> list[str]:
    with open(file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() != ""]


def get_seqname(record: SeqRecord, seqnamefmt: str) -> str:
    name_items = seqnamefmt.split("_")
    seqname = ""
    for name_item in name_items:
        if name_item not in record.annotations:
            available_items = ", ".join(list(record.annotations.keys()))
            txt = "Invalid --seq_name_format element ({}) in {}. Available elements: {}"
            raise ValueError(txt.format(name_item, record.id, available_items))

        try:
            new_name = record.annotations[name_item]
            if isinstance(new_name, list):
                new_name = new_name[0]
            seqname += "_" + str(new_name)
        except Exception:
            available_items = ", ".join(list(record.annotations.keys()))
            txt = "Invalid --seq_name_format element ({}) in {}. Available elements: {}"
            raise ValueError(
                txt.format(name_item, record.id, available_items)
            ) from None
    seqname = re.sub("^_", "", seqname)
    seqname = re.sub(" ", "_", seqname)
    return seqname


def replace_seq2cds(record: SeqRecord) -> SeqRecord | None:
    for feature in record.features:
        if feature.type == "CDS":
            if feature.location is None:
                continue
            record.seq = feature.location.extract(record).seq
            return record
    txt = "Removed from output. No CDS found in: {}\n"
    sys.stderr.write(txt.format(record.id))
    return None


def read_gff(gff_file: Any) -> dict[str, Any]:
    import numpy as np

    header_lines = []
    rows = []
    max_widths = [1] * len(GFF_COLUMNS)
    with open(gff_file, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                header_lines.append(line)
            else:
                fields = [field.strip() for field in line.split("\t")]
                if len(fields) != len(GFF_COLUMNS):
                    raise ValueError(
                        "GFF line {} should contain exactly 9 tab-separated fields.".format(
                            line_number
                        )
                    )
                try:
                    start = int(fields[3])
                    end = int(fields[4])
                except ValueError as exc:
                    raise ValueError(
                        "GFF line {} has a non-integer start or end coordinate.".format(
                            line_number
                        )
                    ) from exc
                row = [*fields[:3], start, end, *fields[5:]]
                rows.append(tuple(row))
                for index, value in enumerate(fields):
                    if index not in (3, 4):
                        max_widths[index] = max(max_widths[index], len(value))
    if len(rows) == 0:
        data = np.array([], dtype=GFF_DTYPE)
    else:
        dtype = [
            (
                name,
                "i8" if name in ("start", "end") else "U{}".format(max_widths[index]),
            )
            for index, name in enumerate(GFF_COLUMNS)
        ]
        data = np.asarray(rows, dtype=dtype)
    sys.stderr.write(
        "Number of input GFF header lines: {:,}\n".format(len(header_lines))
    )
    sys.stderr.write("Number of input GFF records: {:,}\n".format(len(data)))
    sys.stderr.write(
        "Number of input GFF unique seqids: {:,}\n".format(
            len(np.unique(data["seqid"]))
        )
    )
    return {"header": header_lines, "data": data}


def write_gff(gff: dict[str, Any], outfile: Any) -> None:
    import numpy as np

    sys.stderr.write(
        "Number of output GFF header lines: {:,}\n".format(len(gff["header"]))
    )
    sys.stderr.write("Number of output GFF records: {:,}\n".format(len(gff["data"])))
    sys.stderr.write(
        "Number of output GFF unique seqids: {:,}\n".format(
            len(np.unique(gff["data"]["seqid"]))
        )
    )
    with atomic_text_writer(outfile) as f:
        if gff["header"]:
            f.write("\n".join(gff["header"]) + "\n")
        for row in gff["data"]:
            f.write("\t".join(map(str, row)) + "\n")


def coordinates2ranges(gff_coordinates: Sequence[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    if len(gff_coordinates) == 0:
        return ranges
    start = gff_coordinates[0]
    end = gff_coordinates[0]
    for i in range(1, len(gff_coordinates)):
        if gff_coordinates[i] == end + 1:
            end = gff_coordinates[i]
        else:
            ranges.append((start, end))
            start = gff_coordinates[i]
            end = gff_coordinates[i]
    ranges.append((start, end))
    return ranges

"""Deterministic, bounded labels and reusable old/new mapping tables."""

import csv


def iter_fasta_headers(lines):
    """Yield complete FASTA headers, including descriptions, without sequences."""
    for line in lines:
        if line.startswith(">"):
            yield line[1:].rstrip("\r\n")


def map_fasta_headers(lines, mapping):
    """Rewrite complete headers while streaming sequence text and line endings."""
    for line in lines:
        if line.startswith(">"):
            header = line[1:].rstrip("\r\n")
            replacement = mapping.get(header, header)
            if not replacement or "\r" in replacement or "\n" in replacement:
                raise ValueError(
                    "FASTA header replacements must be nonempty single lines"
                )
            line = ">" + replacement + line[len(header) + 1 :]
        yield line


def shorten_labels(labels, max_length):
    labels = list(labels)
    if max_length < 2:
        raise ValueError("max_length must be an integer >= 2")
    if len(labels) != len(set(labels)) or any(not name for name in labels):
        raise ValueError("Sequence labels must be unique and nonempty")
    reserved = set(labels)
    mapping = {}
    for index, name in enumerate((x for x in labels if len(x) > max_length), 1):
        counter = index
        while True:
            suffix = f"_{counter}"
            if len(suffix) >= max_length:
                raise ValueError(
                    "Label length cannot accommodate unique numeric suffixes"
                )
            candidate = name[: max_length - len(suffix)] + suffix
            if candidate not in reserved:
                break
            counter += 1
        mapping[name] = candidate
        reserved.add(candidate)
    return mapping


def write_label_mapping(mapping, path):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("old_name", "new_name"))
        writer.writerows(mapping.items())


def read_label_mapping(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["old_name", "new_name"]:
            raise ValueError("Expected old_name/new_name mapping columns")
        result = {}
        for row in reader:
            old, new = row["old_name"], row["new_name"]
            if None in row or not old or not new or old in result:
                raise ValueError("Invalid or duplicate sequence mapping")
            result[old] = new
    if len(set(result.values())) != len(result):
        raise ValueError("Mapping has duplicate new labels")
    return result


def apply_label_mapping(records, mapping):
    for record in records:
        old = record.id
        if old in mapping:
            record.id = mapping[old]
            record.name = record.id
            description = record.description
            record.description = (
                record.id + description[len(old) :]
                if description.startswith(old)
                else description
            )
        yield record

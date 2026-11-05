"""Quote-aware GFF3/GTF attributes, preserving source syntax."""

import re
import shlex
from collections import defaultdict
from urllib.parse import quote, unquote


def normalize_gff_attribute_value(raw_value):
    value = str(raw_value or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return unquote(value.strip())


def append_gff_attribute(attributes, key, value, *, safe=""):
    """Keep GTF quoted attributes and GFF3 encoded attributes in their syntax."""
    text = str(attributes).rstrip().rstrip(";")
    if re.match(r"^\s*[^\s=;]+\s+", text):
        value = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return text + "; " + key + ' "' + value + '";'
    return text + ";" + key + "=" + quote(str(value), safe=safe)


def parse_gff_attributes(attr_text, *, keys=None):
    """Decode all attributes, or selected keys with identical syntax checks.

    GTF still lexes every field, including unused ones, so a projection never
    hides malformed quoted input. GFF3 skips decoding unused values.
    """
    attrs = defaultdict(list)
    text = str(attr_text or "")
    if re.match(r"^\s*[^\s=;]+\s+", text):
        # GTF quoted scalars have no GFF3 list or percent-escape semantics.
        # Use the same quote-aware lexer as source syntax validation, so a
        # literal comma/semicolon/percent remains one exact attribute value.
        lexer = shlex.shlex(text, posix=True, punctuation_chars=";")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = iter(lexer)
        for key in tokens:
            if key == ";":
                continue
            value = next(tokens, None)
            if value is None or value == ";":
                raise ValueError("Missing GTF attribute value: " + key)
            separator = next(tokens, None)
            if separator not in (None, ";"):
                raise ValueError("Expected ';' after GTF attribute: " + key)
            if value and (keys is None or key in keys):
                attrs[key].append(value)
        return {key: tuple(values) for key, values in attrs.items()}
    fields = [
        raw_field.strip() for raw_field in text.split(";") if raw_field.strip() != ""
    ]
    if (
        len(fields) == 1
        and "=" not in fields[0]
        and not re.search(r"\s", fields[0])
        and fields[0] != "."
    ):
        # GFACS/TreeGenes GTF exports use a lone identifier in column 9.
        # It is an explicit model boundary, so expose it as both feature and
        # authoritative gene identity instead of invoking coordinate rescue.
        value = normalize_gff_attribute_value(fields[0])
        if value != "":
            return {
                key: (value,)
                for key in ("ID", "gene_id")
                if keys is None or key in keys
            }
    for field in fields:
        equal_pos = field.find("=")
        space_pos = field.find(" ")
        if equal_pos != -1 and (space_pos == -1 or equal_pos < space_pos):
            key = field[:equal_pos].strip()
            raw_value = field[equal_pos + 1 :]
        elif space_pos > 0:
            key = field[:space_pos].strip()
            raw_value = field[space_pos + 1 :]
        else:
            continue
        if key == "" or (keys is not None and key not in keys):
            continue
        for raw_item in str(raw_value or "").split(","):
            value = normalize_gff_attribute_value(raw_item)
            if value != "":
                attrs[key].append(value)
    return {key: tuple(values) for key, values in attrs.items()}

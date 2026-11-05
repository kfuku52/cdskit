"""Conservative attribute syntax repair at formatting, validation at consumption.

Recover numeric continuations in funannotate metadata, balanced MAKER mRNA
Notes, and observed AUGUSTUS metadata separators. Structural attributes are
never joined. Work on encoded text so existing escapes stay unchanged.
"""

import gzip
import hashlib
import re
import shlex
from pathlib import Path

GFF_ATTRIBUTE_SYNTAX_VERSION = 8
KEY_VALUE = re.compile(r"^[^\s=;]+=")
# These canonical structural fields never participate in metadata repair.
# Match the complete text; every other form retains the detailed parser.
CANONICAL_STRUCTURAL_ATTRIBUTES = re.compile(
    r"(?:ID|Parent|Name|gene_id|transcript_id)=[^;]*"
    r"(?:;(?:ID|Parent|Name|gene_id|transcript_id)=[^;]*)*;?"
)
NAME_CONTINUATION = re.compile(r"\d+(?:_\d+)?")
PRODUCT_CONTINUATION = re.compile(r"\d+(?:_\d+)?(?:,\s*variant\s+\d+)?")
# Chemical linkage lists in publisher descriptions, e.g. endo-1,3;1,4-beta.
# Restrict repair to this complete lexical pattern, never arbitrary orphan text.
LINKAGE_CONTINUATION = re.compile(r"\d+,\d+-[A-Za-z][^;=]*")
NAP_CONTINUATION = re.compile(
    r"[1-4](?:-like(?: isoform X\d+|, partial)?| isoform X\d+)? \[[A-Za-z][A-Za-z .-]+\]"
)
UNIPROT_CONTINUATION = re.compile(
    r" (?:AltName: (?:Full|Allergen)=[^;=]+|Includes: RecName: Full=[^;=]+|Short=[^;=]+|Flags: (?:Precursor|Fragment)(?: \[[A-Za-z][A-Za-z .-]+\])?)"
)
UNIPROT_METADATA = re.compile(r"^(?:description|Note)=(?:[A-Z0-9_]+ )?RecName: Full=")


def _description_brackets(value):
    stack = []
    for character in value:
        if character in "([":
            stack.append(character)
        elif character in ")]":
            if not stack or stack.pop() != {")": "(", "]": "["}[character]:
                return None
    return stack


def _escape_maker_note_delimiters(text, source, feature):
    """Preserve literal separators inside complete parenthesized MAKER Notes.

    Only mRNA Note values with balanced parentheses/brackets qualify. A real
    key=value boundary always ends the attribute; orphan key:value fragments
    and incomplete descriptions remain errors rather than absorbing identity.
    """
    if source.lower() != "maker" or feature.lower() != "mrna":
        return text
    fields, output = text.split(";"), []
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field.strip().startswith("Note="):
            output.append(field)
            continue
        value = field.split("=", 1)[1]
        fragments = [value]
        stack = _description_brackets(value)
        if not stack and index + 1 < len(fields):
            # These UniProt names repeat their literal numeric separator in
            # a parenthesized alias, proving the continuation independently.
            alias = re.fullmatch(r"([1-4]) \(((?:Nt|Os)NAP1|OsHKT1)", fields[index])
            if (
                alias
                and fields[index + 1].startswith(alias[1] + ")")
                and (
                    (
                        value == "Nucleosome assembly protein 1"
                        and alias[2].endswith("NAP1")
                    )
                    or (value == "Cation transporter HKT1" and alias[2] == "OsHKT1")
                )
            ):
                fragments.append(fields[index])
                index += 1
                stack = _description_brackets(";".join(fragments))
        while stack and index < len(fields):
            continuation = fields[index]
            if not continuation or KEY_VALUE.match(continuation.strip()):
                break
            if "=" in continuation or re.match(r"\s*[\w.-]+\s*:", continuation):
                raise ValueError(
                    "Unrecoverable GFF attribute fragment: " + repr(continuation)
                )
            fragments.append(continuation)
            index += 1
            stack = _description_brackets(";".join(fragments))
        if len(fragments) > 1:
            if stack != [] or "=" in value:
                raise ValueError(
                    "Unrecoverable GFF attribute fragment in incomplete MAKER Note"
                )
            field = (
                field.split("=", 1)[0]
                + "="
                + ";".join(fragments).replace(";", "%3B").replace(",", "%2C")
            )
        output.append(field)
    return ";".join(output)


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_attributes(text):
    """Validate GFF key=value or GTF quoted attributes without decoding them."""
    if text in ("", "."):
        return
    if KEY_VALUE.match(text.strip()):
        for field in text.strip(";").split(";"):
            if field.strip() and not KEY_VALUE.match(field.strip()):
                raise ValueError("Expected a GFF key=value attribute: " + repr(field))
        return
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


def normalise_attributes(text, source, feature, *, allow_bare=False):
    if allow_bare and text not in ("", ".") and not re.search(r"[\s=;]", text):
        return text  # Existing GFACS bare-ID adapter runs before output validation.
    if not KEY_VALUE.match(text.strip()):
        validate_attributes(text)
        return text
    if (
        type(source) is str
        and type(feature) is str
        and CANONICAL_STRUCTURAL_ATTRIBUTES.fullmatch(text)
    ):
        return text
    text = _escape_maker_note_delimiters(text, source, feature)
    repaired = []
    previous_key = None
    for field in text.split(";"):
        if not field:
            repaired.append(field)
            previous_key = None
        elif source.lower() == "augustus" and (
            (feature.lower() == "gene" and field.strip().startswith("Name:"))
            or (feature.lower() == "mrna" and field.strip().startswith("Blast2Go:"))
        ):
            key, value = field.strip().split(":", 1)
            repaired.append(key + "=" + value.replace("=", "%3D").replace(",", "%2C"))
            previous_key = key
        elif (
            previous_key in {"description", "Note"}
            and UNIPROT_METADATA.match(repaired[-1])
            and UNIPROT_CONTINUATION.fullmatch(field)
        ):
            repaired[-1] += "%3B" + field.replace("=", "%3D").replace(",", "%2C")
        elif (
            source == "."
            and feature.lower() == "misc_feature"
            and previous_key == "note"
            and re.fullmatch(
                r"note=Geneious type: Repeat(?: Region|_Regiion)(?:%3B IRa)?",
                repaired[-1],
            )
            and field in {" IRa", " Geneious type: repeat_region"}
        ):
            # These exact Geneious repeat notes contain literal semicolons.
            # Keep the description intact; never consume structural fields.
            repaired[-1] += "%3B" + field
        elif KEY_VALUE.match(field.strip()):
            repaired.append(field)
            previous_key = field.strip().split("=", 1)[0]
        elif (
            source.lower() == "evm"
            and feature.lower() == "gene"
            and field == "HC"
            and previous_key == "Accession"
            and re.fullmatch(r"Accession=GWHG[A-Z]{4}\d+[.]\d+", repaired[-1])
        ):
            repaired.append("gwh_gene_confidence=HC")
            previous_key = None
        else:
            pattern = None
            if (
                previous_key in {"description", "Note"}
                and re.search(r"nucleosome (?:assembly|assly) protein 1$", repaired[-1])
                and NAP_CONTINUATION.fullmatch(field)
            ):
                repaired[-1] += "%3B" + field.replace("=", "%3D").replace(",", "%2C")
                continue
            if (
                previous_key in {"description", "Note"}
                and re.search(r"\d+,\d+$", repaired[-1])
                and LINKAGE_CONTINUATION.fullmatch(field)
            ):
                repaired[-1] = (
                    repaired[-1].replace(",", "%2C") + "%3B" + field.replace(",", "%2C")
                )
                previous_key = None
                continue
            if source.lower() == "funannotate":
                if feature.lower() == "gene" and previous_key == "Name":
                    pattern = NAME_CONTINUATION
                elif feature.lower() == "mrna" and previous_key == "product":
                    pattern = PRODUCT_CONTINUATION
            if pattern is None or pattern.fullmatch(field) is None:
                raise ValueError("Unrecoverable GFF attribute fragment: " + repr(field))
            repaired[-1] += "%3B" + field.replace(",", "%2C")
            previous_key = None  # Multiple orphan fields are ambiguous.
    result = ";".join(repaired)
    validate_attributes(result)
    return result


def normalise_line(line, path, line_number, changes):
    if not line.strip() or line.startswith("#"):
        return line
    parts = line.rstrip("\r\n").split("\t")
    trailing_empty = len(parts) > 9 and all(value == "" for value in parts[9:])
    if trailing_empty:
        parts = parts[:9]
    if len(parts) != 9:
        raise ValueError(f"{path}:{line_number}: Expected nine GFF columns")
    before = parts[8]
    try:
        parts[8] = normalise_attributes(before, parts[1], parts[2], allow_bare=True)
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: {error}") from error
    if parts[8] == before and not trailing_empty:
        return line
    reason = (
        "canonicalised_gwh_confidence_flag"
        if "gwh_gene_confidence=HC" in parts[8]
        and re.search(r"(?:^|;)HC(?:;|$)", before)
        else "escaped_geneious_repeat_note_semicolon"
        if "note=Geneious type: Repeat" in before and parts[2].lower() == "misc_feature"
        else "escaped_NAP_metadata_semicolon"
        if re.search(r"(?:^|;)Note=[^;]*nucleosome", before) and "%3B" in parts[8]
        else "escaped_uniprot_Note_semicolon"
        if re.search(r"(?:^|;)Note=[^;]*RecName: Full=", before)
        else "canonicalised_augustus_metadata_separator"
        if parts[1].lower() == "augustus"
        else "escaped_description_linkage_semicolon"
        if re.search(r"(?:description|Note)=[^;]*\d+,\d+;\d+,\d+-[A-Za-z]", before)
        else "escaped_description_metadata_semicolon"
        if "description=" in before and "description=" in parts[8]
        else "escaped_maker_note_balanced_semicolon"
        if parts[1].lower() == "maker" and parts[2].lower() == "mrna"
        else "escaped_funannotate_metadata_semicolon"
    )
    output = "\t".join(parts) + line[len(line.rstrip("\r\n")) :]
    changes.append(
        {
            "source_line": line_number,
            "before": line.rstrip("\r\n") if trailing_empty else before,
            "after": output.rstrip("\r\n") if trailing_empty else parts[8],
            "reason": "removed_empty_trailing_columns" if trailing_empty else reason,
        }
    )
    return output


def validated_lines(lines, path):
    for line_number, line in enumerate(lines, 1):
        if line.startswith("##FASTA"):
            break
        if line.strip() and not line.startswith("#"):
            parts = line.rstrip("\r\n").split("\t")
            try:
                if len(parts) != 9:
                    raise ValueError("Expected nine GFF columns")
                validate_attributes(parts[8])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: {error}; regenerate with gg_input_generation"
                ) from error
        yield line


def validate_gff(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for _line in validated_lines(handle, path):
            pass


def syntax_audit(source_path, output_path, changes):
    return {
        "version": GFF_ATTRIBUTE_SYNTAX_VERSION,
        "source_sha256": file_sha256(source_path),
        "output_sha256": file_sha256(output_path),
        "changed_rows": len(changes),
        "changes": changes,
    }

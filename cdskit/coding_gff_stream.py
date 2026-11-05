"""Bounded text stream retaining coding features and linked alignment ancestors."""

import gzip
import io


class CodingGffStream(io.TextIOBase):
    """Replayable stream with the historical filtered-GFF row order and no spool."""

    def __init__(self, path, parse_attributes):
        self.path = str(path)
        self.parse_attributes = parse_attributes
        self.pending = ""
        self.lines = self._lines()

    def _lines(self):
        alignment = {
            "match",
            "match_part",
            "protein_match",
            "nucleotide_match",
            "translated_nucleotide_match",
        }
        opener = gzip.open if self.path.endswith(".gz") else open
        known, wanted = set(), set()
        inspected: set[str] = set()
        with opener(self.path, "rt", encoding="utf-8") as source:
            for line in source:
                if line.startswith("##FASTA"):
                    break
                parts = line.rstrip("\r\n").split("\t")
                if line.startswith("#") or len(parts) < 9:
                    yield line
                    continue
                if parts[2].lower() in alignment:
                    continue
                attrs = self.parse_attributes(parts[8])
                known.update(attrs.get("ID", ()))
                for key in ("Parent", "Derives_from", "derives_from"):
                    wanted.update(attrs.get(key, ()))
                yield line
        while wanted - known - inspected:
            pending = wanted - known - inspected
            with opener(self.path, "rt", encoding="utf-8") as source:
                for line in source:
                    if line.startswith("##FASTA"):
                        break
                    parts = line.rstrip("\r\n").split("\t")
                    if (
                        line.startswith("#")
                        or len(parts) < 9
                        or parts[2].lower() not in alignment
                    ):
                        continue
                    attrs = self.parse_attributes(parts[8])
                    if not set(attrs.get("ID", ())) & pending:
                        continue
                    for key in ("Parent", "Derives_from", "derives_from"):
                        wanted.update(attrs.get(key, ()))
                    yield line
            inspected.update(pending)

    def read(self, size=-1):
        if size == 0:
            return ""
        pieces = [self.pending]
        length = len(self.pending)
        self.pending = ""
        while size < 0 or length < size:
            line = next(self.lines, "")
            if not line:
                break
            pieces.append(line)
            length += len(line)
        result = "".join(pieces)
        if size >= 0:
            result, self.pending = result[:size], result[size:]
        return result

    def seek(self, offset, whence=0):
        if offset != 0 or whence != 0:
            raise io.UnsupportedOperation("Only rewinding is supported")
        self.lines.close()
        self.lines = self._lines()
        self.pending = ""
        return 0

    def __iter__(self):
        # The fallback parser requests complete lines after rewinding.
        if self.pending:
            raise io.UnsupportedOperation("Rewind before line iteration")
        return self.lines

    def close(self):
        self.lines.close()
        super().close()

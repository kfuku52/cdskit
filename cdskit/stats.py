from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
import re

import numpy as np

from cdskit.atomicio import atomic_text_writer
from cdskit.tsvio import write_tsv

from cdskit.util import (
    DnaValidation,
    iter_seq_chunks,
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
)

LOWERCASE_DELETE_TABLE = str.maketrans("", "", "abcdefghijklmnopqrstuvwxyz")
_LOWERCASE_DELETE_BYTES = b"abcdefghijklmnopqrstuvwxyz"


def _long_ascii_record_stats(sequence):
    # Keep temporary comparisons bounded for chromosome-sized records.
    array = np.frombuffer(sequence, dtype=np.uint8)
    masked = guanine = cytosine = unknown = gap = 0
    for start in range(0, len(array), 65536):
        block = array[start : start + 65536]
        upper = block & 223
        masked += int(np.count_nonzero((block >= 97) & (block <= 122)))
        guanine += int(np.count_nonzero(upper == 71))
        cytosine += int(np.count_nonzero(upper == 67))
        unknown += int(np.count_nonzero(upper == 78))
        gap += int(np.count_nonzero(block == 45))
    return {
        "bp_masked": masked,
        "bp_all": len(sequence),
        "bp_G": guanine,
        "bp_C": cytosine,
        "bp_N": unknown,
        "bp_gap": gap,
    }


def num_masked_bp(seq):
    seq_str = str(seq)
    return len(seq_str) - len(seq_str.translate(LOWERCASE_DELETE_TABLE))


def record_stats(record):
    seq_str = str(record.seq)
    if type(seq_str) is str and seq_str.isascii():
        sequence = seq_str.encode("ascii")
        # Below this size, byte-string methods avoid NumPy's fixed overhead.
        if len(sequence) >= 8192:
            return _long_ascii_record_stats(sequence)
        upper = sequence.upper()
        return {
            "bp_masked": len(sequence)
            - len(sequence.translate(None, _LOWERCASE_DELETE_BYTES)),
            "bp_all": len(sequence),
            "bp_G": upper.count(b"G"),
            "bp_C": upper.count(b"C"),
            "bp_N": upper.count(b"N"),
            "bp_gap": sequence.count(b"-"),
        }
    seq_upper = seq_str.upper()
    return {
        "bp_masked": num_masked_bp(seq_str),
        "bp_all": len(seq_str),
        "bp_G": seq_upper.count("G"),
        "bp_C": seq_upper.count("C"),
        "bp_N": seq_upper.count("N"),
        "bp_gap": seq_str.count("-"),
    }


def sequence_stats_main(args):
    validation = DnaValidation()
    thread_error = None
    try:
        threads = resolve_threads(getattr(args, "threads", 1))
    except (ValueError, TypeError) as exc:
        # Keep input errors ahead of thread errors, including for direct callers.
        thread_error = exc
        threads = 1
    num_seq = bp_masked = bp_all = bp_G = bp_C = bp_N = bp_gap = 0
    for records in iter_seq_chunks(
        seqfile=args.seqfile, seqformat=args.inseqformat, max_chunk_residues=8_000_000
    ):
        validation.inspect(records)
        if validation.invalid or thread_error is not None:
            continue
        num_seq += len(records)
        for row in parallel_map_ordered(records, record_stats, threads):
            bp_masked += row["bp_masked"]
            bp_all += row["bp_all"]
            bp_G += row["bp_G"]
            bp_C += row["bp_C"]
            bp_N += row["bp_N"]
            bp_gap += row["bp_gap"]
    validation.raise_if_invalid()
    if thread_error is not None:
        raise thread_error
    print("Number of sequences: {:,}".format(num_seq))
    print("Total length: {:,}".format(bp_all))
    print("Total softmasked length: {:,}".format(bp_masked))
    print("Total N length: {:,}".format(bp_N))
    print("Total gap (-) length: {:,}".format(bp_gap))
    gc_content = 0.0
    if bp_all > 0:
        gc_content = ((bp_G + bp_C) / bp_all) * 100
    print("GC content: {:,.1f}%".format(gc_content))


ALPHABETS = {
    "dna": "ACGTKMRYSWBVHDXNO-?",
    "aa": "ACDEFGHIKLMNPQRSTVWYBJZX.*-?",
}
MISSING = {"dna": frozenset("XNO-?"), "aa": frozenset("X.*-?")}
STATES = {"dna": frozenset("ACGT"), "aa": frozenset("ACDEFGHIKLMNPQRSTVWY")}
_ALIGNMENT_ALPHABET_PATTERNS = {
    kind: re.compile(f"[{re.escape(alphabet)}]*")
    for kind, alphabet in ALPHABETS.items()
}
_PROTEIN_STATE_BITS = np.zeros(256, dtype=np.uint32)
_PROTEIN_STATE_CODES = np.zeros(256, dtype=np.uint8)
for _index, _residue in enumerate(sorted(STATES["aa"])):
    _PROTEIN_STATE_BITS[ord(_residue)] = 1 << _index
    _PROTEIN_STATE_CODES[ord(_residue)] = _index + 1
_PROTEIN_STATE_BITS.flags.writeable = False
_PROTEIN_STATE_CODES.flags.writeable = False
_DNA_STATE_BITS = np.zeros(256, dtype=np.uint8)
_DNA_STATE_BITS[list(b"ACGT")] = (1, 2, 4, 8)
_DNA_STATE_BITS.flags.writeable = False
_ALIGNMENT_STATS_BLOCK = 65_536
SUMMARY_COLUMNS = (
    "Alignment_name",
    "No_of_taxa",
    "Alignment_length",
    "Total_matrix_cells",
    "Undetermined_characters",
    "Missing_percent",
    "No_variable_sites",
    "Proportion_variable_sites",
    "Parsimony_informative_sites",
    "Proportion_parsimony_informative",
    "AT_content",
    "GC_content",
)


def _alignment_character_counts(sequence, alphabet):
    if len(sequence) < 256:
        return Counter(sequence)
    counts = np.zeros(128, dtype=np.int64)
    for start in range(0, len(sequence), _ALIGNMENT_STATS_BLOCK):
        encoded = sequence[start : start + _ALIGNMENT_STATS_BLOCK].encode("ascii")
        counts += np.bincount(np.frombuffer(encoded, dtype=np.uint8), minlength=128)
    return {letter: int(counts[ord(letter)]) for letter in alphabet}


def _alignment_site_bit_counts(sequences, length, state_bits):
    variable = informative = 0
    for start in range(0, length, _ALIGNMENT_STATS_BLOCK):
        end = min(start + _ALIGNMENT_STATS_BLOCK, length)
        seen = np.zeros(end - start, dtype=state_bits.dtype)
        repeated = np.zeros_like(seen)
        for sequence in sequences:
            codes = np.frombuffer(sequence[start:end].encode("ascii"), dtype=np.uint8)
            residues = state_bits[codes]
            repeated |= seen & residues
            seen |= residues
        # Clearing the lowest set bit leaves a nonzero value when there are
        # two or more distinct canonical residues in the corresponding mask.
        variable += int(np.count_nonzero(seen & (seen - 1)))
        informative += int(np.count_nonzero(repeated & (repeated - 1)))
    return variable, informative


def _protein_site_counts(block):
    # Code zero contains ambiguous or missing residues. Each column
    # has its own group of bins, so the matrix needs only one count.
    width = block.shape[1]
    bins = len(STATES["aa"]) + 1
    offsets = np.arange(0, width * bins, bins, dtype=np.intp)
    keys = _PROTEIN_STATE_CODES[block] + offsets
    counts = np.bincount(keys.ravel(), minlength=width * bins).reshape(width, bins)[
        :, 1:
    ]
    variable = np.count_nonzero(np.count_nonzero(counts, axis=1) >= 2)
    informative = np.count_nonzero(np.count_nonzero(counts >= 2, axis=1) >= 2)
    return int(variable), int(informative)


def _alignment_site_counts(sequences, length, seq_type):
    if length >= 256:
        bits = _PROTEIN_STATE_BITS if seq_type == "aa" else _DNA_STATE_BITS
        return _alignment_site_bit_counts(sequences, length, bits)
    variable = informative = 0
    # Protein counting uses an integer key for each cell. Keep that array
    # bounded by limiting the number of columns per block.
    max_cells = 262_144 if seq_type == "aa" else 1_000_000
    block_width = max(1, min(length, max_cells // len(sequences)))
    for start in range(0, length, block_width):
        width = min(block_width, length - start)
        block = np.frombuffer(
            "".join(sequence[start : start + width] for sequence in sequences).encode(
                "ascii"
            ),
            dtype=np.uint8,
        ).reshape(len(sequences), width)
        if seq_type == "aa":
            block_variable, block_informative = _protein_site_counts(block)
            variable += block_variable
            informative += block_informative
            continue
        present = np.zeros(width, dtype=np.uint8)
        repeated = np.zeros(width, dtype=np.uint8)
        for state in STATES[seq_type]:
            counts_by_site = np.count_nonzero(block == ord(state), axis=0)
            present += counts_by_site > 0
            repeated += counts_by_site >= 2
        variable += int(np.count_nonzero(present > 1))
        informative += int(np.count_nonzero(repeated >= 2))
    return variable, informative


def summarize_alignment(records, seq_type, name="-"):
    """Return one summary row; reject malformed input instead of dropping taxa."""
    if seq_type not in ALPHABETS:
        raise ValueError("Alignment sequence type must be dna or aa.")
    if not records or not len(records[0]):
        raise ValueError("Alignment must contain nonempty sequences.")
    length = len(records[0])
    identifiers = set()
    sequences = []
    counts: Counter[str] = Counter()
    at_fractions = {}
    for record in records:
        if not record.id or not record.id.strip():
            raise ValueError("Alignment sequence IDs must not be empty.")
        if record.id in identifiers:
            raise ValueError(f"Duplicate sequence ID: {record.id}")
        identifiers.add(record.id)
        if len(record) != length:
            raise ValueError("Alignment sequences must have equal lengths.")
        sequence = str(record.seq).upper()
        if _ALIGNMENT_ALPHABET_PATTERNS[seq_type].fullmatch(sequence) is None:
            invalid = set(sequence) - set(ALPHABETS[seq_type])
            raise ValueError(
                f"Invalid {seq_type} characters in {record.id}: {sorted(invalid)}"
            )
        sequences.append(sequence)
        local_counts = _alignment_character_counts(sequence, ALPHABETS[seq_type])
        counts.update(local_counts)
        if seq_type == "dna":
            at = sum(local_counts[base] for base in "ATW")
            gc = sum(local_counts[base] for base in "GCS")
            at_fractions[record.id] = round(at / (at + gc), 3) if at + gc else 0
    variable, informative = _alignment_site_counts(sequences, length, seq_type)
    cells = len(records) * length
    missing = sum(counts[base] for base in MISSING[seq_type])
    row = dict(
        zip(
            SUMMARY_COLUMNS,
            (
                name,
                len(records),
                length,
                cells,
                missing,
                round(missing / cells * 100, 3),
                variable,
                round(variable / length, 3),
                informative,
                round(informative / length, 3),
                "NA",
                "NA",
            ),
            strict=True,
        )
    )
    if seq_type == "dna":
        # AMAS averages already-rounded per-taxon AT fractions, then derives GC.
        # Even a taxon without ATW/GCS contributes zero AT (documented legacy rule).
        at_content = round(
            sum(at_fractions[label] for label in sorted(at_fractions)) / len(records), 3
        )
        row["AT_content"] = at_content
        row["GC_content"] = round(1 - at_content, 3)
    row.update({base: counts[base] for base in ALPHABETS[seq_type]})
    return row


def stats_main(args):
    mode = getattr(args, "mode", "sequence")
    seq_type = getattr(args, "seq_type", "dna")
    outfile = getattr(args, "outfile", "-")
    if mode == "alignment":
        resolve_threads(getattr(args, "threads", 1))
        records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
        row = summarize_alignment(records, seq_type, Path(args.seqfile).name)
        write_tsv(outfile, [row], [*SUMMARY_COLUMNS, *ALPHABETS[seq_type]])
    elif mode == "sequence":
        if seq_type != "dna":
            raise ValueError("Protein statistics require --mode alignment.")
        if outfile == "-":
            sequence_stats_main(args)
        else:
            with atomic_text_writer(outfile) as handle, redirect_stdout(handle):
                sequence_stats_main(args)
    else:
        raise ValueError("Statistics mode must be sequence or alignment.")

from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

from cdskit.atomicio import atomic_text_writer
from cdskit.tsvio import write_tsv

from cdskit.util import (
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
    stop_if_not_dna,
)

LOWERCASE_DELETE_TABLE = str.maketrans("", "", "abcdefghijklmnopqrstuvwxyz")


def num_masked_bp(seq):
    seq_str = str(seq)
    return len(seq_str) - len(seq_str.translate(LOWERCASE_DELETE_TABLE))


def record_stats(record):
    seq_str = str(record.seq)
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
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=records, label="--seq_file")
    threads = resolve_threads(getattr(args, "threads", 1))
    num_seq = len(records)
    record_count_stats = parallel_map_ordered(
        items=records, worker=record_stats, threads=threads
    )
    bp_masked = sum(x["bp_masked"] for x in record_count_stats)
    bp_all = sum(x["bp_all"] for x in record_count_stats)
    bp_G = sum(x["bp_G"] for x in record_count_stats)
    bp_C = sum(x["bp_C"] for x in record_count_stats)
    bp_N = sum(x["bp_N"] for x in record_count_stats)
    bp_gap = sum(x["bp_gap"] for x in record_count_stats)
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
        invalid = set(sequence) - set(ALPHABETS[seq_type])
        if invalid:
            raise ValueError(
                f"Invalid {seq_type} characters in {record.id}: {sorted(invalid)}"
            )
        sequences.append(sequence)
        local_counts = Counter(sequence)
        counts.update(local_counts)
        if seq_type == "dna":
            at = sum(local_counts[base] for base in "ATW")
            gc = sum(local_counts[base] for base in "GCS")
            at_fractions[record.id] = round(at / (at + gc), 3) if at + gc else 0
    variable = informative = 0
    for column in zip(*sequences, strict=True):
        states = Counter(base for base in column if base in STATES[seq_type])
        variable += len(states) > 1
        informative += sum(count >= 2 for count in states.values()) >= 2
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

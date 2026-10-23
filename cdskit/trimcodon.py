import copy
import math
import sys

import numpy as np

from cdskit.codonarray import codon_features

from cdskit.codonutil import (
    CODON_SEMANTICS_VERSION,
    CODON_AMBIGUOUS,
    CODON_CLEAN,
    CODON_MISSING,
    CODON_STOP,
    classify_codon,
    analyze_codon,
)
from cdskit.atomicio import atomic_output_paths, atomic_write_json
from cdskit.util import (
    read_seqs,
    replace_record_sequence,
    stop_if_invalid_codontable,
    stop_if_not_aligned,
    stop_if_not_dna,
    stop_if_not_multiple_of_three,
    write_seqs,
)
from cdskit.tsvio import write_sectioned_tsv


def validate_fraction(name, value):
    value = float(value)
    if (not math.isfinite(value)) or (value < 0.0) or (value > 1.0):
        txt = "{} should be between 0 and 1 inclusive. Exiting.\n"
        raise ValueError(txt.format(name))
    return value


def summarize_codon_site(seq_strings, codon_site, codontable):
    clean = 0
    missing = 0
    ambiguous = 0
    stop = 0
    possible_stop = 0
    context_dependent = 0
    start = codon_site * 3
    end = start + 3
    for seq_str in seq_strings:
        codon = seq_str[start:end]
        meaning = analyze_codon(codon, codontable)
        ambiguous += int(meaning.ambiguous or meaning.invalid)
        possible_stop += int(meaning.possible_stop)
        context_dependent += int(meaning.context_dependent)
        state = classify_codon(codon=codon, codontable=codontable)
        if state == CODON_CLEAN:
            clean += 1
            continue
        if state == CODON_MISSING:
            missing += 1
            continue
        if state == CODON_AMBIGUOUS:
            continue
        if state == CODON_STOP:
            stop += 1
            continue
    return {
        "codon_site_1based": codon_site + 1,
        "clean_codons": clean,
        "missing_codons": missing,
        "ambiguous_codons": ambiguous,
        "stop_codons": stop,
        "possible_stop_codons": possible_stop,
        "context_dependent_codons": context_dependent,
    }


def summarize_codon_sites(seq_strings, codontable):
    """Collect site counts a row at a time without an alignment-sized matrix."""
    num_sites = len(seq_strings[0]) // 3 if seq_strings else 0
    totals = np.zeros((num_sites, 6), dtype=np.int64)
    for sequence in seq_strings:
        values = codon_features(sequence, codontable)
        if values is None or len(sequence) != num_sites * 3:
            return [
                summarize_codon_site(seq_strings, site, codontable)
                for site in range(num_sites)
            ]
        totals += values[0]
    fields = (
        "clean_codons",
        "missing_codons",
        "ambiguous_codons",
        "stop_codons",
        "possible_stop_codons",
        "context_dependent_codons",
    )
    return [
        {
            "codon_site_1based": site + 1,
            **dict(zip(fields, map(int, counts), strict=True)),
        }
        for site, counts in enumerate(totals)
    ]


def choose_kept_codon_sites(site_summaries, num_sequences, min_clean_fraction):
    kept_sites = list()
    for summary in site_summaries:
        clean_fraction = 0.0
        if num_sequences > 0:
            clean_fraction = summary["clean_codons"] / num_sequences
        summary["clean_fraction"] = clean_fraction
        summary["unclean_codons"] = num_sequences - summary["clean_codons"]
        keep = clean_fraction >= min_clean_fraction
        summary["keep"] = keep
        if keep:
            kept_sites.append(summary["codon_site_1based"] - 1)
    return kept_sites


def trim_record_to_codon_sites(record, kept_sites):
    trimmed = copy.copy(record)
    seq_str = str(record.seq)
    replace_record_sequence(
        trimmed,
        "".join(seq_str[site * 3 : site * 3 + 3] for site in kept_sites),
    )
    return trimmed


def _trim_records_to_codon_sites(records, kept_sites):
    """Reuse contiguous ranges for the sorted sites selected by the command."""
    ranges: list[tuple[int, int]] = []
    for site in kept_sites:
        start = site * 3
        end = start + 3
        if ranges and ranges[-1][1] == start:
            ranges[-1] = (ranges[-1][0], end)
        else:
            ranges.append((start, end))
    trimmed_records = []
    for record in records:
        trimmed = copy.copy(record)
        seq_str = str(record.seq)
        replace_record_sequence(
            trimmed, "".join(seq_str[start:end] for start, end in ranges)
        )
        trimmed_records.append(trimmed)
    return trimmed_records


def build_trimcodon_summary(site_summaries, kept_sites, num_sequences, args):
    removed_sites = [
        summary["codon_site_1based"]
        for summary in site_summaries
        if not summary["keep"]
    ]
    return {
        "num_sequences": num_sequences,
        "codon_semantics_version": CODON_SEMANTICS_VERSION,
        "codon_table": getattr(args, "codontable", 1),
        "num_input_codon_sites": len(site_summaries),
        "num_output_codon_sites": len(kept_sites),
        "num_removed_codon_sites": len(removed_sites),
        "min_clean_fraction": args.min_clean_fraction,
        "kept_codon_sites_1based": [site + 1 for site in kept_sites],
        "removed_codon_sites_1based": removed_sites,
        "site_summaries": site_summaries,
    }


def write_trimcodon_report(report_path, summary):
    if report_path == "":
        return
    if report_path.lower().endswith(".json"):
        atomic_write_json(report_path, summary, indent=2)
        return
    rows = [
        {"section": "summary", "metric": key, "value": summary[key]}
        for key in [
            "num_sequences",
            "codon_semantics_version",
            "codon_table",
            "num_input_codon_sites",
            "num_output_codon_sites",
            "num_removed_codon_sites",
            "min_clean_fraction",
        ]
    ]
    for source_row in summary["site_summaries"]:
        row = dict(source_row)
        row["section"] = "site"
        row["clean_fraction"] = "{:.6f}".format(row["clean_fraction"])
        row["keep"] = "yes" if row["keep"] else "no"
        rows.append(row)
    write_sectioned_tsv(
        path=report_path,
        fieldnames=[
            "section",
            "metric",
            "value",
            "codon_site_1based",
            "clean_fraction",
            "clean_codons",
            "unclean_codons",
            "missing_codons",
            "ambiguous_codons",
            "stop_codons",
            "possible_stop_codons",
            "context_dependent_codons",
            "keep",
        ],
        rows=rows,
    )


def trimcodon_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=records, label="--seq_file")
    stop_if_not_aligned(records=records)
    stop_if_not_multiple_of_three(records=records)
    stop_if_invalid_codontable(args.codontable)
    args.min_clean_fraction = validate_fraction(
        name="--min_clean_fraction", value=args.min_clean_fraction
    )
    seq_strings = [str(record.seq) for record in records]
    site_summaries = summarize_codon_sites(seq_strings, args.codontable)
    kept_sites = choose_kept_codon_sites(
        site_summaries=site_summaries,
        num_sequences=len(records),
        min_clean_fraction=args.min_clean_fraction,
    )
    summary = build_trimcodon_summary(
        site_summaries=site_summaries,
        kept_sites=kept_sites,
        num_sequences=len(records),
        args=args,
    )
    sys.stderr.write(
        "Removed codon sites: {:,}\n".format(summary["num_removed_codon_sites"])
    )
    out_records = _trim_records_to_codon_sites(records, kept_sites)
    outputs = [path for path in (args.outfile, args.report) if path not in ("", "-")]
    with atomic_output_paths(outputs) as temporary_paths:
        staged = dict(zip(outputs, temporary_paths, strict=True))
        write_seqs(
            records=out_records,
            outfile=staged.get(args.outfile, args.outfile),
            outseqformat=args.outseqformat,
        )
        write_trimcodon_report(
            report_path=staged.get(args.report, args.report), summary=summary
        )

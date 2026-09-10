from collections import defaultdict
from collections import deque
import sys
import hashlib
import json
from pathlib import Path
from typing import Any

from cdskit import __version__
from functools import partial

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from cdskit.translate import translate_sequence_string
from cdskit.atomicio import (
    atomic_output_paths,
    atomic_write_json,
    validate_distinct_paths,
)
from cdskit.column_mapping import (
    alignment_columns,
    analyze_mapping,
    read_kept_sites,
    validate_mapping,
)

from cdskit.util import (
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
    stop_if_not_aligned,
    stop_if_invalid_codontable,
    stop_if_not_dna,
    stop_if_not_multiple_of_three,
    write_seqs,
)


def check_same_seq_num(cdn_records, pep_records):
    err_txt = (
        "The numbers of seqs did not match: seqfile={} and trimmed_aa_aln={}".format(
            len(cdn_records), len(pep_records)
        )
    )
    if len(cdn_records) != len(pep_records):
        raise ValueError(err_txt)


def get_record_map(records, label):
    record_map = dict()
    for record in records:
        if record.id in record_map:
            txt = "Sequence IDs must be unique in {}. Duplicated ID: {}"
            raise ValueError(txt.format(label, record.id))
        record_map[record.id] = record
    return record_map


def reorder_aa_records_by_cds_ids(cdn_records, pep_records):
    cdn_record_map = get_record_map(cdn_records, "--seq_file")
    pep_record_map = get_record_map(pep_records, "--trimmed_aa_aln")
    cdn_ids = set(cdn_record_map.keys())
    pep_ids = set(pep_record_map.keys())
    if cdn_ids != pep_ids:
        missing_in_cds = sorted(list(pep_ids - cdn_ids))
        missing_in_aa = sorted(list(cdn_ids - pep_ids))
        txt = "Sequence IDs did not match between CDS (--seq_file) and trimmed amino acid alignment (--trimmed_aa_aln)."
        if len(missing_in_cds) > 0:
            txt += " Missing in CDS: {}.".format(",".join(missing_in_cds))
        if len(missing_in_aa) > 0:
            txt += " Missing in trimmed amino acid alignment: {}.".format(
                ",".join(missing_in_aa)
            )
        raise ValueError(txt)
    return [pep_record_map[record.id] for record in cdn_records]


def build_column_index(seq_strings):
    col_index: defaultdict[str, deque[int]] = defaultdict(deque)
    if len(seq_strings) == 0:
        return col_index
    for ci, col_chars in enumerate(zip(*seq_strings, strict=False)):
        key = "".join(col_chars)
        col_index[key].append(ci)
    return col_index


def find_kept_aa_sites(tcdn_strings, pep_strings):
    # Alignment case and the two accepted gap symbols do not change a site.
    tcdn_strings = [seq.upper().replace(".", "-") for seq in tcdn_strings]
    pep_strings = [seq.upper().replace(".", "-") for seq in pep_strings]
    kept_aa_sites = []
    multiple_matches: set[int] = set()
    last_kept_site = -1
    remaining_site_count = len(tcdn_strings[0])
    tcdn_col_index = build_column_index(tcdn_strings)

    for pi, pep_col_chars in enumerate(zip(*pep_strings, strict=False)):
        if remaining_site_count == 0:
            break
        key = "".join(pep_col_chars)
        same_sites = tcdn_col_index.get(key)
        if (same_sites is None) or (len(same_sites) == 0):
            txt = "The codon site {} could not be matched to trimmed protein sites. "
            txt += "The site may contain only missing, ambiguous, and/or stop codons. "
            txt += "The site will be excluded from the output.\n"
            sys.stderr.write(txt.format(pi))
            continue
        if len(same_sites) == 1:
            kept_site = same_sites.popleft()
            if kept_site <= last_kept_site:
                txt = "The codon site {} would violate codon order at trimmed protein site {}. "
                txt += "The site will be excluded from the output.\n"
                sys.stderr.write(txt.format(kept_site, pi))
                del tcdn_col_index[key]
                continue
            kept_aa_sites.append(kept_site)
            last_kept_site = kept_site
            remaining_site_count -= 1
            del tcdn_col_index[key]
            continue

        all_same_sites = tuple(same_sites)
        multiple_matches.update(all_same_sites)
        # Prefer the first candidate after the previously selected codon site
        # so codon order follows the trimmed amino acid alignment.
        kept_site = None
        for candidate_site in same_sites:
            if candidate_site > last_kept_site:
                kept_site = candidate_site
                break
        if kept_site is None:
            txt = (
                "The trimmed protein site {} has multiple matches to codon sites({}), "
            )
            txt += "but none preserve codon order. The site will be excluded from the output. "
            txt = txt.format(pi, ",".join([str(ss) for ss in all_same_sites]))
            txt += "Site pattern: {}\n".format(key)
            sys.stderr.write(txt)
            continue

        txt = "The trimmed protein site {} has multiple matches to codon sites({}). Reporting codon site {}. "
        txt = txt.format(pi, ",".join([str(ss) for ss in all_same_sites]), kept_site)
        txt += "Site pattern: {}\n".format(key)
        sys.stderr.write(txt)
        same_sites.remove(kept_site)
        kept_aa_sites.append(kept_site)
        last_kept_site = kept_site
        remaining_site_count -= 1

    num_trimmed_multiple_hit_sites = len(multiple_matches - set(kept_aa_sites))
    return kept_aa_sites, num_trimmed_multiple_hit_sites


def trim_codon_records(cdn_records, kept_aa_sites):
    cdn_strings = [str(record.seq) for record in cdn_records]
    trimmed_cdn_records = []
    for i, record in enumerate(cdn_records):
        trimmed_seq = "".join(
            [
                cdn_strings[i][codon_site * 3 : codon_site * 3 + 3]
                for codon_site in kept_aa_sites
            ]
        )
        trimmed_record = SeqRecord(
            seq=Seq(trimmed_seq),
            id=record.id,
            name="",
            description="",
        )
        trimmed_cdn_records.append(trimmed_record)
    return trimmed_cdn_records


def translate_record_to_aa_string(record, codontable):
    return translate_sequence_string(
        seq_str=str(record.seq),
        codontable=codontable,
        to_stop=False,
    )


def trim_codon_record(record, kept_aa_sites):
    seq_str = str(record.seq)
    trimmed_seq = "".join(
        [seq_str[codon_site * 3 : codon_site * 3 + 3] for codon_site in kept_aa_sites]
    )
    return SeqRecord(
        seq=Seq(trimmed_seq),
        id=record.id,
        name="",
        description="",
    )


def codon_sites_to_nucleotide_ranges(codon_sites):
    if len(codon_sites) == 0:
        return list()
    ranges = list()
    run_start = codon_sites[0]
    run_end = codon_sites[0]
    for site in codon_sites[1:]:
        if site == run_end + 1:
            run_end = site
            continue
        ranges.append((run_start * 3, (run_end + 1) * 3))
        run_start = site
        run_end = site
    ranges.append((run_start * 3, (run_end + 1) * 3))
    return ranges


def trim_codon_record_with_ranges(record, nucleotide_ranges):
    seq_str = str(record.seq)
    trimmed_seq = "".join([seq_str[start:end] for start, end in nucleotide_ranges])
    return SeqRecord(
        seq=Seq(trimmed_seq),
        id=record.id,
        name="",
        description="",
    )


def _record_digest(records):
    # Canonical record content also works for stdin and preserves row order/case.
    payload = [[record.id, str(record.seq)] for record in records]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True).encode()).hexdigest()


def select_kept_sites(args, source, target, report):
    analysis = analyze_mapping(source, target)
    report.update(
        inference_status=analysis.status,
        complete_mapping_exists=analysis.status != "unmatched",
        inferred_unique=analysis.status == "unique",
        mapping_examples=[analysis.leftmost, analysis.rightmost]
        if analysis.status == "ambiguous"
        else ([analysis.leftmost] if analysis.status == "unique" else []),
    )
    if getattr(args, "kept_sites", None):
        sites = read_kept_sites(args.kept_sites, args.kept_sites_format, len(source))
        validate_mapping(source, target, sites)
        report["source"] = "provided"
        return sites
    if getattr(args, "mapping_policy", "legacy") == "strict":
        if analysis.status != "unique":
            raise ValueError(
                "AA column mapping is {}. Supply --kept_sites with the trimmer's "
                "retained column positions; no codon alignment was written.".format(
                    analysis.status
                )
            )
        return analysis.leftmost
    return None


def _write_backtrim_outputs(args, records, report):
    report_path = getattr(args, "mapping_report", None)
    if report_path and args.outfile != "-":
        with atomic_output_paths([args.outfile, report_path]) as paths:
            write_seqs(records, paths[0], args.outseqformat)
            atomic_write_json(paths[1], report, indent=2)
    else:
        # stdout cannot be rolled back; all mapping validation is finished first.
        if report_path:
            atomic_write_json(report_path, report, indent=2)
        write_seqs(records, args.outfile, args.outseqformat)


def backtrim_main(args):
    kept_path = getattr(args, "kept_sites", None)
    kept_format = getattr(args, "kept_sites_format", None)
    report_path = getattr(args, "mapping_report", None)
    policy = getattr(args, "mapping_policy", "legacy")
    if policy not in {"legacy", "strict"}:
        raise ValueError("Unknown mapping policy.")
    if bool(kept_path) != bool(kept_format):
        raise ValueError(
            "--kept_sites and --kept_sites_format must be supplied together."
        )
    if report_path == "-" or kept_path == "-":
        raise ValueError(
            "Mapping report and kept sites require file paths, not stdout/stdin."
        )
    validate_distinct_paths(
        inputs=[args.seqfile, args.trimmed_aa_aln, kept_path],
        outputs=[args.outfile, report_path],
    )
    report: dict[str, Any] = dict(
        schema_version=1,
        cdskit_version=__version__,
        coordinate_system="source_aligned_amino_acid",
        index_base=0,
        policy=policy,
        source="provided" if kept_path else policy,
        kept_sites_format=kept_format,
        status="invalid",
        selected_sites=[],
    )
    try:
        _backtrim_main(args, report)
    except ValueError as error:
        report.update(status="failed", error=str(error))
        if report_path:
            atomic_write_json(report_path, report, indent=2)
        raise


def _backtrim_main(args, report):
    cdn_records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=cdn_records, label="--seq_file")
    stop_if_invalid_codontable(args.codontable)
    pep_records = read_seqs(seqfile=args.trimmed_aa_aln, seqformat=args.inseqformat)
    threads = resolve_threads(getattr(args, "threads", 1))
    stop_if_not_multiple_of_three(cdn_records)
    check_same_seq_num(cdn_records, pep_records)
    stop_if_not_aligned(records=cdn_records)
    stop_if_not_aligned(records=pep_records)
    pep_records = reorder_aa_records_by_cds_ids(
        cdn_records=cdn_records, pep_records=pep_records
    )
    report.update(
        codontable=args.codontable,
        source_records_sha256=_record_digest(cdn_records),
        trimmed_records_sha256=_record_digest(pep_records),
    )
    kept_path = getattr(args, "kept_sites", None)
    if kept_path:
        report["kept_sites_sha256"] = hashlib.sha256(
            Path(kept_path).read_bytes()
        ).hexdigest()
    translate_worker = partial(
        translate_record_to_aa_string, codontable=args.codontable
    )
    tcdn_strings = parallel_map_ordered(
        items=cdn_records, worker=translate_worker, threads=threads
    )
    pep_strings = [str(record.seq) for record in pep_records]
    source = alignment_columns(tcdn_strings)
    target = alignment_columns(pep_strings)
    report.update(source_column_count=len(source), target_column_count=len(target))
    kept_aa_sites = select_kept_sites(args, source, target, report)
    if kept_aa_sites is None:
        kept_aa_sites, multiple_count = (
            find_kept_aa_sites(tcdn_strings, pep_strings) if cdn_records else ([], 0)
        )
        txt = "{} codon sites matched to {} protein sites. "
        txt += "Trimmed {} codon sites that matched to multiple protein sites.\n"
        sys.stderr.write(txt.format(len(kept_aa_sites), len(target), multiple_count))
        report["legacy_trimmed_multiple_hit_sites"] = multiple_count
    # Reconstruct the legacy output's target positions for an explicit omission report.
    matched_target_sites = []
    cursor = 0
    for site in kept_aa_sites:
        while cursor < len(target) and target[cursor] != source[site]:
            cursor += 1
        matched_target_sites.append(cursor)
        cursor += 1
    report.update(
        status="success",
        selected_sites=kept_aa_sites,
        matched_target_sites=matched_target_sites,
        unmatched_target_sites=sorted(
            set(range(len(target))) - set(matched_target_sites)
        ),
        output_complete=len(kept_aa_sites) == len(target),
    )
    nucleotide_ranges = codon_sites_to_nucleotide_ranges(codon_sites=kept_aa_sites)
    trim_worker = partial(
        trim_codon_record_with_ranges, nucleotide_ranges=nucleotide_ranges
    )
    trimmed_cdn_records = parallel_map_ordered(
        items=cdn_records, worker=trim_worker, threads=threads
    )
    sys.stderr.write(
        "Number of aligned nucleotide sites in untrimmed codon sequences: {}\n".format(
            len(source) * 3
        )
    )
    sys.stderr.write(
        "Number of aligned nucleotide sites in trimmed codon sequences: {}\n".format(
            len(kept_aa_sites) * 3
        )
    )
    _write_backtrim_outputs(args, trimmed_cdn_records, report)

import copy
import json
import sys

from Bio.Seq import Seq

from cdskit.codonutil import degeneracy_fold_by_position
from cdskit.split import resolve_output_prefix
from cdskit.util import (
    read_seqs,
    resolve_threads,
    stop_if_invalid_codontable,
    stop_if_not_aligned,
    stop_if_not_dna,
    stop_if_not_multiple_of_three,
    write_seqs,
)


def build_degeneracy_output_path(prefix, fold, outseqformat):
    return f'{prefix}_{fold}fold_positions.{outseqformat}'


def classify_alignment_columns(records, codontable):
    assignments = list()
    counts_by_fold = {0: 0, 2: 0, 3: 0, 4: 0}
    num_conflict_sites = 0
    num_unassigned_sites = 0
    for codon_start in range(0, len(records[0].seq), 3):
        fold_values_by_position = [list(), list(), list()]
        for record in records:
            folds = degeneracy_fold_by_position(
                codon=str(record.seq)[codon_start:codon_start + 3],
                codontable=codontable,
            )
            if folds is None:
                continue
            for pos in range(3):
                fold_values_by_position[pos].append(folds[pos])
        for pos in range(3):
            values = fold_values_by_position[pos]
            if len(values) == 0:
                assignments.append(None)
                num_unassigned_sites += 1
                continue
            if len(set(values)) != 1:
                assignments.append(None)
                num_conflict_sites += 1
                continue
            fold = values[0]
            assignments.append(fold)
            counts_by_fold[fold] += 1
    return {
        'assignments': assignments,
        'counts_by_fold': counts_by_fold,
        'num_conflict_sites': num_conflict_sites,
        'num_unassigned_sites': num_unassigned_sites,
    }


def trim_record_to_positions(record, positions):
    trimmed = copy.copy(record)
    seq_str = str(record.seq)
    trimmed.seq = Seq(''.join(seq_str[pos] for pos in positions))
    return trimmed


def write_degeneracy_report(report_path, summary):
    if report_path == '':
        return
    if report_path.lower().endswith('.json'):
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('metric\tvalue\n')
        for key in [
            'num_sequences',
            'num_input_nt_sites',
            'num_input_codon_sites',
            'num_conflict_sites',
            'num_unassigned_sites',
        ]:
            f.write(f'{key}\t{summary[key]}\n')
        for fold in [0, 2, 3, 4]:
            f.write(f'num_{fold}fold_sites\t{summary["counts_by_fold"][str(fold)]}\n')


def degeneracy_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    _ = resolve_threads(getattr(args, 'threads', 1))
    stop_if_not_dna(records=records, label='--seqfile')
    stop_if_not_aligned(records=records)
    stop_if_not_multiple_of_three(records=records)
    stop_if_invalid_codontable(args.codontable)
    selected_folds = sorted(set(int(fold) for fold in args.fold))
    summary = {
        'num_sequences': len(records),
        'num_input_nt_sites': 0,
        'num_input_codon_sites': 0,
        'num_conflict_sites': 0,
        'num_unassigned_sites': 0,
        'counts_by_fold': {'0': 0, '2': 0, '3': 0, '4': 0},
        'selected_folds': selected_folds,
    }
    prefix_str = resolve_output_prefix(args)
    if len(records) == 0:
        write_degeneracy_report(report_path=args.report, summary=summary)
        for fold in selected_folds:
            outfile = build_degeneracy_output_path(prefix=prefix_str, fold=fold, outseqformat=args.outseqformat)
            write_seqs(records=records, outfile=outfile, outseqformat=args.outseqformat)
        return
    classification = classify_alignment_columns(records=records, codontable=args.codontable)
    assignments = classification['assignments']
    summary = {
        'num_sequences': len(records),
        'num_input_nt_sites': len(records[0].seq),
        'num_input_codon_sites': len(records[0].seq) // 3,
        'num_conflict_sites': classification['num_conflict_sites'],
        'num_unassigned_sites': classification['num_unassigned_sites'],
        'counts_by_fold': {str(key): value for key, value in classification['counts_by_fold'].items()},
        'selected_folds': selected_folds,
    }
    write_degeneracy_report(report_path=args.report, summary=summary)
    for fold in selected_folds:
        positions = [idx for idx, assignment in enumerate(assignments) if assignment == fold]
        sys.stderr.write(f'Writing {fold}-fold positions ({len(positions)} sites).\n')
        out_records = [trim_record_to_positions(record=record, positions=positions) for record in records]
        outfile = build_degeneracy_output_path(prefix=prefix_str, fold=fold, outseqformat=args.outseqformat)
        write_seqs(records=out_records, outfile=outfile, outseqformat=args.outseqformat)

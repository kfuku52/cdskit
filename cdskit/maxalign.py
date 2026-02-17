import sys
from collections import Counter

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.util import (
    read_seqs,
    stop_if_not_aligned,
    stop_if_not_multiple_of_three,
    write_seqs,
)


DEFAULT_MISSING_CHARS = '-?.'


def codon_is_present(codon, missing_chars):
    for ch in codon:
        if ch in missing_chars:
            return False
    return True


def build_codon_presence_matrix(records, missing_chars):
    seq_strings = [str(record.seq) for record in records]
    return [
        [codon_is_present(seq[i:i + 3], missing_chars) for i in range(0, len(seq), 3)]
        for seq in seq_strings
    ]


def count_complete_codon_columns(codon_presence_matrix, kept_indices):
    if not kept_indices:
        return 0
    num_codon_sites = len(codon_presence_matrix[0]) if codon_presence_matrix else 0
    num_complete = 0
    for codon_site in range(num_codon_sites):
        complete = True
        for seq_idx in kept_indices:
            if not codon_presence_matrix[seq_idx][codon_site]:
                complete = False
                break
        if complete:
            num_complete += 1
    return num_complete


def alignment_area(codon_presence_matrix, kept_indices):
    complete_codon_columns = count_complete_codon_columns(
        codon_presence_matrix=codon_presence_matrix,
        kept_indices=kept_indices,
    )
    return len(kept_indices) * complete_codon_columns, complete_codon_columns


def subset_support_bitmasks(codon_presence_matrix):
    num_seqs = len(codon_presence_matrix)
    num_sites = len(codon_presence_matrix[0]) if codon_presence_matrix else 0
    masks = list()
    for codon_site in range(num_sites):
        mask = 0
        for seq_idx in range(num_seqs):
            if codon_presence_matrix[seq_idx][codon_site]:
                mask |= (1 << seq_idx)
        masks.append(mask)
    return masks


def get_subset_indices(mask, num_seqs):
    return [i for i in range(num_seqs) if ((mask >> i) & 1) == 1]


def is_better_solution(candidate, best):
    if best is None:
        return True
    if candidate['area'] != best['area']:
        return candidate['area'] > best['area']
    if candidate['num_kept'] != best['num_kept']:
        return candidate['num_kept'] > best['num_kept']
    return candidate['kept_indices'] < best['kept_indices']


def solve_exact(codon_presence_matrix):
    num_seqs = len(codon_presence_matrix)
    support_counts = Counter(subset_support_bitmasks(codon_presence_matrix))
    max_mask = 1 << num_seqs
    best = None
    for subset_mask in range(1, max_mask):
        num_kept = subset_mask.bit_count()
        complete_codon_columns = 0
        for support_mask, site_count in support_counts.items():
            if (support_mask & subset_mask) == subset_mask:
                complete_codon_columns += site_count
        area = num_kept * complete_codon_columns
        candidate = {
            'mask': subset_mask,
            'area': area,
            'num_kept': num_kept,
            'complete_codon_columns': complete_codon_columns,
            'kept_indices': get_subset_indices(subset_mask, num_seqs),
        }
        if is_better_solution(candidate=candidate, best=best):
            best = candidate
    return best


def solve_greedy(codon_presence_matrix):
    active_indices = list(range(len(codon_presence_matrix)))
    current_area, current_complete_codon_columns = alignment_area(
        codon_presence_matrix=codon_presence_matrix,
        kept_indices=active_indices,
    )
    while len(active_indices) > 1:
        best_next = None
        for remove_idx in active_indices:
            kept_indices = [idx for idx in active_indices if idx != remove_idx]
            area, complete_codon_columns = alignment_area(
                codon_presence_matrix=codon_presence_matrix,
                kept_indices=kept_indices,
            )
            candidate = {
                'remove_idx': remove_idx,
                'kept_indices': kept_indices,
                'area': area,
                'complete_codon_columns': complete_codon_columns,
            }
            if best_next is None:
                best_next = candidate
                continue
            if candidate['area'] != best_next['area']:
                if candidate['area'] > best_next['area']:
                    best_next = candidate
                continue
            if candidate['kept_indices'] < best_next['kept_indices']:
                best_next = candidate
        if best_next is None:
            break
        if best_next['area'] <= current_area:
            break
        active_indices = best_next['kept_indices']
        current_area = best_next['area']
        current_complete_codon_columns = best_next['complete_codon_columns']
    return {
        'kept_indices': active_indices,
        'num_kept': len(active_indices),
        'area': current_area,
        'complete_codon_columns': current_complete_codon_columns,
    }


def parse_missing_chars(missing_char_arg):
    if missing_char_arg == '':
        return set(DEFAULT_MISSING_CHARS)
    return set(missing_char_arg)


def extract_complete_codon_indices(codon_presence_matrix, kept_indices):
    if not kept_indices:
        return list()
    num_sites = len(codon_presence_matrix[0]) if codon_presence_matrix else 0
    complete_indices = list()
    for codon_site in range(num_sites):
        complete = True
        for seq_idx in kept_indices:
            if not codon_presence_matrix[seq_idx][codon_site]:
                complete = False
                break
        if complete:
            complete_indices.append(codon_site)
    return complete_indices


def slice_record_to_codon_sites(record, codon_site_indices):
    seq = str(record.seq)
    new_seq = ''.join(seq[(site * 3):((site * 3) + 3)] for site in codon_site_indices)
    return SeqRecord(
        seq=Seq(new_seq),
        id=record.id,
        name=record.name,
        description=record.description,
    )


def pick_solver_mode(num_records, mode, max_exact_sequences):
    if mode == 'auto':
        if num_records <= max_exact_sequences:
            return 'exact'
        return 'greedy'
    return mode


def maxalign_main(args):
    original_records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    if len(original_records) == 0:
        write_seqs(records=original_records, outfile=args.outfile, outseqformat=args.outseqformat)
        return
    stop_if_not_aligned(records=original_records)
    stop_if_not_multiple_of_three(records=original_records)
    missing_chars = parse_missing_chars(args.missing_char)
    codon_presence_matrix = build_codon_presence_matrix(
        records=original_records,
        missing_chars=missing_chars,
    )
    initial_indices = list(range(len(original_records)))
    initial_area, initial_complete_codon_columns = alignment_area(
        codon_presence_matrix=codon_presence_matrix,
        kept_indices=initial_indices,
    )
    solver_mode = pick_solver_mode(
        num_records=len(original_records),
        mode=args.mode,
        max_exact_sequences=args.max_exact_sequences,
    )
    if solver_mode == 'exact':
        if len(original_records) > args.max_exact_sequences:
            txt = '--mode exact requires <= --max_exact_sequences input records. Got {:,} > {:,}. Exiting.\n'
            raise Exception(txt.format(len(original_records), args.max_exact_sequences))
        solution = solve_exact(codon_presence_matrix=codon_presence_matrix)
    elif solver_mode == 'greedy':
        solution = solve_greedy(codon_presence_matrix=codon_presence_matrix)
    else:
        raise Exception('Unknown mode: {}'.format(solver_mode))
    kept_indices = solution['kept_indices']
    complete_codon_indices = extract_complete_codon_indices(
        codon_presence_matrix=codon_presence_matrix,
        kept_indices=kept_indices,
    )
    removed_ids = [
        original_records[i].id
        for i in range(len(original_records))
        if i not in set(kept_indices)
    ]
    txt = 'maxalign mode: {}\n'
    sys.stderr.write(txt.format(solver_mode))
    txt = 'Initial alignment area (codon units): {:,} (= {:,} seqs x {:,} complete codon sites)\n'
    sys.stderr.write(txt.format(initial_area, len(original_records), initial_complete_codon_columns))
    txt = 'Final alignment area (codon units): {:,} (= {:,} seqs x {:,} complete codon sites)\n'
    sys.stderr.write(txt.format(solution['area'], len(kept_indices), len(complete_codon_indices)))
    txt = 'Removed sequences: {:,}\n'
    sys.stderr.write(txt.format(len(removed_ids)))
    for seq_id in removed_ids:
        sys.stderr.write('Removed: {}\n'.format(seq_id))
    output_records = [
        slice_record_to_codon_sites(
            record=original_records[i],
            codon_site_indices=complete_codon_indices,
        )
        for i in kept_indices
    ]
    write_seqs(records=output_records, outfile=args.outfile, outseqformat=args.outseqformat)

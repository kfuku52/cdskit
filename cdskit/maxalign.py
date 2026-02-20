import json
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.util import (
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
    stop_if_not_aligned,
    stop_if_not_multiple_of_three,
    write_seqs,
)


DEFAULT_MISSING_CHARS = '-?.'


def popcount(mask):
    count = 0
    while mask:
        mask &= mask - 1
        count += 1
    return count


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


def build_support_masks(records, missing_chars):
    if len(records) == 0:
        return list()
    num_sites = len(records[0].seq) // 3
    masks = [0] * num_sites
    missing_chars = set(missing_chars)
    for seq_idx, record in enumerate(records):
        seq = str(record.seq)
        seq_bit = 1 << seq_idx
        site_idx = 0
        for pos in range(0, len(seq), 3):
            if (
                (seq[pos] not in missing_chars)
                and (seq[pos + 1] not in missing_chars)
                and (seq[pos + 2] not in missing_chars)
            ):
                masks[site_idx] |= seq_bit
            site_idx += 1
    return masks


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


def indices_to_bitmask(indices):
    mask = 0
    for idx in indices:
        mask |= (1 << idx)
    return mask


def support_counts_from_matrix(codon_presence_matrix):
    return Counter(subset_support_bitmasks(codon_presence_matrix))


def support_counts_from_masks(support_masks):
    return Counter(support_masks)


def is_better_solution(candidate, best):
    if best is None:
        return True
    if candidate['area'] != best['area']:
        return candidate['area'] > best['area']
    if candidate['num_kept'] != best['num_kept']:
        return candidate['num_kept'] > best['num_kept']
    return candidate['kept_indices'] < best['kept_indices']


def count_complete_columns_with_support(support_counts, kept_mask):
    complete_codon_columns = 0
    for support_mask, site_count in support_counts.items():
        if (support_mask & kept_mask) == kept_mask:
            complete_codon_columns += site_count
    return complete_codon_columns


def mask_to_indices(mask, num_sequences):
    return get_subset_indices(mask=mask, num_seqs=num_sequences)


def build_variable_subset_masks(variable_indices):
    max_mask = 1 << len(variable_indices)
    subset_masks = [0] * max_mask
    for subset_mask in range(1, max_mask):
        lsb = subset_mask & -subset_mask
        bit_idx = lsb.bit_length() - 1
        subset_masks[subset_mask] = subset_masks[subset_mask ^ lsb] | (1 << variable_indices[bit_idx])
    return subset_masks


def evaluate_exact_subset_range(
    subset_start,
    subset_end,
    subset_variable_masks,
    required_mask,
    required_count,
    support_counts,
    max_removed,
    total_sequences,
    num_sequences,
):
    best = None
    for subset_mask in range(subset_start, subset_end):
        num_kept = required_count + popcount(subset_mask)
        if num_kept == 0:
            continue
        num_removed = total_sequences - num_kept
        if (max_removed is not None) and (num_removed > max_removed):
            continue
        kept_mask = required_mask | subset_variable_masks[subset_mask]
        complete_codon_columns = count_complete_columns_with_support(
            support_counts=support_counts,
            kept_mask=kept_mask,
        )
        area = num_kept * complete_codon_columns
        if best is None:
            kept_indices = mask_to_indices(mask=kept_mask, num_sequences=num_sequences)
            best = {
                'area': area,
                'num_kept': num_kept,
                'complete_codon_columns': complete_codon_columns,
                'kept_indices': kept_indices,
                'steps': list(),
            }
            continue
        if area > best['area']:
            kept_indices = mask_to_indices(mask=kept_mask, num_sequences=num_sequences)
            best = {
                'area': area,
                'num_kept': num_kept,
                'complete_codon_columns': complete_codon_columns,
                'kept_indices': kept_indices,
                'steps': list(),
            }
            continue
        if area < best['area']:
            continue
        if num_kept > best['num_kept']:
            kept_indices = mask_to_indices(mask=kept_mask, num_sequences=num_sequences)
            best = {
                'area': area,
                'num_kept': num_kept,
                'complete_codon_columns': complete_codon_columns,
                'kept_indices': kept_indices,
                'steps': list(),
            }
            continue
        if num_kept < best['num_kept']:
            continue
        kept_indices = mask_to_indices(mask=kept_mask, num_sequences=num_sequences)
        if kept_indices < best['kept_indices']:
            best = {
                'area': area,
                'num_kept': num_kept,
                'complete_codon_columns': complete_codon_columns,
                'kept_indices': kept_indices,
                'steps': list(),
            }
    return best


def evaluate_exact_subset_chunk(
    subset_range,
    subset_variable_masks,
    required_mask,
    required_count,
    support_counts,
    max_removed,
    total_sequences,
    num_sequences,
):
    return evaluate_exact_subset_range(
        subset_start=subset_range[0],
        subset_end=subset_range[1],
        subset_variable_masks=subset_variable_masks,
        required_mask=required_mask,
        required_count=required_count,
        support_counts=support_counts,
        max_removed=max_removed,
        total_sequences=total_sequences,
        num_sequences=num_sequences,
    )


def solve_exact(
    codon_presence_matrix,
    candidate_indices=None,
    required_indices=None,
    max_removed=None,
    total_sequences=None,
    threads=1,
    support_counts=None,
    num_sequences=None,
):
    if num_sequences is None:
        if codon_presence_matrix is None:
            raise Exception('num_sequences is required when codon_presence_matrix is None.')
        num_sequences = len(codon_presence_matrix)
    if total_sequences is None:
        total_sequences = num_sequences
    if candidate_indices is None:
        candidate_indices = list(range(num_sequences))
    candidate_indices = sorted(candidate_indices)
    candidate_set = set(candidate_indices)
    if required_indices is None:
        required_indices = list()
    required_indices = sorted(required_indices)
    required_set = set(required_indices)
    if not required_set.issubset(candidate_set):
        raise Exception('required_indices must be a subset of candidate_indices.')

    variable_indices = [idx for idx in candidate_indices if idx not in required_set]
    if support_counts is None:
        if codon_presence_matrix is None:
            raise Exception('support_counts or codon_presence_matrix is required.')
        support_counts = support_counts_from_matrix(codon_presence_matrix)
    max_mask = 1 << len(variable_indices)
    subset_variable_masks = build_variable_subset_masks(variable_indices=variable_indices)
    required_mask = indices_to_bitmask(required_indices)
    required_count = len(required_indices)
    threads = resolve_threads(threads=threads)
    if (threads <= 1) or (max_mask <= 1):
        return evaluate_exact_subset_range(
            subset_start=0,
            subset_end=max_mask,
            subset_variable_masks=subset_variable_masks,
            required_mask=required_mask,
            required_count=required_count,
            support_counts=support_counts,
            max_removed=max_removed,
            total_sequences=total_sequences,
            num_sequences=num_sequences,
        )

    chunk_size = max(1, max_mask // (threads * 8))
    ranges = list()
    start = 0
    while start < max_mask:
        end = min(start + chunk_size, max_mask)
        ranges.append((start, end))
        start = end

    # For larger exact searches, use process-based parallelism to avoid the GIL.
    # For small searches, threads are typically faster due to lower startup overhead.
    use_process = max_mask >= 8192
    chunk_best = None
    if use_process:
        try:
            max_workers = min(threads, len(ranges))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        evaluate_exact_subset_chunk,
                        subset_range=subset_range,
                        subset_variable_masks=subset_variable_masks,
                        required_mask=required_mask,
                        required_count=required_count,
                        support_counts=support_counts,
                        max_removed=max_removed,
                        total_sequences=total_sequences,
                        num_sequences=num_sequences,
                    )
                    for subset_range in ranges
                ]
                chunk_best = [future.result() for future in futures]
        except (OSError, PermissionError):
            sys.stderr.write('Process-based parallelism unavailable; falling back to threads.\n')

    if chunk_best is None:
        worker = partial(
            evaluate_exact_subset_chunk,
            subset_variable_masks=subset_variable_masks,
            required_mask=required_mask,
            required_count=required_count,
            support_counts=support_counts,
            max_removed=max_removed,
            total_sequences=total_sequences,
            num_sequences=num_sequences,
        )
        chunk_best = parallel_map_ordered(items=ranges, worker=worker, threads=threads)

    best = None
    for candidate in chunk_best:
        if candidate is None:
            continue
        if is_better_solution(candidate=candidate, best=best):
            best = candidate
    return best


def greedy_complete_and_single_missing_counts(support_counts, active_mask):
    complete_codon_columns = 0
    single_missing_counts = dict()
    for support_mask, site_count in support_counts.items():
        absent_mask = active_mask & ~support_mask
        if absent_mask == 0:
            complete_codon_columns += site_count
            continue
        # Only one active sequence missing at this codon site.
        if (absent_mask & (absent_mask - 1)) == 0:
            missing_idx = absent_mask.bit_length() - 1
            single_missing_counts[missing_idx] = single_missing_counts.get(missing_idx, 0) + site_count
    return complete_codon_columns, single_missing_counts


def solve_greedy(
    codon_presence_matrix,
    active_indices=None,
    protected_indices=None,
    max_removed=None,
    total_sequences=None,
    threads=1,
    support_counts=None,
):
    if total_sequences is None:
        if codon_presence_matrix is None:
            raise Exception('total_sequences is required when codon_presence_matrix is None.')
        total_sequences = len(codon_presence_matrix)
    if active_indices is None:
        active_indices = list(range(total_sequences))
    active_indices = sorted(active_indices)
    if protected_indices is None:
        protected_indices = list()
    protected_set = set(protected_indices)
    if not protected_set.issubset(set(active_indices)):
        raise Exception('protected_indices must be a subset of active_indices.')
    if support_counts is None:
        if codon_presence_matrix is None:
            raise Exception('support_counts or codon_presence_matrix is required.')
        support_counts = support_counts_from_matrix(codon_presence_matrix)
    _ = resolve_threads(threads=threads)
    active_mask = indices_to_bitmask(active_indices)
    current_complete_codon_columns = count_complete_columns_with_support(
        support_counts=support_counts,
        kept_mask=active_mask,
    )
    current_area = len(active_indices) * current_complete_codon_columns
    steps = list()
    while len(active_indices) > 1:
        num_removed_now = total_sequences - len(active_indices)
        if (max_removed is not None) and (num_removed_now >= max_removed):
            break
        removable = [idx for idx in active_indices if idx not in protected_set]
        if len(removable) == 0:
            break
        active_count = len(active_indices)
        current_complete_codon_columns, single_missing_counts = greedy_complete_and_single_missing_counts(
            support_counts=support_counts,
            active_mask=active_mask,
        )
        best_remove_idx = None
        best_area = None
        best_complete_codon_columns = None
        num_kept_next = active_count - 1
        for remove_idx in removable:
            complete_codon_columns = current_complete_codon_columns + single_missing_counts.get(remove_idx, 0)
            candidate_area = num_kept_next * complete_codon_columns
            if best_area is None:
                best_remove_idx = remove_idx
                best_area = candidate_area
                best_complete_codon_columns = complete_codon_columns
                continue
            if candidate_area > best_area:
                best_remove_idx = remove_idx
                best_area = candidate_area
                best_complete_codon_columns = complete_codon_columns
                continue
            # For active_indices sorted ascending, when area ties, removing a larger index
            # yields lexicographically smaller kept_indices.
            if (candidate_area == best_area) and (remove_idx > best_remove_idx):
                best_remove_idx = remove_idx
                best_area = candidate_area
                best_complete_codon_columns = complete_codon_columns
        if best_remove_idx is None:
            break
        if best_area <= current_area:
            break
        active_mask &= ~(1 << best_remove_idx)
        active_indices = [idx for idx in active_indices if idx != best_remove_idx]
        current_area = best_area
        current_complete_codon_columns = best_complete_codon_columns
        steps.append({
            'action': 'remove',
            'removed_index': best_remove_idx,
            'kept_indices': list(active_indices),
            'num_kept': len(active_indices),
            'complete_codon_columns': current_complete_codon_columns,
            'area': current_area,
        })
    return {
        'kept_indices': active_indices,
        'num_kept': len(active_indices),
        'area': current_area,
        'complete_codon_columns': current_complete_codon_columns,
        'steps': steps,
    }


def parse_missing_chars(missing_char_arg):
    if missing_char_arg == '':
        return set(DEFAULT_MISSING_CHARS)
    return set(missing_char_arg)


def parse_csv_patterns(pattern_text):
    if pattern_text is None:
        return list()
    return [part.strip() for part in pattern_text.split(',') if part.strip() != '']


def validate_patterns(patterns, option_name):
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as e:
            txt = 'Invalid regex in {}: {} ({})'
            raise Exception(txt.format(option_name, pattern, e))


def select_indices_by_patterns(records, patterns):
    if len(patterns) == 0:
        return list()
    selected = list()
    for i, record in enumerate(records):
        if any(re.fullmatch(pattern, record.name) for pattern in patterns):
            selected.append(i)
    return selected


def parse_max_removed(max_removed, num_records):
    if max_removed is None:
        return None
    max_removed = int(max_removed)
    if max_removed < 0:
        txt = '--max_removed should be >= 0, but got {}. Exiting.\n'
        raise Exception(txt.format(max_removed))
    if max_removed > num_records:
        txt = '--max_removed ({:,}) is greater than number of sequences ({:,}). Set to {:,}\n'
        sys.stderr.write(txt.format(max_removed, num_records, num_records))
        return num_records
    return max_removed


def extract_complete_codon_indices(codon_presence_matrix=None, kept_indices=None, support_masks=None):
    if not kept_indices:
        return list()
    kept_mask = indices_to_bitmask(kept_indices)
    if support_masks is None:
        support_masks = subset_support_bitmasks(codon_presence_matrix)
    return [
        codon_site
        for codon_site, support_mask in enumerate(support_masks)
        if (support_mask & kept_mask) == kept_mask
    ]


def slice_record_to_codon_sites(record, codon_site_indices):
    seq = str(record.seq)
    new_seq = ''.join(seq[(site * 3):((site * 3) + 3)] for site in codon_site_indices)
    return SeqRecord(
        seq=Seq(new_seq),
        id=record.id,
        name=record.name,
        description=record.description,
    )


def codon_sites_to_nucleotide_ranges(codon_site_indices):
    if len(codon_site_indices) == 0:
        return list()
    ranges = list()
    run_start = codon_site_indices[0]
    run_end = codon_site_indices[0]
    for site in codon_site_indices[1:]:
        if site == run_end + 1:
            run_end = site
            continue
        ranges.append((run_start * 3, (run_end + 1) * 3))
        run_start = site
        run_end = site
    ranges.append((run_start * 3, (run_end + 1) * 3))
    return ranges


def slice_record_to_nucleotide_ranges(record, nucleotide_ranges):
    seq = str(record.seq)
    new_seq = ''.join(seq[start:end] for start, end in nucleotide_ranges)
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


def build_step_snapshot(label, kept_indices, records, removed_id='', support_masks=None, codon_presence_matrix=None):
    if support_masks is None:
        support_masks = subset_support_bitmasks(codon_presence_matrix)
    kept_mask = indices_to_bitmask(kept_indices)
    complete_codon_columns = 0
    for support_mask in support_masks:
        if (support_mask & kept_mask) == kept_mask:
            complete_codon_columns += 1
    area = len(kept_indices) * complete_codon_columns
    removed_ids = [records[i].id for i in range(len(records)) if i not in set(kept_indices)]
    return {
        'label': label,
        'num_kept': len(kept_indices),
        'complete_codon_sites': complete_codon_columns,
        'area': area,
        'removed_id': removed_id,
        'removed_ids': removed_ids,
    }


def write_report(report_path, report_data):
    if report_path == '':
        return
    if report_path.lower().endswith('.json'):
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        return

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('metric\tvalue\n')
        summary_keys = [
            'mode',
            'num_input_sequences',
            'num_kept_sequences',
            'num_removed_sequences',
            'initial_complete_codon_sites',
            'final_complete_codon_sites',
            'initial_area',
            'final_area',
            'area_delta',
            'forced_keep_ids',
            'kept_ids',
            'removed_ids',
        ]
        for key in summary_keys:
            value = report_data.get(key, '')
            if isinstance(value, list):
                value = ','.join(value)
            f.write(f'{key}\t{value}\n')
        f.write('\n')
        f.write('steps\n')
        f.write('label\tnum_kept\tcomplete_codon_sites\tarea\tremoved_id\tremoved_ids\n')
        for step in report_data['steps']:
            removed_ids = ','.join(step['removed_ids'])
            f.write(
                f"{step['label']}\t{step['num_kept']}\t{step['complete_codon_sites']}\t"
                f"{step['area']}\t{step['removed_id']}\t{removed_ids}\n"
            )


def maxalign_main(args):
    mode = getattr(args, 'mode', 'auto')
    max_exact_sequences = int(getattr(args, 'max_exact_sequences', 16))
    missing_char_arg = getattr(args, 'missing_char', DEFAULT_MISSING_CHARS)
    keep_arg = getattr(args, 'keep', '')
    max_removed_arg = getattr(args, 'max_removed', None)
    report_path = getattr(args, 'report', '')
    threads = resolve_threads(getattr(args, 'threads', 1))

    original_records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    if len(original_records) == 0:
        write_seqs(records=original_records, outfile=args.outfile, outseqformat=args.outseqformat)
        return
    stop_if_not_aligned(records=original_records)
    stop_if_not_multiple_of_three(records=original_records)

    keep_patterns = parse_csv_patterns(keep_arg)
    validate_patterns(keep_patterns, '--keep')

    keep_indices = select_indices_by_patterns(records=original_records, patterns=keep_patterns)
    keep_set = set(keep_indices)
    candidate_indices = list(range(len(original_records)))
    if (len(keep_patterns) > 0) and (len(keep_set) == 0):
        sys.stderr.write('No sequence names matched --keep patterns.\n')

    max_removed = parse_max_removed(max_removed_arg, len(original_records))

    missing_chars = parse_missing_chars(missing_char_arg)
    support_masks = build_support_masks(
        records=original_records,
        missing_chars=missing_chars,
    )
    support_counts = support_counts_from_masks(support_masks=support_masks)
    initial_indices = list(range(len(original_records)))
    initial_kept_mask = indices_to_bitmask(initial_indices)
    initial_complete_codon_columns = count_complete_columns_with_support(
        support_counts=support_counts,
        kept_mask=initial_kept_mask,
    )
    initial_area = len(initial_indices) * initial_complete_codon_columns

    if len(candidate_indices) == 0:
        solution = {
            'kept_indices': list(),
            'num_kept': 0,
            'area': 0,
            'complete_codon_columns': 0,
            'steps': list(),
        }
        solver_mode = pick_solver_mode(num_records=0, mode=mode, max_exact_sequences=max_exact_sequences)
    else:
        solver_mode = pick_solver_mode(
            num_records=len(candidate_indices),
            mode=mode,
            max_exact_sequences=max_exact_sequences,
        )
        if solver_mode == 'exact':
            if len(candidate_indices) > max_exact_sequences:
                txt = '--mode exact requires <= --max_exact_sequences candidate records. Got {:,} > {:,}. Exiting.\n'
                raise Exception(txt.format(len(candidate_indices), max_exact_sequences))
            solution = solve_exact(
                codon_presence_matrix=None,
                candidate_indices=candidate_indices,
                required_indices=sorted(keep_set),
                max_removed=max_removed,
                total_sequences=len(original_records),
                threads=threads,
                support_counts=support_counts,
                num_sequences=len(original_records),
            )
        elif solver_mode == 'greedy':
            solution = solve_greedy(
                codon_presence_matrix=None,
                active_indices=candidate_indices,
                protected_indices=sorted(keep_set),
                max_removed=max_removed,
                total_sequences=len(original_records),
                threads=threads,
                support_counts=support_counts,
            )
        else:
            raise Exception('Unknown mode: {}'.format(solver_mode))

        if solution is None:
            txt = 'No feasible solution found under current constraints. Exiting.\n'
            raise Exception(txt)

    kept_indices = sorted(solution['kept_indices'])
    complete_codon_indices = extract_complete_codon_indices(
        kept_indices=kept_indices,
        support_masks=support_masks,
    )
    nucleotide_ranges = codon_sites_to_nucleotide_ranges(codon_site_indices=complete_codon_indices)
    kept_set = set(kept_indices)
    removed_ids = [
        original_records[i].id
        for i in range(len(original_records))
        if i not in kept_set
    ]
    forced_keep_ids = [original_records[i].id for i in sorted(keep_set)]

    txt = 'maxalign mode: {}\n'
    sys.stderr.write(txt.format(solver_mode))
    txt = 'Initial alignment area (codon units): {:,} (= {:,} seqs x {:,} complete codon sites)\n'
    sys.stderr.write(txt.format(initial_area, len(original_records), initial_complete_codon_columns))
    txt = 'Final alignment area (codon units): {:,} (= {:,} seqs x {:,} complete codon sites)\n'
    sys.stderr.write(txt.format(solution['area'], len(kept_indices), len(complete_codon_indices)))
    txt = 'Removed sequences: {:,}\n'
    sys.stderr.write(txt.format(len(removed_ids)))
    if len(forced_keep_ids) > 0:
        txt = 'Protected by keep constraints: {:,}\n'
        sys.stderr.write(txt.format(len(forced_keep_ids)))
    for seq_id in removed_ids:
        sys.stderr.write('Removed: {}\n'.format(seq_id))

    output_records = [
        slice_record_to_nucleotide_ranges(
            record=original_records[i],
            nucleotide_ranges=nucleotide_ranges,
        )
        for i in kept_indices
    ]
    write_seqs(records=output_records, outfile=args.outfile, outseqformat=args.outseqformat)

    if report_path != '':
        steps = [build_step_snapshot(
            label='initial',
            kept_indices=initial_indices,
            records=original_records,
            support_masks=support_masks,
        )]
        for greedy_step in solution.get('steps', list()):
            removed_id = original_records[greedy_step['removed_index']].id
            steps.append(build_step_snapshot(
                label='greedy_remove',
                kept_indices=greedy_step['kept_indices'],
                records=original_records,
                removed_id=removed_id,
                support_masks=support_masks,
            ))
        if (len(steps) == 0) or (steps[-1]['removed_ids'] != removed_ids):
            steps.append(build_step_snapshot(
                label='final',
                kept_indices=kept_indices,
                records=original_records,
                support_masks=support_masks,
            ))

        report_data = {
            'mode': solver_mode,
            'num_input_sequences': len(original_records),
            'num_kept_sequences': len(kept_indices),
            'num_removed_sequences': len(removed_ids),
            'initial_complete_codon_sites': initial_complete_codon_columns,
            'final_complete_codon_sites': len(complete_codon_indices),
            'initial_area': initial_area,
            'final_area': solution['area'],
            'area_delta': solution['area'] - initial_area,
            'forced_keep_ids': forced_keep_ids,
            'kept_ids': [original_records[i].id for i in kept_indices],
            'removed_ids': removed_ids,
            'steps': steps,
        }
        write_report(report_path=report_path, report_data=report_data)

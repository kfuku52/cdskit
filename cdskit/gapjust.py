import numpy
import sys
import re
from cdskit.util import *

def update_gap_ranges(gap_ranges, gap_start, edit_len):
    """
    Shifts all gap ranges to the right of `gap_start` by `edit_len`.
    This is used to keep track of future gap coordinates as we iteratively
    insert or delete bases in the FASTA.
    """
    for i in range(len(gap_ranges)):
        start, end = gap_ranges[i]
        if start > gap_start:
            gap_ranges[i] = (start + edit_len, end + edit_len)
    return gap_ranges

def vectorized_coordinate_update(
    seq_gff_start_coordinates,
    seq_gff_end_coordinates,
    justifications
):
    """
    Updates GFF feature coordinates in place for a list of gap
    justifications. Does NOT touch phase.

    We assume that `original_edit_start` in each justification
    is a 0-based index (as used in Python strings), whereas
    GFF is 1-based. Hence we add +1 when applying the shift.

    We also apply a 'cumulative_offset' so that each gap edit
    is replayed in ascending order, just like the iterative
    edits to the FASTA.
    """
    if len(justifications) == 0:
        return seq_gff_start_coordinates, seq_gff_end_coordinates

    # Accept both legacy dict format and internal compact tuple format.
    # dict: {'original_edit_start': int, 'edit_length': int}
    # tuple: (original_edit_start, edit_length)
    if isinstance(justifications[0], dict):
        justifications_sorted = sorted(justifications, key=lambda x: x['original_edit_start'])
        original_starts_1based = numpy.fromiter(
            (j['original_edit_start'] + 1 for j in justifications_sorted),
            dtype=numpy.int64,
            count=len(justifications_sorted),
        )
        edit_lengths = numpy.fromiter(
            (j['edit_length'] for j in justifications_sorted),
            dtype=numpy.int64,
            count=len(justifications_sorted),
        )
    else:
        # Internal call path already preserves ascending original_edit_start.
        original_starts_1based = numpy.fromiter(
            (j[0] + 1 for j in justifications),
            dtype=numpy.int64,
            count=len(justifications),
        )
        edit_lengths = numpy.fromiter(
            (j[1] for j in justifications),
            dtype=numpy.int64,
            count=len(justifications),
        )

    if not numpy.any(edit_lengths):
        return seq_gff_start_coordinates, seq_gff_end_coordinates

    cumulative_before = numpy.concatenate(([0], numpy.cumsum(edit_lengths[:-1], dtype=numpy.int64)))
    actual_edit_starts_1based = original_starts_1based + cumulative_before
    cumulative_edits = numpy.cumsum(edit_lengths, dtype=numpy.int64)

    # Keep only non-zero edits; zero-length entries have no coordinate effect.
    nonzero = (edit_lengths != 0)
    actual_edit_starts_1based = actual_edit_starts_1based[nonzero]
    cumulative_edits = cumulative_edits[nonzero]

    # Fast vectorized path assumes monotonic edit starts after cumulative shifts.
    # In pathological cases where this is violated, use a safe fallback loop.
    if numpy.any(actual_edit_starts_1based[1:] < actual_edit_starts_1based[:-1]):
        for coords in [seq_gff_start_coordinates, seq_gff_end_coordinates]:
            for i in range(coords.shape[0]):
                value = int(coords[i])
                cumulative_offset = 0
                for j in range(len(edit_lengths)):
                    edit_len = int(edit_lengths[j])
                    actual_edit_start_1based = int(original_starts_1based[j] + cumulative_offset)
                    if (edit_len != 0) and (value > actual_edit_start_1based):
                        value += edit_len
                    cumulative_offset += edit_len
                coords[i] = value
        return seq_gff_start_coordinates, seq_gff_end_coordinates

    def apply_coordinate_shift(coords):
        position = numpy.searchsorted(actual_edit_starts_1based, coords, side='left')
        has_shift = (position > 0)
        if not numpy.any(has_shift):
            return coords
        updated = coords.copy()
        updated[has_shift] = updated[has_shift] + cumulative_edits[position[has_shift] - 1]
        return updated

    updated_starts = apply_coordinate_shift(seq_gff_start_coordinates)
    updated_ends = apply_coordinate_shift(seq_gff_end_coordinates)
    return updated_starts, updated_ends


def should_justify_gap(gap_length, target_gap_length, gap_just_min=None, gap_just_max=None):
    """
    Returns True when a gap should be justified to `target_gap_length`.

    Rules:
      - If `gap_length` already equals target, skip.
      - Gap extension (gap_length < target) follows `gap_just_min` when set.
      - Gap shortening (gap_length > target) follows `gap_just_max` when set.
    """
    if gap_length == target_gap_length:
        return False

    if gap_length < target_gap_length:
        if gap_just_min is not None and gap_length < gap_just_min:
            return False
        return True

    if gap_just_max is not None and gap_length > gap_just_max:
        return False
    return True


def gapjust_main(args):
    """
    Main routine for:
      1) Reading FASTA and replacing all gap lengths with a uniform length (args.gap_len).
      2) Tracking each insertion/deletion in 'justifications'.
      3) Writing the updated FASTA.
      4) Updating GFF coordinates accordingly (without modifying phase),
         then writing the updated GFF.
    """

    # -- 1) Read FASTA, replace 'n' with 'N' for consistency
    gap_just_min = getattr(args, 'gap_just_min', None)
    gap_just_max = getattr(args, 'gap_just_max', None)

    for value, label in [
        (args.gap_len, '--gap_len'),
        (gap_just_min, '--gap_just_min'),
        (gap_just_max, '--gap_just_max'),
    ]:
        if value is not None and value < 0:
            raise ValueError(f'{label} must be >= 0. Got {value}.')

    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    num_justifications = 0
    min_original_gap_length = None
    max_original_gap_length = 0
    justifications_by_seq = dict()

    # -- 2) For each FASTA record, rebuild sequence once while normalizing gap lengths
    for i in range(len(records)):
        seq_str = str(records[i].seq).replace('n', 'N')
        seq_justifications = []
        rebuilt = []
        cursor = 0

        for match in re.finditer('N+', seq_str):
            start = match.start()
            end = match.end()
            gap_length = end - start
            rebuilt.append(seq_str[cursor:start])

            justify_gap = True
            if gap_length == args.gap_len:
                justify_gap = False
            elif gap_length < args.gap_len:
                if (gap_just_min is not None) and (gap_length < gap_just_min):
                    justify_gap = False
            elif (gap_just_max is not None) and (gap_length > gap_just_max):
                justify_gap = False

            if justify_gap:
                rebuilt.append('N' * args.gap_len)
                edit_len = args.gap_len - gap_length
                seq_justifications.append((start, edit_len))
                num_justifications += 1
                if (min_original_gap_length is None) or (gap_length < min_original_gap_length):
                    min_original_gap_length = gap_length
                if gap_length > max_original_gap_length:
                    max_original_gap_length = gap_length
            else:
                rebuilt.append(seq_str[start:end])

            cursor = end

        rebuilt.append(seq_str[cursor:])
        records[i].seq = Bio.Seq.Seq(''.join(rebuilt))

        if seq_justifications:
            justifications_by_seq[records[i].id] = seq_justifications

    # -- 3) Write the updated FASTA
    sys.stderr.write(f'Number of gap justifications: {num_justifications}\n')
    if num_justifications > 0:
        sys.stderr.write(
            f'Minimum and maximum original gap lengths: {min_original_gap_length} and {max_original_gap_length}\n'
        )
    else:
        sys.stderr.write('No gap edits were made.\n')

    write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)

    # -- 4) Read and update GFF coordinates
    if args.ingff is not None:
        gff = read_gff(args.ingff)
        seqid_to_gff_indices = dict()
        for ix, seqid in enumerate(gff['data']['seqid']):
            if seqid not in seqid_to_gff_indices:
                seqid_to_gff_indices[seqid] = []
            seqid_to_gff_indices[seqid].append(ix)

        # For each relevant seqid, update the GFF features
        num_justified_start_coordinate = 0
        num_justified_end_coordinate = 0
        num_justified_gff_gene = 0

        for seqid, seqid_justs in justifications_by_seq.items():
            if seqid not in seqid_to_gff_indices:
                continue

            # Index of features belonging to this seqid
            index_gff_seq = numpy.array(seqid_to_gff_indices[seqid], dtype=int)

            # Extract start/end as arrays
            seq_gff_start_original = gff['data']['start'][index_gff_seq].copy()
            seq_gff_end_original = gff['data']['end'][index_gff_seq].copy()

            # We do not touch 'phase' at all; keep it unchanged

            # Copy the arrays for updating
            seq_gff_start_updated = seq_gff_start_original.copy()
            seq_gff_end_updated   = seq_gff_end_original.copy()

            # Vectorized coordinate shift (no phase updates)
            seq_gff_start_updated, seq_gff_end_updated = vectorized_coordinate_update(
                seq_gff_start_updated,
                seq_gff_end_updated,
                seqid_justs
            )

            # Update the GFF in memory
            gff['data']['start'][index_gff_seq] = seq_gff_start_updated
            gff['data']['end'][index_gff_seq] = seq_gff_end_updated
                # phase remains as is

            # Count how many changed
            is_gene = (gff['data']['type'][index_gff_seq] == 'gene')
            changed_start = (seq_gff_start_original != seq_gff_start_updated)
            changed_end = (seq_gff_end_original != seq_gff_end_updated)

            num_justified_start_coordinate += changed_start.sum()
            num_justified_end_coordinate   += changed_end.sum()
            justified_changes = numpy.logical_and(is_gene, numpy.logical_or(changed_start, changed_end))
            num_justified_gff_gene += justified_changes.sum()

        # Summary
        sys.stderr.write(
            f'Number of justified GFF start coordinates: {num_justified_start_coordinate}\n'
        )
        sys.stderr.write(
            f'Number of justified GFF end coordinates: {num_justified_end_coordinate}\n'
        )
        sys.stderr.write(
            f'Number of justified GFF gene features: {num_justified_gff_gene}\n'
        )

        # -- Write the updated GFF
        write_gff(gff, args.outgff)

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
    # Sort justifications by the leftmost gap coordinate
    justifications_sorted = sorted(justifications, key=lambda x: x['original_edit_start'])

    # Keep track of how many bases have been inserted/deleted so far
    cumulative_offset = 0

    for justification in justifications_sorted:
        original_edit_start = justification['original_edit_start']
        edit_len = justification['edit_length']

        # Convert 0-based (FASTA) to 1-based (GFF), then add the offset
        actual_edit_start_1based = (original_edit_start + 1) + cumulative_offset

        if edit_len != 0:
            # Shift start coordinates greater than actual_edit_start_1based
            mask_start = seq_gff_start_coordinates > actual_edit_start_1based
            seq_gff_start_coordinates[mask_start] += edit_len

            # Shift end coordinates greater than actual_edit_start_1based
            mask_end = seq_gff_end_coordinates > actual_edit_start_1based
            seq_gff_end_coordinates[mask_end] += edit_len

        # Accumulate offset for subsequent justifications
        cumulative_offset += edit_len

    return seq_gff_start_coordinates, seq_gff_end_coordinates


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
    justifications = []

    # -- 2) For each FASTA record, find all gap ranges, fix them to 'args.gap_len'
    for i in range(len(records)):
        # Normalize all gaps to uppercase N
        records[i].seq = records[i].seq.replace('n', 'N')

        # Collect zero-based positions of 'N'
        gap_coordinates = [m.start() for m in re.finditer('N', str(records[i].seq))]
        original_gap_ranges = coordinates2ranges(gap_coordinates)
        updated_gap_ranges = original_gap_ranges.copy()

        # Gap lengths
        gap_lengths = [end - start + 1 for (start, end) in original_gap_ranges]

        # Iterate over each gap and insert/delete to make its length = args.gap_len
        for j in range(len(gap_lengths)):
            gap_length = gap_lengths[j]
            if not should_justify_gap(
                gap_length=gap_length,
                target_gap_length=args.gap_len,
                gap_just_min=gap_just_min,
                gap_just_max=gap_just_max,
            ):
                continue

            updated_gap_start = updated_gap_ranges[j][0]
            edit_len = args.gap_len - gap_length  # positive = insertion, negative = deletion

            # Perform the edit on the FASTA sequence
            if edit_len > 0:
                # Insertion of extra Ns
                records[i].seq = (
                    records[i].seq[:updated_gap_start]
                    + 'N' * edit_len
                    + records[i].seq[updated_gap_start:]
                )
            else:
                # Deletion
                records[i].seq = (
                    records[i].seq[:updated_gap_start]
                    + records[i].seq[updated_gap_start - edit_len:]
                )

            # Update future gap range positions
            updated_gap_ranges = update_gap_ranges(
                gap_ranges=updated_gap_ranges,
                gap_start=updated_gap_start,
                edit_len=edit_len
            )

            # Store the justification info
            justifications.append({
                'seq': records[i].id,
                'original_gap_length': gap_length,
                'original_edit_start': original_gap_ranges[j][0],
                'edit_length': edit_len
            })

    # -- 3) Write the updated FASTA
    sys.stderr.write(f'Number of gap justifications: {len(justifications)}\n')
    all_lengths = [j["original_gap_length"] for j in justifications]
    if all_lengths:
        sys.stderr.write(
            f'Minimum and maximum original gap lengths: {min(all_lengths)} and {max(all_lengths)}\n'
        )
    else:
        sys.stderr.write('No gap edits were made.\n')

    write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)

    # -- 4) Read and update GFF coordinates
    if args.ingff is not None:
        gff = read_gff(args.ingff)

        # Figure out which seqids had gap edits
        justification_seqids = set(j['seq'] for j in justifications)

        # For each relevant seqid, update the GFF features
        num_justified_start_coordinate = 0
        num_justified_end_coordinate = 0
        num_justified_gff_gene = 0

        for record_idx in range(len(records)):
            seqid = records[record_idx].id
            if seqid not in justification_seqids:
                continue

            # Index of features belonging to this seqid
            index_gff_seq = [
                ix for ix in range(len(gff['data']))
                if gff['data'][ix]['seqid'] == seqid
            ]

            # Extract start/end as arrays
            seq_gff_start_original = numpy.array([gff['data'][ix]['start'] for ix in index_gff_seq])
            seq_gff_end_original   = numpy.array([gff['data'][ix]['end']   for ix in index_gff_seq])

            # We do not touch 'phase' at all; keep it unchanged

            # Copy the arrays for updating
            seq_gff_start_updated = seq_gff_start_original.copy()
            seq_gff_end_updated   = seq_gff_end_original.copy()

            # Gather all justifications for this seqid
            seqid_justs = [j for j in justifications if j['seq'] == seqid]

            # Vectorized coordinate shift (no phase updates)
            seq_gff_start_updated, seq_gff_end_updated = vectorized_coordinate_update(
                seq_gff_start_updated,
                seq_gff_end_updated,
                seqid_justs
            )

            # Update the GFF in memory
            for k, ix in enumerate(index_gff_seq):
                gff['data'][ix]['start'] = seq_gff_start_updated[k]
                gff['data'][ix]['end']   = seq_gff_end_updated[k]
                # phase remains as is

            # Count how many changed
            is_gene = numpy.array([
                (gff['data'][ix]['type'] == 'gene')
                for ix in index_gff_seq
            ], dtype=bool)
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

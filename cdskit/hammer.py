import numpy

from cdskit.util import *

GAP_ONLY_CHARS = frozenset('-NXnx')


def codon_is_gap_like(codon):
    for ch in codon:
        if ch not in GAP_ONLY_CHARS:
            return False
    return True


def hammer_main(args):
    original_records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    if len(original_records) == 0:
        write_seqs(records=original_records, outfile=args.outfile, outseqformat=args.outseqformat)
        return
    stop_if_not_multiple_of_three(original_records)
    stop_if_not_aligned(original_records)
    if args.nail=='all':
        nail_value = len(original_records)
        txt = '--nail all was specified. Set to {:,}\n'
        sys.stderr.write(txt.format(nail_value))
    elif (int(args.nail)>len(original_records)):
        nail_value = len(original_records)
        txt = '--nail ({:,}) is greater than the number of input sequences ({:,}). Decreased to {:,}\n'
        sys.stderr.write(txt.format(int(args.nail), len(original_records), nail_value))
    else:
        nail_value = int(args.nail)
    aa_records = translate_records(records=original_records, codontable=args.codontable)
    aa_array = records2array(aa_records)
    max_len = aa_array.shape[1]
    missing_site = numpy.isin(aa_array, ['-', '?', 'X', '*']).sum(axis=0)
    non_missing_site = len(original_records) - missing_site
    seq_strs = [str(record.seq) for record in original_records]
    seq_codon_lists = [[seq[j:j+3] for j in range(0, len(seq), 3)] for seq in seq_strs]
    selected_non_missing_idx = None
    last_non_missing_idx = numpy.array([], dtype=int)
    codon_gap_like_matrix = None
    if args.prevent_gap_only:
        codon_gap_like_matrix = numpy.array(
            [
                [codon_is_gap_like(codon) for codon in seq_codons]
                for seq_codons in seq_codon_lists
            ],
            dtype=bool,
        )
    for current_nail in range(nail_value, 0, -1):
        non_missing_idx = numpy.flatnonzero(non_missing_site >= current_nail)
        non_missing_idx_list = non_missing_idx.tolist()
        last_non_missing_idx = non_missing_idx
        num_removed_site = max_len - non_missing_idx.shape[0]
        sys.stderr.write('{:,} out of {:,} codon sites will be removed.\n'.format(num_removed_site, max_len))
        if args.prevent_gap_only:
            gap_only_mask = numpy.all(codon_gap_like_matrix[:, non_missing_idx], axis=1)
            if numpy.any(gap_only_mask):
                for i in numpy.flatnonzero(gap_only_mask):
                    txt = 'A gap-only sequence was generated with --nail {}. Will try --nail {}: {}\n'
                    sys.stderr.write(txt.format(current_nail, current_nail-1, original_records[i].name))
                continue
        selected_non_missing_idx = non_missing_idx
        break
    if selected_non_missing_idx is None:
        selected_non_missing_idx = last_non_missing_idx
    selected_non_missing_idx_list = selected_non_missing_idx.tolist()
    records = list()
    for i in range(len(original_records)):
        new_seq = ''.join([seq_codon_lists[i][codon_site] for codon_site in selected_non_missing_idx_list])
        new_record = Bio.SeqRecord.SeqRecord(seq=Bio.Seq.Seq(new_seq), id=original_records[i].id, name=original_records[i].name, description=original_records[i].description)
        records.append(new_record)
    write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)

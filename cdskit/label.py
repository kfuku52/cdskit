import sys
from collections import Counter

from cdskit.util import read_seqs, resolve_threads, stop_if_not_dna, write_seqs


def parse_replace_chars(replace_chars):
    parts = replace_chars.split('--')
    if len(parts) != 2:
        txt = '--replace_chars must include exactly one "--": FROM1FROM2...--TO. Exiting.\n'
        raise Exception(txt)
    from_part, to_part = parts
    if from_part == '':
        txt = '--replace_chars FROM part should not be empty. Exiting.\n'
        raise Exception(txt)
    if to_part == '':
        txt = '--replace_chars TO part should not be empty. Exiting.\n'
        raise Exception(txt)
    if len(to_part) != 1:
        txt = '--replace_chars TO part should be exactly one character, but got "{}". Exiting.\n'
        raise Exception(txt.format(to_part))
    from_chars = list(from_part)
    to_char = to_part
    return from_chars, to_char


def apply_char_replacement(records, from_chars, to_char):
    replace_count = 0
    translation_table = str.maketrans(''.join(from_chars), to_char * len(from_chars))
    for record in records:
        replaced_id = record.id.translate(translation_table)
        if replaced_id != record.id:
            replace_count += 1
            record.id = replaced_id
            record.description = ''
    return replace_count


def clip_label_ids(records, clip_len):
    clip_count = 0
    for record in records:
        if len(record.id) > clip_len:
            clip_count += 1
            record.id = record.id[:clip_len]
            record.description = ''
    return clip_count


def uniquify_label_ids(records):
    nonunique_count = 0
    name_counts = Counter([record.id for record in records])
    suffix_counts = {}
    for record in records:
        if name_counts[record.id] > 1:
            nonunique_count += 1
            if record.id not in suffix_counts:
                suffix_counts[record.id] = 1
            else:
                suffix_counts[record.id] += 1
            record.id += '_{}'.format(suffix_counts[record.id])
            record.description = ''
    nonunique_names = [name for name in name_counts if name_counts[name] > 1]
    return nonunique_count, nonunique_names


def replace_record_id(record_id, from_chars, to_char):
    translation_table = str.maketrans(''.join(from_chars), to_char * len(from_chars))
    replaced_id = record_id.translate(translation_table)
    return replaced_id, (replaced_id != record_id)


def clip_record_id(record_id, clip_len):
    if len(record_id) > clip_len:
        return record_id[:clip_len], True
    return record_id, False


def validate_clip_len(clip_len):
    if clip_len < 0:
        txt = '--clip_len should be >= 0, but got {}. Exiting.\n'
        raise Exception(txt.format(clip_len))


def label_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=records, label='--seqfile')
    validate_clip_len(args.clip_len)
    _ = resolve_threads(getattr(args, 'threads', 1))
    if args.replace_chars != '':
        from_chars, to_char = parse_replace_chars(args.replace_chars)
        replace_count = apply_char_replacement(records=records, from_chars=from_chars, to_char=to_char)
        sys.stderr.write('Number of character-replaced sequence labels: {:,}\n'.format(replace_count))
    if args.clip_len != 0:
        clip_count = clip_label_ids(records=records, clip_len=args.clip_len)
        sys.stderr.write('Number of clipped sequence labels: {:,}\n'.format(clip_count))
    if args.unique:
        nonunique_count, nonunique_names = uniquify_label_ids(records)
        sys.stderr.write('Number of resolved non-unique sequence labels: {:,}\n'.format(nonunique_count))
        sys.stderr.write('Non-unique sequence labels:\n{}\n'.format('\n'.join(nonunique_names)))
    write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)

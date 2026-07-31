import Bio.Data.CodonTable
import Bio.Seq
import Bio.SeqIO
import numpy as np

import io
import json
import os
import re
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from cdskit.cliutil import resolve_threads as resolve_cli_threads

GFF_DTYPE = [
    ('seqid', 'U100'),
    ('source', 'U100'),
    ('type', 'U100'),
    ('start', 'i4'),
    ('end', 'i4'),
    ('score', 'U100'),
    ('strand', 'U10'),
    ('phase', 'U10'),
    ('attributes', 'U500')
]
GFF_COLUMNS = ('seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes')
DNA_ALLOWED_CHARS = frozenset('ACGTRYSWKMBDHVNXacgtryswkmbdhvnx-?.')
PROTEIN_ALLOWED_CHARS = frozenset(
    'ABCDEFGHIKLMNPQRSTVWXYZJUO*abcdefghiklmnpqrstvwxyzjuo-?.'
)
DEFAULT_MAX_SEQUENCE_RECORDS = 1_000_000
DEFAULT_MAX_SEQUENCE_RESIDUES = 1_000_000_000
DEFAULT_MAX_SEQUENCE_ID_LENGTH = 4096
DEFAULT_MAX_REGEX_LENGTH = 512
QUALITY_REQUIRED_FORMATS = frozenset({
    'fastq',
    'fastq-sanger',
    'fastq-illumina',
    'fastq-solexa',
    'qual',
})


def _normalized_path(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def validate_distinct_paths(inputs=(), outputs=()):
    """Reject input/output and output/output path collisions before writing."""
    input_paths = [
        path
        for path in inputs
        if path is not None and str(path) not in ('', '-')
    ]
    output_paths = [
        path
        for path in outputs
        if path is not None and str(path) not in ('', '-')
    ]
    normalized_inputs = {_normalized_path(path): str(path) for path in input_paths}
    seen_outputs = {}
    for path in output_paths:
        normalized = _normalized_path(path)
        if normalized in normalized_inputs:
            raise ValueError(
                'Input and output paths should be different: {}.'.format(path)
            )
        if normalized in seen_outputs:
            raise ValueError(
                'Output paths should be different: {} and {}.'.format(
                    seen_outputs[normalized],
                    path,
                )
            )
        seen_outputs[normalized] = str(path)


@contextmanager
def atomic_output_path(path):
    """Yield a same-directory temporary path and atomically replace *path*."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix='.{}.'.format(destination.name),
        suffix='.tmp',
        dir=str(destination.parent),
    )
    os.close(fd)
    try:
        yield temporary
        with open(temporary, 'rb') as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def atomic_output_paths(paths):
    """Stage multiple outputs and commit them together with rollback on error."""
    destinations = [Path(path) for path in paths]
    validate_distinct_paths(outputs=destinations)
    temporary_paths = []
    backup_paths = []
    commit_succeeded = False
    try:
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                prefix='.{}.'.format(destination.name),
                suffix='.tmp{}'.format(destination.suffix),
                dir=str(destination.parent),
            )
            os.close(fd)
            temporary_paths.append(Path(temporary))
        yield [str(path) for path in temporary_paths]
        for temporary in temporary_paths:
            with open(temporary, 'rb') as handle:
                os.fsync(handle.fileno())

        committed = 0
        try:
            for destination in destinations:
                if destination.exists() or destination.is_symlink():
                    fd, backup = tempfile.mkstemp(
                        prefix='.{}.'.format(destination.name),
                        suffix='.bak',
                        dir=str(destination.parent),
                    )
                    os.close(fd)
                    os.unlink(backup)
                    os.replace(destination, backup)
                    backup_paths.append(Path(backup))
                else:
                    backup_paths.append(None)

            for temporary, destination in zip(temporary_paths, destinations):
                os.replace(temporary, destination)
                committed += 1
            commit_succeeded = True
        except Exception as commit_error:
            rollback_errors = []
            for index, destination in enumerate(destinations):
                backup = backup_paths[index] if index < len(backup_paths) else None
                try:
                    if backup is not None and backup.exists():
                        os.replace(backup, destination)
                    elif index < committed and (
                        destination.exists() or destination.is_symlink()
                    ):
                        os.unlink(destination)
                except OSError as rollback_error:
                    rollback_errors.append(str(rollback_error))
            if rollback_errors:
                retained = [
                    str(backup)
                    for backup in backup_paths
                    if backup is not None and backup.exists()
                ]
                raise RuntimeError(
                    'Failed to roll back an atomic multi-output update. '
                    'Recovery backups were retained at {}. Errors: {}'.format(
                        retained,
                        rollback_errors,
                    )
                ) from commit_error
            raise
    finally:
        for temporary in temporary_paths:
            if temporary.exists():
                temporary.unlink()
        if commit_succeeded:
            for backup in backup_paths:
                if backup is not None and backup.exists():
                    try:
                        backup.unlink()
                    except OSError as exc:
                        warnings.warn(
                            'Committed outputs, but could not remove recovery '
                            'backup {}: {}'.format(backup, exc),
                            RuntimeWarning,
                        )


@contextmanager
def atomic_text_writer(path, encoding='utf-8', newline=None):
    """Open a text destination transactionally."""
    with atomic_output_path(path) as temporary:
        with open(temporary, 'w', encoding=encoding, newline=newline) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())


def atomic_write_json(path, payload, **kwargs):
    """Write standards-compliant JSON transactionally."""
    kwargs.setdefault('ensure_ascii', False)
    kwargs.setdefault('allow_nan', False)
    with atomic_text_writer(path, encoding='utf-8') as handle:
        json.dump(payload, handle, **kwargs)
        handle.write('\n')


def _repeated_group_has_ambiguous_content(pattern):
    stack = []
    in_character_class = False
    escaped = False
    for index, character in enumerate(pattern):
        if escaped:
            escaped = False
            continue
        if character == '\\':
            escaped = True
            continue
        if character == '[':
            in_character_class = True
            continue
        if character == ']' and in_character_class:
            in_character_class = False
            continue
        if in_character_class:
            continue
        if character == '(':
            stack.append(index)
            continue
        if character != ')' or not stack:
            continue
        start = stack.pop()
        following = pattern[index + 1:index + 2]
        if following not in ('*', '+', '?', '{'):
            continue
        content = pattern[start + 1:index]
        content = re.sub(
            r'^\?(?::|=|!|<[=!]|P<[^>]+>)',
            '',
            content,
            count=1,
        )
        content_without_escapes = re.sub(r'\\.', '', content)
        if re.search(r'[*+?{|]', content_without_escapes):
            return True
    return False


def compile_safe_regex(pattern, label='regular expression'):
    """Compile a bounded regex while rejecting common nested-repeat DoS forms."""
    pattern = str(pattern)
    if len(pattern) > DEFAULT_MAX_REGEX_LENGTH:
        raise ValueError(
            '{} should be at most {} characters.'.format(
                label,
                DEFAULT_MAX_REGEX_LENGTH,
            )
        )
    if (
        _repeated_group_has_ambiguous_content(pattern)
        or re.search(r'\\[1-9]|\(\?P=', pattern)
    ):
        raise ValueError(
            '{} contains a potentially unsafe repeated group or backreference.'.format(
                label
            )
        )
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ValueError('Invalid {}: {}.'.format(label, exc))


def replace_record_sequence(record, sequence, preserve_features=False):
    """Replace sequence data without leaving stale per-letter annotations."""
    sequence = Bio.Seq.Seq(sequence)
    if len(sequence) != len(record.seq):
        record.letter_annotations = {}
        if not preserve_features:
            record.features = []
    record.seq = sequence
    return record


def resolve_threads(threads):
    try:
        return resolve_cli_threads(threads)
    except ValueError as exc:
        raise Exception(str(exc) + ' Exiting.\n')


def parallel_map_ordered(items, worker, threads):
    items = list(items)
    if (threads <= 1) or (len(items) <= 1):
        return [worker(item) for item in items]

    max_workers = min(threads, len(items))
    if len(items) <= (max_workers * 2):
        return [worker(item) for item in items]

    # Run multiple items per submitted task to reduce scheduler overhead
    # when per-item work is small.
    if len(items) <= (max_workers * 256):
        chunk_size = max(1, (len(items) + max_workers - 1) // max_workers)
    else:
        chunk_size = max(1, len(items) // (max_workers * 8))
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    def process_chunk(chunk):
        return [worker(item) for item in chunk]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunk_results = list(executor.map(process_chunk, chunks))

    out = list()
    for chunk_result in chunk_results:
        out.extend(chunk_result)
    return out


def read_seqs(seqfile, seqformat):
    parsed = sys.stdin if seqfile == '-' else seqfile
    max_records = int(os.environ.get('CDSKIT_MAX_SEQUENCE_RECORDS', DEFAULT_MAX_SEQUENCE_RECORDS))
    max_residues = int(os.environ.get('CDSKIT_MAX_SEQUENCE_RESIDUES', DEFAULT_MAX_SEQUENCE_RESIDUES))
    max_id_length = int(
        os.environ.get(
            'CDSKIT_MAX_SEQUENCE_ID_LENGTH',
            DEFAULT_MAX_SEQUENCE_ID_LENGTH,
        )
    )
    records = []
    residues = 0
    for record in Bio.SeqIO.parse(parsed, seqformat):
        if len(record.id) > max_id_length:
            raise ValueError(
                'Sequence identifier exceeds {} characters. Set '
                'CDSKIT_MAX_SEQUENCE_ID_LENGTH to change the safety limit.'.format(
                    max_id_length
                )
            )
        records.append(record)
        residues += len(record.seq)
        if len(records) > max_records:
            raise ValueError(
                'Input exceeds {} sequence records. Set CDSKIT_MAX_SEQUENCE_RECORDS to change the safety limit.'.format(
                    max_records
                )
            )
        if residues > max_residues:
            raise ValueError(
                'Input exceeds {:,} residues. Set CDSKIT_MAX_SEQUENCE_RESIDUES to change the safety limit.'.format(
                    max_residues
                )
            )
    sys.stderr.write('Number of input sequences: {:,}\n'.format(len(records)))
    return records


def write_seqs(records, outfile, outseqformat):
    sys.stderr.write('Number of output sequences: {:,}\n'.format(len(records)))
    if str(outseqformat).lower() in QUALITY_REQUIRED_FORMATS:
        for record in records:
            quality_keys = [
                key
                for key in ('phred_quality', 'solexa_quality')
                if key in record.letter_annotations
            ]
            if not quality_keys:
                raise ValueError(
                    'Output format {} requires per-base quality scores, but '
                    'record {} has none. Use FASTA for transformed sequences.'.format(
                        outseqformat,
                        record.id,
                    )
                )
            for key in quality_keys:
                if len(record.letter_annotations[key]) != len(record.seq):
                    raise ValueError(
                        'Quality length mismatch for record {} ({}).'.format(
                            record.id,
                            key,
                        )
                    )
    if outfile == '-':
        Bio.SeqIO.write(records, sys.stdout, outseqformat)
    else:
        with atomic_output_path(outfile) as temporary:
            Bio.SeqIO.write(records, temporary, outseqformat)


def stop_if_not_multiple_of_three(records):
    has_non_triplet_sequence = False
    for record in records:
        if len(record.seq) % 3 != 0:
            txt = 'Sequence length is not multiple of three: {}\n'.format(record.id)
            sys.stderr.write(txt)
            has_non_triplet_sequence = True
    if has_non_triplet_sequence:
        txt = 'Input sequence length should be multiple of three. ' \
              'Consider applying `cdskit pad` if the input is truncated coding sequences. Exiting.\n'
        raise Exception(txt)


def stop_if_not_aligned(records):
    if len(records) <= 1:
        return
    first_len = len(records[0].seq)
    for record in records[1:]:
        if len(record.seq) != first_len:
            txt = 'Sequence lengths were not identical. Please make sure input sequences are correctly aligned. Exiting.\n'
            raise Exception(txt)


def stop_if_not_dna(records, label='--seq_file'):
    invalid_ids = list()
    invalid_chars = set()
    for record in records:
        seq_str = str(record.seq)
        is_invalid = False
        for ch in seq_str:
            if ch not in DNA_ALLOWED_CHARS:
                invalid_chars.add(ch)
                is_invalid = True
        if is_invalid:
            invalid_ids.append(record.id)
    if len(invalid_ids) == 0:
        return
    max_show = 10
    shown = ','.join(invalid_ids[:max_show])
    if len(invalid_ids) > max_show:
        shown += ',...'
    chars = ''.join(sorted(invalid_chars))
    txt = (
        'Invalid non-DNA character(s) were detected in {} ({}) [chars: {}]. '
        'DNA-only input is required (use T instead of U). Exiting.\n'
    )
    raise Exception(txt.format(label, shown, chars))


def stop_if_not_protein(records, label='--seq_file'):
    invalid_ids = list()
    invalid_chars = set()
    for record in records:
        seq_str = str(record.seq)
        is_invalid = False
        for ch in seq_str:
            if ch not in PROTEIN_ALLOWED_CHARS:
                invalid_chars.add(ch)
                is_invalid = True
        if is_invalid:
            invalid_ids.append(record.id)
    if len(invalid_ids) == 0:
        return
    max_show = 10
    shown = ','.join(invalid_ids[:max_show])
    if len(invalid_ids) > max_show:
        shown += ',...'
    chars = ''.join(sorted(invalid_chars))
    txt = (
        'Invalid non-protein character(s) were detected in {} ({}) [chars: {}]. '
        'Protein-only input is required. Exiting.\n'
    )
    raise Exception(txt.format(label, shown, chars))


def stop_if_not_seqtype(records, seqtype='auto', label='--seq_file'):
    seqtype_value = str(seqtype).lower()
    if seqtype_value == 'dna':
        stop_if_not_dna(records=records, label=label)
        return
    if seqtype_value == 'protein':
        stop_if_not_protein(records=records, label=label)
        return
    if seqtype_value == 'auto':
        allowed_chars = DNA_ALLOWED_CHARS | PROTEIN_ALLOWED_CHARS
        invalid_ids = list()
        invalid_chars = set()
        for record in records:
            seq_str = str(record.seq)
            is_invalid = False
            for ch in seq_str:
                if ch not in allowed_chars:
                    invalid_chars.add(ch)
                    is_invalid = True
            if is_invalid:
                invalid_ids.append(record.id)
        if len(invalid_ids) == 0:
            return
        max_show = 10
        shown = ','.join(invalid_ids[:max_show])
        if len(invalid_ids) > max_show:
            shown += ',...'
        chars = ''.join(sorted(invalid_chars))
        txt = (
            'Invalid sequence character(s) were detected in {} ({}) [chars: {}]. '
            'DNA or protein input is required when --seq_type=auto. Exiting.\n'
        )
        raise Exception(txt.format(label, shown, chars))

    txt = 'Invalid --seq_type: {}. Choose from dna, protein, auto. Exiting.\n'
    raise Exception(txt.format(seqtype))


def stop_if_invalid_codontable(codontable, label='--codon_table'):
    try:
        Bio.Data.CodonTable.unambiguous_dna_by_id[int(codontable)]
        return
    except (KeyError, TypeError, ValueError):
        pass
    try:
        Bio.Data.CodonTable.unambiguous_dna_by_name[str(codontable)]
        return
    except KeyError:
        txt = 'Invalid {}: {}. Exiting.\n'
        raise Exception(txt.format(label, codontable))


def translate_records(records, codontable):
    from cdskit.translate import translate_sequence_string
    return [
        Bio.SeqRecord.SeqRecord(
            seq=Bio.Seq.Seq(
                translate_sequence_string(
                    seq_str=str(record.seq),
                    codontable=codontable,
                    to_stop=False,
                )
            ),
            id=record.id,
        )
        for record in records
    ]


def records2array(records):
    return np.array([list(record.seq) for record in records])


def read_item_per_line_file(file):
    with open(file, 'r') as f:
        return [line.strip() for line in f if line.strip() != '']


def get_seqname(record, seqnamefmt):
    name_items = seqnamefmt.split('_')
    seqname = ''
    for name_item in name_items:
        if name_item not in record.annotations:
            available_items = ', '.join(list(record.annotations.keys()))
            txt = 'Invalid --seq_name_format element ({}) in {}. Available elements: {}'
            raise Exception(txt.format(name_item, record.id, available_items))

        try:
            new_name = record.annotations[name_item]
            if isinstance(new_name, list):
                new_name = new_name[0]
            seqname += '_' + new_name
        except Exception:
            available_items = ', '.join(list(record.annotations.keys()))
            txt = 'Invalid --seq_name_format element ({}) in {}. Available elements: {}'
            raise Exception(txt.format(name_item, record.id, available_items))
    seqname = re.sub('^_', '', seqname)
    seqname = re.sub(' ', '_', seqname)
    return seqname


def replace_seq2cds(record):
    for feature in record.features:
        if feature.type == "CDS":
            record.seq = feature.location.extract(record).seq
            return record
    txt = 'Removed from output. No CDS found in: {}\n'
    sys.stderr.write(txt.format(record.id))
    return None


def read_gff(gff_file):
    header_lines = []
    data_lines = []
    with open(gff_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line[0] == '#':
                header_lines.append(line)
            else:
                data_lines.append(line)
    if len(data_lines) == 0:
        data = np.array([], dtype=GFF_DTYPE)
    else:
        data = np.genfromtxt(
            io.StringIO('\n'.join(data_lines)),
            delimiter='\t',
            dtype=None,
            names=GFF_COLUMNS,
            encoding='utf-8',
            autostrip=True,
            comments=None,
        )
        # Handle single record case: np.genfromtxt returns 0-d array for single line
        if data.ndim == 0:
            data = np.array([data], dtype=data.dtype)
    sys.stderr.write('Number of input GFF header lines: {:,}\n'.format(len(header_lines)))
    sys.stderr.write('Number of input GFF records: {:,}\n'.format(len(data)))
    sys.stderr.write('Number of input GFF unique seqids: {:,}\n'.format(len(np.unique(data['seqid']))))
    return {'header': header_lines, 'data': data}


def write_gff(gff, outfile):
    sys.stderr.write('Number of output GFF header lines: {:,}\n'.format(len(gff['header'])))
    sys.stderr.write('Number of output GFF records: {:,}\n'.format(len(gff['data'])))
    sys.stderr.write('Number of output GFF unique seqids: {:,}\n'.format(len(np.unique(gff['data']['seqid']))))
    with atomic_text_writer(outfile) as f:
        if gff['header']:
            f.write('\n'.join(gff['header']) + '\n')
        for row in gff['data']:
            f.write('\t'.join(map(str, row)) + '\n')


def coordinates2ranges(gff_coordinates):
    ranges = []
    if len(gff_coordinates) == 0:
        return ranges
    start = gff_coordinates[0]
    end = gff_coordinates[0]
    for i in range(1, len(gff_coordinates)):
        if gff_coordinates[i] == end + 1:
            end = gff_coordinates[i]
        else:
            ranges.append((start, end))
            start = gff_coordinates[i]
            end = gff_coordinates[i]
    ranges.append((start, end))
    return ranges

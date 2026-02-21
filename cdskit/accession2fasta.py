from Bio import Entrez
from Bio import SeqIO

import sys
import time
from functools import partial

from cdskit.util import (
    get_seqname,
    parallel_map_ordered,
    read_item_per_line_file,
    replace_seq2cds,
    resolve_threads,
    stop_if_not_dna,
    write_seqs,
)


def accession_batch_ranges(num_accession, batch_size):
    for start in range(0, num_accession, batch_size):
        end = min(start + batch_size, num_accession)
        yield start, end


def find_missing_accessions(accessions, seq_records):
    missing_ids = []
    for accession in accessions:
        found = False
        for seq_record in seq_records:
            if accession_matches_record_id(accession=accession, record_id=seq_record.id):
                found = True
                break
        if not found:
            missing_ids.append(accession)
    return missing_ids


def accession_matches_record_id(accession, record_id):
    record_head = str(record_id).strip().split()[0]
    if record_head == accession:
        return True
    if record_head.startswith(accession + '.'):
        return True
    for token in record_head.split('|'):
        if token == accession:
            return True
        if token.startswith(accession + '.'):
            return True
    return False


def prepare_accession_record(record, seqnamefmt, extract_cds=False, list_seqname_keys=False):
    if list_seqname_keys:
        sys.stderr.write(str(record.annotations) + '\n')
    record.name = ''
    record.description = ''
    record.id = get_seqname(record, seqnamefmt=seqnamefmt)
    if extract_cds:
        return replace_seq2cds(record=record)
    return record


def accession2seq_record(accessions, database, batch_size=1000):
    num_accession = len(accessions)
    start_time = time.time()
    seq_records = []
    for start, end in accession_batch_ranges(num_accession, batch_size):
        sys.stderr.write('Retrieving accessions: {:,}-{:,}, {:,} [sec]\n'.format(start, end, int(time.time()-start_time)))
        handle = Entrez.efetch(db=database, id=accessions[start:end], rettype="gb", retmode="text", retmax=batch_size)
        try:
            seq_records += list(SeqIO.parse(handle, 'gb'))
        finally:
            close_fn = getattr(handle, 'close', None)
            if callable(close_fn):
                close_fn()
    sys.stderr.write('Number of input accessions: {:,}\n'.format(num_accession))
    sys.stderr.write('Number of retrieved records: {:,}\n'.format(len(seq_records)))
    if (num_accession!=len(seq_records)):
        missing_ids = find_missing_accessions(accessions, seq_records)
        sys.stderr.write('Accessions failed to retrieve: {}\n'.format(' '.join(missing_ids)))
    elapsed_time = int(time.time() - start_time)
    sys.stderr.write("Elapsed_time for sequence record retrieval: {:,} [sec]\n".format(elapsed_time))
    return seq_records

def accession2fasta_main(args):
    accession_file = getattr(args, 'accession_file', '')
    if accession_file in ('', None):
        txt = '--accession_file is required. Exiting.\n'
        raise Exception(txt)

    if args.email!='':
        Entrez.email = args.email
    threads = resolve_threads(getattr(args, 'threads', 1))
    if args.list_seqname_keys:
        # Keep deterministic key listing order in stderr.
        threads = 1
    accessions = read_item_per_line_file(file=accession_file)
    records = accession2seq_record(accessions, args.ncbi_database)
    stop_if_not_dna(records=records, label='retrieved records')
    worker = partial(
        prepare_accession_record,
        seqnamefmt=args.seqnamefmt,
        extract_cds=args.extract_cds,
        list_seqname_keys=args.list_seqname_keys,
    )
    records = parallel_map_ordered(items=records, worker=worker, threads=threads)
    records = [record for record in records if record is not None]
    write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)

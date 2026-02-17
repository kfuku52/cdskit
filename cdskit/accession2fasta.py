from Bio import Entrez
from Bio import SeqIO

import sys
import time

from cdskit.util import get_seqname, read_item_per_line_file, replace_seq2cds, write_seqs


def accession_batch_ranges(num_accession, batch_size):
    for start in range(0, num_accession, batch_size):
        end = min(start + batch_size, num_accession)
        yield start, end


def find_missing_accessions(accessions, seq_records):
    missing_ids = []
    for accession in accessions:
        found = False
        for seq_record in seq_records:
            if accession in seq_record.id:
                found = True
                break
        if not found:
            missing_ids.append(accession)
    return missing_ids


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
        seq_records += list(SeqIO.parse(handle, 'gb'))
    sys.stderr.write('Number of input accessions: {:,}\n'.format(num_accession))
    sys.stderr.write('Number of retrieved records: {:,}\n'.format(len(seq_records)))
    if (num_accession!=len(seq_records)):
        missing_ids = find_missing_accessions(accessions, seq_records)
        sys.stderr.write('Accessions failed to retrieve: {}\n'.format(' '.join(missing_ids)))
    elapsed_time = int(time.time() - start_time)
    sys.stderr.write("Elapsed_time for sequence record retrieval: {:,} [sec]\n".format(elapsed_time))
    return seq_records

def accession2fasta_main(args):
    if args.email!='':
        Entrez.email = args.email
    accessions = read_item_per_line_file(file=args.accession_file)
    records = accession2seq_record(accessions, args.ncbi_database)
    records = [
        prepare_accession_record(
            record=record,
            seqnamefmt=args.seqnamefmt,
            extract_cds=args.extract_cds,
            list_seqname_keys=args.list_seqname_keys,
        )
        for record in records
    ]
    records = [record for record in records if record is not None]
    write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)

from Bio import Entrez
from Bio import SeqIO
import numpy

import sys
import time

from cdskit.util import *

def accession2seq_record(accessions, database, quiet, batch_size=1000):
    num_accession = len(accessions)
    start_time = time.time()
    seq_records = list()
    for i in numpy.arange(numpy.ceil(num_accession//batch_size)+1):
        start = int(i*batch_size)
        end = int(((i+1)*batch_size)-1) if num_accession >= int(((i+1)*batch_size)-1) else num_accession
        if not quiet:
            sys.stderr.write('Retrieving accessions: {:,}-{:,}, {:,} [sec]\n'.format(start, end, int(time.time()-start_time)))
        handle = Bio.Entrez.efetch(db=database, id=accessions[start:end], rettype="gb", retmode="text", retmax=batch_size)
        seq_records += list(Bio.SeqIO.parse(handle, 'gb'))
    if not quiet:
        sys.stderr.write('Number of input accessions: {:,}\n'.format(num_accession))
        sys.stderr.write('Number of retrieved records: {:,}\n'.format(len(seq_records)))
    if (num_accession!=len(seq_records)):
        missing_ids = []
        for accession in accessions:
            flag_found = False
            for seq_record in seq_records:
                if (accession in seq_record.id):
                    flag_found = True
                    break
            if not flag_found:
                missing_ids.append(accession)
        if not quiet:
            sys.stderr.write('Accessions failed to retrieve: {}\n'.format(' '.join(missing_ids)))
    elapsed_time = int(time.time() - start_time)
    if not quiet:
        sys.stderr.write("Elapsed_time for sequence record retrieval: {:,} [sec]\n".format(elapsed_time))
    return seq_records

def accession2fasta_main(args):
    if args.email!='':
        Bio.Entrez.email = args.email
    accessions = read_item_per_line_file(file=args.accession_file)
    records = accession2seq_record(accessions, args.ncbi_database, args.quiet)
    for i in range(len(records)):
        records[i].name = ''
        records[i].description = ''
        records[i].id = get_seqname(records[i], seqnamefmt=args.seqnamefmt)
        if args.extract_cds:
            records[i] = replace_seq2cds(record=records[i])
    write_seqs(records, args)
    if not args.quiet:
        sys.stderr.write('cdskit accession2fasta: end\n')
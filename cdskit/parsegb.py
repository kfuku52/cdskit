import os
import re

from cdskit.util import *

def parsegb_main(args):
    if not args.quiet:
        sys.stderr.write('cdskit parsegb: start\n')
    if (args.verbose)&(not args.quiet):
        sys.stderr.write(str(args)+'\n')
    name_items = args.seqnamefmt.split('_')
    records = read_seqs(seqfile=args.seqfile, seqformat='genbank', quiet=args.quiet)
    for i in range(len(records)):
        record = records[i]
        seqname = ''
        for name_item in name_items:
            try:
                new_name = record.annotations[name_item]
                if type(new_name) is list:
                    new_name = new_name[0]
                seqname += '_'+new_name
            except:
                available_items = ', '.join(list(record.annotations.keys()))
                txt = 'Invalid --seqnamefmt element ({}) in {}. Available elements: {}'
                raise Exception(txt.format(name_item, record.id, available_items))
        seqname = re.sub('^_', '', seqname)
        seqname = re.sub(' ', '_', seqname)
        record.name = ''
        record.description = ''
        record.id = seqname
        if args.extract_cds:
            flag_no_cds = True
            for feature in record.features:
                if feature.type=="CDS":
                    seq = feature.location.extract(record).seq
                    record.seq = seq
                    flag_no_cds = False
                    break
            if flag_no_cds:
                txt = 'Removed from output. No CDS found in: {}\n'
                os.stderr.write(txt.format(record.id))
    write_seqs(records, args)
    if not args.quiet:
        sys.stderr.write('cdskit parsegb: end\n')
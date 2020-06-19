import numpy
from cdskit.util import *

def hammer_main(args):
    if not args.quiet:
        sys.stderr.write('cdskit hammer: start\n')
    if (args.verbose)&(not args.quiet):
        sys.stderr.write(str(args)+'\n')
    records = read_seqs(args)
    max_len = max([ len(r.seq) for r in records ])//3
    missing_site = numpy.zeros(shape=[max_len,], dtype=int)
    for record in records:
        aaseq = record.seq.translate(table=args.codontable, to_stop=False, gap="-")
        for i in numpy.arange(len(aaseq)):
            if aaseq[i] in ['-','?','X','*']:
                missing_site[i] += 1
        if len(aaseq)<missing_site.shape[0]:
            missing_site[len(aaseq):] += 1
    non_missing_site = len(records) - missing_site
    non_missing_idx = numpy.argwhere(non_missing_site>=args.nail)
    non_missing_idx = numpy.reshape(non_missing_idx, newshape=[non_missing_idx.shape[0],])
    num_removed_site = max_len - non_missing_idx.shape[0]
    if not args.quiet:
        sys.stderr.write('{:,} out of {:,} codon sites will be removed.\n'.format(num_removed_site, max_len))
    for record in records:
        seq = str(record.seq)
        new_seq = ''.join([ seq[nmi*3:nmi*3+3] for nmi in non_missing_idx ])
        record.seq = Bio.Seq.Seq(new_seq)
    write_seqs(records, args)
    if not args.quiet:
        sys.stderr.write('cdskit hammer: end\n')
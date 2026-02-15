#!/usr/bin/env python

import Bio.Seq
from cdskit.util import *

def mask_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_multiple_of_three(records)
    mask_triplet = args.maskchar * 3
    mask_ambiguous = (args.ambiguouscodon == 'yes')
    mask_stop = (args.stopcodon == 'yes')
    for record in records:
        nucseq = str(record.seq)
        codons = [nucseq[i:i+3] for i in range(0, len(nucseq), 3)]

        # Mask partial-gap codons (e.g. A-G) but keep full gaps (---).
        flag1 = False
        for i, codon in enumerate(codons):
            if ('-' in codon) and (codon != '---'):
                codons[i] = mask_triplet
                flag1 = True

        if not mask_ambiguous and not mask_stop:
            if flag1:
                record.seq = Bio.Seq.Seq(''.join(codons))
            continue

        if flag1:
            translated_source = ''.join(codons)
            aaseq = str(Bio.Seq.Seq(translated_source).translate(table=args.codontable, to_stop=False, gap="-"))
        else:
            aaseq = str(record.seq.translate(table=args.codontable, to_stop=False, gap="-"))
        flag2 = False
        if mask_ambiguous and mask_stop:
            for i, aa in enumerate(aaseq):
                if aa in ['X', '*']:
                    codons[i] = mask_triplet
                    flag2 = True
        elif mask_ambiguous:
            for i, aa in enumerate(aaseq):
                if aa == 'X':
                    codons[i] = mask_triplet
                    flag2 = True
        else:
            for i, aa in enumerate(aaseq):
                if aa == '*':
                    codons[i] = mask_triplet
                    flag2 = True

        if flag1 or flag2:
            record.seq = Bio.Seq.Seq(''.join(codons))
    write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)

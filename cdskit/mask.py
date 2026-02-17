#!/usr/bin/env python

import Bio.Seq

from cdskit.util import read_seqs, stop_if_not_multiple_of_three, write_seqs


def codon_chunks(nucseq):
    return [nucseq[i:i + 3] for i in range(0, len(nucseq), 3)]


def mask_partial_gap_codons(codons, mask_triplet):
    changed = False
    for i, codon in enumerate(codons):
        if ('-' in codon) and (codon != '---'):
            codons[i] = mask_triplet
            changed = True
    return changed


def should_mask_amino_acid(aa, mask_ambiguous, mask_stop):
    if mask_ambiguous and mask_stop:
        return aa in ['X', '*']
    if mask_ambiguous:
        return aa == 'X'
    if mask_stop:
        return aa == '*'
    return False


def mask_translated_codons(codons, aaseq, mask_triplet, mask_ambiguous, mask_stop):
    changed = False
    for i, aa in enumerate(aaseq):
        if should_mask_amino_acid(aa, mask_ambiguous, mask_stop):
            codons[i] = mask_triplet
            changed = True
    return changed


def mask_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_multiple_of_three(records)
    mask_triplet = args.maskchar * 3
    mask_ambiguous = (args.ambiguouscodon == 'yes')
    mask_stop = (args.stopcodon == 'yes')
    for record in records:
        nucseq = str(record.seq)
        codons = codon_chunks(nucseq)

        # Mask partial-gap codons (e.g. A-G) but keep full gaps (---).
        flag1 = mask_partial_gap_codons(codons, mask_triplet)

        if not mask_ambiguous and not mask_stop:
            if flag1:
                record.seq = Bio.Seq.Seq(''.join(codons))
            continue

        if flag1:
            translated_source = ''.join(codons)
            aaseq = str(Bio.Seq.Seq(translated_source).translate(table=args.codontable, to_stop=False, gap="-"))
        else:
            aaseq = str(record.seq.translate(table=args.codontable, to_stop=False, gap="-"))
        flag2 = mask_translated_codons(
            codons=codons,
            aaseq=aaseq,
            mask_triplet=mask_triplet,
            mask_ambiguous=mask_ambiguous,
            mask_stop=mask_stop,
        )

        if flag1 or flag2:
            record.seq = Bio.Seq.Seq(''.join(codons))
    write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)

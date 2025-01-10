#!/usr/bin/env python

import Bio.Seq
import Bio.SeqIO
import numpy

from cdskit.util import *

import sys

class padseqs:
    def __init__(self, original_seq, codon_table='Standard', padchar='N'):
        self.new_seqs = list()
        self.num_stops = list()
        self.headn = list()
        self.tailn = list()
        self.original_seq = str(original_seq)
        self.codon_table = codon_table
        self.padchar = padchar
    def add(self, headn=0, tailn=0):
        new_seq = Bio.Seq.Seq((self.padchar*headn)+self.original_seq+(self.padchar*tailn))
        self.new_seqs.append(new_seq)
        self.num_stops.append(str(new_seq[:-3].translate(self.codon_table)).count('*'))
        self.headn.append(headn)
        self.tailn.append(tailn)
    def get_minimum_num_stop(self):
        min_index = numpy.argmin(self.num_stops)
        out = {
            'new_seq':self.new_seqs[min_index],
            'num_stop':self.num_stops[min_index],
            'headn':self.headn[min_index],
            'tailn':self.tailn[min_index],
        }
        return out

def pad_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    is_no_stop = list()
    seqnum_padded = 0
    for i in range(len(records)):
        clean_seq = str(records[i].seq).replace('X', 'N')
        seqlen = len(clean_seq)
        if seqlen % 3 == 0:
            adjlen = seqlen
            tailpad_seq = Bio.Seq.Seq(clean_seq)
        else:
            adjlen = ((seqlen//3)+1)*3
            tailpad_seq = Bio.Seq.Seq(str(clean_seq).ljust(adjlen, args.padchar))
        num_stop_input = str(tailpad_seq[:-3].translate(args.codontable)).count('*')
        if ((num_stop_input)|(seqlen % 3)):
            num_missing = adjlen - seqlen
            seqs = padseqs(original_seq=clean_seq, codon_table=args.codontable, padchar=args.padchar)
            if num_stop_input:
                if (num_missing==0)|(num_missing==3):
                    seqs.add(headn=0, tailn=0)
                    seqs.add(headn=1, tailn=2)
                    seqs.add(headn=2, tailn=1)
                elif num_missing==1:
                    seqs.add(headn=0, tailn=1)
                    seqs.add(headn=1, tailn=0)
                    seqs.add(headn=2, tailn=2)
                elif num_missing==2:
                    seqs.add(headn=0, tailn=2)
                    seqs.add(headn=2, tailn=0)
                    seqs.add(headn=1, tailn=1)
            if ((~num_stop_input)&(seqlen % 3)):
                seqs.add(headn=0, tailn=num_missing)
            best_padseq = seqs.get_minimum_num_stop()
            records[i].seq = best_padseq['new_seq']
            if best_padseq['num_stop']==0:
                is_no_stop.append(True)
            else:
                is_no_stop.append(False)
            txt = f'{records[i].name}, original_seqlen={seqlen}, head_padding={best_padseq["headn"]}, tail_padding={best_padseq["tailn"]}, '
            txt += f'original_num_stop={num_stop_input}, new_num_stop={best_padseq["num_stop"]}\n'
            sys.stderr.write(txt)
            if not ((best_padseq['headn']==0)&(best_padseq['tailn']==0)):
                seqnum_padded += 1
        else:
            is_no_stop.append(True)
    if args.nopseudo:
        records = [ records[i] for i in range(len(records)) if is_no_stop[i] ]
    sys.stderr.write('Number of padded sequences: {:,} / {:,}\n'.format(seqnum_padded, len(records)))
    write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)

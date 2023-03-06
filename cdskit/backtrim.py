import numpy
from cdskit.util import *

def check_same_seq_num(cdn_records, pep_records):
    err_txt = 'The numbers of seqs did not match: seqfile={} and trimmed_aa_aln={}'.format(len(cdn_records), len(pep_records))
    assert len(cdn_records)==len(pep_records), err_txt

def backtrim_main(args):
    cdn_records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    pep_records = read_seqs(seqfile=args.trimmed_aa_aln, seqformat=args.inseqformat)
    check_same_seq_num(cdn_records, pep_records)
    check_aligned(records=cdn_records)
    check_aligned(records=pep_records)
    # check_same_order()
    tcdn_records = translate_records(records=cdn_records, codontable=args.codontable)
    tcdn_array = records2array(records=tcdn_records)
    pep_array = records2array(records=pep_records)
    cdn_array = records2array(records=cdn_records)
    kept_aa_sites = list()
    remaining_tcdn_sites = numpy.arange(tcdn_array.shape[1])
    multiple_matches = set()
    for pi in numpy.arange(pep_array.shape[1]):
        same_sites = list()
        if remaining_tcdn_sites.shape[0]==0:
            break
        for ci in remaining_tcdn_sites:
            if numpy.array_equal(pep_array[:,pi], tcdn_array[:,ci]):
                same_sites.append(ci)
        if (len(same_sites)==1):
            kept_aa_sites.append(same_sites[0])
            remaining_tcdn_sites = remaining_tcdn_sites[remaining_tcdn_sites!=same_sites[0]]
        elif (len(same_sites)>1):
            multiple_matches = multiple_matches | set(same_sites)
            txt = 'The trimmed protein site {} has multiple matches to codon sites({}). Reporting the first match. '
            txt = txt.format(pi, ','.join([ str(ss) for ss in same_sites ]))
            txt += 'Site pattern: {}\n'.format(''.join(pep_array[:,pi]))
            sys.stderr.write(txt)
            kept_aa_sites.append(same_sites[0])
            remaining_tcdn_sites = remaining_tcdn_sites[remaining_tcdn_sites!=same_sites[0]]
        elif (len(same_sites)==0):
            txt = 'The codon site {} could not be matched to trimmed protein sites. '
            txt += 'The site may contain only missing, ambiguous, and/or stop codons. '
            txt += 'The site will be excluded from the output.\n'
            sys.stderr.write(txt.format(ci))
    num_trimmed_multiple_hit_sites = len(multiple_matches-set(kept_aa_sites))
    txt = '{} codon sites matched to {} protein sites. '
    txt = txt+'Trimmed {} codon sites that matched to multiple protein sites.\n'
    txt = txt.format(len(kept_aa_sites), pep_array.shape[1], num_trimmed_multiple_hit_sites)
    sys.stderr.write(txt)
    kept_aa_sites = numpy.array(kept_aa_sites)
    codon_pos1 = kept_aa_sites * 3 + 0
    codon_pos2 = kept_aa_sites * 3 + 1
    codon_pos3 = kept_aa_sites * 3 + 2
    kept_cdn_sites = numpy.sort(numpy.concatenate([codon_pos1, codon_pos2, codon_pos3]))
    trimmed_cdn_records = list()
    for i in numpy.arange(len(cdn_records)):
        trimmed_seq = ''.join(cdn_array[i,kept_cdn_sites])
        trimmed_record = Bio.SeqRecord.SeqRecord(seq=Bio.Seq.Seq(trimmed_seq),
                                                 id=cdn_records[i].id, name='', description='')
        trimmed_cdn_records.append(trimmed_record)
    txt = 'Number of aligned nucleotide sites in untrimmed codon sequences: {}\n'
    sys.stderr.write(txt.format(len(cdn_records[0].seq)))
    txt = 'Number of aligned nucleotide sites in trimmed codon sequences: {}\n'
    sys.stderr.write(txt.format(len(trimmed_cdn_records[0].seq)))
    write_seqs(records=trimmed_cdn_records, args=args)

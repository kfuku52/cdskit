import os,json,pathlib,random,subprocess
import pandas as pd
import numpy as np
from Bio.Data import CodonTable
p=pathlib.Path('/audit/localize-compatibility');p.mkdir(exist_ok=True)
rng=random.Random(174);alphabet='ACDEFGHIKLMNPQRSTVWY'
sequences=[]
for i in range(32):
 seq='M'+''.join(rng.choice(alphabet if i%3 else 'ALSVT') for _ in range(rng.randrange(30,360)))
 if i%4==0:seq=seq[:-3]+'SKL'
 if i%4==1:seq='M'+'LAS'*7+seq[22:]
 sequences.append(seq)
protein=p/'proteins.fa';protein.write_text(''.join(f'>seq{i}\n{s}\n' for i,s in enumerate(sequences)))
codons={aa:codon for codon,aa in CodonTable.unambiguous_dna_by_id[1].forward_table.items()}
dna=p/'cds.fa';dna.write_text(''.join(f'>seq{i}\n'+''.join(codons[a] for a in s)+'\n' for i,s in enumerate(sequences)))
summary=[]
for model in ('targeting5-v1','targeting5-perox-deeploc21-et-v1'):
 for group in ('plant','non_plant','unknown'):
  for seqtype,seqfile in (('protein',protein),('dna',dna)):
   tables={}
   for version,py_path in [('1.5.2','/tmp/sklearn152:/cdskit'),('1.9.0','/cdskit')]:
    prefix=p/f'{model}-{group}-{seqtype}-{version}';out=prefix.with_suffix(prefix.suffix+'.tsv')
    env=dict(os.environ,PYTHONPATH=py_path,CDSKIT_MODEL_DIR='/audit/fix-model-cache')
    cmd=['cdskit','localize','--seq_file',str(seqfile),'--seq_type',seqtype,'--model',model,'--model_download','no','--organism_group',group,'--threads','1','--report',str(out)]
    with open(str(prefix)+'.log','w') as log:
     subprocess.run(cmd,env=env,stdout=log,stderr=log,check=True,timeout=90,cwd='/tmp')
    tables[version]=pd.read_csv(out,sep='\t')
   a,b=tables.values();pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=0,atol=1e-14)
   numeric=a.select_dtypes(include='number').columns
   diff=float(np.nanmax(np.abs(a[numeric].to_numpy()-b[numeric].to_numpy())))
   summary.append(dict(model=model,group=group,seq_type=seqtype,rows=len(a),numeric_columns=len(numeric),max_abs_diff=diff))
   print(summary[-1],flush=True)
(p/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')

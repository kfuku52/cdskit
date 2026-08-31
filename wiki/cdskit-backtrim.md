# cdskit backtrim

`cdskit backtrim` back-translates a trimmed protein alignment.

![Backtrimming workflow](https://raw.githubusercontent.com/kfuku52/cdskit/master/img/backtrim.svg)

## Example
In this example, CDSKIT is combined with [SeqKit](https://github.com/shenwei356/seqkit) and [ClipKIT](https://github.com/JLSteenwyk/ClipKIT) to generate a trimmed codon alignment.

### Command
```
seqkit translate untrimmed_codon.fasta > untrimmed_aa.fasta

clipkit untrimmed_aa.fasta -o trimmed_aa.fasta

cdskit backtrim --seq_file untrimmed_codon.fasta --trimmed_aa_aln trimmed_aa.fasta --out_file trimmed_codon.fasta
```

### untrimmed_codon.fasta
```
>Drosophila_melanogaster_AE014298.5_cds_AAF48408.2_3478
ATGAACCCAGCCGCTCAACTGCTGCGCATGCGCAGCGCT
>Drosophila_melanogaster_AE014298.5_cds_ADV37672.1_3479
ATGAACCCAGCCGCTCAACTGCTGCGCATGCGCAGCGCT
>Drosophila_melanogaster_AE014298.5_cds_AFH07387.1_3480
ATGAACCCAGCCGCTCAACTGCTGCGCATGCGCAGCGCT
>Drosophila_melanogaster_AE014298.5_cds_AHN59727.1_3481
ATGAACCCAGCCGCTCAACTGCTGCGCATGCGCAGCGCT
>Drosophila_melanogaster_AE014134.6_cds_AAF52246.1_6873
---------------------------ATGAACAGCAAG
>Drosophila_melanogaster_AE013599.5_cds_AAF58513.1_13350
---------------------------ATGGCCATGATA
>Drosophila_melanogaster_AE013599.5_cds_AAF46628.2_15780
---------------------------ATGAGCTGTGAG
>Drosophila_melanogaster_AE013599.5_cds_AAF46629.1_15781
---------------------------ATGGCGTCCACC
>Drosophila_melanogaster_AE013599.5_cds_AAF47206.1_16961
---------------------------ATGCCGACAAAG
>Drosophila_melanogaster_AE014296.5_cds_AAF50738.3_18793
---------------------------ATGGGTGAATTG
>Drosophila_melanogaster_AE014296.5_cds_AAF50737.2_18794
---------------------------ATGGCTGAAATG
>Drosophila_melanogaster_AE014296.5_cds_AGB94148.1_18795
---------------------------ATGGCTGAAATG
>Drosophila_melanogaster_AE014297.3_cds_AAF54758.1_25318
---------------------------ATGTCCAAGTTA
>Drosophila_melanogaster_AE014297.3_cds_AAF55696.2_27167
---------------------------------------
>Drosophila_melanogaster_AE014297.3_cds_AAF55697.2_27168
---------------------------------------
>Drosophila_melanogaster_AE014297.3_cds_AAF55698.1_27169
---------------------------ATGTTGGACCTC
>Drosophila_melanogaster_AE014297.3_cds_AAF55699.1_27170
---------------------------------------
>Drosophila_melanogaster_AE014297.3_cds_AAF55700.2_27171
---------------------------ATGAATCGCTCG
>Drosophila_melanogaster_AE014297.3_cds_AAF56245.1_28374
------------------ATGACTTCAAAGCTACTGCCC
```

### untrimmed_aa.fasta
```
>Drosophila_melanogaster_AE014298.5_cds_AAF48408.2_3478
MNPAAQLLRMRSA
>Drosophila_melanogaster_AE014298.5_cds_ADV37672.1_3479
MNPAAQLLRMRSA
>Drosophila_melanogaster_AE014298.5_cds_AFH07387.1_3480
MNPAAQLLRMRSA
>Drosophila_melanogaster_AE014298.5_cds_AHN59727.1_3481
MNPAAQLLRMRSA
>Drosophila_melanogaster_AE014134.6_cds_AAF52246.1_6873
---------MNSK
>Drosophila_melanogaster_AE013599.5_cds_AAF58513.1_13350
---------MAMI
>Drosophila_melanogaster_AE013599.5_cds_AAF46628.2_15780
---------MSCE
>Drosophila_melanogaster_AE013599.5_cds_AAF46629.1_15781
---------MAST
>Drosophila_melanogaster_AE013599.5_cds_AAF47206.1_16961
---------MPTK
>Drosophila_melanogaster_AE014296.5_cds_AAF50738.3_18793
---------MGEL
>Drosophila_melanogaster_AE014296.5_cds_AAF50737.2_18794
---------MAEM
>Drosophila_melanogaster_AE014296.5_cds_AGB94148.1_18795
---------MAEM
>Drosophila_melanogaster_AE014297.3_cds_AAF54758.1_25318
---------MSKL
>Drosophila_melanogaster_AE014297.3_cds_AAF55696.2_27167
-------------
>Drosophila_melanogaster_AE014297.3_cds_AAF55697.2_27168
-------------
>Drosophila_melanogaster_AE014297.3_cds_AAF55698.1_27169
---------MLDL
>Drosophila_melanogaster_AE014297.3_cds_AAF55699.1_27170
-------------
>Drosophila_melanogaster_AE014297.3_cds_AAF55700.2_27171
---------MNRS
>Drosophila_melanogaster_AE014297.3_cds_AAF56245.1_28374
------MTSKLLP
```

### trimmed_aa.fasta
```
>Drosophila_melanogaster_AE014298.5_cds_AAF48408.2_3478
LLRMRSA
>Drosophila_melanogaster_AE014298.5_cds_ADV37672.1_3479
LLRMRSA
>Drosophila_melanogaster_AE014298.5_cds_AFH07387.1_3480
LLRMRSA
>Drosophila_melanogaster_AE014298.5_cds_AHN59727.1_3481
LLRMRSA
>Drosophila_melanogaster_AE014134.6_cds_AAF52246.1_6873
---MNSK
>Drosophila_melanogaster_AE013599.5_cds_AAF58513.1_13350
---MAMI
>Drosophila_melanogaster_AE013599.5_cds_AAF46628.2_15780
---MSCE
>Drosophila_melanogaster_AE013599.5_cds_AAF46629.1_15781
---MAST
>Drosophila_melanogaster_AE013599.5_cds_AAF47206.1_16961
---MPTK
>Drosophila_melanogaster_AE014296.5_cds_AAF50738.3_18793
---MGEL
>Drosophila_melanogaster_AE014296.5_cds_AAF50737.2_18794
---MAEM
>Drosophila_melanogaster_AE014296.5_cds_AGB94148.1_18795
---MAEM
>Drosophila_melanogaster_AE014297.3_cds_AAF54758.1_25318
---MSKL
>Drosophila_melanogaster_AE014297.3_cds_AAF55696.2_27167
-------
>Drosophila_melanogaster_AE014297.3_cds_AAF55697.2_27168
-------
>Drosophila_melanogaster_AE014297.3_cds_AAF55698.1_27169
---MLDL
>Drosophila_melanogaster_AE014297.3_cds_AAF55699.1_27170
-------
>Drosophila_melanogaster_AE014297.3_cds_AAF55700.2_27171
---MNRS
>Drosophila_melanogaster_AE014297.3_cds_AAF56245.1_28374
MTSKLLP
```

### trimmed_codon.fasta
```
>Drosophila_melanogaster_AE014298.5_cds_AAF48408.2_3478
CTGCTGCGCATGCGCAGCGCT
>Drosophila_melanogaster_AE014298.5_cds_ADV37672.1_3479
CTGCTGCGCATGCGCAGCGCT
>Drosophila_melanogaster_AE014298.5_cds_AFH07387.1_3480
CTGCTGCGCATGCGCAGCGCT
>Drosophila_melanogaster_AE014298.5_cds_AHN59727.1_3481
CTGCTGCGCATGCGCAGCGCT
>Drosophila_melanogaster_AE014134.6_cds_AAF52246.1_6873
---------ATGAACAGCAAG
>Drosophila_melanogaster_AE013599.5_cds_AAF58513.1_13350
---------ATGGCCATGATA
>Drosophila_melanogaster_AE013599.5_cds_AAF46628.2_15780
---------ATGAGCTGTGAG
>Drosophila_melanogaster_AE013599.5_cds_AAF46629.1_15781
---------ATGGCGTCCACC
>Drosophila_melanogaster_AE013599.5_cds_AAF47206.1_16961
---------ATGCCGACAAAG
>Drosophila_melanogaster_AE014296.5_cds_AAF50738.3_18793
---------ATGGGTGAATTG
>Drosophila_melanogaster_AE014296.5_cds_AAF50737.2_18794
---------ATGGCTGAAATG
>Drosophila_melanogaster_AE014296.5_cds_AGB94148.1_18795
---------ATGGCTGAAATG
>Drosophila_melanogaster_AE014297.3_cds_AAF54758.1_25318
---------ATGTCCAAGTTA
>Drosophila_melanogaster_AE014297.3_cds_AAF55696.2_27167
---------------------
>Drosophila_melanogaster_AE014297.3_cds_AAF55697.2_27168
---------------------
>Drosophila_melanogaster_AE014297.3_cds_AAF55698.1_27169
---------ATGTTGGACCTC
>Drosophila_melanogaster_AE014297.3_cds_AAF55699.1_27170
---------------------
>Drosophila_melanogaster_AE014297.3_cds_AAF55700.2_27171
---------ATGAATCGCTCG
>Drosophila_melanogaster_AE014297.3_cds_AAF56245.1_28374
ATGACTTCAAAGCTACTGCCC
```

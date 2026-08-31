# cdskit split

`cdskit split` writes the first, second, and third codon positions of an
aligned CDS to three separate sequence files.

## Example

### Command
```
cdskit split --seq_file input.fasta --prefix output
```

### input.fasta
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

### output_1st_codon_positions.fasta
```
>Drosophila_melanogaster_AE014298.5_cds_AAF48408.2_3478
AACGGCCCCACAG
>Drosophila_melanogaster_AE014298.5_cds_ADV37672.1_3479
AACGGCCCCACAG
>Drosophila_melanogaster_AE014298.5_cds_AFH07387.1_3480
AACGGCCCCACAG
>Drosophila_melanogaster_AE014298.5_cds_AHN59727.1_3481
AACGGCCCCACAG
>Drosophila_melanogaster_AE014134.6_cds_AAF52246.1_6873
---------AAAA
>Drosophila_melanogaster_AE013599.5_cds_AAF58513.1_13350
---------AGAA
>Drosophila_melanogaster_AE013599.5_cds_AAF46628.2_15780
---------AATG
>Drosophila_melanogaster_AE013599.5_cds_AAF46629.1_15781
---------AGTA
>Drosophila_melanogaster_AE013599.5_cds_AAF47206.1_16961
---------ACAA
>Drosophila_melanogaster_AE014296.5_cds_AAF50738.3_18793
---------AGGT
>Drosophila_melanogaster_AE014296.5_cds_AAF50737.2_18794
---------AGGA
>Drosophila_melanogaster_AE014296.5_cds_AGB94148.1_18795
---------AGGA
>Drosophila_melanogaster_AE014297.3_cds_AAF54758.1_25318
---------ATAT
>Drosophila_melanogaster_AE014297.3_cds_AAF55696.2_27167
-------------
>Drosophila_melanogaster_AE014297.3_cds_AAF55697.2_27168
-------------
>Drosophila_melanogaster_AE014297.3_cds_AAF55698.1_27169
---------ATGC
>Drosophila_melanogaster_AE014297.3_cds_AAF55699.1_27170
-------------
>Drosophila_melanogaster_AE014297.3_cds_AAF55700.2_27171
---------AACT
>Drosophila_melanogaster_AE014297.3_cds_AAF56245.1_28374
------AATACCC
```

### output_2nd_codon_positions.fasta
```
>Drosophila_melanogaster_AE014298.5_cds_AAF48408.2_3478
TACCCATTGTGGC
>Drosophila_melanogaster_AE014298.5_cds_ADV37672.1_3479
TACCCATTGTGGC
>Drosophila_melanogaster_AE014298.5_cds_AFH07387.1_3480
TACCCATTGTGGC
>Drosophila_melanogaster_AE014298.5_cds_AHN59727.1_3481
TACCCATTGTGGC
>Drosophila_melanogaster_AE014134.6_cds_AAF52246.1_6873
---------TAGA
>Drosophila_melanogaster_AE013599.5_cds_AAF58513.1_13350
---------TCTT
>Drosophila_melanogaster_AE013599.5_cds_AAF46628.2_15780
---------TGGA
>Drosophila_melanogaster_AE013599.5_cds_AAF46629.1_15781
---------TCCC
>Drosophila_melanogaster_AE013599.5_cds_AAF47206.1_16961
---------TCCA
>Drosophila_melanogaster_AE014296.5_cds_AAF50738.3_18793
---------TGAT
>Drosophila_melanogaster_AE014296.5_cds_AAF50737.2_18794
---------TCAT
>Drosophila_melanogaster_AE014296.5_cds_AGB94148.1_18795
---------TCAT
>Drosophila_melanogaster_AE014297.3_cds_AAF54758.1_25318
---------TCAT
>Drosophila_melanogaster_AE014297.3_cds_AAF55696.2_27167
-------------
>Drosophila_melanogaster_AE014297.3_cds_AAF55697.2_27168
-------------
>Drosophila_melanogaster_AE014297.3_cds_AAF55698.1_27169
---------TTAT
>Drosophila_melanogaster_AE014297.3_cds_AAF55699.1_27170
-------------
>Drosophila_melanogaster_AE014297.3_cds_AAF55700.2_27171
---------TAGC
>Drosophila_melanogaster_AE014297.3_cds_AAF56245.1_28374
------TCCATTC
```

### output_3rd_codon_positions.fasta
```
>Drosophila_melanogaster_AE014298.5_cds_AAF48408.2_3478
GCACTAGGCGCCT
>Drosophila_melanogaster_AE014298.5_cds_ADV37672.1_3479
GCACTAGGCGCCT
>Drosophila_melanogaster_AE014298.5_cds_AFH07387.1_3480
GCACTAGGCGCCT
>Drosophila_melanogaster_AE014298.5_cds_AHN59727.1_3481
GCACTAGGCGCCT
>Drosophila_melanogaster_AE014134.6_cds_AAF52246.1_6873
---------GCCG
>Drosophila_melanogaster_AE013599.5_cds_AAF58513.1_13350
---------GCGA
>Drosophila_melanogaster_AE013599.5_cds_AAF46628.2_15780
---------GCTG
>Drosophila_melanogaster_AE013599.5_cds_AAF46629.1_15781
---------GGCC
>Drosophila_melanogaster_AE013599.5_cds_AAF47206.1_16961
---------GGAG
>Drosophila_melanogaster_AE014296.5_cds_AAF50738.3_18793
---------GTAG
>Drosophila_melanogaster_AE014296.5_cds_AAF50737.2_18794
---------GTAG
>Drosophila_melanogaster_AE014296.5_cds_AGB94148.1_18795
---------GTAG
>Drosophila_melanogaster_AE014297.3_cds_AAF54758.1_25318
---------GCGA
>Drosophila_melanogaster_AE014297.3_cds_AAF55696.2_27167
-------------
>Drosophila_melanogaster_AE014297.3_cds_AAF55697.2_27168
-------------
>Drosophila_melanogaster_AE014297.3_cds_AAF55698.1_27169
---------GGCC
>Drosophila_melanogaster_AE014297.3_cds_AAF55699.1_27170
-------------
>Drosophila_melanogaster_AE014297.3_cds_AAF55700.2_27171
---------GTCG
>Drosophila_melanogaster_AE014297.3_cds_AAF56245.1_28374
------GTAGAGC
```

## Output naming

`--prefix PREFIX` produces
`PREFIX_1st_codon_positions.FORMAT`,
`PREFIX_2nd_codon_positions.FORMAT`, and
`PREFIX_3rd_codon_positions.FORMAT`. If no prefix is supplied, the input path
is used; standard input uses `stdin`. Input sequences must be nucleotide data
with lengths divisible by three.

# cdskit intersection

`cdskit intersection` drops non-overlapping sequence labels between two sequences files or between a sequence file and a gff file.

## Example 1: Sequence vs sequence intersection

### Command
```
cdskit intersection --seq_file input1.fasta --seq_file_2 input2.fasta --out_file output1.fasta --out_file_2 output2.fasta
```

### input1.fasta
```
>seq1
AAAAAAAAAA
>seq2
TTTTTTTTTT
>seq3
GGGGGGGGGG
>seq4
CCCCCCCCCC
```

### input2.fasta
```
>seq3
GGGGGGGGGGCCCCCCCCCC
>seq4
CCCCCCCCCCATATATATAT
>seq5
ATATATATATAGAGAGAGAG
>seq6
AGAGAGAGAGGGGGGGGGGG
```

### output1.fasta
```
>seq3
GGGGGGGGGG
>seq4
CCCCCCCCCC
```

### output2.fasta
```
>seq3
GGGGGGGGGGCCCCCCCCCC
>seq4
CCCCCCCCCCATATATATAT
```


## Example 2: Sequence vs gff intersection

### Command
```
cdskit intersection --seq_file input.fasta --in_gff input.gff --out_file output.fasta --out_gff output.gff
```

### input.fasta
```
>seq1
AAAAAAAAAA
>seq2
TTTTTTTTTT
>seq3
GGGGGGGGGG
>seq4
CCCCCCCCCC
```

### input.gff
```
seq1	cdskit	gene	1	10	.	+	.	ID=gene1;Name=gene1
seq1	cdskit	mRNA	1	10	.	+	.	ID=gene1-mRNA1;Parent=gene1;Name=gene1-mRNA1
seq1	cdskit	exon	1	3	.	+	.	ID=gene1-mRNA1:exon1;Parent=gene1-mRNA1
seq1	cdskit	exon	5	10	.	+	.	ID=gene1-mRNA1:exon2;Parent=gene1-mRNA1
seq1	cdskit	CDS	1	3	.	+	0	ID=gene1-mRNA1-cds1;Parent=gene1-mRNA1
seq1	cdskit	CDS	5	10	.	+	2	ID=gene1-mRNA1-cds2;Parent=gene1-mRNA1
seq2	cdskit	gene	1	10	.	+	.	ID=gene2;Name=gene2
seq2	cdskit	mRNA	1	10	.	+	.	ID=gene2-mRNA1;Parent=gene2;Name=gene2-mRNA1
seq2	cdskit	exon	1	3	.	+	.	ID=gene2-mRNA1:exon1;Parent=gene2-mRNA1
seq2	cdskit	exon	5	10	.	+	.	ID=gene2-mRNA1:exon2;Parent=gene2-mRNA1
seq2	cdskit	CDS	1	3	.	+	0	ID=gene2-mRNA1-cds1;Parent=gene2-mRNA1
seq2	cdskit	CDS	5	10	.	+	2	ID=gene2-mRNA1-cds2;Parent=gene2-mRNA1
seq5	cdskit	gene	1	10	.	+	.	ID=gene5;Name=gene5
seq5	cdskit	mRNA	1	10	.	+	.	ID=gene5-mRNA1;Parent=gene5;Name=gene5-mRNA1
seq5	cdskit	exon	1	3	.	+	.	ID=gene5-mRNA1:exon1;Parent=gene5-mRNA1
seq5	cdskit	exon	5	10	.	+	.	ID=gene5-mRNA1:exon2;Parent=gene5-mRNA1
seq5	cdskit	CDS	1	3	.	+	0	ID=gene5-mRNA1-cds1;Parent=gene5-mRNA1
seq5	cdskit	CDS	5	10	.	+	2	ID=gene5-mRNA1-cds2;Parent=gene5-mRNA1
seq6	cdskit	gene	1	10	.	+	.	ID=gene6;Name=gene6
seq6	cdskit	mRNA	1	10	.	+	.	ID=gene6-mRNA1;Parent=gene6;Name=gene6-mRNA1
seq6	cdskit	exon	1	3	.	+	.	ID=gene6-mRNA1:exon1;Parent=gene6-mRNA1
seq6	cdskit	exon	5	10	.	+	.	ID=gene6-mRNA1:exon2;Parent=gene6-mRNA1
seq6	cdskit	CDS	1	3	.	+	0	ID=gene6-mRNA1-cds1;Parent=gene6-mRNA1
seq6	cdskit	CDS	5	10	.	+	2	ID=gene6-mRNA1-cds2;Parent=gene6-mRNA1
```

### output.fasta
```
>seq1
AAAAAAAAAA
>seq2
TTTTTTTTTT
```

### output.gff
```
seq1	cdskit	gene	1	10	.	+	.	ID=gene1;Name=gene1
seq1	cdskit	mRNA	1	10	.	+	.	ID=gene1-mRNA1;Parent=gene1;Name=gene1-mRNA1
seq1	cdskit	exon	1	3	.	+	.	ID=gene1-mRNA1:exon1;Parent=gene1-mRNA1
seq1	cdskit	exon	5	10	.	+	.	ID=gene1-mRNA1:exon2;Parent=gene1-mRNA1
seq1	cdskit	CDS	1	3	.	+	0	ID=gene1-mRNA1-cds1;Parent=gene1-mRNA1
seq1	cdskit	CDS	5	10	.	+	2	ID=gene1-mRNA1-cds2;Parent=gene1-mRNA1
seq2	cdskit	gene	1	10	.	+	.	ID=gene2;Name=gene2
seq2	cdskit	mRNA	1	10	.	+	.	ID=gene2-mRNA1;Parent=gene2;Name=gene2-mRNA1
seq2	cdskit	exon	1	3	.	+	.	ID=gene2-mRNA1:exon1;Parent=gene2-mRNA1
seq2	cdskit	exon	5	10	.	+	.	ID=gene2-mRNA1:exon2;Parent=gene2-mRNA1
seq2	cdskit	CDS	1	3	.	+	0	ID=gene2-mRNA1-cds1;Parent=gene2-mRNA1
seq2	cdskit	CDS	5	10	.	+	2	ID=gene2-mRNA1-cds2;Parent=gene2-mRNA1
```

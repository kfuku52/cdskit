# ORF coordinate recovery on annotated transcripts

We compared fixed search policies on nuclear, spliced transcripts already
oriented 5′ to 3′. A prediction counted as exact when the forward strand and
both annotated CDS coordinates matched, including the terminal stop. Each
panel uses one eligible transcript per gene, chosen by the lowest SHA-256
of a fixed seed followed by the versioned transcript ID. The ORF scores did
not enter that selection.

| Search policy | GENCODE human (19,573 genes) | Ensembl Arabidopsis (27,381 genes) |
| --- | ---: | ---: |
| Default, both strands and table starts | 9,501 (48.54%) | 19,329 (70.59%) |
| Both strands and table starts, length first | 1,546 (7.90%) | 6,074 (22.18%) |
| Forward strand and table starts | 9,830 (50.22%) | 19,651 (71.77%) |
| Both strands and ATG starts | 18,111 (92.53%) | 26,971 (98.50%) |
| Forward strand and ATG starts | 18,569 (94.87%) | 27,290 (99.67%) |

The combined restriction recovered 9,077 additional human annotations and
7,967 additional plant annotations. It lost nine and six exact matches,
respectively, all in the annotated non-ATG groups (44 human and 49 plant
transcripts). For CDSs up to 300 nt, exact matches rose from 306/1,149 to
856/1,149 in human and from 1,405/2,180 to 2,132/2,180 in Arabidopsis.
CDSKIT exposes the two restrictions independently for workflows with
corresponding input evidence.

The human panel joined the protein-coding FASTA to basic GTF transcripts
with both start and stop annotations, protein-coding transcript type,
and no incomplete CDS boundary tags. The plant panel joined cDNA and CDS
FASTA by transcript ID on nuclear chromosomes 1–5 and required a unique
match of the annotated CDS in the cDNA. Both panels used code 1, concrete
ACGT sequences, a CDS in frame with a terminal stop, and no internal stop.
After the coordinate joins, 85 human and 48 plant entries failed those
sequence checks. The non-ATG groups refer to the reference sequence
boundaries. They do not independently establish translation initiation.

The study measures agreement with these two reference releases on transcripts
that already have annotated CDSs. It does not measure coding detection in
noncoding transcripts, fragment recovery, or agreement with other genetic
codes and taxa. The standard code lists ATG alongside alternative initiators.
[NCBI genetic codes](https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi?chapter=tgencodes)

## Reproduce

Download the files into a scratch directory and run the commands from the
repository root in the core test environment. Results include per-policy
strand, frame, start, stop, and exact-match counts, paired gains/losses, and
input hashes.

| Input | SHA-256 |
| --- | --- |
| [GENCODE v47 protein-coding transcripts](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.pc_transcripts.fa.gz) | `5fed4bff68c467447873c5908739d2905190b026fe56af7f70a182ec0cf4ebb3` |
| [GENCODE v47 basic GTF](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.basic.annotation.gtf.gz) | `8858ec2557c2904f91e44aeda762759cd998ce663774245710cb52ad4dc334ee` |
| [Ensembl Plants 63 TAIR10 cDNA](https://ftp.ebi.ac.uk/ensemblgenomes/pub/release-63/plants/fasta/arabidopsis_thaliana/cdna/Arabidopsis_thaliana.TAIR10.cdna.all.fa.gz) | `4a4302c4d5320cd8c26437bbb307145de63d749585931a144c40f5ffe47cbfe8` |
| [Ensembl Plants 63 TAIR10 CDS](https://ftp.ebi.ac.uk/ensemblgenomes/pub/release-63/plants/fasta/arabidopsis_thaliana/cds/Arabidopsis_thaliana.TAIR10.cds.all.fa.gz) | `3580fd49dc079892359c1e5b7ffe76b7c2aa13718daa4daa6f397882ad41795d` |

```bash
.venvs/core-3.12/bin/python scripts/evaluate_orf_reference.py human \
  --transcripts /tmp/gencode.v47.pc_transcripts.fa.gz \
  --annotation /tmp/gencode.v47.basic.annotation.gtf.gz \
  --output /tmp/orf-human.json --workers 5

.venvs/core-3.12/bin/python scripts/evaluate_orf_reference.py arabidopsis \
  --transcripts /tmp/Arabidopsis_thaliana.TAIR10.cdna.all.fa.gz \
  --annotation /tmp/Arabidopsis_thaliana.TAIR10.cds.all.fa.gz \
  --output /tmp/orf-arabidopsis.json --workers 5
```

## Known clipped coding ends

The clipped-end comparison used GENCODE human 47 and mouse M36 basic
annotations. It supplied the missing sides from `cds_start_NF` and
`cds_end_NF` tags and scored agreement with the annotated partial CDS.
These tags mark an unconfirmed coding boundary. The panel required the
corresponding CDS to reach the input edge within two bases, so the benchmark
tests partial fragments with known missing sides. [GENCODE tags](https://www.gencodegenes.org/pages/tags.html)

The evaluator trimmed any partial leading codon using the first CDS phase
and retained only whole codons at a clipped 3′ end. It required a terminal
stop or ATG at each intact end as appropriate, concrete ACGT and no internal
stop. One transcript per gene per missing-side group was selected by the
lowest hash of its versioned ID. Four mouse entries failed the sequence
checks. All three methods used code 1, forward orientation and ATG-only
start detection.

| Panel | Missing ends | Complete-first | Longest | Clipped-end selection |
| --- | --- | ---: | ---: | ---: |
| Human 47 | 5′ | 1/49 | 1/49 | 46/49 |
| Human 47 | 3′ | 2/59 | 5/59 | 55/59 |
| Human 47 | Both | 0/22 | 22/22 | 22/22 |
| Mouse M36 | 5′ | 0/22 | 0/22 | 21/22 |
| Mouse M36 | 3′ | 0/23 | 8/23 | 21/23 |
| Mouse M36 | Both | 0/15 | 15/15 | 15/15 |

The results measure recovery given known clipping. They do not measure
automatic detection of clipping or recovery of sequence beyond the input.

Run the same command separately for each release with the matching FASTA
and GTF. The mouse files are [protein-coding transcript FASTA](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M36/gencode.vM36.pc_transcripts.fa.gz) (`52cffd84fd954f3eddc317db1d0dc588dae9403aadcc9f16658353e395ec0a12`)
and [basic GTF](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M36/gencode.vM36.basic.annotation.gtf.gz) (`acc6f6650b1a32d200a50b18494749ab39a4a176b22d80182552efb713a483a7`).

```bash
.venvs/core-3.12/bin/python scripts/evaluate_partial_orf_reference.py \
  --transcripts /tmp/gencode.vM36.pc_transcripts.fa.gz \
  --annotation /tmp/gencode.vM36.basic.annotation.gtf.gz \
  --output /tmp/partial-orf-mouse.json
```

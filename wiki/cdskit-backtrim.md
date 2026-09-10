# cdskit backtrim

`cdskit backtrim` selects codon triplets from an untrimmed CDS alignment using a trimmed protein alignment.

Retain the trimmer's source column positions whenever possible: identical AA columns can contain different synonymous codons. AA agreement alone cannot identify which source column survived. Without positions, the default `--mapping_policy legacy` preserves historical first-match selection and warnings; use `--mapping_policy strict` to reject ambiguous or incomplete mappings.

![Backtrimming workflow](https://raw.githubusercontent.com/kfuku52/cdskit/master/img/backtrim.svg)

## Example
In this example, CDSKIT is combined with [SeqKit](https://github.com/shenwei356/seqkit) and [ClipKIT](https://github.com/JLSteenwyk/ClipKIT) to generate a trimmed codon alignment.

### Command
```
seqkit translate untrimmed_codon.fasta > untrimmed_aa.fasta

clipkit untrimmed_aa.fasta -o trimmed_aa.fasta -l

cdskit backtrim --seq_file untrimmed_codon.fasta --trimmed_aa_aln trimmed_aa.fasta --out_file trimmed_codon.fasta \
  --kept_sites trimmed_aa.fasta.log --kept_sites_format clipkit-log \
  --mapping_report mapping.json
```

### Column positions and mapping policies

Coordinates refer to the **original aligned protein columns, including gaps**. They are not genomic positions, ungapped residue numbers, or nucleotide offsets. Internally, source AA column `k` is zero-based and selects CDS bases `[3*k:3*k+3]`, preserving the original bases and gap symbols.

| `--kept_sites_format` | Input contract |
|---|---|
| `indices0` | One zero-based retained column integer per nonblank line. |
| `indices1` | One one-based retained column integer per nonblank line. |
| `clipkit-log` | Four whitespace-separated fields: one-based position, `keep`/`trim`, site class, unavailable fraction. Every original column must occur once in order. Uses only position and action. |
| `trimal-colnumbering` | Exactly one `#ColumnsMap` record with comma-separated zero-based retained positions, produced by `-colnumbering`. Keep alignment output separate using trimAl `-out`. |

The external formats have been exercised with ClipKIT 2.3.0 and trimAl v1.5.rev1 (upstream commit `5b5f730430843d39c579acdf8094969780c3c23e`). Other versions must satisfy the same input contracts.

Supply both `--kept_sites` and `--kept_sites_format`; neither the format nor the coordinate origin is guessed. Blank integer lists represent no retained columns. An empty trimAl map is written as `#ColumnsMap`. ClipKIT logs for all-deleted input must still list every original column as `trim`.

The trimmed AA alignment remains required. CDS and AA IDs must match uniquely (row order may differ); positions must be strictly increasing, unique, in range, and equal in count to the trimmed AA columns. Every selected column must match the supplied AA alignment after case and `.`/`-` normalization. `X` and `?` are not additional wildcards here. Position input errors always stop: there is no fallback to inferred positions. Logs must describe the actual output, not its complementary alignment. Sequence removal, residue masking, and column reordering are outside this column-deletion contract.

With no positions, `strict` checks all columns together: no complete ordered mapping or more than one complete mapping causes an error before writing the codon alignment. Repeated column patterns can still have a unique complete mapping (`AA` → `AA`); these are accepted. Even when alternative source positions give identical DNA, strict mode rejects their positional ambiguity. `legacy` keeps its historical warnings and partial-output behavior. Its local multiple-hit count is **not** a count of globally ambiguous positions. A future default change will require a separate announced compatibility release.

For example, CDS rows `GCTGCC` and `GCTGCT` both translate to `AA`. For trimmed rows `A`/`A`, `indices0` containing `1` produces `GCC`/`GCT`; legacy chooses `0` and produces `GCT`/`GCT`. Strict inference stops because both positions fit.

### Mapping report and provenance

`--mapping_report PATH` writes JSON schema version 1 to a file (not stdout). All report positions are zero-based. Fields include `source` (`provided`, `strict`, or `legacy`), policy, format, source/target column counts, genetic code, `inference_status` (`unique`, `ambiguous`, `unmatched`), existence/uniqueness of complete inferred maps, and `mapping_examples` (two witnesses for ambiguity, not an exhaustive list). Even a provided map can have ambiguous AA-only inference; the explicit positions resolve the requested selection.

On success, `selected_sites` lists source positions, `matched_target_sites` lists corresponding target positions, `unmatched_target_sites` records legacy omissions, and `output_complete` describes coverage. `legacy_trimmed_multiple_hit_sites` retains the old diagnostic separately. On validation failure, `status` is `failed` and `error` explains the failure; available analysis fields and witnesses remain, without a codon output. Option/path preflight failures may precede report creation. File alignment and success report outputs are committed together. Stdout cannot be rolled back, so all mapping validation precedes it.

The report records SHA-256 of the position file bytes and canonical CDS/AA record content (JSON lists of `[id, sequence]`, using Python `json.dumps(..., ensure_ascii=True)` defaults and UTF-8; AA rows reordered to CDS ID order). Preserve the original files, the trimmer version and exact command separately. A wrong map selecting an indistinguishable AA column cannot be detected from translation alone. For sequential trimming, compose maps back to the original alignment; a second-stage log refers to that stage's input.

Use `backalign` on the **complete** protein alignment before trimming, then save trimming positions for `backtrim`. `backalign` consumes CDS codons in order and intentionally rejects unmatched non-stop codons; it cannot recover positions deleted from a trimmed protein alignment.

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

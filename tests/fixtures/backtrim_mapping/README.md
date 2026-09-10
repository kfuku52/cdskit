# External trimmer mapping fixtures

Generated on 2026-09-10 from `source.aa.fa` (translations of `source.cds.fa`). Both AA columns have pattern A/A, but their codons differ. The known retained mask is zero-based `[1]`; expected CDS is `GCC` / `GCT`.

ClipKIT 2.3.0, locally installed executable:

```sh
clipkit source.aa.fa -m cst -a clipkit.cst.tsv -o clipkit.aa.fa -l
```

trimAl v1.5.rev1 build[2025-11-25], built from upstream branch `trimAl`, commit `5b5f730430843d39c579acdf8094969780c3c23e`:

```sh
trimal -in source.aa.fa -out trimal.aa.fa -selectcols '{' 0 '}' -colnumbering > trimal.columns
```

The checked-in log/map and AA outputs are the actual tool outputs. Tools are not required for ordinary regression tests; tests read these fixtures. Blank lines in trimAl stdout are preserved. Position adapters target these formats; other versions require compatible output, not automatic guessing. These fixtures validate mapping interoperability, not biological alignment quality.

Additional integration evaluation used both trimmers on two existing alignments, with ClipKIT `-m gappy -g 0.5` and trimAl `-gt 0.5 -keepseqs -colnumbering`. Trimming logs were parsed independently for a direct triplet-slicing oracle. All rows, IDs, and nucleotide strings matched exactly:

| Existing CDS fixture | Sequences | Original AA columns | ClipKIT retained | trimAl retained |
|---|---:|---:|---:|---:|
| `example_backtrim.cds.aln.fasta` | 21 | 319 | 198 | 200 |
| `backtrim_02/untrimmed_codon.fasta` | 19 | 13 | 4 | 4 |

The two trimmers need not select identical columns. The acceptance check is that each map reproduces its own retained codons exactly. No dS or phylogenetic-accuracy claim follows from this check.

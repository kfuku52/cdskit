# Alignment summary reference fixtures

These synthetic alignments are repository-owned. The `.tsv` files were generated
with AMAS commit `9ffc9d688bdb701847be931c2e107d9831caa129` on 2026-09-11:

```bash
python AMAS.py summary -i dna_edges.fasta -f fasta -d dna -o dna_edges.tsv
```

Use `-d aa` for the protein inputs. Random fixtures use Python `random.Random(527)`
and nine sequences of length 173, drawn from each AMAS alphabet in DNA then AA
order. Edge fixtures cover ambiguous/missing characters and informative sites;
the all-missing DNA case captures the legacy GC=1 convention.
No AMAS source is bundled and no AMAS runtime dependency is added.

`dna_rounding` has 23 missing cells out of 320, where AMAS reports 7.187%.
It catches changing the floating-point operation order to multiply before divide.

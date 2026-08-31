# cdskit rmseq

`cdskit rmseq` removes records whose IDs fully match a regular expression or
whose sequences contain too many configured problematic characters.

## Example

### Command
```
cdskit rmseq --seq_file input.fasta --seq_name_regex "Arabidopsis_thaliana.*" --problematic_percent 50 --out_file output.fasta
```

### input.fasta
```
>Aquilegia_coerulea_1
AGAGTTCAATATGCTTTGAGTCGAATTCGTAACAATGCTAGAAATCTTCTTACTCTTGAT
>Aquilegia_coerulea_2
AGAGTTCAATATGCTTTAAGTCGAATTCGAAACAATGCTAGAAATCTTCTCACTCTGGAT
>Aquilegia_coerulea_3
AGAGTTCAATATGCTTTAAGTCGAATTCGTAACAATGCAAGAAATCTTCTTACACTTGAT
>Hylocereus_undatus_1
AGGGTCCAATATGTTCTGAGCCGTATCCNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
>Hylocereus_undatus_2
AGGGTTCAATACGTTCTGAGCCGTATCCGTAATGCTGCAAGGCATCTTCTTACCCTGGAT
>Hylocereus_undatus_3
NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGCGGCAAGGCACCTTCTCACTCTGGAT
>Arabidopsis_thaliana_1
AGAGTTCAATATACACTTAGCAGAATCCGTAATGCTGCAAGAGAACTCTTAACTCTTGAT
>Arabidopsis_thaliana_2
AGAGTGCAGTACTCTCTTAGCCGTATCCGTAATGCTGCTAGAGATCTTTTGACTCTTGAT
```

### output.fasta
```
>Aquilegia_coerulea_1
AGAGTTCAATATGCTTTGAGTCGAATTCGTAACAATGCTAGAAATCTTCTTACTCTTGAT
>Aquilegia_coerulea_2
AGAGTTCAATATGCTTTAAGTCGAATTCGAAACAATGCTAGAAATCTTCTCACTCTGGAT
>Aquilegia_coerulea_3
AGAGTTCAATATGCTTTAAGTCGAATTCGTAACAATGCAAGAAATCTTCTTACACTTGAT
>Hylocereus_undatus_2
AGGGTTCAATACGTTCTGAGCCGTATCCGTAATGCTGCAAGGCATCTTCTTACCCTGGAT
```

## Key options

- `--seq_name_regex REGEX` removes complete-ID matches. Add `.*` around a
  substring when needed.
- `--problematic_chars CHARS` defines characters counted by
  `--problematic_percent`; the default is `NX-?` and matching is
  case-insensitive.
- `--problematic_percent FLOAT` removes a record when the configured
  characters occupy at least that percentage of its sequence. `0` disables
  this rule.

The ID and problematic-character rules are combined with OR: satisfying either
rule removes the record.

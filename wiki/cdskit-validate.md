# cdskit validate

`cdskit validate` runs basic quality checks for CDS alignments and prints a summary.

### Example

#### Command

    cdskit validate --seqfile alignment.fasta --codontable 1 --report validate.json

### What is checked

- Whether all sequence lengths are identical (`aligned`).
- Sequence lengths that are not multiples of three.
- Duplicate sequence IDs.
- Gap-only sequences.
- Sequences containing internal stop codons.
- Ambiguous codons and ambiguous-codon rate.

### Key options

- `--codontable INT`: NCBI codon table ID used for internal stop checks.
- `--report PATH`: Optional output report path (`.json` or tab-separated text).  
  No report file is produced unless this option is specified.

### Notes

- Validation output is printed to standard output.
- This command does not modify input sequences.

# cdskit label

`cdskit label` cleans, truncates, or uniquifies FASTA/sequence record IDs. The
sequence contents and record order are preserved.

## Example

```bash
cdskit label \
  --seq_file input.fasta \
  --out_file labeled.fasta \
  --replace_chars " !|/--_" \
  --clip_len 60 \
  --unique yes
```

`--replace_chars FROM--TO` replaces every character in `FROM` with the single
character `TO`. In the example, spaces, `!`, `|`, and `/` become `_`. Quote the
value because shells treat several of these characters specially.

Operations run in this order:

1. character replacement;
2. clipping to `--clip_len` when it is greater than zero;
3. duplicate resolution when `--unique yes`.

Duplicate IDs receive `_1`, `_2`, and subsequent suffixes. Because clipping can
create new duplicates, enable `--unique yes` when using `--clip_len` on IDs that
may share a prefix.

## Key options

- `--replace_chars FROM--TO`: Character replacement rule; empty by default.
- `--clip_len INT`: Maximum ID length; `0` disables clipping.
- `--unique yes|no`: Add suffixes to duplicate IDs; default `no`.
- `--seq_type dna|protein|auto`: Expected input type; default `auto`.

Label editing runs serially and has no `--threads` option.

Counts of changed, clipped, and duplicate IDs are written to standard error.

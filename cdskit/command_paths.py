"""Explicit file roles for public commands, independent of their implementations."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from cdskit.atomicio import validate_distinct_paths, validate_output_paths


@dataclass(frozen=True)
class CommandPaths:
    inputs: tuple[str, ...] = ("seqfile",)
    outputs: tuple[str, ...] = ("outfile",)


COMMAND_PATHS = {
    name: CommandPaths()
    for name in (
        "aggregate",
        "hammer",
        "label",
        "longestorf",
        "longestcds",
        "mask",
        "pad",
        "parsegb",
        "rmseq",
        "translate",
        "plot",
    )
}
COMMAND_PATHS.update(
    {
        "accession2fasta": CommandPaths(inputs=("accession_file",)),
        "backalign": CommandPaths(inputs=("seqfile", "aa_aln")),
        "backtrim": CommandPaths(inputs=("seqfile", "trimmed_aa_aln")),
        "gapjust": CommandPaths(inputs=("seqfile", "ingff")),
        "intersection": CommandPaths(inputs=("seqfile", "seqfile2", "ingff")),
        "filter": CommandPaths(outputs=("outfile", "report")),
        "trimcodon": CommandPaths(outputs=("outfile", "report")),
        "maxalign": CommandPaths(outputs=("outfile", "report")),
        "localize": CommandPaths(inputs=("seqfile", "model"), outputs=("report",)),
        "localize-learn": CommandPaths(
            inputs=("training_tsv", "esm_model_local_dir"),
            outputs=("model_out", "report", "uniprot_out_tsv"),
        ),
        "validate": CommandPaths(outputs=("report",)),
        "printseq": CommandPaths(outputs=()),
        "stats": CommandPaths(outputs=()),
        "codonstats": CommandPaths(outputs=()),
        # These commands generate filenames from a prefix, not --out_file itself.
        "split": CommandPaths(outputs=()),
        "degeneracy": CommandPaths(outputs=("report",)),
    }
)


def command_paths(args: Namespace) -> tuple[list[str], list[str]]:
    """Resolve only the paths a command actually reads or writes."""

    spec = COMMAND_PATHS[args.command]
    inputs = [getattr(args, name, None) for name in spec.inputs]
    if args.command == "localize" and args.model:
        # An alias is not a file named after the alias. Its resolved cache file
        # is checked by localize_main before the model is loaded.
        model_path = Path(str(args.model)).expanduser()
        if not model_path.exists() and not model_path.is_symlink():
            inputs = [args.seqfile]
    outputs = [getattr(args, name, None) for name in spec.outputs]
    if args.command in ("gapjust", "intersection") and args.ingff is not None:
        outputs.append(args.outgff)
    elif args.command == "intersection" and args.seqfile2 is not None:
        outputs.append(args.outfile2)
    if args.command in ("split", "degeneracy"):
        from cdskit.split import build_split_output_paths, resolve_output_prefix

        prefix = resolve_output_prefix(args)
        if args.command == "split":
            outputs.extend(build_split_output_paths(prefix, args.outseqformat))
        else:
            from cdskit.degeneracy import build_degeneracy_output_path

            outputs.extend(
                build_degeneracy_output_path(prefix, fold, args.outseqformat)
                for fold in sorted(set(args.fold))
            )
    return (
        [str(path) for path in inputs if path not in (None, "", "-")],
        [str(path) for path in outputs if path not in (None, "", "-")],
    )


def validate_command_paths(args: Namespace) -> None:
    inputs, outputs = command_paths(args)
    validate_distinct_paths(inputs=inputs, outputs=outputs)
    validate_output_paths(outputs)

"""Typed, transactional output helpers shared by cdskit commands."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import warnings
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeAlias


Pathish: TypeAlias = str | os.PathLike[str]
_ACTIVE_STAGED_PATHS: set[str] = set()


def normalized_path(path: Pathish) -> str:
    """Return a normalized absolute path suitable for collision checks."""

    return os.path.normcase(
        os.path.realpath(os.path.abspath(os.path.expanduser(os.fspath(path))))
    )


def _same_path(first: str, second: str) -> bool:
    if first == second:
        return True
    try:
        # Also catches hard links and case aliases on case-insensitive volumes.
        return os.path.samefile(first, second)
    except OSError:
        return False


def _contains_path(parent: str, child: str) -> bool:
    return any(_same_path(parent, str(ancestor)) for ancestor in Path(child).parents)


def validate_distinct_paths(
    inputs: Iterable[Pathish | None] = (),
    outputs: Iterable[Pathish | None] = (),
) -> None:
    """Reject input/output and output/output path collisions before writing."""

    input_paths = [
        path for path in inputs if path is not None and str(path) not in ("", "-")
    ]
    output_paths = [
        path for path in outputs if path is not None and str(path) not in ("", "-")
    ]
    normalized_inputs = {normalized_path(path): str(path) for path in input_paths}
    seen_outputs: dict[str, str] = {}
    for path in output_paths:
        normalized = normalized_path(path)
        if any(
            _same_path(normalized, source)
            or _contains_path(normalized, source)
            or (Path(source).is_dir() and _contains_path(source, normalized))
            for source in normalized_inputs
        ):
            raise ValueError(f"Input and output paths should be different: {path}.")
        for previous, previous_path in seen_outputs.items():
            if (
                _same_path(normalized, previous)
                or _contains_path(normalized, previous)
                or _contains_path(previous, normalized)
            ):
                raise ValueError(
                    "Output paths should be different and must not contain one "
                    f"another: {previous_path} and {path}."
                )
        seen_outputs[normalized] = str(path)


def validate_output_paths(paths: Iterable[Pathish]) -> None:
    """Reject directories and special files before creating any output."""

    for path in paths:
        destination = Path(path)
        if destination.exists():
            if not stat.S_ISREG(destination.stat().st_mode):
                raise ValueError(f"Output path must be a regular file: {path}.")
        elif not destination.is_symlink():
            # A dangling symlink may be replaced, but other non-regular entries
            # and invalid parent components must never be moved to a backup.
            try:
                mode = destination.lstat().st_mode
            except FileNotFoundError:
                mode = None
            if mode is not None and not stat.S_ISREG(mode):
                raise ValueError(f"Output path must be a regular file: {path}.")
        for parent in destination.parents:
            if parent.exists():
                if not parent.is_dir():
                    raise ValueError(f"Output parent must be a directory: {parent}.")
                break


@contextmanager
def atomic_output_path(path: Pathish) -> Iterator[str]:
    """Yield a same-directory temporary path and atomically replace *path*."""

    destination = Path(path)
    normalized_destination = normalized_path(destination)
    if normalized_destination in _ACTIVE_STAGED_PATHS:
        yield str(destination)
        return
    validate_output_paths([destination])
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        yield str(temporary)
        # Windows rejects fsync on a read-only descriptor, so reopen read/write.
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        validate_output_paths([destination])
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def atomic_output_paths(paths: Iterable[Pathish]) -> Iterator[list[str]]:
    """Stage multiple outputs and commit them together with rollback on error."""

    destinations = [Path(path) for path in paths]
    validate_distinct_paths(outputs=destinations)
    validate_output_paths(destinations)
    temporary_paths: list[Path] = []
    backup_paths: list[Path | None] = []
    commit_succeeded = False
    try:
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=f".tmp{destination.suffix}",
                dir=str(destination.parent),
            )
            os.close(fd)
            temporary_paths.append(Path(temporary_name))
        normalized_temporary_paths = {normalized_path(path) for path in temporary_paths}
        _ACTIVE_STAGED_PATHS.update(normalized_temporary_paths)
        try:
            yield [str(path) for path in temporary_paths]
        finally:
            _ACTIVE_STAGED_PATHS.difference_update(normalized_temporary_paths)
        for temporary_path in temporary_paths:
            with temporary_path.open("rb+") as handle:
                os.fsync(handle.fileno())

        committed = 0
        try:
            validate_output_paths(destinations)
            for destination in destinations:
                if destination.exists() or destination.is_symlink():
                    fd, backup_name = tempfile.mkstemp(
                        prefix=f".{destination.name}.",
                        suffix=".bak",
                        dir=str(destination.parent),
                    )
                    os.close(fd)
                    os.unlink(backup_name)
                    backup_path = Path(backup_name)
                    os.replace(destination, backup_path)
                else:
                    backup_path = None
                backup_paths.append(backup_path)

            for temporary_path, destination in zip(
                temporary_paths,
                destinations,
                strict=True,
            ):
                os.replace(temporary_path, destination)
                committed += 1
            commit_succeeded = True
        except Exception as commit_error:
            rollback_errors: list[str] = []
            for index, destination in enumerate(destinations):
                backup_path = backup_paths[index] if index < len(backup_paths) else None
                try:
                    if backup_path is not None and (
                        backup_path.exists() or backup_path.is_symlink()
                    ):
                        os.replace(backup_path, destination)
                    elif index < committed and (
                        destination.exists() or destination.is_symlink()
                    ):
                        destination.unlink()
                except OSError as rollback_error:
                    rollback_errors.append(str(rollback_error))
            if rollback_errors:
                retained = [
                    str(backup_path)
                    for backup_path in backup_paths
                    if backup_path is not None
                    and (backup_path.exists() or backup_path.is_symlink())
                ]
                raise RuntimeError(
                    "Failed to roll back an atomic multi-output update. "
                    "Recovery backups were retained at {}. Errors: {}".format(
                        retained,
                        rollback_errors,
                    )
                ) from commit_error
            raise
    finally:
        for temporary_path in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()
        if commit_succeeded:
            for backup_path in backup_paths:
                if backup_path is not None and (
                    backup_path.exists() or backup_path.is_symlink()
                ):
                    try:
                        backup_path.unlink()
                    except OSError as exc:
                        warnings.warn(
                            "Committed outputs, but could not remove recovery "
                            f"backup {backup_path}: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )


@contextmanager
def atomic_text_writer(
    path: Pathish,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> Iterator[Any]:
    """Open a text destination transactionally."""

    with atomic_output_path(path) as temporary:
        with open(temporary, "w", encoding=encoding, newline=newline) as handle:
            yield handle


def atomic_write_json(
    path: Pathish,
    payload: Mapping[str, Any] | list[Any],
    **kwargs: Any,
) -> None:
    """Write standards-compliant JSON transactionally."""

    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("allow_nan", False)
    with atomic_text_writer(path, encoding="utf-8") as handle:
        json.dump(payload, handle, **kwargs)
        handle.write("\n")

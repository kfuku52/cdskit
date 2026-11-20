"""Publish a multi-file dataset only after all of its files are complete."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def new_directory(destination):
    if not destination or str(destination) == "-":
        raise ValueError("A new output directory path is required.")
    destination = Path(destination)
    if os.path.lexists(destination):
        raise ValueError(f"Output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        stage = Path(temp) / "dataset"
        stage.mkdir()
        yield stage
        if os.path.lexists(destination):
            raise ValueError(f"Output directory already exists: {destination}")
        stage.rename(destination)

"""Failure-path tests: transactions must preserve pre-existing user data."""

import os
from pathlib import Path

import pytest

from cdskit import atomicio


@pytest.mark.parametrize("multiple", [False, True])
def test_directory_destination_is_rejected_before_staging(tmp_path, multiple):
    directory = tmp_path / "important"
    directory.mkdir()
    (directory / "data.txt").write_text("original")
    output = tmp_path / "new-parent" / "output.txt"
    context = (
        atomicio.atomic_output_paths([output, directory])
        if multiple
        else atomicio.atomic_output_path(directory)
    )
    with pytest.raises(ValueError, match="regular file"), context:
        pytest.fail("An invalid destination must be rejected before yielding")
    assert (directory / "data.txt").read_text() == "original"
    assert not output.parent.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["important"]


def test_output_ancestor_is_rejected_before_creating_parents(tmp_path):
    parent = tmp_path / "output"
    with (
        pytest.raises(ValueError, match="must not contain"),
        atomicio.atomic_output_paths([parent, parent / "child"]),
    ):
        pytest.fail("Overlapping destinations must not be staged")
    assert not parent.exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_special_file_destination_is_never_replaced(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with (
        pytest.raises(ValueError, match="regular file"),
        atomicio.atomic_output_paths([fifo]),
    ):
        pytest.fail("A FIFO must not be staged")
    assert fifo.exists()


def test_rollback_restores_dangling_symlink(tmp_path, monkeypatch):
    first = tmp_path / "link"
    try:
        first.symlink_to("missing-target")
    except OSError:
        pytest.skip("Symlinks unavailable")
    second = tmp_path / "second.txt"
    second.write_text("old")
    replace = os.replace

    def fail_second_commit(src, dst):
        if str(src).endswith(".tmp.txt") and Path(dst) == second:
            raise OSError("simulated commit failure")
        replace(src, dst)

    monkeypatch.setattr(atomicio.os, "replace", fail_second_commit)
    with pytest.raises(OSError, match="simulated commit failure"):
        with atomicio.atomic_output_paths([first, second]) as staged:
            for path in staged:
                Path(path).write_text("new")
    assert first.is_symlink()
    assert os.readlink(first) == "missing-target"
    assert second.read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["link", "second.txt"]


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_input_aliases_are_protected(tmp_path, link_kind):
    source = tmp_path / "input"
    source.write_text("old")
    alias = tmp_path / "alias"
    try:
        if link_kind == "symlink":
            alias.symlink_to(source)
        else:
            os.link(source, alias)
    except OSError:
        pytest.skip("Links unavailable")
    with pytest.raises(ValueError, match="Input and output paths"):
        atomicio.validate_distinct_paths(inputs=[source], outputs=[alias])
    assert source.read_text() == "old"


def test_input_directory_case_alias_cannot_contain_an_output(tmp_path):
    source = tmp_path / "Model"
    source.mkdir()
    alias = tmp_path / "MODEL"
    if not alias.exists():
        pytest.skip("Case-sensitive filesystem")
    with pytest.raises(ValueError, match="Input and output paths"):
        atomicio.validate_distinct_paths(inputs=[source], outputs=[alias / "new"])
    assert list(source.iterdir()) == []

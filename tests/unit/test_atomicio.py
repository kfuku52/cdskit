"""Failure-path tests: transactions must preserve pre-existing user data."""

import os
from pathlib import Path

import pytest

from cdskit import atomicio


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows resolves parent components before directory symlinks",
)
def test_duplicate_new_outputs_through_symlink_parent_are_rejected(tmp_path):
    directory = tmp_path / "actual"
    child = directory / "child"
    child.mkdir(parents=True)
    link = tmp_path / "link"
    try:
        link.symlink_to(child, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable")
    output = directory / "new.txt"
    with (
        pytest.raises(ValueError, match="Output paths should be different"),
        atomicio.atomic_output_paths([output, link / ".." / "new.txt"]),
    ):
        pytest.fail("Aliased outputs must be rejected before staging")
    assert not output.exists()
    assert list(directory.iterdir()) == [child]


def test_literal_tilde_does_not_collide_with_home(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    literal = tmp_path / "~"
    literal.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    source = home / "input.txt"
    source.write_text("source")
    atomicio.validate_distinct_paths(inputs=[source], outputs=["~/input.txt"])
    with atomicio.atomic_text_writer("~/input.txt") as handle:
        handle.write("output")
    assert source.read_text() == "source"
    assert (literal / "input.txt").read_text() == "output"


@pytest.mark.parametrize("error_type", [KeyboardInterrupt])
@pytest.mark.parametrize("phase", ["backup", "install"])
@pytest.mark.parametrize("after_replace", [False, True])
@pytest.mark.parametrize("existing_first", [False, True])
def test_interrupted_commit_restores_original_outputs(
    tmp_path, monkeypatch, error_type, phase, after_replace, existing_first
):
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    if existing_first:
        first.write_text("first original")
    second.write_text("second original")
    replace = os.replace
    interrupted = False

    def interrupt_replace(src, dst):
        nonlocal interrupted
        target = Path(src) == second if phase == "backup" else Path(dst) == first
        if target and not interrupted:
            interrupted = True
            if after_replace:
                replace(src, dst)
            raise error_type("interrupted commit")
        replace(src, dst)

    monkeypatch.setattr(atomicio.os, "replace", interrupt_replace)
    with pytest.raises(error_type, match="interrupted commit"):
        with atomicio.atomic_output_paths([first, second]) as staged:
            for path in staged:
                Path(path).write_text("replacement")
    assert interrupted
    assert second.read_text() == "second original"
    if existing_first:
        assert first.read_text() == "first original"
    else:
        assert not first.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == (
        ["first.txt", "second.txt"] if existing_first else ["second.txt"]
    )


def test_directory_destination_is_rejected_before_staging(tmp_path):
    directory = tmp_path / "important"
    directory.mkdir()
    (directory / "data.txt").write_text("original")
    output = tmp_path / "new-parent" / "output.txt"
    context = atomicio.atomic_output_paths([output, directory])
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


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
@pytest.mark.parametrize("multiple", [False, True])
def test_atomic_replacement_preserves_existing_mode(tmp_path, multiple):
    destination = tmp_path / "shared.txt"
    destination.write_text("old")
    destination.chmod(0o640)
    if multiple:
        with atomicio.atomic_output_paths([destination]) as paths:
            Path(paths[0]).write_text("replacement")
    else:
        with atomicio.atomic_text_writer(destination) as output:
            output.write("replacement")
    assert destination.read_text() == "replacement"
    assert destination.stat().st_mode & 0o777 == 0o640


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_atomic_permission_failure_preserves_outputs(tmp_path, monkeypatch):
    first, second = tmp_path / "a", tmp_path / "b"
    for path in (first, second):
        path.write_text("old")
        path.chmod(0o644)
    chmod = Path.chmod

    def fail_second(path, mode, *args, **kwargs):
        if path.name.startswith(".b."):
            raise OSError("permission update failed")
        return chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "chmod", fail_second)
    with pytest.raises(OSError, match="permission update failed"):
        with atomicio.atomic_output_paths([first, second]) as paths:
            for path in paths:
                Path(path).write_text("new")
    assert all(
        path.read_text() == "old" and path.stat().st_mode & 0o777 == 0o644
        for path in (first, second)
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a", "b"]

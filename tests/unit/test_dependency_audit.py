"""The dependency audit must not silently pass an unexamined CPU wheel."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "dependency_audit", Path(__file__).parents[2] / "scripts/audit_dependencies.py"
)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def test_audit_snapshot_maps_only_cpu_torch_and_excludes_editables(capsys):
    def distribution(name, version, editable=False):
        return SimpleNamespace(
            metadata={"Name": name},
            version=version,
            read_text=lambda _: json.dumps({"dir_info": {"editable": editable}}),
        )

    requirements = audit_module.audit_requirements(
        [
            distribution("torch", "2.13.0+cpu"),
            distribution("numpy", "2.5.2"),
            distribution("private_package", "1.0+local"),
            distribution("cdskit", "0.28.2", editable=True),
        ]
    )
    assert requirements == [
        "numpy==2.5.2",
        "private-package==1.0+local",
        "torch==2.13.0",
    ]
    assert "CPU torch 2.13.0+cpu against upstream 2.13.0" in capsys.readouterr().out


@pytest.mark.parametrize(
    "dependency,returncode,expected",
    [
        ({"name": "torch", "version": "2.13.0", "vulns": []}, 0, 0),
        ({"name": "torch", "skip_reason": "Not found on PyPI"}, 0, 1),
        ({"name": "torch", "vulns": [{"id": "TEST-VULNERABILITY"}]}, 1, 1),
    ],
)
def test_audit_rejects_skipped_or_vulnerable_dependencies(
    tmp_path, monkeypatch, dependency, returncode, expected
):
    def run(command, check):
        assert command[:3] == [audit_module.sys.executable, "-m", "pip_audit"]
        assert "--disable-pip" in command and "--no-deps" in command
        source = Path(command[command.index("--requirement") + 1])
        assert source.read_text() == "torch==2.13.0\n"
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({"dependencies": [dependency]}))
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(audit_module.subprocess, "run", run)
    assert audit_module.audit(["torch==2.13.0"]) == expected

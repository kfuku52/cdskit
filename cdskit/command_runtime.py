"""Shared runtime for lazily loaded CLI commands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import datetime
from importlib import import_module
from typing import Any


CommandHandler = Callable[[argparse.Namespace], Any]


def lazy_command(
    *,
    command: str,
    module_name: str,
    function_name: str,
    warning: str | None = None,
) -> CommandHandler:
    """Create a logged command handler without importing its implementation yet."""

    def run(args: argparse.Namespace) -> Any:
        if warning is not None:
            sys.stderr.write(warning.rstrip("\n") + "\n")
        handler = getattr(import_module(module_name), function_name)
        if not callable(handler):
            raise TypeError(f"{module_name}.{function_name} is not callable")
        sys.stderr.write(f"cdskit {command}: started at {datetime.now()}\n")
        result = handler(args)
        sys.stderr.write(f"cdskit {command}: ended at {datetime.now()}\n")
        return result

    run.__name__ = f"command_{command.replace('-', '_')}"
    return run

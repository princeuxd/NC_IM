"""Configuration loader helpers."""

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:  # noqa: D401
    mod = import_module("config.core")
    value = getattr(mod, name)
    globals()[name] = value
    return value

__all__ = ["parse_args", "Settings"] 
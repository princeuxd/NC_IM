"""Configuration loader helpers."""

from importlib import import_module
from types import ModuleType


def __getattr__(name: str) -> ModuleType:  # noqa: D401
    mod = import_module("config.core")
    value = getattr(mod, name)
    globals()[name] = value
    return value

__all__ = ["parse_args", "Settings"] 
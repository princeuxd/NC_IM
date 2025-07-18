"""Authentication helpers.

This sub-package exposes the YouTube auth helpers via::

    from auth import get_public_service, get_oauth_service
"""

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:  # noqa: D401
    mod = import_module("auth.youtube")
    value = getattr(mod, name)
    globals()[name] = value  # cache
    return value


__all__ = [
    "get_public_service",
    "get_oauth_service",
] 
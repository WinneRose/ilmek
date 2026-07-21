"""Backends: pluggable engines behind one analysis contract.

``native`` is always available and offline. ``stanza`` and ``zemberek`` are optional and
imported lazily so a missing optional dependency never breaks ``import ilmek``.
"""

from __future__ import annotations

from .base import Backend, BaseBackend
from .native import NativeBackend

__all__ = ["Backend", "BaseBackend", "NativeBackend", "get_backend"]


def get_backend(name: str = "native", **kwargs) -> BaseBackend:
    """Return a backend by name, importing optional ones lazily."""
    name = name.lower()
    if name == "native":
        return NativeBackend(**kwargs)
    if name == "stanza":
        from .stanza import StanzaBackend

        return StanzaBackend(**kwargs)
    if name == "zemberek":
        from .zemberek import ZemberekBackend

        return ZemberekBackend(**kwargs)
    raise ValueError(f"Unknown backend: {name!r}. Available: native, stanza, zemberek.")

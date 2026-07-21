"""Zemberek backend (optional, reserved).

Adapts Zemberek-NLP (Java, via JPype) for comparison, word generation, spell checking, and
noisy-text normalization. Kept a separate optional integration because of the Java runtime
requirement, and detected at runtime. Instantiation fails loudly until implemented.
"""

from __future__ import annotations

from .base import BaseBackend


class ZemberekBackend(BaseBackend):  # pragma: no cover - reserved
    name = "zemberek"

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "The Zemberek backend is an optional Java integration and is not implemented "
            'yet. It requires a Java runtime and `pip install "ilmek[zemberek]"`.'
        )

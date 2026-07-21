"""Stanza backend (optional, reserved for v0.2).

Provides contextual lemmatization, POS/feature tagging, and dependency parsing by adapting
Stanza's Turkish UD models into the shared :class:`AnalysisResult` schema. Stanza and its
models are an *optional* dependency downloaded via the model registry — never a required
core dependency. Instantiation fails loudly with install guidance until v0.2 lands.
"""

from __future__ import annotations

from .base import BaseBackend


class StanzaBackend(BaseBackend):  # pragma: no cover - reserved for v0.2
    name = "stanza"

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "The Stanza backend arrives in v0.2. Install extras with "
            '`pip install "ilmek[stanza]"` once released.'
        )

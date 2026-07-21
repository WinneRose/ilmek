"""Test fixtures and import shim (so tests run without an editable install)."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from ilmek import Pipeline
from ilmek.morphology.analyzer import default_analyzer


@pytest.fixture(scope="session")
def analyzer():
    return default_analyzer()


@pytest.fixture(scope="session")
def nlp():
    return Pipeline()


def analyses_for(analyzer, word):
    """Return the list of analyses for a word."""
    return analyzer.analyze(word)


def has_analysis(analyzer, word, *, lemma=None, pos=None, features=None, morphemes=None):
    """True if some analysis of ``word`` matches all provided constraints."""
    for a in analyzer.analyze(word):
        if lemma is not None and a.lemma != lemma:
            continue
        if pos is not None and a.pos != pos:
            continue
        if morphemes is not None and a.morphemes != list(morphemes):
            continue
        if features is not None and not all(a.features.get(k) == v for k, v in features.items()):
            continue
        return True
    return False

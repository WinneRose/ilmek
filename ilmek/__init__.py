"""ilmek — explainable Turkish morphology (stemmer, lemmatizer, analyzer).

Public API (stable contract):

    >>> import ilmek as trnlp
    >>> trnlp.stem("kitaplarımızdan")
    'kitap'
    >>> trnlp.lemmatize("kitaplarımızdan")
    'kitap'
    >>> [a.lemma for a in trnlp.analyze("kitaplarımızdan")]
    ['kitap']
    >>> doc = trnlp.analyze_sentence("Kitaplarımızı masaya bıraktık.")

``analyze`` returns every valid analysis (best first) as :class:`AnalysisResult` objects;
``stem`` / ``lemmatize`` return the best single string; ``analyze_sentence`` returns a
:class:`Document`. ``load()`` returns a reusable :class:`Pipeline` (``nlp = trnlp.load()``;
``nlp("...")``).
"""

from __future__ import annotations

from .core.document import AnalysisResult, Document, Token
from .morphology.analyzer import Analyzer, default_analyzer
from .morphology.lexicon import Lexicon
from .pipeline.pipeline import Pipeline

__version__ = "0.1.0a0"

__all__ = [
    "__version__",
    "AnalysisResult",
    "Document",
    "Token",
    "Analyzer",
    "Lexicon",
    "Pipeline",
    "load",
    "stem",
    "lemmatize",
    "analyze",
    "analyze_sentence",
]

_PIPELINE: Pipeline | None = None


def load() -> Pipeline:
    """Return a reusable native pipeline (constructed once, then cached)."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = Pipeline()
    return _PIPELINE


def stem(word: str) -> str:
    """Return the stem of ``word`` (best analysis)."""
    return default_analyzer().analyze(word)[0].stem


def lemmatize(word: str) -> str:
    """Return the lemma of ``word`` (best analysis)."""
    return default_analyzer().analyze(word)[0].lemma


def analyze(word: str) -> list[AnalysisResult]:
    """Return every valid analysis of ``word``, best candidate first."""
    return default_analyzer().analyze(word)


def analyze_sentence(text: str) -> Document:
    """Tokenize and analyze a whole sentence, returning a :class:`Document`."""
    return load().analyze_sentence(text)

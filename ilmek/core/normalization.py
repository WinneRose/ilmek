"""Turkish-aware text normalization.

The one rule the spec repeats: **never trust locale-naive ``str.lower()`` /
``str.upper()`` for Turkish.** Python's default casing maps ``I -> i`` and mishandles
``İ``; Turkish needs ``I -> ı`` and ``İ -> i``. All casing in the library goes through
:func:`turkish_lower` / :func:`turkish_upper`, which use the explicit pair tables in
:mod:`ilmek.core.alphabet`.
"""

from __future__ import annotations

import unicodedata

from .alphabet import LOWER_TO_UPPER, UPPER_TO_LOWER

#: Apostrophe-like characters that Turkish text uses to attach a suffix to a proper noun
#: (e.g. ``Ankara'da``). We fold every variant to the ASCII apostrophe so the tokenizer
#: sees exactly one shape.
_APOSTROPHES = {
    "’": "'",  # ’ right single quote (most common in the wild)
    "‘": "'",  # ‘ left single quote
    "ʼ": "'",  # ʼ modifier letter apostrophe
    "′": "'",  # ′ prime
    "`": "'",  # ` grave accent
    "´": "'",  # ´ acute accent
    "＇": "'",  # ＇ fullwidth apostrophe
}

_APOSTROPHE_TABLE = {ord(k): v for k, v in _APOSTROPHES.items()}

#: A stray combining dot above (U+0307) can appear after NFD-decomposing ``İ``; drop it
#: once casing has been resolved so it never leaks into a lemma.
_COMBINING_DOT_ABOVE = "̇"

#: Circumflex-marked vowels fold to their plain counterparts for LOOKUP ONLY (never in
#: :func:`normalize`, so the surface/token keeps the circumflex). The circumflex (düzeltme
#: işareti) marks vowel length / a palatalized preceding consonant in Ottoman-origin loans —
#: kâğıt, hâlâ, âlim, kâr — but the modern lexicon stores the plain spelling (kağıt, hala,
#: alim), so folding lets a circumflex surface match its plain root. Only the three vowels that
#: actually carry the mark in Turkish (â/î/û — never *ô/*ê); the uppercase Â/Î/Û reach these via
#: :func:`turkish_lower`'s ``ch.lower()`` fallback before the fold runs. NFC (guaranteed by
#: :func:`normalize`) keeps each as a single composed codepoint, so a one-char translate suffices.
_CIRCUMFLEX_FOLD = {ord("â"): "a", ord("î"): "i", ord("û"): "u"}


def turkish_lower(text: str) -> str:
    """Lowercase ``text`` using Turkish casing rules (``I -> ı``, ``İ -> i``)."""
    out = []
    for ch in text:
        mapped = UPPER_TO_LOWER.get(ch)
        out.append(mapped if mapped is not None else ch.lower())
    return "".join(out)


def turkish_upper(text: str) -> str:
    """Uppercase ``text`` using Turkish casing rules (``i -> İ``, ``ı -> I``)."""
    out = []
    for ch in text:
        mapped = LOWER_TO_UPPER.get(ch)
        out.append(mapped if mapped is not None else ch.upper())
    return "".join(out)


def standardize_apostrophes(text: str) -> str:
    """Fold every apostrophe variant to a single ASCII ``'``."""
    return text.translate(_APOSTROPHE_TABLE)


def normalize(
    text: str,
    *,
    form: str = "NFC",
    unify_apostrophes: bool = True,
    strip_combining_dot: bool = True,
) -> str:
    """Normalize Turkish text without destroying case (case matters for proper nouns).

    Steps: Unicode ``form`` normalization (NFC by default) → optional apostrophe folding
    → optional removal of a stray combining dot above. Casing is *not* changed here; use
    :func:`turkish_lower` / :func:`turkish_upper` explicitly when you need it.
    """
    text = unicodedata.normalize(form, text)
    if unify_apostrophes:
        text = standardize_apostrophes(text)
    if strip_combining_dot:
        text = text.replace(_COMBINING_DOT_ABOVE, "")
    return text


def fold_for_lookup(text: str) -> str:
    """Canonical form used for lexicon lookup: normalized + Turkish-lowercased + circumflex-folded.

    The circumflex fold (â/î/û -> a/i/u) is applied AFTER Turkish-lowercasing so a circumflex
    surface (kâğıt, hâlâ, âlim) matches its plain lexicon root (kağıt, hala, alim). It lives
    here, not in :func:`normalize`, so the preserved surface/token keeps its circumflex — only
    the lookup key is folded.
    """
    return turkish_lower(normalize(text)).translate(_CIRCUMFLEX_FOLD)

"""Rule-based Turkish tokenizer.

Splits normalized text into :class:`Token` spans and classifies each: ``word``,
``number``, ``date``, ``time``, ``abbr``, ``url``, ``email``, ``mention``, ``hashtag``,
``emoticon``, ``punct``. Words keep their apostrophe suffix separated (``Ankara'da`` ->
text ``Ankara'da`` with ``apostrophe_suffix='da'``) so a proper noun and its inflection
stay independently analyzable.

The tokenizer only *segments and classifies*; it never lowercases or morphologically
analyzes. Case is preserved for downstream proper-noun handling.
"""

from __future__ import annotations

import re

from .alphabet import LOWER, UPPER
from .document import Token
from .normalization import normalize, turkish_lower

_LETTERS = LOWER + UPPER
_L = re.escape(_LETTERS)

#: Common Turkish abbreviations that legitimately carry an internal/trailing dot, so the
#: dot is not mistaken for a sentence boundary. Data, not code — extend freely.
ABBREVIATIONS = frozenset(
    {
        "dr",
        "prof",
        "doç",
        "av",
        "sn",
        "bkz",
        "örn",
        "vb",
        "vs",
        "yrd",
        "arş",
        "gör",
        "mah",
        "cad",
        "sok",
        "apt",
        "no",
        "tel",
        "gsm",
        "bl",
        "böl",
        "fak",
        "üni",
        "tc",
        "tbmm",
        "abd",
        "ab",
        "kdv",
        "pk",
        "vd",
        "çev",
        "ed",
        "haz",
        "bs",
    }
)

# Ordered alternation. First branch that matches at a position wins (Python re semantics).
# ``§L§`` is a placeholder for the Turkish letter class, substituted via str.replace so
# literal ``%`` in the number pattern is not mistaken for a format specifier.
_MASTER = re.compile(
    r"""
    (?P<url>(?:https?://|www\.)[^\s]+)
  | (?P<email>[^\s@]+@[^\s@]+\.[^\s@]+)
  | (?P<mention>@[§L§0-9_]+)
  | (?P<hashtag>\#[§L§0-9_]+)
  | (?P<time>\d{1,2}:\d{2}(?::\d{2})?)
  | (?P<date>\d{1,2}[./]\d{1,2}[./]\d{2,4})
  | (?P<number>\d+(?:[.,]\d+)*%?)
  | (?P<word>[§L§]+(?:'[§L§]+)*)
  | (?P<emoticon>[:;=][-^]?[)(DPpOo/\\|]+)
  | (?P<punct>[^\s§L§0-9])
    """.replace("§L§", _L),
    re.VERBOSE,
)

# A word that is an abbreviation followed immediately by a dot, e.g. "Dr." / "vb."
_ABBR_DOT = re.compile(r"(?P<abbr>[§L§]+)\.(?=\s|$)".replace("§L§", _L))


def _split_apostrophe(surface: str) -> str | None:
    """Return the suffix after the last apostrophe in a word, or ``None``."""
    if "'" in surface:
        head, tail = surface.rsplit("'", 1)
        if head and tail:
            return tail
    return None


def tokenize(text: str, *, normalize_text: bool = True) -> list[Token]:
    """Tokenize ``text`` into a list of :class:`Token` with character offsets."""
    if normalize_text:
        text = normalize(text)

    tokens: list[Token] = []
    pos = 0
    n = len(text)

    while pos < n:
        if text[pos].isspace():
            pos += 1
            continue

        # Prefer an abbreviation-with-dot at this position (keeps "vb." as one token).
        abbr = _ABBR_DOT.match(text, pos)
        if abbr is not None and turkish_lower(abbr.group("abbr")) in ABBREVIATIONS:
            tokens.append(Token(abbr.group(0), abbr.start(), abbr.end(), kind="abbr"))
            pos = abbr.end()
            continue

        m = _MASTER.match(text, pos)
        if m is None:  # pragma: no cover - master regex covers every non-space char
            pos += 1
            continue

        kind = m.lastgroup or "word"
        surface = m.group()
        apo = _split_apostrophe(surface) if kind == "word" else None
        tokens.append(Token(surface, m.start(), m.end(), kind=kind, apostrophe_suffix=apo))
        pos = m.end()

    return tokens

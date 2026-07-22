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

# --- Numeric-token fragments ---------------------------------------------------------
# Defined once here (data, not code) and interpolated into BOTH the master tokenizer regex
# and the ``classify_numeric`` classifiers below, so the tokenizer and the analyzer's NUM
# fast path can never disagree on what counts as a number / date / time.
_TIME_RE = r"\d{1,2}:\d{2}(?::\d{2})?"
_DATE_RE = r"\d{1,2}[./]\d{1,2}[./]\d{2,4}"
#: Percent, Turkish (leading ``%25``) or Anglo (trailing ``25%``). The ``%`` is mandatory on
#: this branch so it is distinguishable from a plain number (which carries an *optional* %).
_PERCENT_RE = r"%\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*%"
#: Ordinal dot: digits + a dot NOT followed by another digit, so ``3.`` is ONE token while a
#: grouped/decimal figure keeps its dot/comma on the plain branch (``1.000,50``, ``3,5``).
_ORDINAL_BODY_RE = r"\d+\."
_ORDINAL_RE = _ORDINAL_BODY_RE + r"(?!\d)"
#: Plain number: a bare cardinal or a thousands/decimal-grouped figure, optional trailing %.
_PLAIN_NUMBER_RE = r"\d+(?:[.,]\d+)*%?"
#: The tokenizer ``number`` group: percent and ordinal-dot come BEFORE the plain branch so a
#: mandatory-% or a trailing-dot form is preferred over the plain (optional-%) reading.
_NUMBER_RE = rf"(?:{_PERCENT_RE}|{_ORDINAL_RE}|{_PLAIN_NUMBER_RE})"

# Ordered alternation. First branch that matches at a position wins (Python re semantics).
# ``§L§`` is a placeholder for the Turkish letter class, substituted via str.replace so
# literal ``%`` in the number pattern is not mistaken for a format specifier; the numeric
# fragments above are interpolated the same way.
_MASTER = re.compile(
    r"""
    (?P<url>(?:https?://|www\.)[^\s]+)
  | (?P<email>[^\s@]+@[^\s@]+\.[^\s@]+)
  | (?P<mention>@[§L§0-9_]+)
  | (?P<hashtag>\#[§L§0-9_]+)
  | (?P<time>§TIME§)
  | (?P<date>§DATE§)
  | (?P<number>§NUMBER§)
  | (?P<word>[§L§]+(?:'[§L§]+)*)
  | (?P<emoticon>[:;=][-^]?[)(DPpOo/\\|]+)
  | (?P<punct>[^\s§L§0-9])
    """.replace("§TIME§", _TIME_RE)
    .replace("§DATE§", _DATE_RE)
    .replace("§NUMBER§", _NUMBER_RE)
    .replace("§L§", _L),
    re.VERBOSE,
)

# A word that is an abbreviation followed immediately by a dot, e.g. "Dr." / "vb."
_ABBR_DOT = re.compile(r"(?P<abbr>[§L§]+)\.(?=\s|$)".replace("§L§", _L))

#: A dotted acronym written with internal periods (``T.C.``, ``A.Ş.``): two or more
#: single-UPPERCASE-letter + dot units. Kept whole (``kind="abbr"``) so its internal periods
#: are not mistaken for sentence boundaries. Uppercase-only, so it never eats a lowercase
#: word followed by a full stop (``kedi.`` still splits into word + punct).
_DOTTED_ACRONYM = re.compile(r"(?:[§U§]\.){2,}".replace("§U§", re.escape(UPPER)))

#: Full-match numeric classifiers, in priority order, built from the SAME fragments as
#: :data:`_MASTER`. The returned label is the token's numeric kind; ``None`` means the whole
#: string is not a single numeric token (so a mixed token like ``3x4`` / ``v2`` is not NUM).
_NUMERIC_CLASSIFIERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("time", re.compile(rf"(?:{_TIME_RE})\Z")),
    ("date", re.compile(rf"(?:{_DATE_RE})\Z")),
    ("percent", re.compile(rf"(?:{_PERCENT_RE})\Z")),
    ("ordinal", re.compile(rf"(?:{_ORDINAL_BODY_RE})\Z")),
    ("cardinal", re.compile(r"\d+\Z")),
    ("formatted", re.compile(rf"(?:{_PLAIN_NUMBER_RE})\Z")),
)


def classify_numeric(surface: str) -> str | None:
    """Classify a whole token as a numeric kind, or ``None`` if it is not one numeric token.

    Returns one of ``"time"``, ``"date"``, ``"percent"``, ``"ordinal"``, ``"cardinal"`` or
    ``"formatted"`` (a thousands/decimal-grouped figure). It requires a FULL match, so a
    mixed token (``3x4``, ``v2``) is ``None`` and stays on the word/guesser path. Shared with
    the analyzer so a token the tokenizer classes as number/date/time resolves as NUM there.
    """
    for label, pattern in _NUMERIC_CLASSIFIERS:
        if pattern.match(surface):
            return label
    return None


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

        # A dotted acronym (T.C., A.Ş.) is kept whole so its internal periods are not read as
        # sentence boundaries. Checked before the abbreviation-dot and the master alternation
        # (the master would otherwise split it into single letters and punctuation).
        dotted = _DOTTED_ACRONYM.match(text, pos)
        if dotted is not None:
            tokens.append(Token(dotted.group(0), dotted.start(), dotted.end(), kind="abbr"))
            pos = dotted.end()
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

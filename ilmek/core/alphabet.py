"""Turkish alphabet, vowel/consonant classification, and harmony/voicing tables.

This module is deliberately data-only (frozen sets + dict tables + tiny pure helpers).
Every higher layer — normalization, phonology, morphotactics — reads its Turkish facts
from here so the language model lives in one auditable place, per the design principle
"keep language rules in data, not scattered through code".
"""

from __future__ import annotations

# --- Letters -------------------------------------------------------------------------

LOWER = "abcçdefgğhıijklmnoöprsştuüvyz"
UPPER = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"

#: The two case pairs that break locale-naive ``str.lower()`` / ``str.upper()``.
#: dotless ı <-> I, dotted i <-> İ. Everything else in :data:`LOWER`/:data:`UPPER`
#: maps 1:1 by position.
LOWER_TO_UPPER = dict(zip(LOWER, UPPER, strict=True))
UPPER_TO_LOWER = dict(zip(UPPER, LOWER, strict=True))

# --- Vowels --------------------------------------------------------------------------

VOWELS = frozenset("aeıioöuü")
VOWELS_UPPER = frozenset("AEIİOÖUÜ")

BACK_VOWELS = frozenset("aıou")
FRONT_VOWELS = frozenset("eiöü")
ROUNDED_VOWELS = frozenset("oöuü")
UNROUNDED_VOWELS = frozenset("aeıi")
#: "wide"/open vowels vs. "narrow"/high vowels (used for -Iyor narrowing).
WIDE_VOWELS = frozenset("aeoö")
NARROW_VOWELS = frozenset("ıiuü")

# --- Consonants ----------------------------------------------------------------------

CONSONANTS = frozenset("bcçdfgğhjklmnprsştvyz")

#: Voiceless consonants ("sert ünsüzler" — the mnemonic *fıstıkçı şahap*).
#: A suffix-initial voiced stop hardens after one of these (D -> t, etc.).
VOICELESS = frozenset("fstkçşhp")

#: Consonant softening on a stem-final stop before a vowel-initial suffix
#: (ünsüz yumuşaması): kitap -> kitab-ı, ağaç -> ağac-ı, kanat -> kanad-ı, renk -> reng-i.
#: The *nk -> ng* case is handled specially in phonology because only the k softens.
VOICING = {"p": "b", "ç": "c", "t": "d", "k": "ğ"}
DEVOICING = {v: k for k, v in VOICING.items()}

# --- Vowel-harmony resolution --------------------------------------------------------


#: Archiphoneme ``A`` (low/wide harmony): back -> "a", front -> "e".
def resolve_A(last_vowel: str) -> str:
    return "a" if last_vowel in BACK_VOWELS else "e"


#: Archiphoneme ``I`` (high/narrow four-way harmony): back+unrounded -> "ı",
#: front+unrounded -> "i", back+rounded -> "u", front+rounded -> "ü".
def resolve_I(last_vowel: str) -> str:
    back = last_vowel in BACK_VOWELS
    rounded = last_vowel in ROUNDED_VOWELS
    if back and not rounded:
        return "ı"
    if not back and not rounded:
        return "i"
    if back and rounded:
        return "u"
    return "ü"


def last_vowel(text: str) -> str | None:
    """Return the last vowel in ``text`` (drives progressive harmony), or ``None``."""
    for ch in reversed(text):
        if ch in VOWELS:
            return ch
    return None


def first_vowel(text: str) -> str | None:
    for ch in text:
        if ch in VOWELS:
            return ch
    return None


def is_vowel(ch: str) -> bool:
    return ch in VOWELS


def ends_with_vowel(text: str) -> bool:
    return bool(text) and text[-1] in VOWELS


def ends_with_voiceless(text: str) -> bool:
    return bool(text) and text[-1] in VOICELESS

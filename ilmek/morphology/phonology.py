"""Morphophonemics: turn an archiphonemic suffix template into its surface form.

A suffix is written once, abstractly, and realized in context. The template alphabet:

* ``A`` — low-vowel harmony archiphoneme, becomes ``a``/``e``.
* ``I`` — high-vowel (4-way) harmony archiphoneme, becomes ``ı``/``i``/``u``/``ü``.
* ``D`` — alternating stop, becomes ``d`` normally, ``t`` after a voiceless consonant.
* ``C`` — alternating affricate (agentive ``-CI``), becomes ``c`` normally, ``ç`` after a
  voiceless consonant (yol+CI -> yolcu, but kitap+CI -> kitapçı, iş+CI -> işçi).
* ``(y)`` ``(s)`` ``(n)`` ``(ş)`` — **buffer consonant**: inserted only when the left
  context ends in a **vowel** (prevents vowel–vowel hiatus), e.g. ``kapı+(y)I -> kapıyı``.
* ``(A)`` ``(I)`` — **linking vowel**: inserted only when the left context ends in a
  **consonant**, e.g. ``ev+(I)m -> evim`` but ``kapı+(I)m -> kapım``.
* any lowercase letter — literal.

Harmony/voicing decisions always read the *running* surface (left context plus whatever
this suffix has emitted so far), so stacked archiphonemes like ``lArI`` harmonize correctly
(``lar`` then ``ı`` off the new ``a``). Root-boundary alternations (consonant voicing,
vowel drop) are applied by the analyzer, which owns the lexicon; this module is purely
about suffix shape.
"""

from __future__ import annotations

from ..core.alphabet import (
    SUFFIX_ALTERNATIONS,
    VOWELS,
    ends_with_voiceless,
    ends_with_vowel,
    last_vowel,
    resolve_A,
    resolve_I,
)

_BUFFER_CONSONANTS = frozenset("ysnş")
_ARCHI_VOWELS = frozenset("AI")
#: Fallback vowel when a stem carries no vowel at all (abbreviations, some loanwords).
_DEFAULT_LAST_VOWEL = "a"


def _harmony_vowel(archi: str, ctx: str) -> str:
    lv = last_vowel(ctx) or _DEFAULT_LAST_VOWEL
    return resolve_A(lv) if archi == "A" else resolve_I(lv)


def realize(template: str, left_context: str) -> str:
    """Realize an archiphonemic ``template`` given ``left_context`` (surface to its left)."""
    ctx = left_context
    out: list[str] = []
    i = 0
    n = len(template)
    while i < n:
        ch = template[i]
        if ch == "(":
            close = template.index(")", i)
            inner = template[i + 1 : close]
            i = close + 1
            if inner in _BUFFER_CONSONANTS:
                # Buffer consonant: only between two vowels.
                if ends_with_vowel(ctx):
                    out.append(inner)
                    ctx += inner
            elif inner in _ARCHI_VOWELS:
                # Linking vowel: only after a consonant.
                if not ends_with_vowel(ctx):
                    v = _harmony_vowel(inner, ctx)
                    out.append(v)
                    ctx += v
            else:  # pragma: no cover - defensive: unknown parenthesized symbol
                out.append(inner)
                ctx += inner
            continue

        if ch in _ARCHI_VOWELS:
            v = _harmony_vowel(ch, ctx)
            out.append(v)
            ctx += v
        elif ch in SUFFIX_ALTERNATIONS:
            voiced, hardened = SUFFIX_ALTERNATIONS[ch]
            letter = hardened if ends_with_voiceless(ctx) else voiced
            out.append(letter)
            ctx += letter
        else:
            out.append(ch)
            ctx += ch
        i += 1

    return "".join(out)


def starts_with_vowel(realized: str) -> bool:
    """Whether a realized suffix begins with a vowel (triggers root-boundary voicing)."""
    return bool(realized) and realized[0] in VOWELS

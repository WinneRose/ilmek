"""Morphophonemic realizer: harmony (A/I), voicing (D), buffers, linking vowels."""

from __future__ import annotations

import pytest

from ilmek.morphology.phonology import realize, starts_with_vowel


@pytest.mark.positive
@pytest.mark.parametrize(
    "template,ctx,expected",
    [
        # A-type (2-way) harmony
        ("lAr", "kitap", "lar"),
        ("lAr", "ev", "ler"),
        ("lAr", "okul", "lar"),
        ("lAr", "göz", "ler"),
        # I-type (4-way) harmony
        ("(s)I", "ev", "i"),
        ("(s)I", "okul", "u"),
        ("(s)I", "göz", "ü"),
        ("(s)I", "kapı", "sı"),
        # D devoicing after voiceless
        ("DAn", "kitap", "tan"),
        ("DAn", "ev", "den"),
        ("DA", "sokak", "ta"),
        ("DA", "deniz", "de"),
    ],
)
def test_harmony_and_voicing(template, ctx, expected):
    assert realize(template, ctx) == expected


@pytest.mark.positive
def test_buffer_consonant_only_after_vowel():
    assert realize("(y)I", "kapı") == "yı"  # vowel -> buffer y inserted
    assert realize("(y)I", "ev") == "i"  # consonant -> no buffer


@pytest.mark.positive
def test_linking_vowel_only_after_consonant():
    assert realize("(I)m", "ev") == "im"  # consonant -> linking vowel present
    assert realize("(I)m", "kapı") == "m"  # vowel -> linking vowel dropped


@pytest.mark.positive
def test_stacked_archiphonemes_harmonize_progressively():
    # lArI: 'lar' then 'ı' harmonizing off the freshly added 'a'
    assert realize("lArI", "kapı") == "ları"
    assert realize("lArI", "ev") == "leri"


@pytest.mark.positive
def test_starts_with_vowel():
    assert starts_with_vowel("ım")
    assert not starts_with_vowel("lar")
    assert not starts_with_vowel("")


# --- Front-harmony loan flag: realize(..., front_root=True) ---------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "template,ctx,expected",
    [
        ("(s)I", "saat", "i"),  # a -> fronted i (saati, not saatı)
        ("lAr", "saat", "ler"),  # a -> fronted e (saatler)
        ("(y)A", "saat", "e"),  # dative fronts too (saate)
        ("DA", "kalp", "te"),  # front + D hardening after voiceless p (kalpte)
        ("(s)I", "kabul", "ü"),  # u -> fronted ü (kabulü)
        ("(s)I", "usul", "ü"),  # u -> fronted ü (usulü)
    ],
)
def test_front_root_harmony(template, ctx, expected):
    assert realize(template, ctx, front_root=True) == expected


@pytest.mark.negative
@pytest.mark.parametrize(
    "template,ctx,expected",
    [
        ("(s)I", "saat", "ı"),  # flag off: back harmony (the parallel non-loan sanatı)
        ("lAr", "saat", "lar"),
        ("(s)I", "kabul", "u"),
    ],
)
def test_front_root_flag_off_is_back_harmony(template, ctx, expected):
    # Without the flag the very same context stays back-harmonic, so the flag is the only
    # thing turning saat into a front-harmony root (no leakage into ordinary words).
    assert realize(template, ctx) == expected


@pytest.mark.consistency
def test_front_root_clears_after_first_vowel():
    # The fronting applies only to the FIRST emitted vowel; a following suffix reads the
    # emitted front vowel and harmonizes normally. saat + ler + DA -> saatlerde: the DA is
    # realized off "saatler" (already front), so no flag is needed for it.
    ler = realize("lAr", "saat", front_root=True)
    assert ler == "ler"
    assert realize("DA", "saat" + ler) == "de"  # off "saatler", plain harmony -> de

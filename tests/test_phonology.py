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

"""Turkish-aware normalization: casing, apostrophes, Unicode."""

from __future__ import annotations

import pytest

from ilmek.core.normalization import (
    fold_for_lookup,
    normalize,
    standardize_apostrophes,
    turkish_lower,
    turkish_upper,
)


@pytest.mark.positive
@pytest.mark.parametrize(
    "text,expected",
    [
        ("İSTANBUL", "istanbul"),  # İ -> i (dotted)
        ("IRMAK", "ırmak"),  # I -> ı (dotless)
        ("İYİ", "iyi"),
        ("AĞAÇ", "ağaç"),
        ("ÇOCUK", "çocuk"),
        ("GÜNEŞ", "güneş"),
    ],
)
def test_turkish_lower(text, expected):
    assert turkish_lower(text) == expected


@pytest.mark.positive
@pytest.mark.parametrize(
    "text,expected",
    [
        ("iyi", "İYİ"),  # i -> İ
        ("ırmak", "IRMAK"),  # ı -> I
        ("ağaç", "AĞAÇ"),
        ("güneş", "GÜNEŞ"),
    ],
)
def test_turkish_upper(text, expected):
    assert turkish_upper(text) == expected


@pytest.mark.negative
def test_lower_differs_from_python_default_on_dotless_i():
    # This is the whole point: the locale-naive default is wrong for Turkish.
    assert "I".lower() != turkish_lower("I")
    assert turkish_lower("I") == "ı"


@pytest.mark.positive
@pytest.mark.parametrize("variant", ["’", "‘", "ʼ", "´", "`", "′"])
def test_apostrophe_variants_folded(variant):
    assert standardize_apostrophes(f"Ankara{variant}da") == "Ankara'da"


@pytest.mark.positive
def test_normalize_is_nfc_and_preserves_case():
    out = normalize("İzmir’de")
    assert out == "İzmir'de"  # apostrophe folded, case preserved


@pytest.mark.positive
def test_fold_for_lookup_lowercases_turkishly():
    assert fold_for_lookup("KİTAP") == "kitap"
    assert fold_for_lookup("IŞIK") == "ışık"


@pytest.mark.positive
@pytest.mark.parametrize(
    "text,expected",
    [
        ("kâğıt", "kağıt"),  # â -> a
        ("hâlâ", "hala"),  # two â in one word
        ("âlim", "alim"),  # word-initial â
        ("HÂLÂ", "hala"),  # uppercase Â reaches the fold via turkish_lower
        ("Û", "u"),  # û -> u
        ("târîh", "tarih"),  # î -> i (and â -> a)
    ],
)
def test_fold_for_lookup_folds_circumflex(text, expected):
    # The circumflex (düzeltme işareti) is folded ONLY on the lookup path so a circumflex
    # surface matches its plain lexicon root.
    assert fold_for_lookup(text) == expected


@pytest.mark.negative
@pytest.mark.parametrize("text", ["kâğıt", "hâlâ", "âlim", "Û"])
def test_normalize_preserves_circumflex(text):
    # normalize() must NOT fold the circumflex — the surface/token keeps it; only the lookup
    # key (fold_for_lookup) is folded.
    assert normalize(text) == text

"""Tokenizer: words, numbers, dates, abbreviations, apostrophes, social tokens."""

from __future__ import annotations

import pytest

from ilmek.core.tokenization import tokenize


def kinds(text):
    return [(t.text, t.kind) for t in tokenize(text)]


@pytest.mark.positive
def test_basic_words_and_punct():
    assert kinds("Kediyi gördüm.") == [
        ("Kediyi", "word"),
        ("gördüm", "word"),
        (".", "punct"),
    ]


@pytest.mark.positive
def test_apostrophe_proper_noun_suffix_split():
    (tok,) = tokenize("Ankara'da")
    assert tok.text == "Ankara'da"
    assert tok.kind == "word"
    assert tok.apostrophe_suffix == "da"
    assert tok.stem_text == "Ankara"


@pytest.mark.positive
def test_numbers_dates_times():
    ks = dict(kinds("Saat 09:30, 01.01.2020 tarihinde 3,5 kg aldık."))
    assert ks["09:30"] == "time"
    assert ks["01.01.2020"] == "date"
    assert ks["3,5"] == "number"


@pytest.mark.positive
def test_abbreviation_with_dot_is_one_token():
    ks = kinds("Dr. Ahmet geldi.")
    assert ("Dr.", "abbr") in ks


@pytest.mark.positive
def test_leading_percent_is_one_number_token():
    # Turkish writes the percent sign BEFORE the figure (%25 = "yüzde 25").
    assert kinds("%25 arttı") == [("%25", "number"), ("arttı", "word")]


@pytest.mark.positive
def test_ordinal_dot_is_one_number_token():
    # "3." (üçüncü) keeps the ordinal dot; the following word is separate.
    assert kinds("Fiyat 3. sırada") == [
        ("Fiyat", "word"),
        ("3.", "number"),
        ("sırada", "word"),
    ]


@pytest.mark.positive
def test_dotted_acronym_kept_as_one_abbr():
    # "T.C." must NOT split on its internal periods (previously 4 tokens: T . C .).
    assert kinds("T.C. vatandaşı") == [("T.C.", "abbr"), ("vatandaşı", "word")]


@pytest.mark.negative
def test_decimal_and_grouped_numbers_stay_one_token():
    # The ordinal-dot branch must not split a decimal/grouped figure: the dot/comma is
    # internal (followed by a digit), so "3,5" and "1.000,50" each stay one number token.
    assert kinds("3,5") == [("3,5", "number")]
    assert kinds("1.000,50") == [("1.000,50", "number")]


@pytest.mark.negative
def test_dotted_acronym_does_not_eat_lowercase_word_dot():
    # The dotted-acronym pattern is UPPERCASE-only: a lowercase word + period still splits.
    assert kinds("Geldi kedi.") == [("Geldi", "word"), ("kedi", "word"), (".", "punct")]


@pytest.mark.negative
def test_non_abbreviation_word_then_period_splits():
    # "kedi" is not an abbreviation, so the period is separate.
    ks = kinds("kedi.")
    assert ks == [("kedi", "word"), (".", "punct")]


@pytest.mark.positive
def test_social_media_tokens():
    ks = dict(kinds("@user #haber https://example.com adresinde"))
    assert ks["@user"] == "mention"
    assert ks["#haber"] == "hashtag"
    assert ks["https://example.com"] == "url"


@pytest.mark.positive
def test_offsets_are_correct():
    tokens = tokenize("ev okul")
    assert (tokens[0].start, tokens[0].end) == (0, 2)
    assert (tokens[1].start, tokens[1].end) == (3, 7)

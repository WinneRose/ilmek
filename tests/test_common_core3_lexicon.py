"""Data-driven checks for the first high-yield TDK lexicon expansion batch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ilmek import analyze, lemmatize, stem

_BATCH_PATH = Path(__file__).parents[1] / "ilmek" / "data" / "lexicon" / "common_core3.json"
_BATCH = json.loads(_BATCH_PATH.read_text(encoding="utf-8"))
_ENTRIES = tuple(_BATCH["entries"])


@pytest.mark.positive
@pytest.mark.parametrize(
    "entry",
    _ENTRIES,
    ids=[f"{entry['lemma']}-{entry['pos']}" for entry in _ENTRIES],
)
def test_common_core3_bare_roots_are_lexicon_verified(analyzer, entry):
    assert any(
        result.lemma == entry["lemma"] and result.pos == entry["pos"] and result.source == "lexicon"
        for result in analyzer.analyze(entry["lemma"])
    )


@pytest.mark.consistency
@pytest.mark.parametrize(
    "word,lemma,pos,feature_key,feature_value",
    [
        ("çalışanları", "çalışan", "NOUN", "number", "plural"),
        ("öğrencilerimizin", "öğrenci", "NOUN", "possessive", "1pl"),
        ("yönetiminde", "yönetim", "NOUN", "case", "locative"),
        ("başarılıydı", "başarılı", "ADJ", "copula", "past"),
        ("öğrenecek", "öğren", "VERB", "tense", "future"),
    ],
)
def test_common_core3_roots_survive_inflection_and_api_consistency(
    analyzer, word, lemma, pos, feature_key, feature_value
):
    candidates = analyzer.analyze(word)
    expected = next(
        result
        for result in candidates
        if result.lemma == lemma and result.pos == pos and result.source == "lexicon"
    )
    assert expected.features.get(feature_key) == feature_value
    assert lemmatize(word) == lemma
    assert stem(word) == expected.stem
    assert any(result.lemma == lemma and result.pos == pos for result in analyze(word))


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,features",
    [
        (
            "öğrencilerimizin",
            "öğrenci",
            {"number": "plural", "possessive": "1pl", "case": "genitive"},
        ),
        (
            "çalışanlarından",
            "çalışan",
            {"number": "plural", "possessive": "3sg", "case": "ablative"},
        ),
        (
            "öğrenmeyecek",
            "öğren",
            {"polarity": "negative", "tense": "future", "person": "3sg"},
        ),
    ],
)
def test_common_core3_long_chains_keep_lexicon_root_and_features(analyzer, word, lemma, features):
    result = analyzer.analyze(word)[0]
    assert result.lemma == lemma
    assert result.source == "lexicon"
    for key, value in features.items():
        assert result.features[key] == value


@pytest.mark.consistency
def test_common_core3_does_not_add_unverified_phonology_attributes():
    """The batch must not claim voicing or vowel-drop facts without evidence."""
    for entry in _ENTRIES:
        assert set(entry) == {"lemma", "pos"}


@pytest.mark.negative
@pytest.mark.parametrize("word", ["öğrencilerimizzz", "yönetimindee", "çalışanlarımz"])
def test_common_core3_does_not_turn_near_misspellings_into_lexicon_roots(analyzer, word):
    assert not any(result.source == "lexicon" for result in analyzer.analyze(word))

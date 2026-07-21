"""Public API: stem / lemmatize / analyze / analyze_sentence, schema, provenance."""

from __future__ import annotations

import pytest
from conftest import has_analysis

import ilmek as trnlp


@pytest.mark.positive
def test_top_level_stem_and_lemmatize():
    assert trnlp.stem("kitaplarımızdan") == "kitap"
    assert trnlp.lemmatize("kitaplarımızdan") == "kitap"


@pytest.mark.positive
def test_analyze_returns_list_of_results():
    results = trnlp.analyze("kitaplarımızdan")
    assert isinstance(results, list)
    assert results[0].lemma == "kitap"
    assert results[0].pos == "NOUN"


@pytest.mark.consistency
@pytest.mark.parametrize(
    "word", ["kitaplarımızdan", "evlerimizden", "geliyorum", "gelmeyecekmişsiniz", "ağacı"]
)
def test_stem_lemma_analyze_agree(word):
    best = trnlp.analyze(word)[0]
    # In v0.1 (inflection only) stem == lemma, and all three views agree.
    assert trnlp.stem(word) == best.stem
    assert trnlp.lemmatize(word) == best.lemma
    assert best.stem == best.lemma


@pytest.mark.positive
def test_analysis_result_schema_and_to_dict():
    result = trnlp.analyze("kitaplarımızdan")[0]
    data = result.to_dict()
    assert set(data) >= {
        "surface",
        "lemma",
        "stem",
        "pos",
        "morphemes",
        "features",
        "confidence",
        "backend",
        "source",
        "analyses",
    }
    assert data["surface"] == "kitaplarımızdan"
    assert data["backend"] == "native"
    assert data["source"] == "lexicon"
    # We never fabricate a confidence score.
    assert data["confidence"] is None


@pytest.mark.positive
def test_lexicon_words_are_sourced_from_lexicon(analyzer):
    assert analyzer.analyze("evler")[0].source == "lexicon"


@pytest.mark.negative
def test_unknown_word_is_marked_guess_not_lexicon(analyzer):
    results = analyzer.analyze("fl789zzptlerden")
    assert results[0].source == "guess"
    # A guess must never claim to be a lexicon-verified analysis.
    assert all(r.source != "lexicon" for r in results)


@pytest.mark.positive
def test_proper_noun_apostrophe(analyzer):
    result = analyzer.analyze("Ankara'da")[0]
    assert result.lemma == "ankara"
    assert result.pos == "PROPN"
    assert result.source == "rule"
    assert result.features.get("case") == "locative"
    assert result.surface == "Ankara'da"


@pytest.mark.positive
def test_ambiguous_word_returns_multiple_analyses(analyzer):
    results = analyzer.analyze("yüz")
    pos_tags = {r.pos for r in results}
    # yüz = hundred (NUM) / face (NOUN) / swim! (VERB imperative)
    assert {"NUM", "NOUN", "VERB"} <= pos_tags
    assert all(r.lemma == "yüz" for r in results)


@pytest.mark.positive
def test_analyze_sentence_returns_document():
    doc = trnlp.analyze_sentence("Kitaplarımızı masaya bıraktık.")
    assert doc.lemmas[:3] == ["kitap", "masa", "bırak"]
    # Punctuation token has no analysis.
    assert doc.analyses[-1] is None


@pytest.mark.positive
def test_pipeline_load_is_cached_and_callable():
    nlp = trnlp.load()
    assert nlp is trnlp.load()
    doc = nlp("Denizde yüzüyor.")
    assert has_analysis  # imported symbol available
    assert doc.tokens[0].text == "Denizde"
    assert nlp.analyze("evler").lemma == "ev"


@pytest.mark.positive
def test_token_offsets_align_with_document_text():
    # Token spans must index into Document.text (which is the normalized text).
    doc = trnlp.analyze_sentence("İzmir’de üç kişi vardı.")
    for token in doc.tokens:
        assert doc.text[token.start : token.end] == token.text


@pytest.mark.negative
def test_stray_leading_apostrophe_does_not_make_empty_proper_noun(analyzer):
    # A stray leading apostrophe is NOT a proper-noun boundary: it is dropped and the
    # residual analyzed normally, never yielding an empty-lemma PROPN via the apostrophe rule.
    result = analyzer.analyze("'da")[0]
    assert result.lemma != ""
    assert result.pos != "PROPN"
    assert result.source != "rule"
    # For an out-of-lexicon residual the stripped-apostrophe path still falls through to the
    # guesser, unchanged. (The bare clitic "da" itself became a lexicon function word in the
    # closed-class milestone, so it now resolves as lexicon rather than a guess.)
    oov = analyzer.analyze("'zonklardan")[0]
    assert oov.lemma == "zonk" and oov.source == "guess"


@pytest.mark.positive
def test_denizde_and_yuzuyor_from_the_spec_sentence():
    doc = trnlp.analyze_sentence("Yüz kişi denizde yüzüyor.")
    lemmas = [a.lemma if a else None for a in doc.analyses]
    assert "deniz" in lemmas  # denizde -> deniz (locative)
    assert "yüz" in lemmas  # yüzüyor -> yüz (verb, progressive)

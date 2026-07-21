"""Common adjective/adverb/noun batch: sampled new roots inflect correctly.

This batch (see ``docs`` / the milestone) adds ~90 high-frequency Turkish words across four
data files, with NO engine change — the FSM, phonology, and ek-fiil already handle them:

* ``adjectives.json`` — plain ADJ roots (taze, hasta-class) and a voicing subset whose final
  stop softens before a vowel (yumuşak->yumuşağı, korkak->korkağı, uzak->uzağa).
* ``nouns.json`` — plain NOUN roots plus a voicing subset (oyuncak->oyuncağı, kasap->kasabı).
* ``other.json`` — declinable time adverbs that inflect via the nominal FSM (dün->dünden).
* ``function_words.json`` "irregular" — INDECLINABLE adverbs enumerated whole (bazen), so a
  run-on like *bazenler stays an honest guess instead of a lexicon-verified plural.

Per the testing contract the sample proves each flag class in real Turkish: ``voicing``
softens the final stop ONLY before a vowel suffix, monosyllabic / loan stop-final roots left
*unflagged* (tok, sert, market, alt, üst) must NOT voice, and the newly lexicalized whole
words (akıllı, tuzlu, sessiz) coexist with their still-valid derived split (akıl+lI).
"""

from __future__ import annotations

import json
from collections import Counter

import pytest
from conftest import has_analysis

import ilmek
from ilmek.morphology.lexicon import _DATA_DIR


def _sourced_from_lexicon(analyzer, word, lemma):
    return any(a.lemma == lemma and a.source == "lexicon" for a in analyzer.analyze(word))


# --- Data integrity: no accidental duplicate regular (lemma, pos) entries ------------


@pytest.mark.consistency
def test_no_duplicate_regular_lemma_pos_entries():
    # Guards this batch (and future ones) against re-adding a word that already exists.
    # Scoped to REGULAR root entries: the ``irregular`` lists intentionally repeat a
    # (lemma, pos) across harmony/spelling variants (mi/mı/mu/mü, hala/hâlâ).
    pairs: Counter[tuple[str, str]] = Counter()
    for path in sorted(_DATA_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data.get("entries", ()) if isinstance(data, dict) else data
        for entry in entries:
            pairs[(entry["lemma"], entry.get("pos", "NOUN").upper())] += 1
    duplicates = {pair: n for pair, n in pairs.items() if n > 1}
    assert not duplicates, f"duplicate (lemma, pos) regular entries: {duplicates}"


# --- Positive: bare new roots resolve, lexicon-verified ------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,pos",
    [
        ("taze", "taze", "ADJ"),
        ("hasta", "hasta", "ADJ"),  # pre-existing, pinned here as the milestone headline
        ("tok", "tok", "ADJ"),
        ("pahalı", "pahalı", "ADJ"),
        ("çalışkan", "çalışkan", "ADJ"),
        ("kardeş", "kardeş", "NOUN"),
        ("teyze", "teyze", "NOUN"),
        ("market", "market", "NOUN"),
        ("bilgisayar", "bilgisayar", "NOUN"),
        ("dün", "dün", "ADV"),
    ],
)
def test_new_root_resolves_from_lexicon(analyzer, word, lemma, pos):
    assert has_analysis(analyzer, word, lemma=lemma, pos=pos)
    assert _sourced_from_lexicon(analyzer, word, lemma)


# --- Positive: voicing roots soften the final stop before a vowel suffix -------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,features",
    [
        ("yumuşağı", "yumuşak", {"case": "accusative"}),  # k -> ğ
        ("korkağı", "korkak", {"case": "accusative"}),
        ("kalabalığı", "kalabalık", {"case": "accusative"}),
        ("gerçeği", "gerçek", {"case": "accusative"}),
        ("gevşeği", "gevşek", {"case": "accusative"}),
        ("parlağı", "parlak", {"case": "accusative"}),
        ("sönüğü", "sönük", {"case": "accusative"}),
        ("çıplağı", "çıplak", {"case": "accusative"}),
        ("uzağa", "uzak", {"case": "dative"}),
        ("oyuncağı", "oyuncak", {"case": "accusative"}),  # NOUN, k -> ğ
        ("kasabı", "kasap", {"case": "accusative"}),  # p -> b
        ("şimşeği", "şimşek", {"case": "accusative"}),
    ],
)
def test_new_voicing_roots_soften_before_vowel(analyzer, word, lemma, features):
    assert has_analysis(analyzer, word, lemma=lemma, features=features)


# --- Negative: the voiced bound form never appears before a consonant suffix ---------


@pytest.mark.negative
def test_voicing_root_keeps_stop_before_consonant_suffix(analyzer):
    # -DA is consonant-initial: the stop stays (oyuncakta), the voiced form is invalid.
    assert has_analysis(analyzer, "oyuncakta", lemma="oyuncak", features={"case": "locative"})
    assert not has_analysis(analyzer, "oyuncakda", lemma="oyuncak")
    # ek-fiil past on an ADJ predicate is consonant-initial too: yumuşaktı, never yumuşağdı.
    assert has_analysis(analyzer, "yumuşaktı", lemma="yumuşak", features={"copula": "past"})
    assert not has_analysis(analyzer, "yumuşağdı", lemma="yumuşak")


# --- Negative: unflagged stop-final roots must NOT voice -----------------------------


@pytest.mark.negative
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("toğu", "tok"),  # monosyllabic -k never voices
        ("serdi", "sert"),  # serdi is the verb ser-+past, not sert voicing
        ("markedi", "market"),  # loanword -t stays t
    ],
)
def test_unflagged_stop_final_roots_do_not_false_voice(analyzer, word, lemma):
    assert not has_analysis(analyzer, word, lemma=lemma)


# --- Positive: unflagged stop-final roots keep the stop before a vowel suffix --------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("toku", "tok"),  # tok -> toku (k stays)
        ("serti", "sert"),  # sert -> serti (t stays)
        ("marketi", "market"),  # market -> marketi (loan t stays)
        ("altı", "alt"),  # alt -> altı (monosyllabic t stays)
        ("üstü", "üst"),  # üst -> üstü
    ],
)
def test_unflagged_stop_final_roots_keep_stop(analyzer, word, lemma):
    assert has_analysis(analyzer, word, lemma=lemma)


# --- Positive: the ek-fiil (copula) works on the new ADJ predicates ------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,features",
    [
        ("tazeyim", "taze", {"person": "1sg"}),  # y-buffer, zero-copula present
        ("ucuzdu", "ucuz", {"copula": "past"}),
        ("sertti", "sert", {"copula": "past"}),
    ],
)
def test_ek_fiil_on_new_adjectives(analyzer, word, lemma, features):
    assert has_analysis(analyzer, word, lemma=lemma, pos="ADJ", features=features)


# --- Positive: plain nominal inflection on the new roots -----------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,features",
    [
        ("yaşlılar", "yaşlı", {"number": "plural"}),
        ("kardeşim", "kardeş", {"possessive": "1sg"}),
        ("teyzesi", "teyze", {"possessive": "3sg"}),  # s-buffer after vowel stem
        ("bilgisayarlardan", "bilgisayar", {"number": "plural", "case": "ablative"}),
    ],
)
def test_new_noun_inflection(analyzer, word, lemma, features):
    assert has_analysis(analyzer, word, lemma=lemma, features=features)


# --- Positive: declinable time adverbs inflect via the nominal FSM -------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [("dünden", "dün"), ("yarından", "yarın"), ("sonradan", "sonra"), ("erkenden", "erken")],
)
def test_declinable_adverbs_take_ablative(analyzer, word, lemma):
    assert has_analysis(analyzer, word, lemma=lemma, pos="ADV", features={"case": "ablative"})


# --- Positive: indeclinable adverbs are whole-surface lexicon entries ----------------


@pytest.mark.positive
@pytest.mark.parametrize("surface", ["bazen", "nadiren", "yalnızca", "özellikle", "birlikte"])
def test_indeclinable_adverbs_are_lexicon(analyzer, surface):
    best = analyzer.analyze(surface)[0]
    assert best.lemma == surface
    assert best.pos == "ADV"
    assert best.source == "lexicon"
    assert best.morphemes == []  # indeclinable: no suffixes
    assert best.stem == surface


# --- Negative: indeclinable adverbs do not inflect via the lexicon -------------------


@pytest.mark.negative
@pytest.mark.parametrize("word,lemma", [("bazenler", "bazen"), ("özellikleyi", "özellikle")])
def test_forced_inflection_on_indeclinable_adverbs_not_lexicon(analyzer, word, lemma):
    # A guesser strip may coincidentally reach the base lemma, but only as source=guess:
    # the frozen adverb licenses no lexicon-verified plural/accusative.
    assert not any(a.source == "lexicon" and a.lemma == lemma for a in analyzer.analyze(word))


# --- Exception / ambiguity: homographs coexist, none is erased -----------------------


@pytest.mark.exception
def test_alti_numeral_and_alt_possessive_coexist(analyzer):
    # "altı" = the numeral six AND alt+3sg-possessive ("its bottom"): both must survive.
    assert has_analysis(analyzer, "altı", lemma="altı", pos="NUM")
    assert has_analysis(analyzer, "altı", lemma="alt", features={"possessive": "3sg"})
    assert has_analysis(analyzer, "altı", lemma="alt", features={"case": "accusative"})


@pytest.mark.exception
def test_siki_adjective_and_sik_accusative_coexist(analyzer):
    # "sıkı" = the ADJ "tight" AND sık+accusative: both readings are valid.
    assert has_analysis(analyzer, "sıkı", lemma="sıkı", pos="ADJ")
    assert has_analysis(analyzer, "sıkı", lemma="sık", features={"case": "accusative"})


@pytest.mark.exception
@pytest.mark.parametrize(
    "word,root_lemma,derived_lemma,derivation",
    [
        ("akıllı", "akıllı", "akıl", ("li",)),
        ("tuzlu", "tuzlu", "tuz", ("li",)),
        ("sessiz", "sessiz", "ses", ("siz",)),
    ],
)
def test_lexicalized_whole_word_and_derived_split_coexist(
    analyzer, word, root_lemma, derived_lemma, derivation
):
    # The new whole-word root becomes primary (elli/güçlü precedent) but the derived
    # X+lI / X+sIz split is never lost.
    results = analyzer.analyze(word)
    assert results[0].lemma == root_lemma
    assert "derivation" not in results[0].features
    assert any(
        a.lemma == derived_lemma and a.features.get("derivation") == derivation for a in results
    )


# --- Consistency: stem / lemma / analyze agree on a sampled root ---------------------


@pytest.mark.consistency
def test_stem_lemma_analyze_agree_for_batch():
    # Bare adjective: three views of one analysis agree, and it is inflection-free.
    assert ilmek.lemmatize("taze") == "taze"
    assert ilmek.stem("taze") == "taze"
    best = ilmek.analyze("taze")[0]
    assert best.stem == best.lemma == "taze"
    # A voiced accusative reduces to the free lemma across all three views.
    assert ilmek.lemmatize("yumuşağı") == "yumuşak"
    assert ilmek.stem("yumuşağı") == "yumuşak"

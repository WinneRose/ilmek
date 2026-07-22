"""Three cheap, fully-productive suffixes: the equative/adverbial -CA (güzelce), the relational
-sAl (toplumsal), and the ordinal -(I)ncI (birinci).

* -CA turns a NOUN/ADJ into an ADVERB (manner/equative). The C archiphoneme hardens to ç after
  a voiceless consonant for free (çocukça). It lands in a TERMINAL state (no *güzelceyi), and it
  fires on NOUN/ADJ only (not NUM: *birce, not VERB: *gelce). The pronominal equatives
  (bence, sence) and the language name türkçe are lexicalized instead.
* -sAl turns a NOUN into an ADJECTIVE (toplumsal), inflecting and hosting the ek-fiil for free
  (toplumsaldır). It fires on NOUN only (*güzelsel, *gelsel).
* -(I)ncI is an INFLECTIONAL ordinal on a NUM (birinci -> lemma/stem bir, num_type=ordinal),
  plus the curated ADJ "son" -> sonuncu (attribute-gated). It never fires on a NOUN/ADJ
  (*evinci) nor for the guesser.
"""

from __future__ import annotations

import pytest
from conftest import has_analysis

import ilmek


def _lemma_sourced(analyzer, word, lemma, *, source="lexicon"):
    return any(a.lemma == lemma and a.source == source for a in analyzer.analyze(word))


# =====================================================================================
# 1. EQUATIVE / ADVERBIAL -CA (güzelce, çocukça, bence, türkçe)
# =====================================================================================


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("güzelce", "güzel"),  # ADJ -> ADV
        ("hızlıca", "hızlı"),
        ("çocukça", "çocuk"),  # pins ç-hardening after voiceless k
        ("insanca", "insan"),  # NOUN -> ADV
        ("aptalca", "aptal"),
    ],
)
def test_ca_positive(analyzer, word, lemma):
    assert has_analysis(analyzer, word, lemma=lemma, pos="ADV", features={"derivation": ("ca",)})
    assert _lemma_sourced(analyzer, word, lemma)


@pytest.mark.positive
def test_ca_adverb_has_no_fabricated_nominal_features(analyzer):
    # güzelce is an adverb: it carries ONLY the derivation, no fabricated case/number/possessive.
    a = next(r for r in analyzer.analyze("güzelce") if r.pos == "ADV")
    assert "case" not in a.features
    assert "number" not in a.features
    assert "possessive" not in a.features
    assert a.stem == "güzelce"


@pytest.mark.positive
@pytest.mark.parametrize("word,lemma", [("bence", "ben"), ("sence", "sen"), ("bizce", "biz")])
def test_ca_pronominal_equatives(analyzer, word, lemma):
    # bence/sence/bizce are enumerated (the pronoun base is suppletive), pos ADV, derivation ca.
    assert has_analysis(
        analyzer, word, lemma=lemma, pos="ADV", morphemes=["ce"], features={"derivation": ("ca",)}
    )


@pytest.mark.negative
@pytest.mark.parametrize("word", ["gelce", "birce", "güzelceyi", "güzelceler"])
def test_ca_does_not_overgenerate(analyzer, word):
    # gelce: VERB (never reaches the deriv slot); birce: NUM (excluded from applies_to);
    # güzelceyi/güzelceler: N_ADV_CA is terminal, so no case/plural.
    assert not any(
        a.source == "lexicon" and a.features.get("derivation") == ("ca",)
        for a in analyzer.analyze(word)
    )


@pytest.mark.exception
def test_turkce_language_name_is_primary(analyzer):
    # türkçe is a lexicalized NOUN (language names inflect: türkçeyi -> türkçe), primary over any
    # productive reading by the longer-lemma sort.
    assert analyzer.analyze("türkçe")[0].lemma == "türkçe"
    assert analyzer.analyze("türkçe")[0].pos == "NOUN"
    assert has_analysis(analyzer, "türkçeyi", lemma="türkçe", pos="NOUN")


@pytest.mark.exception
def test_karinca_lexicalized_over_ca_split(analyzer):
    # karınca (ant) is a whole-word NOUN; the karın+ca ADV parse never wins for it.
    results = analyzer.analyze("karınca")
    assert results[0].lemma == "karınca"
    assert results[0].pos == "NOUN"
    # ...but the productive split survives as a ranked alternative.
    assert any(a.lemma == "karın" and a.features.get("derivation") == ("ca",) for a in results)


# =====================================================================================
# 2. RELATIONAL -sAl (toplumsal, bölgesel)
# =====================================================================================


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("toplumsal", "toplum"),
        ("kimyasal", "kimya"),
        ("bölgesel", "bölge"),  # front harmony: -sEl
        ("tarihsel", "tarih"),
        ("tarımsal", "tarım"),
    ],
)
def test_sal_positive(analyzer, word, lemma):
    assert has_analysis(analyzer, word, lemma=lemma, pos="ADJ", features={"derivation": ("sal",)})
    assert _lemma_sourced(analyzer, word, lemma)


@pytest.mark.positive
def test_sal_hosts_ekfiil(analyzer):
    # toplumsaldır = toplum + sal + DIr: the derived ADJ takes the assertive copula (via N_DERIV).
    assert has_analysis(
        analyzer, "toplumsaldır", lemma="toplum", pos="ADJ", features={"copula": "assertive"}
    )


@pytest.mark.negative
@pytest.mark.parametrize("word", ["güzelsel", "gelsel", "vebasal"])
def test_sal_does_not_overgenerate(analyzer, word):
    # güzelsel: ADJ (applies_to={NOUN}); gelsel: VERB; vebasal: OOV guess (derivation off).
    assert not any(
        a.source == "lexicon" and a.features.get("derivation") == ("sal",)
        for a in analyzer.analyze(word)
    )


@pytest.mark.exception
def test_kumsal_lexicalized_over_sal_split(analyzer):
    # kumsal (sandy beach) is a whole-word NOUN, primary; the kum+sal ADJ split stays an alt.
    results = analyzer.analyze("kumsal")
    assert results[0].lemma == "kumsal"
    assert results[0].pos == "NOUN"
    assert any(a.lemma == "kum" and a.features.get("derivation") == ("sal",) for a in results)


# =====================================================================================
# 3. ORDINAL -(I)ncI (birinci, dördüncü, sonuncu)
# =====================================================================================


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("birinci", "bir"),  # (I) linking vowel after a consonant
        ("ikinci", "iki"),  # no linking vowel after a vowel
        ("üçüncü", "üç"),
        ("dördüncü", "dört"),  # voicing bound form (dörd)
        ("beşinci", "beş"),
        ("onuncu", "on"),
        ("yüzüncü", "yüz"),  # NUM reading
    ],
)
def test_ordinal_positive(analyzer, word, lemma):
    assert has_analysis(analyzer, word, lemma=lemma, pos="NUM", features={"num_type": "ordinal"})
    assert _lemma_sourced(analyzer, word, lemma)


@pytest.mark.positive
def test_ordinal_is_inflectional_lemma_equals_stem(analyzer):
    # The ordinal is INFLECTIONAL: lemma == stem == the bare numeral (no derivation recorded).
    a = next(r for r in analyzer.analyze("birinci") if r.features.get("num_type") == "ordinal")
    assert a.lemma == "bir"
    assert a.stem == "bir"
    assert "derivation" not in a.features


@pytest.mark.positive
def test_ordinal_further_inflection(analyzer):
    assert has_analysis(analyzer, "birincisi", lemma="bir", features={"possessive": "3sg"})
    assert has_analysis(analyzer, "ikincide", lemma="iki", features={"case": "locative"})
    assert has_analysis(analyzer, "birinciydi", lemma="bir", features={"copula": "past"})


@pytest.mark.positive
def test_ordinal_sonuncu_attribute_gated(analyzer):
    # son is an ADJ, not a numeral, so it needs the curated ordinal_host attribute (ORD_SON).
    assert has_analysis(analyzer, "sonuncu", lemma="son", features={"num_type": "ordinal"})
    assert _lemma_sourced(analyzer, "sonuncu", "son")


@pytest.mark.negative
@pytest.mark.parametrize("word", ["evinci", "güzelinci", "zorbinci"])
def test_ordinal_does_not_overgenerate(analyzer, word):
    # evinci: NOUN (applies_to={NUM}); güzelinci: ADJ without ordinal_host; zorbinci: OOV guess
    # (the guesser synthesizes only NOUN/VERB roots, so the {NUM} ordinal never fires).
    assert not any(a.features.get("num_type") == "ordinal" for a in analyzer.analyze(word))


# =====================================================================================
# 4. Consistency
# =====================================================================================


@pytest.mark.consistency
@pytest.mark.parametrize(
    "word,lemma",
    [("güzelce", "güzel"), ("toplumsal", "toplum"), ("birinci", "bir"), ("dördüncü", "dört")],
)
def test_ca_sal_ordinal_lemma_analyze_agree(word, lemma):
    assert ilmek.lemmatize(word) == lemma
    assert ilmek.analyze(word)[0].lemma == lemma

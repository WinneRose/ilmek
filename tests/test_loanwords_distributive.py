"""Distributive numeral -(ş)Ar, loanword exceptions (gemination, front-harmony), and the
common-word fill (incl. the noun "anı").

Three independent milestone pieces, one file (all data + a tiny declarative engine change):

* Distributive -(ş)Ar: an inflectional numeral suffix ("n each"). The ş buffer appears after
  a vowel (ikişer, altışar) and is absent after a consonant (birer, onar). It is NOT
  derivational, so stem/lemma stay the bare numeral (birer -> bir). ``applies_to={NUM}`` is
  the overgeneration guard: no *ever from ev, no OOV zom+ar.
* Gemination: a loanword's final consonant doubles before a vowel suffix (hakkı, affı, reddi);
  ret also voices as it doubles. Before a consonant suffix the single free form is kept (hakta).
* Front-harmony loans: a back-spelled loan harmonizes as if front (saati, kalbe, rolü, kabulü,
  usulü); kalp also voices (kalbi) but keeps its stop before a consonant (kalpte).
* Fill: ~95 high-frequency words, headlined by "anı" (memory) so anıydı -> anı via the ek-fiil.
"""

from __future__ import annotations

import pytest
from conftest import has_analysis

import ilmek
from ilmek.morphology import morphotactics as mt


def _lemma_sourced(analyzer, word, lemma, *, source="lexicon"):
    return any(a.lemma == lemma and a.source == source for a in analyzer.analyze(word))


# =====================================================================================
# 1. DISTRIBUTIVE -(ş)Ar
# =====================================================================================


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("birer", "bir"),  # consonant-final: no ş buffer
        ("ikişer", "iki"),  # vowel-final: ş buffer
        ("üçer", "üç"),  # üç does NOT voice (üçü) -> üçer, not *üçğer
        ("dörder", "dört"),  # root voicing: dörd + er
        ("beşer", "beş"),
        ("altışar", "altı"),  # ş buffer after vowel, back harmony
        ("yedişer", "yedi"),
        ("sekizer", "sekiz"),
        ("dokuzar", "dokuz"),
        ("onar", "on"),
        ("biner", "bin"),
        ("ellişer", "elli"),
    ],
)
def test_distributive_positive(analyzer, word, lemma):
    assert has_analysis(
        analyzer, word, lemma=lemma, pos="NUM", features={"num_type": "distributive"}
    )
    assert _lemma_sourced(analyzer, word, lemma)


@pytest.mark.negative
@pytest.mark.parametrize("word", ["ever", "üçar", "altıar", "oner"])
def test_distributive_does_not_overgenerate(analyzer, word):
    # "ever" (ev is a NOUN, applies_to=NUM blocks it), wrong-harmony "üçar"/"oner", and the
    # buffer-less "altıar": none may parse as a lexicon distributive.
    assert not any(
        a.source == "lexicon" and a.features.get("num_type") == "distributive"
        for a in analyzer.analyze(word)
    )


@pytest.mark.negative
def test_distributive_never_from_verb_aorist(analyzer):
    # koşar is only the verb aorist koş+ar; it has no NUM / distributive reading.
    results = analyzer.analyze("koşar")
    assert all(a.features.get("num_type") != "distributive" for a in results)
    assert not any(a.pos == "NUM" for a in results)


@pytest.mark.negative
def test_distributive_guesser_byte_identity(analyzer):
    # An OOV like "zomar" must NOT be split as zom + Ar distributive (the guesser synthesizes
    # only NOUN/VERB roots, never NUM), so it stays an honest guess.
    results = analyzer.analyze("zomar")
    assert results[0].source == "guess"
    assert all(a.features.get("num_type") != "distributive" for a in results)


@pytest.mark.exception
def test_yuzer_keeps_both_num_and_verb_readings(analyzer):
    # yüzer = yüz(NUM) distributive AND yüz(VERB) aorist ("swims"): neither erased.
    results = analyzer.analyze("yüzer")
    assert any(a.pos == "NUM" and a.features.get("num_type") == "distributive" for a in results)
    assert any(a.pos == "VERB" and a.features.get("tense") == "aorist" for a in results)
    assert all(a.lemma == "yüz" for a in results)


@pytest.mark.exception
@pytest.mark.xfail(
    reason="ş-buffer after a consonant is a lexical exception (yarımşar); yarım is not in "
    "the lexicon and the productive rule inserts ş only after a vowel."
)
def test_yarimsar_distributive_deferred(analyzer):
    assert has_analysis(analyzer, "yarımşar", lemma="yarım", features={"num_type": "distributive"})


@pytest.mark.consistency
@pytest.mark.parametrize("word,lemma", [("ikişer", "iki"), ("dörder", "dört"), ("birer", "bir")])
def test_distributive_stem_lemma_analyze_agree(word, lemma):
    assert ilmek.stem(word) == lemma
    assert ilmek.lemmatize(word) == lemma
    assert ilmek.analyze(word)[0].lemma == lemma


# =====================================================================================
# 2. GEMINATION (hak -> hakkı, ret -> reddi)
# =====================================================================================


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("hakkı", "hak"),
        ("hakka", "hak"),
        ("affı", "af"),
        ("reddi", "ret"),
        ("hissi", "his"),
        ("haccı", "hac"),
        ("zannı", "zan"),
    ],
)
def test_gemination_positive(analyzer, word, lemma):
    assert has_analysis(analyzer, word, lemma=lemma, pos="NOUN")
    assert _lemma_sourced(analyzer, word, lemma)


@pytest.mark.positive
def test_gemination_accusative_and_possessive_readings(analyzer):
    # hakkı is both the accusative (hak+ı) and the 3sg-possessive (hak+ı): both survive.
    assert has_analysis(analyzer, "hakkı", lemma="hak", features={"case": "accusative"})
    assert has_analysis(analyzer, "hakkı", lemma="hak", features={"possessive": "3sg"})


@pytest.mark.positive
@pytest.mark.parametrize("word", ["hakkında", "hakkımızdan", "haklar"])
def test_gemination_long_chains(analyzer, word):
    # poss3sg + pronominal locative, poss1pl + ablative, and the free form before a consonant
    # suffix (haklar) all reduce to hak.
    assert has_analysis(analyzer, word, lemma="hak")


@pytest.mark.negative
def test_gemination_single_consonant_before_vowel_impossible(analyzer):
    # A single k before a vowel is unrepresentable: "hakı" has no lexicon analysis as hak.
    assert not _lemma_sourced(analyzer, "hakı", "hak")


@pytest.mark.negative
def test_gemination_locative_keeps_single_free_form(analyzer):
    # The locative is consonant-initial -> single free form: hakta parses, hakkta/hakda do not.
    assert has_analysis(analyzer, "hakta", lemma="hak", features={"case": "locative"})
    assert not _lemma_sourced(analyzer, "hakkta", "hak")
    assert not _lemma_sourced(analyzer, "hakda", "hak")


@pytest.mark.negative
@pytest.mark.parametrize("word,lemma", [("redi", "ret"), ("toppu", "top"), ("hakı", "hak")])
def test_gemination_only_on_flagged_roots(analyzer, word, lemma):
    # An unattributed root never geminates (top -> topu, not *toppu); a geminating root's
    # single-consonant surface is never a lexicon parse.
    assert not _lemma_sourced(analyzer, word, lemma)


@pytest.mark.exception
def test_gemination_voicing_split(analyzer):
    # ret both geminates AND voices, so the accusative is reddi and the non-geminated "reti"
    # is impossible; hak geminates WITHOUT voicing (hakkı, never *hağğı) — a paired assertion
    # over the two gemination sub-classes. (Note "retti" IS valid — ret + ek-fiil copula
    # past, "it was a veto" — so the discriminating negative is the single-consonant "reti".)
    assert _lemma_sourced(analyzer, "reddi", "ret")
    assert not _lemma_sourced(analyzer, "reti", "ret")
    assert _lemma_sourced(analyzer, "hakkı", "hak")
    assert not _lemma_sourced(analyzer, "hağğı", "hak")


@pytest.mark.consistency
def test_gemination_stem_lemma_analyze_agree(analyzer):
    assert ilmek.stem("hakkı") == "hak"
    assert ilmek.lemmatize("hakkı") == "hak"
    assert ilmek.analyze("hakkı")[0].lemma == "hak"
    assert ilmek.lemmatize("reddi") == "ret"


# =====================================================================================
# 3. FRONT-HARMONY LOANS (saat, kalp, rol, kabul, usul) + control (hukuk)
# =====================================================================================


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,features",
    [
        ("kalbi", "kalp", {"case": "accusative"}),  # voice + front
        ("kalbe", "kalp", {"case": "dative"}),
        ("kalpte", "kalp", {"case": "locative"}),  # no voicing before consonant
        ("rolü", "rol", {"case": "accusative"}),
        ("role", "rol", {"case": "dative"}),
        ("roller", "rol", {"number": "plural"}),
        ("saati", "saat", {"case": "accusative"}),
        ("saate", "saat", {"case": "dative"}),
        ("saatte", "saat", {"case": "locative"}),
        ("saatler", "saat", {"number": "plural"}),
        ("saatlerde", "saat", {"number": "plural", "case": "locative"}),
        ("hukuku", "hukuk", {"case": "accusative"}),  # control: back harmony, no voicing
        ("hukuka", "hukuk", {"case": "dative"}),
        ("kabulü", "kabul", {"case": "accusative"}),
        ("usulü", "usul", {"case": "accusative"}),
        ("usulüne", "usul", {"case": "dative"}),  # poss3sg + pronominal dative chain
    ],
)
def test_front_harmony_loans_positive(analyzer, word, lemma, features):
    assert has_analysis(analyzer, word, lemma=lemma, features=features)
    assert _lemma_sourced(analyzer, word, lemma)


@pytest.mark.positive
def test_meshgul_front_ekfiil(analyzer):
    # The ADJ loan meşgul fronts under the zero-copula/ek-fiil: meşgulüm, meşguldü.
    assert has_analysis(analyzer, "meşgulüm", lemma="meşgul", pos="ADJ", features={"person": "1sg"})
    assert has_analysis(
        analyzer, "meşguldü", lemma="meşgul", pos="ADJ", features={"copula": "past"}
    )


@pytest.mark.negative
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("saatı", "saat"),  # back harmony forbidden on a front loan
        ("rolu", "rol"),
        ("kabulu", "kabul"),
        ("usulu", "usul"),
        ("hukuğu", "hukuk"),  # no voicing flag -> k never softens
        ("kalpı", "kalp"),  # voicing mandatory before a vowel (must be kalbi)
        ("kalbta", "kalp"),  # ...and never before a consonant (must be kalpte)
    ],
)
def test_front_harmony_wrong_forms_absent(analyzer, word, lemma):
    assert not _lemma_sourced(analyzer, word, lemma)


@pytest.mark.exception
def test_front_harmony_contrast_with_back_loans(analyzer):
    # saat fronts, but the phonologically-parallel, already-present sanat stays BACK harmony;
    # kalp voices AND fronts while the parallel park does NEITHER.
    assert has_analysis(analyzer, "sanatı", lemma="sanat", features={"case": "accusative"})
    assert not _lemma_sourced(analyzer, "sanati", "sanat")
    assert has_analysis(analyzer, "parkı", lemma="park", features={"case": "accusative"})
    assert not _lemma_sourced(analyzer, "parğı", "park")


@pytest.mark.consistency
@pytest.mark.parametrize("word,lemma", [("kalbi", "kalp"), ("saatte", "saat"), ("usulüne", "usul")])
def test_front_harmony_stem_lemma_analyze_agree(word, lemma):
    assert ilmek.stem(word) == lemma
    assert ilmek.lemmatize(word) == lemma
    assert ilmek.analyze(word)[0].lemma == lemma


# =====================================================================================
# 4. COMMON-WORD FILL / anı
# =====================================================================================


@pytest.mark.positive
def test_ani_is_a_noun(analyzer):
    # anı (memory) resolves as a NOUN, primary (its lemma is longer than the moment noun "an").
    results = analyzer.analyze("anı")
    assert results[0].lemma == "anı"
    assert results[0].pos == "NOUN"
    assert results[0].features.get("case") == "nominative"


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,features",
    [
        ("anıydı", {"copula": "past"}),  # milestone headline: ek-fiil past
        ("anılar", {"number": "plural"}),
        ("anısı", {"possessive": "3sg"}),  # s-buffer after a vowel stem
        ("anılarımızda", {"number": "plural", "possessive": "1pl", "case": "locative"}),
    ],
)
def test_ani_inflects(analyzer, word, features):
    assert has_analysis(analyzer, word, lemma="anı", pos="NOUN", features=features)


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,features",
    [
        ("dostu", "dost", {"case": "accusative"}),
        ("suçu", "suç", {"case": "accusative"}),  # NO voicing (suçu, not *sucu-as-suç)
        ("göçü", "göç", {"case": "accusative"}),
        ("tarihi", "tarih", {"case": "accusative"}),
        ("avukatı", "avukat", {"case": "accusative"}),  # loan -t, no voicing
        ("şirketi", "şirket", {"case": "accusative"}),
        ("müziği", "müzik", {"case": "accusative"}),  # loan -k that DOES voice
    ],
)
def test_fill_nouns_inflect(analyzer, word, lemma, features):
    assert has_analysis(analyzer, word, lemma=lemma, pos="NOUN", features=features)
    assert _lemma_sourced(analyzer, word, lemma)


@pytest.mark.positive
def test_fill_verb_bin_aorist(analyzer):
    # biner has BOTH the distributive numeral (bin=1000) and the verb aorist (bin-=to board).
    assert has_analysis(analyzer, "biner", lemma="bin", pos="VERB", features={"tense": "aorist"})


@pytest.mark.negative
@pytest.mark.parametrize("word,lemma", [("sucu", "suç"), ("göcü", "göç")])
def test_fill_non_voicing_roots(analyzer, word, lemma):
    # suç/göç do not voice, so their voiced bound forms are never lexicon parses of them.
    # ("sucu" is still a valid parse of su+CI, but its lemma is "su", not "suç".)
    assert not _lemma_sourced(analyzer, word, lemma)


@pytest.mark.negative
def test_aniydi_has_no_accusative_copula_reading(analyzer):
    # anıydı is anı+copula or an+poss3sg+copula, never an+ACCUSATIVE+copula (N_ACC is terminal).
    for a in analyzer.analyze("anıydı"):
        assert not (a.features.get("case") == "accusative" and "copula" in a.features)


@pytest.mark.exception
def test_ani_ambiguity_order_pinned(analyzer):
    # analyze("anı") = anı(NOUN nom, primary) / an+3sg-poss / an+accusative — all three present,
    # the longer-lemma noun leading, and the two "an" readings retained in generation order.
    results = analyzer.analyze("anı")
    assert results[0].lemma == "anı" and results[0].pos == "NOUN"
    an_poss = next(
        (
            i
            for i, a in enumerate(results)
            if a.lemma == "an" and a.features.get("possessive") == "3sg"
        ),
        None,
    )
    an_acc = next(
        (
            i
            for i, a in enumerate(results)
            if a.lemma == "an" and a.features.get("case") == "accusative"
        ),
        None,
    )
    assert an_poss is not None and an_acc is not None
    assert an_poss < an_acc  # possessive before accusative (generation order)


@pytest.mark.exception
@pytest.mark.parametrize(
    "word,root_lemma,derived_lemma",
    [("sağlığı", "sağlık", "sağ"), ("hastalığı", "hastalık", "hasta")],
)
def test_lexicalized_lik_word_and_derived_split_coexist(analyzer, word, root_lemma, derived_lemma):
    # The whole-word -lIk noun is primary (longer lemma), but the derived X+lIk split survives.
    results = analyzer.analyze(word)
    assert results[0].lemma == root_lemma
    assert any(
        a.lemma == derived_lemma and a.features.get("derivation") == ("lik",) for a in results
    )


@pytest.mark.consistency
def test_ani_stem_lemma_analyze_agree(analyzer):
    assert ilmek.stem("anıydı") == "anı"
    assert ilmek.lemmatize("anıydı") == "anı"
    assert ilmek.analyze("anı")[0].lemma == "anı"


# =====================================================================================
# 5. STRUCTURAL INVARIANT: applies_to is only used as a POS guard, never silently on a
#    plain inflectional suffix (the safety net for lifting the check out of the
#    derivational branch in the analyzer).
# =====================================================================================


@pytest.mark.consistency
def test_applies_to_only_on_derivations_and_distributive():
    seen = {}
    for edges in mt.GRAPH.values():
        for suffix, _ in edges:
            seen[id(suffix)] = suffix
    for suffix in seen.values():
        if suffix.applies_to is not None:
            # Every applies_to guard belongs to a derivation OR to an inflectional numeral
            # suffix (the distributive -(ş)Ar or the ordinal -(I)ncI, both carrying a num_type).
            assert suffix.derivational or suffix.features.get("num_type") in (
                "distributive",
                "ordinal",
            ), f"non-derivational, non-numeral suffix {suffix.name!r} carries applies_to"

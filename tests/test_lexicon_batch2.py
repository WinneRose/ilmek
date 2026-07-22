"""Batch-2 aggressive lexicon expansion: ~550 high-frequency roots across four data files.

Adds ilmek/data/lexicon/{nouns_core2,verbs_core2,adjectives_core2}.json plus targeted
appends to loanwords.json (zam/hat/sır/üs/tıp gemination, hal front-harmony) and
function_words.json (frozen adverbs herhalde/aslında/genellikle/ayrıca/gerçekten/...). No
engine change — the nominal/verbal FSM, phonology, ek-fiil, gemination and front-harmony
machinery already handle every entry.

Per the testing contract the sample proves each flag class in real Turkish:
* ``voicing`` softens the final stop ONLY before a vowel suffix (sorumluluk->sorumluluğu,
  süreç->süreci, talep->talebi), and t-/f-final loans left *unflagged* (hükümet, cumhuriyet,
  şart, taraf) must NOT voice — a wrong flag would silently license *hükümedi;
* ``gemination`` (zam->zammı) makes the single-consonant vowel form (zamı) unrepresentable
  while the consonant-suffix form (zamda) stays valid;
* ``front_harmony`` (hal->hali/halde) blocks the back-harmony *halı;
* the frozen adverbs (herhalde) are whole-surface entries that license no inflection
  (*herhaldeler stays an honest guess);
* the new verb/noun homographs coexist with every pre-existing reading (kurdu = kurt AND
  kur-, arttı = art-, bulundu = bulun- AND bul-+passive), none erased.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest
from conftest import has_analysis

import ilmek
from ilmek.morphology.lexicon import _DATA_DIR

BATCH2_FILES = ("nouns_core2.json", "verbs_core2.json", "adjectives_core2.json")


def _sourced_from_lexicon(analyzer, word, lemma):
    return any(a.lemma == lemma and a.source == "lexicon" for a in analyzer.analyze(word))


# --- Data integrity: the new files add no duplicate (lemma, pos) rows ----------------


@pytest.mark.consistency
def test_batch2_files_add_no_duplicate_lemma_pos():
    # The repo-wide test_no_duplicate_regular_lemma_pos_entries already scans every file;
    # this narrower check pins the batch-2 files specifically and fails loudly if a future
    # edit re-adds a (lemma, pos[, forms]) already present anywhere in the lexicon.
    keys: Counter[tuple] = Counter()
    for path in sorted(_DATA_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data.get("entries", ()) if isinstance(data, dict) else data
        for entry in entries:
            keys[
                (entry["lemma"], entry.get("pos", "NOUN").upper(), tuple(entry.get("forms", ())))
            ] += 1
    duplicates = {k: n for k, n in keys.items() if n > 1}
    assert not duplicates, f"duplicate regular entries: {duplicates}"


@pytest.mark.consistency
def test_batch2_delivered_at_least_500_new_entries():
    total = 0
    for fn in BATCH2_FILES:
        data = json.loads((_DATA_DIR / fn).read_text(encoding="utf-8"))
        total += len(data["entries"])
    assert total >= 500, f"batch-2 shipped only {total} entries (floor is ~500)"


# --- Positive: headline nouns resolve, lexicon-verified ------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,features",
    [
        ("enflasyonun", "enflasyon", {"case": "genitive"}),  # milestone headline
        ("kütüphaneye", "kütüphane", {"case": "dative"}),
        ("sözleşmenin", "sözleşme", {"case": "genitive"}),
        ("toplantıda", "toplantı", {"case": "locative"}),
        ("projenin", "proje", {"case": "genitive"}),
        ("başvurular", "başvuru", {"number": "plural"}),
        ("etkisi", "etki", {"possessive": "3sg"}),
        ("raporları", "rapor", {"number": "plural"}),
    ],
)
def test_headline_nouns_resolve(analyzer, word, lemma, features):
    assert has_analysis(analyzer, word, lemma=lemma, pos="NOUN", features=features)
    assert _sourced_from_lexicon(analyzer, word, lemma)


# --- Positive: voicing nouns soften the final stop before a vowel --------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,features",
    [
        ("sorumluluğu", "sorumluluk", {"case": "accusative"}),  # -lUk k -> ğ
        ("yönetmeliği", "yönetmelik", {"case": "accusative"}),
        ("güvenliğe", "güvenlik", {"case": "dative"}),
        ("süreci", "süreç", {"case": "accusative"}),  # ç -> c
        ("inancı", "inanç", {"case": "accusative"}),
        ("talebi", "talep", {"case": "accusative"}),  # p -> b (Arabic loan)
        ("emeği", "emek", {"case": "accusative"}),  # native -k
        ("ışığı", "ışık", {"case": "accusative"}),
    ],
)
def test_voicing_nouns_soften_before_vowel(analyzer, word, lemma, features):
    assert has_analysis(analyzer, word, lemma=lemma, features=features)
    assert _sourced_from_lexicon(analyzer, word, lemma)


# --- Negative: t-/f-final loans are UNFLAGGED and must NOT voice ----------------------


@pytest.mark.negative
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("hükümedi", "hükümet"),
        ("cumhuriyedi", "cumhuriyet"),
        ("siyasedi", "siyaset"),
        ("şardı", "şart"),
        ("taravı", "taraf"),  # f-final: taraf keeps f (tarafı), the voiced *taravı is invalid
    ],
)
def test_t_f_final_loans_do_not_false_voice(analyzer, word, lemma):
    assert not _sourced_from_lexicon(analyzer, word, lemma)


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,features",
    [
        ("hükümeti", "hükümet", {"case": "accusative"}),  # t kept before a vowel
        ("cumhuriyeti", "cumhuriyet", {"case": "accusative"}),
        ("şartı", "şart", {"case": "accusative"}),
        ("şartlar", "şart", {"number": "plural"}),
        ("tarafı", "taraf", {"case": "accusative"}),  # f kept
    ],
)
def test_t_f_final_loans_keep_stop(analyzer, word, lemma, features):
    assert has_analysis(analyzer, word, lemma=lemma, features=features)
    assert _sourced_from_lexicon(analyzer, word, lemma)


# --- Positive: new verbs (aorist default classes + past) -----------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,features",
    [
        ("ulaştı", "ulaş", {"tense": "past"}),
        ("gerçekleşti", "gerçekleş", {"tense": "past"}),
        ("değerlendirdi", "değerlendir", {"tense": "past"}),
        ("kötüleşti", "kötüleş", {"tense": "past"}),
        ("düzenledi", "düzenle", {"tense": "past"}),
        ("araştırdı", "araştır", {"tense": "past"}),
    ],
)
def test_new_verbs_past(analyzer, word, lemma, features):
    assert has_analysis(analyzer, word, lemma=lemma, pos="VERB", features=features)
    assert _sourced_from_lexicon(analyzer, word, lemma)


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("sağlar", "sağla"),  # vowel-final -> -r aorist
        ("ulaşır", "ulaş"),  # polysyllabic -> -Ir
        ("gerekir", "gerek"),
        ("oluşur", "oluş"),
        ("kurar", "kur"),  # monosyllabic -> -Ar (NOT in the -Ir exception list)
        ("bozar", "boz"),
        ("artar", "art"),
        ("çözer", "çöz"),
    ],
)
def test_new_verb_aorist_classes(analyzer, word, lemma):
    assert has_analysis(analyzer, word, lemma=lemma, pos="VERB", features={"tense": "aorist"})


# --- Positive: new adjectives + ek-fiil ----------------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("ekonomik", "ekonomik"),
        ("siyasi", "siyasi"),
        ("sosyal", "sosyal"),
        ("kültürel", "kültürel"),
        ("ayrıntılı", "ayrıntılı"),
        ("olumsuz", "olumsuz"),
    ],
)
def test_new_adjectives_bare(analyzer, word, lemma):
    assert has_analysis(analyzer, word, lemma=lemma, pos="ADJ")
    assert _sourced_from_lexicon(analyzer, word, lemma)


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,features",
    [
        ("ekonomikti", "ekonomik", {"copula": "past"}),  # ek-fiil past on a new ADJ
        ("sosyaldı", "sosyal", {"copula": "past"}),
        ("olumluydu", "olumlu", {"copula": "past"}),  # y-buffer after vowel stem
    ],
)
def test_ek_fiil_on_new_adjectives(analyzer, word, lemma, features):
    assert has_analysis(analyzer, word, lemma=lemma, pos="ADJ", features=features)


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,features",
    [
        ("ortağı", "ortak", {"case": "accusative"}),  # ADJ voicing k -> ğ
        ("ilginci", "ilginç", {"case": "accusative"}),  # ç -> c
        ("düşüğü", "düşük", {"case": "accusative"}),
    ],
)
def test_adjective_voicing(analyzer, word, lemma, features):
    assert has_analysis(analyzer, word, lemma=lemma, features=features)


# --- Positive: gemination + front-harmony loanword additions -------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [("zammı", "zam"), ("hattı", "hat"), ("sırrı", "sır"), ("üssü", "üs"), ("tıbbı", "tıp")],
)
def test_new_gemination_positive(analyzer, word, lemma):
    assert has_analysis(analyzer, word, lemma=lemma, pos="NOUN")
    assert _sourced_from_lexicon(analyzer, word, lemma)


@pytest.mark.negative
@pytest.mark.parametrize("word,lemma", [("zamı", "zam"), ("hatı", "hat"), ("tıpı", "tıp")])
def test_new_gemination_single_consonant_impossible(analyzer, word, lemma):
    # A single consonant before a vowel is unrepresentable for a geminating root (hakkı/hakı
    # precedent): zamı/hatı/tıpı have no lexicon analysis as the geminating lemma.
    assert not _sourced_from_lexicon(analyzer, word, lemma)


@pytest.mark.positive
def test_new_gemination_locative_keeps_single_form(analyzer):
    # Before a consonant suffix the single free form is kept: zamda (not *zammda) parses.
    assert has_analysis(analyzer, "zamda", lemma="zam", features={"case": "locative"})
    assert has_analysis(analyzer, "hatta", lemma="hat", features={"case": "locative"})


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,features",
    [
        ("hali", {"case": "accusative"}),
        ("hale", {"case": "dative"}),
        ("halde", {"case": "locative"}),
    ],
)
def test_hal_front_harmony(analyzer, word, features):
    # hal (state) harmonizes as front: hali/hale/halde, never back-harmony *halı/*halda.
    assert has_analysis(analyzer, word, lemma="hal", pos="NOUN", features=features)
    assert _sourced_from_lexicon(analyzer, word, "hal")


@pytest.mark.negative
def test_hal_back_harmony_absent(analyzer):
    # *halı (back harmony) must NOT resolve to hal — it would collide with halı "carpet".
    assert not _sourced_from_lexicon(analyzer, "halı", "hal")


# --- Positive / negative: frozen adverbs ---------------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "surface", ["herhalde", "aslında", "genellikle", "ayrıca", "gerçekten", "kesinlikle"]
)
def test_frozen_adverbs_are_lexicon(analyzer, surface):
    best = analyzer.analyze(surface)[0]
    assert best.lemma == surface
    assert best.pos == "ADV"
    assert best.source == "lexicon"
    assert best.morphemes == []  # indeclinable: no suffixes
    assert best.stem == surface


@pytest.mark.negative
@pytest.mark.parametrize("word,lemma", [("herhaldeler", "herhalde"), ("ayrıcayı", "ayrıca")])
def test_frozen_adverbs_do_not_inflect_via_lexicon(analyzer, word, lemma):
    # A guesser strip may coincidentally reach the base, but only as source=guess: the frozen
    # adverb licenses no lexicon-verified plural/accusative.
    assert not _sourced_from_lexicon(analyzer, word, lemma)


# --- Exception / ambiguity: homographs coexist, none erased --------------------------


@pytest.mark.exception
def test_artik_adverb_primary_and_art_verb_coexist(analyzer):
    # "artık" stays the frozen ADV (function_words irregular, prepended) even after the verb
    # art- is added; art-+past "arttı" resolves as the verb.
    assert analyzer.analyze("artık")[0].lemma == "artık"
    assert analyzer.analyze("artık")[0].pos == "ADV"
    assert has_analysis(analyzer, "arttı", lemma="art", pos="VERB", features={"tense": "past"})


@pytest.mark.exception
def test_kurdu_noun_and_verb_coexist(analyzer):
    # "kurdu" = kurt (wolf, t->d voicing) NOUN AND kur-+past VERB: both survive.
    assert has_analysis(analyzer, "kurdu", lemma="kurt", pos="NOUN")
    assert has_analysis(analyzer, "kurdu", lemma="kur", pos="VERB", features={"tense": "past"})


@pytest.mark.exception
def test_bulundu_new_verb_and_passive_coexist(analyzer):
    # "bulundu" = bulun-+past (the new middle/reflexive verb) AND bul-+passive+past: both hold.
    assert has_analysis(analyzer, "bulundu", lemma="bulun", pos="VERB", features={"tense": "past"})
    assert has_analysis(
        analyzer, "bulundu", lemma="bul", pos="VERB", features={"voice": ("passive",)}
    )


# --- Consistency: stem / lemma / analyze agree on sampled roots ----------------------


@pytest.mark.consistency
def test_stem_lemma_analyze_agree_for_batch2():
    # Voiced -lUk accusative reduces to the free lemma across all three views.
    assert ilmek.lemmatize("sorumluluğu") == "sorumluluk"
    assert ilmek.stem("sorumluluğu") == "sorumluluk"
    # Gemination + front-harmony reduce too.
    assert ilmek.lemmatize("zammı") == "zam"
    assert ilmek.stem("halde") == "hal"
    # A bare new adjective is inflection-free.
    best = ilmek.analyze("ekonomik")[0]
    assert best.lemma == "ekonomik"

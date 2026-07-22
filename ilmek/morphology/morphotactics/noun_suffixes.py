"""Nominal inflectional suffixes: plural, possessive, case, and the distributive numeral.

Suffix *instances* only — the graphs that wire them to states (and the accusative split-off)
live in :mod:`.transitions`.
"""

from __future__ import annotations

from ...core import tags
from .suffix import Suffix

PLURAL = Suffix("plural", "lAr", {tags.NUMBER: "plural"})

POSS_1SG = Suffix("poss_1sg", "(I)m", {tags.POSSESSIVE: "1sg"})
POSS_2SG = Suffix("poss_2sg", "(I)n", {tags.POSSESSIVE: "2sg"})
POSS_3SG = Suffix("poss_3sg", "(s)I", {tags.POSSESSIVE: "3sg"})
POSS_1PL = Suffix("poss_1pl", "(I)mIz", {tags.POSSESSIVE: "1pl"})
POSS_2PL = Suffix("poss_2pl", "(I)nIz", {tags.POSSESSIVE: "2pl"})
POSS_3PL = Suffix("poss_3pl", "lArI", {tags.POSSESSIVE: "3pl"})

# Case, plain (after stem / plural / non-3rd possessive).
CASE_ACC = Suffix("acc", "(y)I", {tags.CASE: "accusative"})
CASE_DAT = Suffix("dat", "(y)A", {tags.CASE: "dative"})
CASE_LOC = Suffix("loc", "DA", {tags.CASE: "locative"})
CASE_ABL = Suffix("abl", "DAn", {tags.CASE: "ablative"})
CASE_GEN = Suffix("gen", "(n)In", {tags.CASE: "genitive"})
CASE_INS = Suffix("ins", "(y)lA", {tags.CASE: "instrumental"})

# Case, pronominal (after a 3rd-person possessive: evi -> evi-n-de).
CASE_ACC_N = Suffix("acc", "(n)I", {tags.CASE: "accusative"})
CASE_DAT_N = Suffix("dat", "(n)A", {tags.CASE: "dative"})
CASE_LOC_N = Suffix("loc", "(n)DA", {tags.CASE: "locative"})
CASE_ABL_N = Suffix("abl", "(n)DAn", {tags.CASE: "ablative"})
CASE_GEN_N = Suffix("gen", "(n)In", {tags.CASE: "genitive"})
CASE_INS_N = Suffix("ins", "(y)lA", {tags.CASE: "instrumental"})

_PLAIN_CASES = [CASE_ACC, CASE_DAT, CASE_LOC, CASE_ABL, CASE_GEN, CASE_INS]
_PRONOMINAL_CASES = [CASE_ACC_N, CASE_DAT_N, CASE_LOC_N, CASE_ABL_N, CASE_GEN_N, CASE_INS_N]
_POSSESSIVES_TO_NONE3 = [POSS_1SG, POSS_2SG, POSS_1PL, POSS_2PL]
_POSSESSIVES_TO_3 = [POSS_3SG, POSS_3PL]


# --- Distributive numeral suffix -(ş)Ar (bir->birer, iki->ikişer) --------------------
# A numeral-only *inflectional* suffix ("n each / n at a time"): the ş buffer appears after a
# vowel-final numeral (iki->ikişer, altı->altışar), and is absent after a consonant (bir->birer,
# beş->beşer, on->onar). It is NOT derivational — the lemma and stem stay the bare numeral
# (birer -> bir) for stem(), lemmatize() and analyze() alike. ``applies_to={NUM}`` is the
# declarative overgeneration guard, checked by the analyzer for any suffix (not just
# derivations): a NOUN/ADJ/VERB stem can never take it (no *ever from ev+Ar, no OOV zom+ar).
# It lands in the terminal ``N_DIST``: further inflection/copula on a distributive (birerden,
# birerdi) is deferred, correctness over coverage. The lexical exception yarımşar (ş after a
# consonant) is not modeled (xfailed): it needs a per-word ``ş`` fact, not this productive rule.
# ``excludes_attribute="fraction"`` keeps it off yarım/çeyrek/buçuk entirely: without it the
# productive rule would wrongly accept *yarımar/*çeyreker (real Turkish has no such forms; the
# only attested distributive of yarım is the lexical exception yarımşar, itself still unmodeled).
DIST = Suffix(
    "dist",
    "(ş)Ar",
    {tags.NUM_TYPE: "distributive"},
    applies_to=frozenset({tags.NUM}),
    excludes_attribute="fraction",
)


# --- Ordinal numeral suffix -(I)ncI (bir->birinci, iki->ikinci, dört->dördüncü) ------
# The ordinal "n-th". Like the distributive it is an *inflectional* numeral suffix, NOT a
# derivation: the lemma and stem stay the bare numeral (birinci -> bir), the pos stays NUM, and
# ``features[NUM_TYPE]="ordinal"`` records it. The (I) linking vowel appears after a consonant
# (bir->birinci, on->onuncu) and is absent after a vowel (iki->ikinci, altı->altıncı); dört's
# voicing bound form (dörd) is selected for free because the suffix is vowel-initial (dördüncü).
# ``applies_to={NUM}`` is the overgeneration guard: a NOUN/ADJ/VERB stem can never take it
# (*evinci, *güzelinci), and the guesser (which synthesizes only NOUN/VERB roots) never fires it.
# It lands in N_ORD, which inflects and hosts the ek-fiil (birincisi, ikincide, birinciydi).
# ``excludes_attribute="fraction"`` keeps it off yarım/çeyrek/buçuk: a fraction has no ordinal
# reading in Turkish (there is no "*yarımıncı"/"*çeyreğinci"/"*buçuğuncu" — ordinals attach to
# counting cardinals, not to quantity fractions), so without this guard the shared {NUM} gate
# would wrongly accept them once the fractions joined the lexicon as NUM.
ORD = Suffix(
    "ordinal",
    "(I)ncI",
    {tags.NUM_TYPE: "ordinal"},
    applies_to=frozenset({tags.NUM}),
    excludes_attribute="fraction",
)
# The lexically-limited ordinal on the ADJ "son" (sonuncu "last"): son is an ADJECTIVE, not a
# numeral, so the {NUM}-guarded ORD cannot reach it. This curated twin fires only on an ADJ that
# carries the ``ordinal_host`` attribute (mirrors the reflexive/reciprocal curated-attribute
# pattern), keeping the same template/feature; the stem/lemma stay "son".
ORD_SON = Suffix(
    "ordinal",
    "(I)ncI",
    {tags.NUM_TYPE: "ordinal"},
    applies_to=frozenset({tags.ADJ}),
    requires_attribute="ordinal_host",
)

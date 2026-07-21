"""Morphotactics: the ordered suffix-transition graphs for nouns and verbs.

Encoded as *data* (suffix objects + a state→transitions map) so the ordering of Turkish
morphology is declared, inspectable, and extensible without touching the analyzer. The
analyzer walks these graphs forward, realizing each suffix with :mod:`.phonology` and
keeping paths whose running surface stays a prefix of the target word.

v0.1 scope (correctness over coverage): nouns — plural, six persons of possessive, six
cases, with pronominal buffering after a 3rd-person possessive; verbs — negation, the
progressive/future/past/evidential tense-aspects, one copular (ek-fiil) layer, and both
person paradigms. Aorist, ability, voice, derivation, and clitics are later milestones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core import tags

# --- States --------------------------------------------------------------------------

N_ROOT = "N_ROOT"
N_PL = "N_PL"
N_POSS = "N_POSS"  # non-3rd-person possessive
N_POSS3 = "N_POSS3"  # 3rd-person possessive (triggers pronominal -n- before case)
N_CASE = "N_CASE"

V_ROOT = "V_ROOT"
V_NEG = "V_NEG"
V_T1 = "V_T1"  # after a tense/aspect that takes the type-1 person set
V_T2 = "V_T2"  # after a tense/aspect that takes the type-2 person set
V_COP1 = "V_COP1"  # after an evidential copula (ek-fiil) -> type-1 person
V_COP2 = "V_COP2"  # after a past copula (ek-fiil) -> type-2 person
V_PERS = "V_PERS"


# --- Suffix model --------------------------------------------------------------------


@dataclass
class Suffix:
    name: str
    template: str
    features: dict = field(default_factory=dict)
    #: -Iyor deletes a preceding stem-final vowel (başla+Iyor -> başlıyor).
    drop_preceding: bool = False
    #: This morpheme's final stop softens before a following vowel (-(y)AcAk + Im -> ...eğim).
    voice_final: bool = False


# --- Nominal suffixes ----------------------------------------------------------------

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


def _nominal_graph() -> dict[str, list[tuple[Suffix, str]]]:
    poss = [(s, N_POSS) for s in _POSSESSIVES_TO_NONE3] + [(s, N_POSS3) for s in _POSSESSIVES_TO_3]
    plain_case = [(s, N_CASE) for s in _PLAIN_CASES]
    pronom_case = [(s, N_CASE) for s in _PRONOMINAL_CASES]
    return {
        N_ROOT: [(PLURAL, N_PL), *poss, *plain_case],
        N_PL: [*poss, *plain_case],
        N_POSS: plain_case,
        N_POSS3: pronom_case,
        N_CASE: [],
    }


NOMINAL_GRAPH = _nominal_graph()
NOMINAL_START = N_ROOT
NOMINAL_FINALS = frozenset({N_ROOT, N_PL, N_POSS, N_POSS3, N_CASE})


# --- Verbal suffixes -----------------------------------------------------------------

NEG = Suffix("neg", "mA", {tags.POLARITY: "negative"})

PROG = Suffix(
    "prog", "Iyor", {tags.TENSE: "present", tags.ASPECT: "progressive"}, drop_preceding=True
)
FUT = Suffix("fut", "(y)AcAk", {tags.TENSE: "future"}, voice_final=True)
EVID = Suffix("evid", "mIş", {tags.EVIDENTIAL: True})
PAST = Suffix("past", "DI", {tags.TENSE: "past"})

# Copula (ek-fiil) forms stack on a finished tense and take the (y) buffer after a vowel
# (geldi -> geldiydi / geldiymiş), unlike the primary tense suffixes which attach to the
# bare stem. The past copula records a separate ``copula`` feature so it never overwrites a
# primary tense (gelecekti = future + past-copula).
COP_EVID = Suffix("cop_evid", "(y)mIş", {tags.EVIDENTIAL: True})
COP_PAST = Suffix("cop_past", "(y)DI", {tags.COPULA: "past"})

# Person set type-1 (present/future/evidential/aorist; 3sg is zero).
P1_1SG = Suffix("pers_1sg", "(y)Im", {tags.PERSON: "1sg"})
P1_2SG = Suffix("pers_2sg", "sIn", {tags.PERSON: "2sg"})
P1_1PL = Suffix("pers_1pl", "(y)Iz", {tags.PERSON: "1pl"})
P1_2PL = Suffix("pers_2pl", "sInIz", {tags.PERSON: "2pl"})
P1_3PL = Suffix("pers_3pl", "lAr", {tags.PERSON: "3pl"})

# Person set type-2 (past -DI, conditional; 3sg is zero).
P2_1SG = Suffix("pers_1sg", "m", {tags.PERSON: "1sg"})
P2_2SG = Suffix("pers_2sg", "n", {tags.PERSON: "2sg"})
P2_1PL = Suffix("pers_1pl", "k", {tags.PERSON: "1pl"})
P2_2PL = Suffix("pers_2pl", "nIz", {tags.PERSON: "2pl"})
P2_3PL = Suffix("pers_3pl", "lAr", {tags.PERSON: "3pl"})

_PERSON_T1 = [P1_1SG, P1_2SG, P1_1PL, P1_2PL, P1_3PL]
_PERSON_T2 = [P2_1SG, P2_2SG, P2_1PL, P2_2PL, P2_3PL]


def _verbal_graph() -> dict[str, list[tuple[Suffix, str]]]:
    primary_from = lambda: [  # noqa: E731 - compact, local
        (PROG, V_T1),
        (FUT, V_T1),
        (EVID, V_T1),
        (PAST, V_T2),
    ]
    copula = [(COP_EVID, V_COP1), (COP_PAST, V_COP2)]
    pers_t1 = [(s, V_PERS) for s in _PERSON_T1]
    pers_t2 = [(s, V_PERS) for s in _PERSON_T2]
    return {
        V_ROOT: [(NEG, V_NEG), *primary_from()],
        V_NEG: [*primary_from()],
        V_T1: [*copula, *pers_t1],
        V_T2: [*copula, *pers_t2],
        V_COP1: pers_t1,
        V_COP2: pers_t2,
        V_PERS: [],
    }


VERBAL_GRAPH = _verbal_graph()
VERBAL_START = V_ROOT
# V_NEG is final too: a bare negated stem is a negative imperative (gelme! "don't come").
VERBAL_FINALS = frozenset({V_ROOT, V_NEG, V_T1, V_T2, V_COP1, V_COP2, V_PERS})


# --- Feature closure at acceptance ---------------------------------------------------


def nominal_default_features() -> dict:
    return {tags.NUMBER: "singular", tags.POSSESSIVE: "none", tags.CASE: "nominative"}


def finalize_verbal_features(features: dict) -> dict:
    """Fill implicit verbal features at an accepting state."""
    features.setdefault(tags.POLARITY, "positive")
    finite = any(k in features for k in (tags.TENSE, tags.ASPECT, tags.EVIDENTIAL))
    if not finite and tags.PERSON not in features:
        # A bare verb root standing alone is a 2nd-person-singular imperative.
        features["mood"] = "imperative"
        features[tags.PERSON] = "2sg"
    else:
        features.setdefault(tags.PERSON, "3sg")
    return features

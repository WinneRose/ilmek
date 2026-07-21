"""Morphotactics: the ordered suffix-transition graphs for nouns and verbs.

Encoded as *data* (suffix objects + a state→transitions map) so the ordering of Turkish
morphology is declared, inspectable, and extensible without touching the analyzer. The
analyzer walks these graphs forward, realizing each suffix with :mod:`.phonology` and
keeping paths whose running surface stays a prefix of the target word.

v0.1 scope (correctness over coverage): nouns — plural, six persons of possessive, six
cases, with pronominal buffering after a 3rd-person possessive; verbs — negation, the
progressive/future/past/evidential tense-aspects, one copular (ek-fiil) layer, and both
person paradigms.

Verbal moods & aorist (this milestone): ability -(y)Abil (gelebilir, okuyabilir), the
conditional -sA (gelse, gelseydi via the copula) and optative -(y)A (gele, gelelim), the
negative aorist -mAz (gelmez) with its *defective* person paradigm, and the positive aorist
— which is lexically irregular, so its allomorph (-r / -Ar / -Ir) is a lexicon fact on the
root (:attr:`~ilmek.morphology.lexicon.Root.aorist`) selected declaratively by an edge's
:attr:`Suffix.aorist_class`. Deferred (correctness over coverage): the impossibilitive
-(y)AmA (gelemez), the copular conditional -(y)sA (gelirse), and the negative-aorist 1sg/1pl
(gelmem/gelmeyiz) — all xfailed rather than overgenerated.

Derivation (this milestone): a single, non-recursive derivation slot sits between root and
inflection. Nominal derivations (-lI, -sIz, -lIk, -CI) leave ``N_ROOT`` for ``N_DERIV``;
verbal derivations (-mA, -(y)Iş, -(y)An, -DIk, -(y)AcAk) leave ``V_ROOT``/``V_NEG`` for the
*same* ``N_DERIV``, and the infinitive -mAk lands in a terminal ``V_INF``. ``N_DERIV``'s
outgoing edges are exactly ``N_ROOT``'s inflectional ones, so a derived stem inflects
normally (evli -> evlilerden) but cannot derive again (no stacking this milestone). Which
derivation may fire is gated declaratively by :attr:`Suffix.applies_to` (POS of the current
stem), never by a hardcoded ``if`` in the analyzer. Voice, derivational stacking, and clitics
remain later milestones.
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
N_DERIV = "N_DERIV"  # a derived nominal/adjectival stem (inflects like N_ROOT, cannot re-derive)

V_ROOT = "V_ROOT"
V_NEG = "V_NEG"
V_ABIL = "V_ABIL"  # after ability -(y)Abil; not final (no bare *gelebil), takes further tense
V_T1 = "V_T1"  # after a tense/aspect that takes the type-1 person set
V_T2 = "V_T2"  # after a tense/aspect that takes the type-2 person set
V_COP1 = "V_COP1"  # after an evidential copula (ek-fiil) -> type-1 person
V_COP2 = "V_COP2"  # after a past copula (ek-fiil) -> type-2 person
V_AOR_NEG = "V_AOR_NEG"  # after negative aorist -mAz; final, defective person paradigm
V_OPT = "V_OPT"  # after optative -(y)A; final, its own person paradigm (1pl is -lIm)
V_PERS = "V_PERS"
V_INF = "V_INF"  # infinitive -mAk (a noun); terminal this milestone (no case inflection yet)


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
    #: A *derivational* suffix changes the stem's word class; it opens a new derivation
    #: boundary rather than adding an inflectional feature. Its ``name`` is recorded (in
    #: order) under ``features[tags.DERIVATION]`` so the boundary stays visible.
    derivational: bool = False
    #: The part of speech of the stem *after* this derivation (``None`` -> keep current pos).
    to_pos: str | None = None
    #: The stem POS values this derivation may attach to (``None`` -> unrestricted, e.g. the
    #: verb-side derivations, which are already gated by their position in the graph). This
    #: is the declarative guard against overgeneration: -CI is {NOUN} so *güzelci is blocked.
    applies_to: frozenset[str] | None = None
    #: For the lexically-irregular aorist: the allomorph class (``"r"``/``"Ar"``/``"Ir"``)
    #: this edge realizes. When set, the analyzer walks the edge only if it equals the root's
    #: own :attr:`~ilmek.morphology.lexicon.Root.aorist` — so ``gel`` (marked ``"Ir"``) takes
    #: only ``gelir`` and a synthetic/guessed root (``aorist=None``) takes none. ``None`` on
    #: every non-aorist suffix (and on the post-ability aorist, which is always ``-Ir``).
    aorist_class: str | None = None


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


# --- Nominal derivational suffixes (root/adj -> new nominal stem) ---------------------
# Data-declared: each carries no inflectional feature dict — its record is the derivation
# name appended to features[tags.DERIVATION]. ``applies_to`` is the overgeneration guard.

D_LI = Suffix("li", "lI", derivational=True, to_pos=tags.ADJ, applies_to=frozenset({tags.NOUN}))
D_SIZ = Suffix("siz", "sIz", derivational=True, to_pos=tags.ADJ, applies_to=frozenset({tags.NOUN}))
D_LIK = Suffix(
    "lik",
    "lIk",
    derivational=True,
    to_pos=tags.NOUN,
    applies_to=frozenset({tags.NOUN, tags.ADJ}),
    voice_final=True,  # kitaplık -> kitaplığı (final k softens before a vowel)
)
D_CI = Suffix("ci", "CI", derivational=True, to_pos=tags.NOUN, applies_to=frozenset({tags.NOUN}))

#: Nominal-side derivations, appended after the inflectional edges so inflection-only
#: traversal order (and the guesser, which forbids derivation) is byte-identical to before.
_NOMINAL_DERIVATIONS = [D_LI, D_SIZ, D_LIK, D_CI]


def _nominal_inflection() -> list[tuple[Suffix, str]]:
    poss = [(s, N_POSS) for s in _POSSESSIVES_TO_NONE3] + [(s, N_POSS3) for s in _POSSESSIVES_TO_3]
    plain_case = [(s, N_CASE) for s in _PLAIN_CASES]
    return [(PLURAL, N_PL), *poss, *plain_case]


def _nominal_graph() -> dict[str, list[tuple[Suffix, str]]]:
    inflection = _nominal_inflection()
    poss = [(s, N_POSS) for s in _POSSESSIVES_TO_NONE3] + [(s, N_POSS3) for s in _POSSESSIVES_TO_3]
    plain_case = [(s, N_CASE) for s in _PLAIN_CASES]
    pronom_case = [(s, N_CASE) for s in _PRONOMINAL_CASES]
    deriv = [(s, N_DERIV) for s in _NOMINAL_DERIVATIONS]
    return {
        # Derivation edges appended last: with derivation disabled the prefix of this list
        # is exactly the pre-milestone N_ROOT, so nothing about plain inflection changes.
        N_ROOT: [*inflection, *deriv],
        N_PL: [*poss, *plain_case],
        N_POSS: plain_case,
        N_POSS3: pronom_case,
        N_CASE: [],
        # A derived stem inflects exactly like a bare root, but may not derive again.
        N_DERIV: inflection,
    }


NOMINAL_GRAPH = _nominal_graph()
NOMINAL_START = N_ROOT
NOMINAL_FINALS = frozenset({N_ROOT, N_PL, N_POSS, N_POSS3, N_CASE, N_DERIV})


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

# Optative persons are NOT plain type-1: the 1pl is -lIm (gelelim), never *(y)Iz (*geleyiz).
OPT_1PL = Suffix("pers_1pl", "lIm", {tags.PERSON: "1pl"})
_PERSON_OPT = [P1_1SG, P1_2SG, OPT_1PL, P1_2PL, P1_3PL]

# The negative-aorist person paradigm is *defective*: the 1sg is gelmem and the 1pl is
# gelmeyiz (distinct morphemes), never *gelmezim / *gelmeziz. So -mAz takes only the 2sg,
# 2pl and 3pl personal endings here; the 1sg/1pl readings are deferred (see the tests).
_PERSON_AOR_NEG = [P1_2SG, P1_2PL, P1_3PL]


# --- Ability, aorist, and mood suffixes ----------------------------------------------

# Ability / potential -(y)Abil (gelebilir "can come", okuyabilir): a fully productive slot
# between the root/negation and the tense. After it the stem ends in "bil", so the aorist is
# deterministically -Ir (AOR_ABIL) with no lexical guessing.
ABIL = Suffix("abil", "(y)Abil", {tags.ABILITY: True})

# Aorist (geniş zaman), lexically irregular. Each edge realizes one allomorph and carries the
# matching ``aorist_class``; the analyzer walks it only when it equals the root's aorist
# class, so an unmarked/synthetic root never takes the wrong (or any) aorist.
AOR_R = Suffix("aor", "r", {tags.TENSE: "aorist"}, aorist_class="r")
AOR_AR = Suffix("aor", "Ar", {tags.TENSE: "aorist"}, aorist_class="Ar")
AOR_IR = Suffix("aor", "Ir", {tags.TENSE: "aorist"}, aorist_class="Ir")
_AORISTS = [AOR_R, AOR_AR, AOR_IR]
#: The post-ability aorist is always -Ir (gel-ebil-ir), independent of the root's class, so
#: it carries no ``aorist_class`` guard.
AOR_ABIL = Suffix("aor", "Ir", {tags.TENSE: "aorist"})

# Negative aorist -mAz (gelmez, yapmaz): a dedicated negative morpheme (not NEG + aorist).
NEG_AOR = Suffix("neg_aor", "mAz", {tags.POLARITY: "negative", tags.TENSE: "aorist"})

# Conditional -sA (gelse) -> type-2 persons; the past/evidential copula (gelseydi/gelseymiş)
# stacks for free. Optative -(y)A (gele) -> its own person set (V_OPT).
COND = Suffix("cond", "sA", {tags.MOOD: "conditional"})
OPT = Suffix("opt", "(y)A", {tags.MOOD: "optative"})


# --- Verbal derivational suffixes (verb -> new nominal/adjectival stem) ---------------
# Gated by graph position (only reachable from V_ROOT/V_NEG), so no applies_to is needed.
# verb -> noun:
VN_MA = Suffix("ma", "mA", derivational=True, to_pos=tags.NOUN)  # gelme (verbal noun / act)
VN_IS = Suffix("is", "(y)Iş", derivational=True, to_pos=tags.NOUN)  # geliş, yürüyüş
INF = Suffix("mak", "mAk", derivational=True, to_pos=tags.NOUN)  # gelmek (infinitive)
# verb -> adjective (participles):
PART_AN = Suffix("an", "(y)An", derivational=True, to_pos=tags.ADJ)  # gelen
PART_DIK = Suffix("dik", "DIk", derivational=True, to_pos=tags.ADJ, voice_final=True)  # bildiği
PART_ACAK = Suffix(
    "acak", "(y)AcAk", derivational=True, to_pos=tags.ADJ, voice_final=True
)  # gelecek

#: Verb-side derivations that land in the shared N_DERIV nominal state (then inflect). The
#: infinitive -mAk is handled separately: it lands in the terminal V_INF (no case yet).
_VERBAL_DERIVATIONS_TO_NOMINAL = [VN_MA, VN_IS, PART_AN, PART_DIK, PART_ACAK]


def _verbal_graph() -> dict[str, list[tuple[Suffix, str]]]:
    primary_from = lambda: [  # noqa: E731 - compact, local
        (PROG, V_T1),
        (FUT, V_T1),
        (EVID, V_T1),
        (PAST, V_T2),
    ]
    # Aorist / mood edges leaving the root and the negation, appended after the primary
    # tenses so plain-inflection traversal order is unchanged. The three aorist edges are
    # guarded by ``aorist_class``, so exactly one fires per verb (and none for a guess).
    aorist_mood = lambda: [  # noqa: E731 - compact, local
        (ABIL, V_ABIL),
        *[(s, V_T1) for s in _AORISTS],
        (NEG_AOR, V_AOR_NEG),
        (COND, V_T2),
        (OPT, V_OPT),
    ]
    # Derivation edges appended after the inflectional ones, same as the nominal side.
    deriv = [(s, N_DERIV) for s in _VERBAL_DERIVATIONS_TO_NOMINAL] + [(INF, V_INF)]
    copula = [(COP_EVID, V_COP1), (COP_PAST, V_COP2)]
    pers_t1 = [(s, V_PERS) for s in _PERSON_T1]
    pers_t2 = [(s, V_PERS) for s in _PERSON_T2]
    return {
        V_ROOT: [(NEG, V_NEG), *primary_from(), *aorist_mood(), *deriv],
        # After negation: the primary tenses, ability (gelmeyebilir), conditional (gelmese),
        # optative (gelmeye). The aorist and negative-aorist edges are intentionally absent —
        # the aorist's own negative is -mAz on the bare root, not NEG + aorist (*gelmemez).
        V_NEG: [*primary_from(), (ABIL, V_ABIL), (COND, V_T2), (OPT, V_OPT), *deriv],
        # Ability: primary tenses, the deterministic -Ir aorist, conditional, and the verbal
        # derivations (gelebilmek, gelebilen). Not final -> no bare *gelebil.
        V_ABIL: [*primary_from(), (AOR_ABIL, V_T1), (COND, V_T2), *deriv],
        V_T1: [*copula, *pers_t1],
        V_T2: [*copula, *pers_t2],
        V_COP1: pers_t1,
        V_COP2: pers_t2,
        # Negative aorist: final (gelmez = 3sg), copular stacking (gelmezdi, gelmezmiş,
        # gelmezdim), plus only the defective personal set (2sg/2pl/3pl).
        V_AOR_NEG: [*copula, *[(s, V_PERS) for s in _PERSON_AOR_NEG]],
        # Optative: final (gele = 3sg), with its own person set (1pl is -lIm). No copula yet.
        V_OPT: [(s, V_PERS) for s in _PERSON_OPT],
        V_PERS: [],
        V_INF: [],
    }


VERBAL_GRAPH = _verbal_graph()
VERBAL_START = V_ROOT
# V_NEG is final too: a bare negated stem is a negative imperative (gelme! "don't come").
# V_INF (gelmek) is final: a bare infinitive is a complete word. V_AOR_NEG (gelmez) and
# V_OPT (gele) are final: a bare negative-aorist / optative 3sg is a complete word.
VERBAL_FINALS = frozenset(
    {V_ROOT, V_NEG, V_T1, V_T2, V_COP1, V_COP2, V_AOR_NEG, V_OPT, V_PERS, V_INF}
)


# --- Unified graph -------------------------------------------------------------------
# The nominal and verbal state names are disjoint, so the two transition maps merge into
# one graph the analyzer walks from the POS-appropriate start state. Verbal derivations
# cross into the nominal N_DERIV state, so a single traversal spans both sides.

GRAPH: dict[str, list[tuple[Suffix, str]]] = {**NOMINAL_GRAPH, **VERBAL_GRAPH}
FINALS = NOMINAL_FINALS | VERBAL_FINALS
#: States whose accepting side is *nominal*: they take nominal feature defaults and never
#: run verbal finalization. This includes the shared derived state and the infinitive, so a
#: verb-derived noun (gelme, gelmek) gets no fabricated person/mood but keeps any polarity.
NOMINAL_STATES = frozenset(NOMINAL_FINALS | {V_INF})


# --- Feature closure at acceptance ---------------------------------------------------


def nominal_default_features() -> dict:
    return {tags.NUMBER: "singular", tags.POSSESSIVE: "none", tags.CASE: "nominative"}


def finalize_verbal_features(features: dict) -> dict:
    """Fill implicit verbal features at an accepting state."""
    features.setdefault(tags.POLARITY, "positive")
    # A mood (conditional/optative) counts as finite too, so gelse/gele are not mislabelled
    # imperative: without this, their personless 3sg path would be stamped mood=imperative.
    finite = any(k in features for k in (tags.TENSE, tags.ASPECT, tags.EVIDENTIAL, tags.MOOD))
    if not finite and tags.PERSON not in features:
        # A bare verb root standing alone is a 2nd-person-singular imperative.
        features[tags.MOOD] = "imperative"
        features[tags.PERSON] = "2sg"
    else:
        features.setdefault(tags.PERSON, "3sg")
    return features

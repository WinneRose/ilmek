"""Morphotactics: the ordered suffix-transition graphs for nouns and verbs.

Encoded as *data* (suffix objects + a state→transitions map) so the ordering of Turkish
morphology is declared, inspectable, and extensible without touching the analyzer. The
analyzer walks these graphs forward, realizing each suffix with :mod:`.phonology` and
keeping paths whose running surface stays a prefix of the target word.

v0.1 scope (correctness over coverage): nouns — plural, six persons of possessive, six
cases, with pronominal buffering after a 3rd-person possessive; verbs — negation, the
progressive/future/past/evidential tense-aspects, one copular (ek-fiil) layer, and both
person paradigms.

Verbal moods & aorist: ability -(y)Abil (gelebilir, okuyabilir), the conditional -sA (gelse,
gelseydi via the copula) and optative -(y)A (gele, gelelim), the necessitative -mAlI (gelmeli,
yapmalıyım, gitmeliydik), the negative aorist -mAz (gelmez) with its *defective* person
paradigm, and the positive aorist — which is lexically irregular, so its allomorph
(-r / -Ar / -Ir) is a lexicon fact on the root (:attr:`~ilmek.morphology.lexicon.Root.aorist`)
selected declaratively by an edge's :attr:`Suffix.aorist_class`.

This milestone completes the mood inventory and the remaining inflectional gaps: the
impossibilitive -(y)AmA (gelemez = gel+eme+z, a *distinct* morpheme = the ability-negative,
carrying polarity=negative + ability, NOT NEG+aorist) with its own V_IMPOSS state feeding the
defective -z aorist; the copular conditional -(y)sA stacking on a finished tense (gelirse,
geldiyse, gelecekse, geliyorsa, gelmezse), landing the bare conditional in its own V_COND
state so a restack (*gelseyse) is blocked; the irregular negative-aorist 1sg/1pl (gelmem,
gelmeyiz, and gelemem/gelemeyiz), which attach to the -mA stem, not to -mAz; and the irregular
de-/ye- glide raising (diyor, yiyor, diyecek, diye) via :attr:`Root.raised_form` plus the
:attr:`Suffix.glide_raise` flag, with dedi/demiş/deyiş staying regular. Deferred (correctness
over coverage, xfailed rather than overgenerated): the copular optative and further copular
restacking.

Verb voice / çatı (this milestone): a bounded voice layer sits between the root and the
negation/tense chain, encoding the canonical order reflexive/reciprocal < causative(<=2) <
passive. Reflexive -In and reciprocal -Iş share the ``V_RECIP`` state; the first causative
(-DIr/-t, and the lexically-limited -Ir/-Ar) enters ``V_CAUS1``, a second causative
``V_CAUS2``; the passive (-Il/-In/-n) enters ``V_PASS``. Every voice state then reuses the
bare root's continuation via :func:`_root_continuation`, so a voiced stem inflects for
negation, all tenses, ability, mood and the verb->nominal derivations for free (yaptırabilir,
yapılmayacaktı, görüşme). Crucially the post-voice aorist is the deterministic -Ir
(``AOR_VOICE``), never the root's lexical class (denir, not ``*dener``). Which voice may fire
is gated declaratively — the causative by :attr:`Suffix.causative_class` against the root fact
(second/post-voice causatives by :attr:`Suffix.stem_final_class` instead), the passive by
``stem_final_class`` alone (fully productive), and the semi-productive reflexive/reciprocal by
:attr:`Suffix.requires_attribute`. Voice is recorded (ordered) under ``features[tags.VOICE]``;
it is not derivational, so stem and lemma stay the root (yaptırdı -> stem/lemma yap). The
voice states are deliberately non-final (a bare voiced imperative is deferred), and the
guesser walks no voice edge (``allow_voice=False``), so OOV stripping is byte-identical.

Derivation (this milestone): a single, non-recursive derivation slot sits between root and
inflection. Nominal derivations (-lI, -sIz, -lIk, -CI) leave ``N_ROOT`` for ``N_DERIV``;
verbal derivations (-mA, -(y)Iş, -(y)An, -DIk, -(y)AcAk) leave ``V_ROOT``/``V_NEG`` for the
*same* ``N_DERIV``, and the infinitive -mAk lands in a terminal ``V_INF``. ``N_DERIV``'s
outgoing edges are exactly ``N_ROOT``'s inflectional ones, so a derived stem inflects
normally (evli -> evlilerden) but cannot derive again (no stacking this milestone). Which
derivation may fire is gated declaratively by :attr:`Suffix.applies_to` (POS of the current
stem), never by a hardcoded ``if`` in the analyzer. Voice, derivational stacking, and clitics
remain later milestones.

Nominal ek-fiil (this milestone): NOUN/ADJ/PRON/NUM predicates take the copula ("to be")
*directly* as a suffix (no separate verb). From every nominal final state — the bare root, or
after plural/possessive/a non-accusative case/a derivation — the shared :data:`_NOMINAL_COPULA`
edge set adds the past -(y)DI (güzeldi, hastaydı, evdeydim), evidential -(y)mIş (güzelmiş,
arabaymış), conditional -(y)sA (güzelse, evse), assertive -DIr (güzeldir, evdedir, kitaptır),
and the zero-copula present persons -(y)Im/-sIn/∅/-(y)Iz/-sInIz/-lAr (güzelim, güzelsin,
güzeliz, güzelsiniz). It REUSES the verbal copula states rather than duplicating suffixes:
-(y)DI/-(y)sA feed the type-2 person state (güzeldim, güzelsen), -(y)mIş the type-1 state
(güzelmişim), and the present persons land in ``V_PERS``. The copular ``copula``/``mood``/
``evidential``/``person`` keys are distinct from number/possessive/case, so a case is never
overwritten (evlerimizdeydi keeps plural+1pl+locative+copula-past). The accusative is split
off to the terminal ``N_ACC`` because ``*eviydi`` read as accusative+copula is ungrammatical.
Deferred (correctness over coverage, xfailed rather than overgenerated): -DIr person/plural
stacking (güzeldirim, güzelmiştir) and suppletive personal-pronoun predicates (oydu, bendim —
they are enumerated IrregularForm surfaces, not FSM roots). The verbal copular conditional
-(y)sA (gelirse) is now implemented (see the Verbal moods & aorist note above).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core import tags

# --- States --------------------------------------------------------------------------

N_ROOT = "N_ROOT"
N_PL = "N_PL"
N_POSS = "N_POSS"  # non-3rd-person possessive
N_POSS3 = "N_POSS3"  # 3rd-person possessive (triggers pronominal -n- before case)
N_CASE = "N_CASE"  # a non-accusative case (grammatical ek-fiil predicate host: evdeydi)
N_ACC = "N_ACC"  # the accusative, split off as terminal: the copula never follows it (*eviydi)
N_DERIV = "N_DERIV"  # a derived nominal/adjectival stem (inflects like N_ROOT, cannot re-derive)
N_COP_DIR = "N_COP_DIR"  # after the assertive/generalizing ek-fiil -DIr; terminal this milestone

V_ROOT = "V_ROOT"
# Voice (çatı) states, between the root and the negation/tense layer. Ordered
# reflexive/reciprocal < causative(<=2) < passive, encoded as a bounded (acyclic) chain so
# stacking is finite. None is final this milestone: a bare voiced stem IS a real 2sg
# imperative (yıkan!, görüş!) but making it final would rank it above the homograph nouns
# (sorun, alın) and the -Iş verbal nouns (görüş, geliş) — deferred, correctness over coverage.
V_RECIP = "V_RECIP"  # after reflexive -In or reciprocal -Iş (shared: both precede caus/pass)
V_CAUS1 = "V_CAUS1"  # after the first causative -DIr/-t/-Ir/-Ar
V_CAUS2 = "V_CAUS2"  # after a second (stacked) causative (yap-tır-t); bounded at depth 2
V_PASS = "V_PASS"  # after the passive -Il/-In/-n; takes only the negation/tense continuation
V_NEG = "V_NEG"
V_ABIL = "V_ABIL"  # after ability -(y)Abil; not final (no bare *gelebil), takes further tense
V_IMPOSS = "V_IMPOSS"  # after impossibilitive -(y)AmA (gelemez); non-final, mirrors V_NEG
V_T1 = "V_T1"  # after a tense/aspect that takes the type-1 person set
V_T2 = "V_T2"  # after a tense/aspect that takes the type-2 person set
V_COND = "V_COND"  # after the verbal conditional -sA (gelse); final, no copular -(y)sA restack
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
    #: The verbal voice this edge realizes (``"causative"``/``"passive"``/``"reflexive"``/
    #: ``"reciprocal"``), appended in order to ``features[tags.VOICE]``. ``None`` on every
    #: non-voice suffix. A voice suffix is not derivational (stem/lemma stay the root), and it
    #: is gated off for the guesser (``allow_voice=False``) so OOV stripping is unchanged.
    voice: str | None = None
    #: For the lexically-irregular *first* causative: the allomorph class (``"DIr"``/``"t"``/
    #: ``"Ir"``/``"Ar"``) this edge realizes, matched against the root's own
    #: :attr:`~ilmek.morphology.lexicon.Root.causative` exactly as ``aorist_class`` is. ``None``
    #: on the phonologically-chosen (second / post-voice) causatives and every non-causative.
    causative_class: str | None = None
    #: A declarative *phonological* guard: the edge fires only when the running surface's
    #: final segment falls in this class set — ``"vowel"``/``"l"``/``"r"``/``"other"`` (a
    #: consonant that is not l or r). Used by the passive allomorphs (-In after vowel/l, -Il
    #: after r/other) and the phonologically-chosen causatives. ``None`` -> no phonological
    #: restriction. Kept out of the analyzer as a hardcoded ``if`` — the rule lives in data.
    stem_final_class: frozenset[str] | None = None
    #: The root attribute this edge requires (declared in the lexicon entry), e.g.
    #: ``"reflexive"`` / ``"reciprocal"``. The semi-productive reflexive and reciprocal voices
    #: fire only on a curated verb list, so this is the overgeneration guard for them. ``None``
    #: -> unrestricted (the fully-productive passive and every non-voice suffix).
    requires_attribute: str | None = None
    #: When this suffix is *root-adjacent* (first in the chain), realize it against the root's
    #: :attr:`~ilmek.morphology.lexicon.Root.raised_form` (de->di, ye->yi) instead of its free
    #: form. Models the irregular glide raising of de-/ye- before a vowel-initial suffix
    #: (diyecek, diye, diyen), so it is set only on the vowel-initial edges that trigger it
    #: (FUT, PART_ACAK, OPT, ABIL, PART_AN, IMPOSS). Roots without a raised form are unaffected;
    #: the -(y)Iş verbal noun is deliberately *unflagged* so ``deyiş`` stays regular.
    glide_raise: bool = False


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
# Diminutive / endearment -CIk (kitapçık, evcik, kuşçuk). The C archiphoneme hardens to ç
# after a voiceless consonant for free; ``voice_final`` softens the final k before a vowel
# (kitapçık -> kitapçığı, like -lIk). ``applies_to={NOUN}`` is the overgeneration guard: the
# rule never fires on an ADJ/ADV base (no *güzelcik, *sıcakçık) — the intensive-adjective
# diminutives (sıcacık, küçücük) reshape their stem unpredictably and are enumerated as
# IrregularForm exceptions instead, never generated by this edge.
D_CIK = Suffix(
    "cik",
    "CIk",
    derivational=True,
    to_pos=tags.NOUN,
    applies_to=frozenset({tags.NOUN}),
    voice_final=True,
)

#: Nominal-side derivations, appended after the inflectional edges so inflection-only
#: traversal order (and the guesser, which forbids derivation) is byte-identical to before.
#: -CIk is appended LAST so every pre-existing traversal prefix is unchanged.
_NOMINAL_DERIVATIONS = [D_LI, D_SIZ, D_LIK, D_CI, D_CIK]


def _case_edges(cases: list[Suffix]) -> list[tuple[Suffix, str]]:
    """Case edges, with the accusative retargeted to the terminal ``N_ACC``.

    Every case but the accusative is a grammatical ek-fiil predicate host (evdeydi,
    evdendi, benimleydi-style) and lands in ``N_CASE``, which carries the copula edges. The
    accusative is *not* — ``*eviydi`` read as accusative+copula is ungrammatical — so it goes
    to ``N_ACC`` (terminal, no copula). Split declaratively by the suffix's own case feature,
    not by a hardcoded ``if`` in the analyzer.
    """
    return [(s, N_ACC if s.features.get(tags.CASE) == "accusative" else N_CASE) for s in cases]


def _nominal_inflection() -> list[tuple[Suffix, str]]:
    poss = [(s, N_POSS) for s in _POSSESSIVES_TO_NONE3] + [(s, N_POSS3) for s in _POSSESSIVES_TO_3]
    return [(PLURAL, N_PL), *poss, *_case_edges(_PLAIN_CASES)]


def _nominal_graph(copula: list[tuple[Suffix, str]]) -> dict[str, list[tuple[Suffix, str]]]:
    inflection = _nominal_inflection()
    poss = [(s, N_POSS) for s in _POSSESSIVES_TO_NONE3] + [(s, N_POSS3) for s in _POSSESSIVES_TO_3]
    plain_case = _case_edges(_PLAIN_CASES)
    pronom_case = _case_edges(_PRONOMINAL_CASES)
    deriv = [(s, N_DERIV) for s in _NOMINAL_DERIVATIONS]
    return {
        # Derivation edges appended after inflection, then the ek-fiil (copula) edges last:
        # with derivation *and* copula disabled the prefix of each list is exactly the
        # pre-milestone graph, so plain inflection and the guesser stay byte-identical.
        N_ROOT: [*inflection, *deriv, *copula],
        N_PL: [*poss, *plain_case, *copula],
        N_POSS: [*plain_case, *copula],
        N_POSS3: [*pronom_case, *copula],
        # A non-accusative case is a complete word AND a copular predicate host (evdeydi).
        N_CASE: [*copula],
        # The accusative is terminal: the copula never attaches after it (*eviydi as acc+cop).
        N_ACC: [],
        # A derived stem inflects exactly like a bare root (and hosts the copula: yürüyüştü),
        # but may not derive again.
        N_DERIV: [*inflection, *copula],
        # After the assertive -DIr: terminal (person/plural -DIrlAr stacking deferred).
        N_COP_DIR: [],
    }


NOMINAL_START = N_ROOT
#: Accepting nominal states. ``N_ACC`` (the split-off accusative) is a complete word, so it
#: is final; ``N_COP_DIR`` is *not* here — its closure is the copular-predicate one, below.
NOMINAL_FINALS = frozenset({N_ROOT, N_PL, N_POSS, N_POSS3, N_CASE, N_ACC, N_DERIV})


# --- Verbal suffixes -----------------------------------------------------------------

NEG = Suffix("neg", "mA", {tags.POLARITY: "negative"})

PROG = Suffix(
    "prog", "Iyor", {tags.TENSE: "present", tags.ASPECT: "progressive"}, drop_preceding=True
)
FUT = Suffix("fut", "(y)AcAk", {tags.TENSE: "future"}, voice_final=True, glide_raise=True)
EVID = Suffix("evid", "mIş", {tags.EVIDENTIAL: True})
PAST = Suffix("past", "DI", {tags.TENSE: "past"})

# Copula (ek-fiil) forms stack on a finished tense and take the (y) buffer after a vowel
# (geldi -> geldiydi / geldiymiş), unlike the primary tense suffixes which attach to the
# bare stem. The past copula records a separate ``copula`` feature so it never overwrites a
# primary tense (gelecekti = future + past-copula).
COP_EVID = Suffix("cop_evid", "(y)mIş", {tags.EVIDENTIAL: True})
COP_PAST = Suffix("cop_past", "(y)DI", {tags.COPULA: "past"})
# Copular conditional -(y)sA (güzelse, evse) and the generalizing/assertive -DIr (güzeldir,
# evdedir, kitaptır). These are the *nominal* ek-fiil forms; the copular conditional takes
# the type-2 person set (güzelsen), so it lands in V_COP2 like the past copula. -DIr is
# terminal this milestone (person/plural stacking -DIrlAr is deferred), so it lands in its
# own N_COP_DIR. The VERBAL copular conditional (gelirse) stays deferred and untouched.
COP_COND = Suffix("cop_cond", "(y)sA", {tags.MOOD: "conditional"})
DIR = Suffix("cop_dir", "DIr", {tags.COPULA: "assertive"})

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

#: The nominal ek-fiil layer, shared by every nominal final state (N_ROOT..N_DERIV). It
#: REUSES the verbal copula machinery rather than duplicating it: -(y)DI and -(y)sA feed the
#: type-2 person state V_COP2 (güzeldim, güzelsen), -(y)mIş feeds the type-1 state V_COP1
#: (güzelmişim), and the zero-copula present persons are the very same _PERSON_T1 objects
#: into V_PERS (güzelim, güzelsin, güzeliz, güzelsiniz, and -lAr for güzeller). -DIr is the
#: only edge into a nominal-side terminal (N_COP_DIR). Appended LAST to each nominal final so
#: pre-milestone traversal order (and the derivation-free guesser) is unchanged.
_NOMINAL_COPULA: list[tuple[Suffix, str]] = [
    (COP_PAST, V_COP2),
    (COP_EVID, V_COP1),
    (COP_COND, V_COP2),
    (DIR, N_COP_DIR),
    *[(s, V_PERS) for s in _PERSON_T1],
]

# Optative persons are NOT plain type-1: the 1pl is -lIm (gelelim), never *(y)Iz (*geleyiz).
OPT_1PL = Suffix("pers_1pl", "lIm", {tags.PERSON: "1pl"})
_PERSON_OPT = [P1_1SG, P1_2SG, OPT_1PL, P1_2PL, P1_3PL]

# The negative-aorist person paradigm is *defective*: the 1sg is gelmem and the 1pl is
# gelmeyiz (distinct morphemes), never *gelmezim / *gelmeziz. So -mAz (V_AOR_NEG) takes only
# the 2sg, 2pl and 3pl personal endings; the irregular 1sg/1pl attach to the *bare* negation
# stem instead (gel+me+m, gel+me+yiz), so they are edges from V_NEG (and V_IMPOSS) below.
_PERSON_AOR_NEG = [P1_2SG, P1_2PL, P1_3PL]

# The irregular negative-aorist 1sg/1pl: they attach directly to the -mA negation (gelmem =
# gel+me+m, gelmeyiz = gel+me+yiz) rather than to -mAz, so they are their own suffixes into
# V_PERS. Polarity comes from the preceding NEG; each adds tense=aorist and its person. They
# also serve the impossibilitive (gelemem, gelemeyiz) from V_IMPOSS, whose stem likewise ends
# in the -mA shape (-(y)AmA), so the same two edges apply there.
NEG_AOR_1SG = Suffix("neg_aor_1sg", "m", {tags.TENSE: "aorist", tags.PERSON: "1sg"})
NEG_AOR_1PL = Suffix("neg_aor_1pl", "(y)Iz", {tags.TENSE: "aorist", tags.PERSON: "1pl"})


# --- Ability, aorist, and mood suffixes ----------------------------------------------

# Ability / potential -(y)Abil (gelebilir "can come", okuyabilir): a fully productive slot
# between the root/negation and the tense. After it the stem ends in "bil", so the aorist is
# deterministically -Ir (AOR_ABIL) with no lexical guessing.
ABIL = Suffix("abil", "(y)Abil", {tags.ABILITY: True}, glide_raise=True)

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
#: The post-*voice* aorist is likewise always -Ir: a voiced stem is always consonant-final,
#: so its aorist is deterministically -Ir regardless of the root's lexical class (denir,
#: alınır, okunur, görüşür, yaptırır). Reusing the root-class aorists here would wrongly give
#: ``*dener`` for the passive ``den``; AOR_VOICE carries no ``aorist_class`` guard (like
#: AOR_ABIL) so the class fact is correctly ignored after voice.
AOR_VOICE = Suffix("aor", "Ir", {tags.TENSE: "aorist"})

# Negative aorist -mAz (gelmez, yapmaz): a dedicated negative morpheme (not NEG + aorist).
NEG_AOR = Suffix("neg_aor", "mAz", {tags.POLARITY: "negative", tags.TENSE: "aorist"})

# Conditional -sA (gelse) -> type-2 persons; the past/evidential copula (gelseydi/gelseymiş)
# stacks for free. Optative -(y)A (gele) -> its own person set (V_OPT).
COND = Suffix("cond", "sA", {tags.MOOD: "conditional"})
OPT = Suffix("opt", "(y)A", {tags.MOOD: "optative"}, glide_raise=True)

# Necessitative -mAlI (gelmeli "must come", yapmalıyım): a fully productive mood. It lands in
# V_T1 so it inherits exactly the right continuations — the type-1 persons (gelmeliyim,
# gelmelisin, gelmeliler) and the (y)-buffered copulas (gelmeliydi, gitmeliydik, gelmeliymiş).
NECESS = Suffix("necess", "mAlI", {tags.MOOD: "necessitative"})

# Impossibilitive -(y)AmA (gelemez "cannot come", yapamadı, okuyamam): a distinct morpheme —
# the negative of the ability -(y)Abil, NOT NEG + aorist. It carries polarity=negative AND
# ability=True (so gelemez reads as "cannot come", the ability-negative), and lands in its own
# V_IMPOSS state, whose impossibilitive aorist -z (IMPOSS_AOR) reuses V_AOR_NEG for the
# defective persons (gelemezsin) exactly as the negative aorist does.
IMPOSS = Suffix(
    "imposs", "(y)AmA", {tags.POLARITY: "negative", tags.ABILITY: True}, glide_raise=True
)
IMPOSS_AOR = Suffix("imposs_aor", "z", {tags.TENSE: "aorist"})


# --- Voice (çatı) suffixes -----------------------------------------------------------
# The voice layer sits between the root and negation/tense. Each suffix records its voice
# under features[tags.VOICE] (an ordered tuple, so stacking is preserved). Passive is fully
# productive (phonological guard only); reflexive/reciprocal are semi-productive (a root
# attribute gates them); causative's *first* allomorph is a lexical fact on the root, but a
# second (stacked) causative — and a causative after another voice suffix — is chosen by the
# stem's final segment, since a voiced stem's ending is predictable.

_CVR = frozenset({"vowel", "l", "r"})  # a -t causative attaches after a vowel or l/r
_CONS_OTHER = frozenset({"other"})  # a consonant that is not l or r (D-causative after voice)

# Causative, first (from the bare root / a reciprocal): lexically-irregular allomorph, guarded
# by the root's causative class exactly like the aorist. -DIr (yaptır, güldür), -t (okut,
# oturt), and the lexically-limited -Ir (içir, kaçır) / -Ar (çıkar).
CAUS_DIR = Suffix("caus", "DIr", {}, voice="causative", causative_class="DIr")
CAUS_T = Suffix("caus", "t", {}, voice="causative", causative_class="t")
CAUS_IR = Suffix("caus", "Ir", {}, voice="causative", causative_class="Ir")
CAUS_AR = Suffix("caus", "Ar", {}, voice="causative", causative_class="Ar")
# Causative, post-voice / second (stacked): the allomorph is phonologically predictable, so it
# is guarded by the stem's final-segment class, not the root fact. -t after a vowel/l/r
# (yaptır-t, çıkar-t), -DIr after any other consonant (okut-tur, and görüş-tür after the ş of a
# reciprocal). D->t hardening after a voiceless stop is free in the phonology.
CAUS2_T = Suffix("caus", "t", {}, voice="causative", stem_final_class=_CVR)
CAUS2_DIR = Suffix("caus", "DIr", {}, voice="causative", stem_final_class=_CONS_OTHER)

# Passive, fully productive: -In after a vowel or l (oku-n, de-n, al-ın, bul-un — the (I)
# linking vowel collapses -n / -In for free), -Il after r or any other consonant (yap-ıl,
# yaz-ıl, otur-ul). The final-segment guard is what blocks *oku+l -> "okul" as a fake passive.
PASS_IN = Suffix("pass", "(I)n", {}, voice="passive", stem_final_class=frozenset({"vowel", "l"}))
PASS_IL = Suffix("pass", "(I)l", {}, voice="passive", stem_final_class=frozenset({"r", "other"}))

# Reflexive -In (yıka-n, giy-in, tara-n) and reciprocal/collective -Iş (gör-üş, bak-ış, döv-üş,
# anla-ş): semi-productive, so each is gated by a curated root attribute. -In shares its shape
# with the passive, so an attributed vowel/l-final verb (yıka, al) yields BOTH readings. -Iş
# shares its shape with the verbal-noun -(y)Iş, but note the reciprocal takes NO (y) buffer
# after a vowel (anla-ş, distinct from the verbal noun anla-(y)Iş = anlayış).
REFL_IN = Suffix("refl", "(I)n", {}, voice="reflexive", requires_attribute="reflexive")
RECIP_IS = Suffix("recip", "(I)ş", {}, voice="reciprocal", requires_attribute="reciprocal")

#: The reflexive/reciprocal edges leaving the bare root (both land in the shared V_RECIP).
_VOICE_REFL_RECIP = [(REFL_IN, V_RECIP), (RECIP_IS, V_RECIP)]
#: The lexically-guarded first-causative edges leaving the bare root (into V_CAUS1).
_VOICE_CAUS1 = [
    (CAUS_DIR, V_CAUS1),
    (CAUS_T, V_CAUS1),
    (CAUS_IR, V_CAUS1),
    (CAUS_AR, V_CAUS1),
]
#: The two passive edges (into V_PASS); exactly one fires per stem (guards are disjoint).
_VOICE_PASS = [(PASS_IL, V_PASS), (PASS_IN, V_PASS)]


def _post_voice_caus(target: str) -> list[tuple[Suffix, str]]:
    """Phonologically-guarded causative edges after a reciprocal (-> V_CAUS1) or a first
    causative (-> V_CAUS2). A voiced stem's ending is predictable, so no root fact is used."""
    return [(CAUS2_T, target), (CAUS2_DIR, target)]


# --- Verbal derivational suffixes (verb -> new nominal/adjectival stem) ---------------
# Gated by graph position (only reachable from V_ROOT/V_NEG), so no applies_to is needed.
# verb -> noun:
VN_MA = Suffix("ma", "mA", derivational=True, to_pos=tags.NOUN)  # gelme (verbal noun / act)
VN_IS = Suffix("is", "(y)Iş", derivational=True, to_pos=tags.NOUN)  # geliş, yürüyüş
INF = Suffix("mak", "mAk", derivational=True, to_pos=tags.NOUN)  # gelmek (infinitive)
# verb -> adjective (participles):
PART_AN = Suffix(
    "an", "(y)An", derivational=True, to_pos=tags.ADJ, glide_raise=True
)  # gelen, diyen
PART_DIK = Suffix("dik", "DIk", derivational=True, to_pos=tags.ADJ, voice_final=True)  # bildiği
PART_ACAK = Suffix(
    "acak", "(y)AcAk", derivational=True, to_pos=tags.ADJ, voice_final=True, glide_raise=True
)  # gelecek, diyecek

#: Verb-side derivations that land in the shared N_DERIV nominal state (then inflect). The
#: infinitive -mAk is handled separately: it lands in the terminal V_INF (no case yet).
_VERBAL_DERIVATIONS_TO_NOMINAL = [VN_MA, VN_IS, PART_AN, PART_DIK, PART_ACAK]


def _primary_from() -> list[tuple[Suffix, str]]:
    return [(PROG, V_T1), (FUT, V_T1), (EVID, V_T1), (PAST, V_T2)]


def _root_continuation(aorist_edges: list[tuple[Suffix, str]]) -> list[tuple[Suffix, str]]:
    """The negation / tense / mood / derivation edges shared by the bare root AND every
    voiced stem. The only difference is the aorist: the bare root selects its lexically-
    irregular allomorph via ``aorist_class`` (``aorist_edges`` = the three guarded -r/-Ar/-Ir
    edges), whereas a voiced stem is always consonant-final and takes the deterministic -Ir
    (``aorist_edges`` = the single unguarded AOR_VOICE). Everything else — negation, the four
    primary tenses, ability, the negative aorist, conditional, optative, and the verb->nominal
    derivations — is identical, so görüşme, yaptırabilir, yapılmaz, okutmak all fall out free.
    """
    return [
        (NEG, V_NEG),
        *_primary_from(),
        (ABIL, V_ABIL),
        *aorist_edges,
        (NEG_AOR, V_AOR_NEG),
        (COND, V_COND),
        (OPT, V_OPT),
        *[(s, N_DERIV) for s in _VERBAL_DERIVATIONS_TO_NOMINAL],
        (INF, V_INF),
        # Appended at the end so the pre-existing traversal prefix (and the guesser's ordering)
        # is byte-stable: the necessitative mood and the impossibilitive voice-negative.
        (NECESS, V_T1),
        (IMPOSS, V_IMPOSS),
    ]


def _verbal_graph() -> dict[str, list[tuple[Suffix, str]]]:
    # The bare root's aorist is the three lexically-guarded allomorph edges; a voiced stem's
    # aorist is the single deterministic -Ir (AOR_VOICE), never the root's class.
    root_aorists = [(s, V_T1) for s in _AORISTS]
    voiced_aorist = [(AOR_VOICE, V_T1)]
    deriv = [(s, N_DERIV) for s in _VERBAL_DERIVATIONS_TO_NOMINAL] + [(INF, V_INF)]
    # The copular layer that stacks on a *finished* verbal tense: the evidential -(y)mIş, the
    # past -(y)DI, and — this milestone — the copular conditional -(y)sA (gelirse, geldiyse,
    # gelecekse, geliyorsa, gelmezse). All three take the (y) buffer after a vowel and feed the
    # copula person states, so gelirsem/gelirsek fall out free from V_COP2. COP_COND is appended
    # LAST so the pre-existing traversal prefix of every state using ``copula`` is byte-stable.
    copula = [(COP_EVID, V_COP1), (COP_PAST, V_COP2), (COP_COND, V_COP2)]
    pers_t1 = [(s, V_PERS) for s in _PERSON_T1]
    pers_t2 = [(s, V_PERS) for s in _PERSON_T2]
    # The bare verbal conditional -sA lands in its OWN state V_COND (not V_T2), whose copula is
    # the *old* set WITHOUT COP_COND: this is what blocks a conditional-on-conditional restack
    # (*gelseyse) while V_T2 (reached only by the past -DI) does license geldiyse. Both still
    # take -(y)DI / -(y)mIş (gelseydi, gelseymiş) and the type-2 persons (gelsek, gelsem).
    cond_copula = [(COP_EVID, V_COP1), (COP_PAST, V_COP2)]
    # The voice-progression edges are appended AFTER the bare root's full continuation, so the
    # pre-milestone traversal prefix of V_ROOT is byte-identical: plain inflection and the
    # guesser (which forbids voice) see the same order as before.
    voice_from_root = [*_VOICE_REFL_RECIP, *_VOICE_CAUS1, *_VOICE_PASS]
    return {
        V_ROOT: [*_root_continuation(root_aorists), *voice_from_root],
        # After negation: the primary tenses, ability (gelmeyebilir), conditional (gelmese),
        # optative (gelmeye), the necessitative (gelmemeli). The aorist and negative-aorist -mAz
        # edges are intentionally absent — the aorist's own negative is -mAz on the bare root,
        # not NEG + aorist (*gelmemez) — but the irregular negative-aorist 1sg/1pl DO attach to
        # this -mA stem (gelmem, gelmeyiz), appended last (their surface is not *gelmez-based).
        V_NEG: [
            *_primary_from(),
            (ABIL, V_ABIL),
            (COND, V_COND),
            (OPT, V_OPT),
            *deriv,
            (NECESS, V_T1),
            (NEG_AOR_1SG, V_PERS),
            (NEG_AOR_1PL, V_PERS),
        ],
        # Ability: primary tenses, the deterministic -Ir aorist, conditional, the necessitative
        # (gelebilmeli), and the verbal derivations (gelebilmek, gelebilen). Not final -> no bare
        # *gelebil.
        V_ABIL: [*_primary_from(), (AOR_ABIL, V_T1), (COND, V_COND), *deriv, (NECESS, V_T1)],
        # Impossibilitive -(y)AmA (gelemez, yapamadı): mirrors V_NEG (it is the ability-negative)
        # — the primary tenses (gelemiyor via -Iyor's vowel drop, gelemeyecek), ability
        # (gelemeyebilir), conditional (gelemese), optative (gelemeye), necessitative, and the
        # verb->nominal derivations (gelemeyen, gelememek) — PLUS its aorist -z into V_AOR_NEG
        # (gelemez = 3sg, gelemezsin/gelemezler defective persons, gelemezdi/gelemezse copula)
        # and the irregular 1sg/1pl (gelemem, gelemeyiz). Non-final: bare *geleme is not a word.
        V_IMPOSS: [
            *_primary_from(),
            (ABIL, V_ABIL),
            (COND, V_COND),
            (OPT, V_OPT),
            *deriv,
            (NECESS, V_T1),
            (IMPOSS_AOR, V_AOR_NEG),
            (NEG_AOR_1SG, V_PERS),
            (NEG_AOR_1PL, V_PERS),
        ],
        # Voice states: each takes the shared voiced-stem continuation (with the deterministic
        # -Ir aorist), plus the onward voice progression. None is a final state, so a bare
        # voiced stem is not accepted (see V_RECIP's note above). Order refl/recip < caus(<=2)
        # < pass is enforced by which progression edges each state exposes.
        # After reflexive/reciprocal: a (phonologically-chosen) causative or a passive.
        V_RECIP: [*_root_continuation(voiced_aorist), *_post_voice_caus(V_CAUS1), *_VOICE_PASS],
        # After the first causative: a second (stacked) causative or a passive.
        V_CAUS1: [*_root_continuation(voiced_aorist), *_post_voice_caus(V_CAUS2), *_VOICE_PASS],
        # After a second causative: only a passive (depth is bounded at two causatives).
        V_CAUS2: [*_root_continuation(voiced_aorist), *_VOICE_PASS],
        # After the passive: the continuation only (voice is closed; no double passive here —
        # the impersonal double passive aranıldı is deferred).
        V_PASS: [*_root_continuation(voiced_aorist)],
        V_T1: [*copula, *pers_t1],
        # V_T2 is reached only by the past -DI, so its copula CAN include COP_COND (geldiyse).
        V_T2: [*copula, *pers_t2],
        # V_COND is reached by the bare conditional -sA; its copula omits COP_COND to block the
        # ungrammatical conditional restack (*gelseyse), while keeping gelseydi/gelseymiş/gelsek.
        V_COND: [*cond_copula, *pers_t2],
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
# V_INF (gelmek) is final: a bare infinitive is a complete word. V_AOR_NEG (gelmez), V_OPT
# (gele) and V_COND (gelse) are final: a bare negative-aorist / optative / conditional 3sg is
# a complete word. V_IMPOSS (geleme) is NOT final — a bare impossibilitive is not a word (it
# needs at least the aorist -z or a tense), so it is deliberately absent here.
# The voice states (V_RECIP, V_CAUS1, V_CAUS2, V_PASS) are deliberately NOT final this
# milestone: a bare voiced stem is a real 2sg imperative (yıkan!) but making it final would
# rank it above the homograph nouns (sorun, alın) and the -Iş verbal nouns (görüş, geliş).
VERBAL_FINALS = frozenset(
    {V_ROOT, V_NEG, V_T1, V_T2, V_COND, V_COP1, V_COP2, V_AOR_NEG, V_OPT, V_PERS, V_INF}
)


# --- Unified graph -------------------------------------------------------------------
# The nominal and verbal state names are disjoint, so the two transition maps merge into
# one graph the analyzer walks from the POS-appropriate start state. Verbal derivations
# cross into the nominal N_DERIV state, so a single traversal spans both sides. The nominal
# graph is assembled here (rather than at its definition) because the ek-fiil edges it
# appends reference the verbal person sets and copula suffixes defined above.

NOMINAL_GRAPH = _nominal_graph(_NOMINAL_COPULA)
GRAPH: dict[str, list[tuple[Suffix, str]]] = {**NOMINAL_GRAPH, **VERBAL_GRAPH}
#: The copular (ek-fiil) landing states: the type-1/type-2 person states shared with the
#: verbal copula, the present-person state, and the assertive -DIr terminal. A NOMINAL
#: predicate accepted here takes nominal-predicate finalization; a verbal path reaching the
#: same person/copula states (cur_pos == VERB) still takes verbal finalization.
COPULA_STATES = frozenset({V_COP1, V_COP2, V_PERS, N_COP_DIR})
FINALS = NOMINAL_FINALS | VERBAL_FINALS | {N_COP_DIR}
#: States whose accepting side is *nominal*: they take nominal feature defaults and never
#: run verbal finalization. This includes the shared derived state and the infinitive, so a
#: verb-derived noun (gelme, gelmek) gets no fabricated person/mood but keeps any polarity.
#: (N_COP_DIR is deliberately excluded: its closure is the copular-predicate one.)
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


def finalize_nominal_predicate_features(features: dict) -> dict:
    """Fill implicit features at an ek-fiil (copular) acceptance on a NOMINAL predicate.

    The nominal number/possessive/case fill in *under* whatever the chain accrued, so a
    copular reading never loses its case/number (evlerimizdeydi keeps plural+1pl+locative);
    person defaults to the zero third person. Crucially we do NOT fabricate a verbal
    polarity/tense/mood — the zero copula is represented by ``person`` plus whatever explicit
    ``copula``/``evidential``/``mood`` key the ek-fiil suffix already added (negation of a
    nominal predicate is the separate word ``değil``, out of scope this milestone).
    """
    merged = {**nominal_default_features(), **features}
    merged.setdefault(tags.PERSON, "3sg")
    return merged

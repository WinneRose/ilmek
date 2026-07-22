"""Transitions: the state→(suffix, target) graphs the analyzer walks, plus feature closure.

This module wires the suffix instances (from :mod:`.noun_suffixes`, :mod:`.verb_suffixes`,
:mod:`.derivational_suffixes`) onto the states (:mod:`.states`) into the nominal and verbal
graphs, merges them into the unified :data:`GRAPH`, and defines the accepting-state sets and
the feature-closure functions run at acceptance. Everything that pairs a suffix with a target
state lives here; the suffix definitions themselves stay data-only in their category modules.

See the package docstring (:mod:`ilmek.morphology.morphotactics`) for the full design.
"""

from __future__ import annotations

from ...core import tags
from .derivational_suffixes import (
    _NOMINAL_DERIVATIONS,
    _PART_AORISTS,
    _VERBAL_DERIVATIONS_TO_NOMINAL,
    INF,
    PART_AOR_VOICE,
    PART_MAZ,
)
from .noun_suffixes import (
    _PLAIN_CASES,
    _POSSESSIVES_TO_3,
    _POSSESSIVES_TO_NONE3,
    _PRONOMINAL_CASES,
    DIST,
    PLURAL,
)
from .states import (
    N_ACC,
    N_CASE,
    N_COP_DIR,
    N_DERIV,
    N_DIST,
    N_PART_NEG,
    N_PL,
    N_POSS,
    N_POSS3,
    N_ROOT,
    Q_ROOT,
    V_ABIL,
    V_AOR_NEG,
    V_CAUS1,
    V_CAUS2,
    V_COND,
    V_COP1,
    V_COP2,
    V_IMPOSS,
    V_INF,
    V_NEG,
    V_OPT,
    V_PASS,
    V_PERS,
    V_RECIP,
    V_ROOT,
    V_T1,
    V_T2,
)
from .suffix import Suffix
from .verb_suffixes import (
    _AORISTS,
    _PERSON_AOR_NEG,
    _PERSON_OPT,
    _PERSON_T1,
    _PERSON_T2,
    ABIL,
    AOR_ABIL,
    AOR_VOICE,
    CAUS2_DIR,
    CAUS2_T,
    CAUS_AR,
    CAUS_DIR,
    CAUS_IR,
    CAUS_T,
    COND,
    COP_COND,
    COP_EVID,
    COP_PAST,
    DIR,
    EVID,
    FUT,
    IMPOSS,
    IMPOSS_AOR,
    NECESS,
    NEG,
    NEG_AOR,
    NEG_AOR_1PL,
    NEG_AOR_1SG,
    OPT,
    PASS_IL,
    PASS_IN,
    PAST,
    PROG,
    RECIP_IS,
    REFL_IN,
)

# --- Nominal ek-fiil (copula) landing edges ------------------------------------------

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


# --- Interrogative-particle edges ----------------------------------------------------

#: The only edges leaving the interrogative-particle root Q_ROOT. It REUSES the shared nominal
#: ek-fiil layer (:data:`_NOMINAL_COPULA`) rather than duplicating it, so every inflected
#: question form (midir, misin, miyim, miydi, miymiş, and their harmonic variants) falls out
#: for free — the copula/persons harmonize to the particle's own vowel via the phonology.
#: Two edges of the nominal copula are declaratively filtered out (not by a hardcoded surface
#: list, by the suffix's own feature):
#:   * the PRESENT 3pl person -lAr: it would surface *miler, exactly the bare plural the
#:     particle must not take. (3pl AFTER a copula — mıydılar, mıymışlar — stays licensed,
#:     because it is emitted by V_COP1/V_COP2's own untouched person sets, not this edge.)
#:   * the copular conditional -(y)sA: *miyse/mıysa is at best marginal, so under "correctness
#:     over coverage" it is excluded (a negative test pins the choice).
#: What remains: the past copula -(y)DI, the evidential -(y)mIş, the assertive -DIr, and the
#: present persons 1sg/2sg/1pl/2pl.
_Q_COPULA: list[tuple[Suffix, str]] = [
    (s, t)
    for s, t in _NOMINAL_COPULA
    if s.features.get(tags.PERSON) != "3pl" and s.features.get(tags.MOOD) != "conditional"
]


# --- Voice progression edges ---------------------------------------------------------

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


# --- Nominal graph -------------------------------------------------------------------


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
        # Derivation edges appended after inflection, then the ek-fiil (copula) edges, then the
        # distributive LAST: with derivation, copula *and* distributive disabled/unmatched the
        # prefix of each list is exactly the pre-milestone graph, so plain inflection and the
        # guesser stay byte-identical. The distributive edge fires only on a NUM stem (applies_to).
        N_ROOT: [*inflection, *deriv, *copula, (DIST, N_DIST)],
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
        # The negative-aorist participle -mAz: CASE only (çıkmazda), then N_CASE/N_ACC. It takes
        # NO plural, possessive or copula — the -Im/-Iz those add would resurrect the deliberately
        # -defective finite negative-aorist persons (*gelmezim / *gelmeziz), which an existing
        # test pins as invalid. None of the plain case suffixes (-(y)I/-(y)A/-DA/-DAn/-(n)In/
        # -(y)lA) surface as -Im/-Iz, so the participle stays adjectival. Final (bare çıkmaz).
        N_PART_NEG: [*plain_case],
        # After the assertive -DIr: terminal (person/plural -DIrlAr stacking deferred).
        N_COP_DIR: [],
        # After the distributive -(ş)Ar: terminal this milestone (birerden/birerdi deferred).
        N_DIST: [],
    }


NOMINAL_START = N_ROOT
#: Accepting nominal states. ``N_ACC`` (the split-off accusative) is a complete word, so it
#: is final; ``N_DIST`` (the distributive, birer) is a complete word too. ``N_COP_DIR`` is
#: *not* here — its closure is the copular-predicate one, below.
NOMINAL_FINALS = frozenset(
    {N_ROOT, N_PL, N_POSS, N_POSS3, N_CASE, N_ACC, N_DERIV, N_PART_NEG, N_DIST}
)


# --- Verbal graph --------------------------------------------------------------------


def _primary_from() -> list[tuple[Suffix, str]]:
    return [(PROG, V_T1), (FUT, V_T1), (EVID, V_T1), (PAST, V_T2)]


def _root_continuation(
    aorist_edges: list[tuple[Suffix, str]],
    part_aorist_edges: list[tuple[Suffix, str]],
) -> list[tuple[Suffix, str]]:
    """The negation / tense / mood / derivation edges shared by the bare root AND every
    voiced stem. The only difference is the aorist: the bare root selects its lexically-
    irregular allomorph via ``aorist_class`` (``aorist_edges`` = the three guarded -r/-Ar/-Ir
    edges), whereas a voiced stem is always consonant-final and takes the deterministic -Ir
    (``aorist_edges`` = the single unguarded AOR_VOICE). Everything else — negation, the four
    primary tenses, ability, the negative aorist, conditional, optative, and the verb->nominal
    derivations — is identical, so görüşme, yaptırabilir, yapılmaz, okutmak all fall out free.

    ``part_aorist_edges`` is the aorist *participle* set, parallel to ``aorist_edges`` (the three
    class-guarded edges on the bare root, the single -Ir on a voiced stem). It is appended LAST,
    together with the negative-aorist participle -mAz, so the pre-existing traversal prefix stays
    byte-stable AND the participle scope mirrors the finite side exactly: an aorist/-mAz
    participle is only ever reachable where the finite aorist/negative-aorist already is, so no
    new *bare* surface is accepted (akar/gelir/bitmez all already parse finitely) — only inflected
    continuations (geçmişi, çıkmazda, akarlar) are genuinely new. -mAz lands in the case-only
    N_PART_NEG (see states), never N_DERIV.
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
        # The sıfat-fiil (participle) aorist edges, appended LAST (byte-stable prefix). The -mAz
        # participle is here too (so it fires from the bare root AND voiced stems: yapılmaz-ADJ),
        # but NOT from V_NEG, which builds its edges directly — that is the *gelmemez gap.
        *part_aorist_edges,
        (PART_MAZ, N_PART_NEG),
    ]


def _verbal_graph() -> dict[str, list[tuple[Suffix, str]]]:
    # The bare root's aorist is the three lexically-guarded allomorph edges; a voiced stem's
    # aorist is the single deterministic -Ir (AOR_VOICE), never the root's class.
    root_aorists = [(s, V_T1) for s in _AORISTS]
    voiced_aorist = [(AOR_VOICE, V_T1)]
    # The aorist *participle* edges, parallel to the finite ones: the bare root uses the three
    # class-guarded -r/-Ar/-Ir (akar, gelir, okur), a voiced stem the deterministic -Ir
    # (okunur). All cross into the shared derived-nominal state N_DERIV (so akarlar/geçmişi
    # inflect like the other participles); no aorist participle from V_NEG/V_ABIL this milestone.
    part_root_aorists = [(s, N_DERIV) for s in _PART_AORISTS]
    part_voiced_aorist = [(PART_AOR_VOICE, N_DERIV)]
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
        V_ROOT: [*_root_continuation(root_aorists, part_root_aorists), *voice_from_root],
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
        V_RECIP: [
            *_root_continuation(voiced_aorist, part_voiced_aorist),
            *_post_voice_caus(V_CAUS1),
            *_VOICE_PASS,
        ],
        # After the first causative: a second (stacked) causative or a passive.
        V_CAUS1: [
            *_root_continuation(voiced_aorist, part_voiced_aorist),
            *_post_voice_caus(V_CAUS2),
            *_VOICE_PASS,
        ],
        # After a second causative: only a passive (depth is bounded at two causatives).
        V_CAUS2: [*_root_continuation(voiced_aorist, part_voiced_aorist), *_VOICE_PASS],
        # After the passive: the continuation only (voice is closed; no double passive here —
        # the impersonal double passive aranıldı is deferred).
        V_PASS: [*_root_continuation(voiced_aorist, part_voiced_aorist)],
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
#: The interrogative-particle graph: Q_ROOT's only edges are the filtered ek-fiil ones. It has
#: no incoming edges (it is reached solely as a start state, gated by the root attribute), so a
#: single-entry map merged into GRAPH is enough — its copula targets (V_COP1/V_COP2/V_PERS/
#: N_COP_DIR) already exist on the verbal/nominal side.
Q_GRAPH: dict[str, list[tuple[Suffix, str]]] = {Q_ROOT: _Q_COPULA}
GRAPH: dict[str, list[tuple[Suffix, str]]] = {**NOMINAL_GRAPH, **VERBAL_GRAPH, **Q_GRAPH}
Q_START = Q_ROOT
#: The copular (ek-fiil) landing states: the type-1/type-2 person states shared with the
#: verbal copula, the present-person state, and the assertive -DIr terminal. A NOMINAL
#: predicate accepted here takes nominal-predicate finalization; a verbal path reaching the
#: same person/copula states (cur_pos == VERB) still takes verbal finalization.
COPULA_STATES = frozenset({V_COP1, V_COP2, V_PERS, N_COP_DIR})
# Q_ROOT is final (bare mi is a complete word). It is deliberately kept OUT of
# NOMINAL_FINALS/NOMINAL_STATES/COPULA_STATES so the nominal closure and the guesser's copula
# gate are provably untouched: its own closure (question=True, no fabricated person) runs from
# the dedicated Q_ROOT branch in the analyzer.
FINALS = NOMINAL_FINALS | VERBAL_FINALS | {N_COP_DIR, Q_ROOT}
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


def interrogative_default_features() -> dict:
    """The features a bare interrogative particle starts (and, alone, ends) with.

    Only ``question=True`` — the particle is not a noun, so it fabricates NO
    number/possessive/case. Inflected forms accrue their copula/person keys on top of this.
    """
    return {tags.QUESTION: True}


def finalize_particle_predicate_features(features: dict) -> dict:
    """Fill implicit features at an ek-fiil (copular) acceptance on the interrogative PARTICLE.

    Unlike :func:`finalize_nominal_predicate_features`, it fabricates NO nominal
    number/possessive/case (mi is not a full noun): it only defaults the zero third person, so
    midir = question+copula:assertive+3sg and misin = question+2sg, nothing else. ``question``
    is already present (threaded from :func:`interrogative_default_features`).
    """
    features.setdefault(tags.PERSON, "3sg")
    return features

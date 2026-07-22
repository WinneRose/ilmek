"""Morphotactic state names.

Plain string constants naming every state in the noun and verb suffix-transition graphs.
The nominal (``N_*``) and verbal (``V_*``) names are disjoint, so the two graphs merge into
one map the analyzer walks from the POS-appropriate start state. Kept dependency-free so
both the suffix definitions and the transition graphs can import them without cycles.
"""

from __future__ import annotations

# --- Nominal states ------------------------------------------------------------------

N_ROOT = "N_ROOT"
N_PL = "N_PL"
N_POSS = "N_POSS"  # non-3rd-person possessive
N_POSS3 = "N_POSS3"  # 3rd-person possessive (triggers pronominal -n- before case)
N_CASE = "N_CASE"  # a non-accusative case (grammatical ek-fiil predicate host: evdeydi)
N_ACC = "N_ACC"  # the accusative, split off as terminal: the copula never follows it (*eviydi)
N_DERIV = "N_DERIV"  # a derived nominal/adjectival stem (inflects like N_ROOT, cannot re-derive)
N_COP_DIR = "N_COP_DIR"  # after the assertive/generalizing ek-fiil -DIr; terminal this milestone
N_DIST = "N_DIST"  # after the distributive numeral suffix -(ş)Ar (birer, ikişer); terminal

# --- Interrogative-particle state ----------------------------------------------------

# The root state of the interrogative (question) particle mi/mı/mu/mü. It is FINAL (a bare mi
# is a complete word) and its ONLY outgoing edges are the ek-fiil (nominal-copula) edges —
# the past/evidential/assertive copula and the present persons (midir, misin, miyim, miydi,
# miymiş) — reusing the verbal copula/person landing states. It deliberately takes NO
# plural/possessive/case edge (mi is not a full noun: no *miler, no *mide as mi+locative).
Q_ROOT = "Q_ROOT"

# --- Verbal states -------------------------------------------------------------------

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

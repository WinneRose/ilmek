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
# The locative/genitive case: like N_CASE (a grammatical ek-fiil host: evdeydi, evinindi) BUT
# it ALSO hosts the relative/pronominal -ki (evdeki, evinki, kiminki) — the only two cases that
# do. Split off from N_CASE (which keeps the copula edges as its identical leading prefix) so
# the -ki edge attaches to exactly the locative and genitive and nowhere else (*evdenki blocked).
N_CASE_LG = "N_CASE_LG"
N_ACC = "N_ACC"  # the accusative, split off as terminal: the copula never follows it (*eviydi)
# After the relative/pronominal -ki (evdeki, benimki, dünkü). A complete ADJ/pronominal that
# declines like a pronoun (buffer-n before case: evdekini, evdekinde), pluralizes (evdekiler),
# and hosts the ek-fiil (evdekiydi). Final; reached from N_CASE_LG (evde-ki), from a temporal
# noun/adv at the root (dün-kü), and from a genitive/locative pronoun host (benim-ki).
N_KI = "N_KI"
# After the equative/adverbial -CA (güzelce, insanca, çocukça): a verb-less ADVERB derivation.
# TERMINAL — güzelce takes no further nominal inflection (no *güzelceyi/*güzelceler) — so, like
# the converb state ADV_CVB, it is a dead-end; its accepting closure drops the nominal defaults.
N_ADV_CA = "N_ADV_CA"
# After the ordinal -(I)ncI (birinci, ikinci, sonuncu): a fully nominal ordinal numeral. It
# inflects like a bare root (birincisi, ikincide, birinciler) and hosts the ek-fiil
# (birinciydi), so its edges are the shared nominal inflection plus the copula. Final.
N_ORD = "N_ORD"
N_DERIV = "N_DERIV"  # a derived nominal/adjectival stem (inflects like N_ROOT, cannot re-derive)
# The negative-aorist participle -mAz (çıkmaz, tükenmez). It inflects for CASE only (çıkmazda),
# taking NEITHER the possessive NOR the ek-fiil copula — both attach -Im/-Iz, which would revive
# the deliberately-defective finite negative-aorist persons (*gelmezim / *gelmeziz). So it lands
# here rather than in the full N_DERIV, keeping the participle adjectival. Final (bare çıkmaz).
N_PART_NEG = "N_PART_NEG"
N_COP_DIR = "N_COP_DIR"  # after the assertive/generalizing ek-fiil -DIr; terminal this milestone
N_DIST = "N_DIST"  # after the distributive numeral suffix -(ş)Ar (birer, ikişer); terminal

# --- Relative-particle host state (benimki, seninki, ondaki) -------------------------

# The start state for a genitive/locative PRONOUN that hosts the relative -ki (benim->benimki,
# senin->seninki, onun->onunki, onda->ondaki). NOT final — a bare "benim" is not accepted here
# (it is covered by the enumerated irregular pronoun); its ONLY outgoing edge is the pronominal
# -ki into N_KI, so this host licenses nothing but -ki (no *benimler / *benimde). Reached solely
# as a start state, gated by the ``ki_host`` root attribute, so a guess can never reach it.
KI_HOST_ROOT = "KI_HOST_ROOT"

# --- Interrogative-particle state ----------------------------------------------------

# The root state of the interrogative (question) particle mi/mı/mu/mü. It is FINAL (a bare mi
# is a complete word) and its ONLY outgoing edges are the ek-fiil (nominal-copula) edges —
# the past/evidential/assertive copula and the present persons (midir, misin, miyim, miydi,
# miymiş) — reusing the verbal copula/person landing states. It deliberately takes NO
# plural/possessive/case edge (mi is not a full noun: no *miler, no *mide as mi+locative).
Q_ROOT = "Q_ROOT"

# --- Negative-copula-particle state --------------------------------------------------

# The root state of the negative copula değil (the negative mirror of the ek-fiil: değildi is
# the negative of güzeldi). FINAL — bare değil is a complete word — and, exactly like Q_ROOT,
# its only outgoing edges are the ek-fiil (nominal-copula) layer, reusing the verbal copula/
# person landing states so değildi/değildim/değilim/değilsin/değildir/değilse/değilmiş/değiller
# all fall out for free (harmonizing off değil's front vowel). Unlike Q_ROOT the layer is the
# FULL, unfiltered one (değilse and değiller ARE valid). Polarity=negative is inherent to değil
# (seeded as a base feature, never overwritten by a copula/person suffix). It takes NO
# plural/possessive/case edge (değil is a particle, not a noun: no *değile, no *değilde).
NEG_COP_ROOT = "NEG_COP_ROOT"

# --- Substantive-verb state ----------------------------------------------------------

# The root state of the standalone substantive verb i- (the ek-fiil written as a SEPARATE word:
# idi, imiş, ise, iken). NOT final — a bare "i" is not a word — so acceptance is structurally
# impossible until a copula/converb edge fires. Its only outgoing edges are the four BUFFERLESS
# substantive-verb suffixes (idi, NOT *iydi): the bufferless past -DI / evidential -mIş / con-
# ditional -sA feed the shared copula-person states (V_COP2/V_COP1) so the persons come free
# (idim, imişsin, isem), and a bufferless -ken converb lands in the terminal ADV_CVB (iken).
# It carries NO polarity (i- is polarity-neutral — negation is a separate word, değil).
I_ROOT = "I_ROOT"

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
# After the formal progressive -mAktA (gelmekte, yazmakta): a finite present-continuous. FINAL
# (bare gelmekte = 3sg). Its edges are the copula layer (gelmekteydi/gelmekteymiş), the type-1
# persons (gelmekteyim/gelmekteler), and the assertive -DIr (gelmektedir). -DIr sits HERE only,
# never on V_T1, so the generalizing *gelirdir stays deferred.
V_MAKTA = "V_MAKTA"

# --- Adverbial (converb / zarf-fiil) state -------------------------------------------

# A verb-derived ADVERB (zarf-fiil / converb): gelerek, gelip, gelince, gelmeden, geleli,
# geldikçe, gelmeksizin, gelirken. TERMINAL — a converb is a complete adverb that takes NO
# further inflection (no plural/possessive/case, no copula, no verbal person/mood), so its
# only outgoing edge set is empty. Kept OUT of the nominal N_DERIV precisely so an adverb can
# never fabricate nominal number/possessive/case nor license *gelerekler/*gelerekte — the
# no-case/no-plural guard is structural (a dead-end state), not a filter in the analyzer.
ADV_CVB = "ADV_CVB"

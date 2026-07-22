"""Verbal suffixes: negation, tense/aspect, copula, persons, ability/aorist/mood, and voice.

Suffix *instances* and the person-set groupings only. The graphs that order them (the voice
chain, the copular landings, the person paradigms per state) live in :mod:`.transitions`.
"""

from __future__ import annotations

from ...core import tags
from .suffix import Suffix

# --- Core tense / aspect -------------------------------------------------------------

NEG = Suffix("neg", "mA", {tags.POLARITY: "negative"})

PROG = Suffix(
    "prog", "Iyor", {tags.TENSE: "present", tags.ASPECT: "progressive"}, drop_preceding=True
)
FUT = Suffix("fut", "(y)AcAk", {tags.TENSE: "future"}, voice_final=True, glide_raise=True)
EVID = Suffix("evid", "mIş", {tags.EVIDENTIAL: True})
PAST = Suffix("past", "DI", {tags.TENSE: "past"})

# Formal / written present-continuous -mAktA (gelmekte, yazmakta): the same present +
# progressive as -Iyor, but a distinct morpheme (the name "makta" keeps the two apart — no
# fabricated feature). It is consonant-initial, so it never drops a preceding vowel (okumakta)
# nor triggers root voicing. It lands in its own V_MAKTA state, whose assertive -DIr (gelmektedir)
# and copula/persons (gelmekteydi, gelmekteyim) are wired in :mod:`.transitions`.
MAKTA = Suffix("makta", "mAktA", {tags.TENSE: "present", tags.ASPECT: "progressive"})

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

# BUFFERLESS copula variants for the STANDALONE substantive verb i- (idi, imiş, ise). The
# free-standing ek-fiil is vowel-final ("i") yet takes NO (y) buffer — the forms are idi/imiş/
# ise, never *iydi/*iymiş/*iyse — so the (y)-buffered COP_* above cannot be reused here (they
# would realize the glide after the vowel). Same features/targets as their buffered twins; wired
# ONLY onto I_ROOT (see transitions.I_GRAPH), so nothing else can reach a bufferless copula.
COP_PAST_BARE = Suffix("cop_past", "DI", {tags.COPULA: "past"})
COP_EVID_BARE = Suffix("cop_evid", "mIş", {tags.EVIDENTIAL: True})
COP_COND_BARE = Suffix("cop_cond", "sA", {tags.MOOD: "conditional"})

# --- Person paradigms ----------------------------------------------------------------

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


# --- Ability, aorist, and mood -------------------------------------------------------

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

# Denominal-stem aorist (temizler, taşlar, evlenir, güzelleşir). A NOUN/ADJ root has
# aorist=None, so the lexically class-guarded root aorists (_AORISTS) can never fire after a
# denominal verbalizer (-lA/-lAn/-lAş). Every denominal stem is polysyllabic, and the
# polysyllabic aorist is fully DETERMINISTIC — vowel-final -> -r (temizle->temizler,
# taşla->taşlar), consonant-final -> -Ir (evlen->evlenir, güzelleş->güzelleşir) — so two edges
# guarded by the running surface's final-segment class (like the passive / post-voice
# causatives, NOT a fabricated lexical class) cover it declaratively. Wired ONLY from V_DENOM
# (see transitions); the vowel/consonant guards are disjoint, so exactly one fires per stem.
AOR_DENOM_R = Suffix("aor", "r", {tags.TENSE: "aorist"}, stem_final_class=frozenset({"vowel"}))
AOR_DENOM_IR = Suffix(
    "aor", "Ir", {tags.TENSE: "aorist"}, stem_final_class=frozenset({"l", "r", "other"})
)

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

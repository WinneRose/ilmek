"""Morphotactics: the ordered suffix-transition graphs for nouns and verbs.

Encoded as *data* (suffix objects + a state→transitions map) so the ordering of Turkish
morphology is declared, inspectable, and extensible without touching the analyzer. The
analyzer walks these graphs forward, realizing each suffix with :mod:`.phonology` and
keeping paths whose running surface stays a prefix of the target word.

This package is split by concern for readability, while preserving the original flat module's
public surface (``import ... morphotactics as mt``; ``mt.GRAPH``, ``mt.Suffix``, …):

* :mod:`.states` — the state-name constants.
* :mod:`.suffix` — the :class:`Suffix` model (one declarative edge label).
* :mod:`.noun_suffixes` / :mod:`.verb_suffixes` / :mod:`.derivational_suffixes` — the suffix
  instances, grouped by category (inflection, voice/mood, derivation).
* :mod:`.transitions` — the graphs (``NOMINAL_GRAPH``/``VERBAL_GRAPH``/``GRAPH``), the
  accepting-state sets, and the feature-closure functions.

--- Design (unchanged from the flat module) ---

Nouns — plural, six persons of possessive, six cases, with pronominal buffering after a
3rd-person possessive; verbs — negation, the progressive/future/past/evidential tense-aspects,
one copular (ek-fiil) layer, and both person paradigms.

Verbal moods & aorist: ability -(y)Abil (gelebilir, okuyabilir), the conditional -sA (gelse,
gelseydi via the copula) and optative -(y)A (gele, gelelim), the necessitative -mAlI (gelmeli,
yapmalıyım, gitmeliydik), the negative aorist -mAz (gelmez) with its *defective* person
paradigm, and the positive aorist — which is lexically irregular, so its allomorph
(-r / -Ar / -Ir) is a lexicon fact on the root (:attr:`~ilmek.morphology.lexicon.Root.aorist`)
selected declaratively by an edge's :attr:`~.suffix.Suffix.aorist_class`.

The impossibilitive -(y)AmA (gelemez = gel+eme+z, a *distinct* morpheme = the ability-negative,
carrying polarity=negative + ability, NOT NEG+aorist) has its own V_IMPOSS state feeding the
defective -z aorist; the copular conditional -(y)sA stacks on a finished tense (gelirse,
geldiyse, gelecekse, geliyorsa, gelmezse), landing the bare conditional in its own V_COND
state so a restack (*gelseyse) is blocked; the irregular negative-aorist 1sg/1pl (gelmem,
gelmeyiz, and gelemem/gelemeyiz) attach to the -mA stem, not to -mAz; and the irregular
de-/ye- glide raising (diyor, yiyor, diyecek, diye) is handled via :attr:`Root.raised_form`
plus the :attr:`~.suffix.Suffix.glide_raise` flag, with dedi/demiş/deyiş staying regular.

Verb voice / çatı: a bounded voice layer sits between the root and the negation/tense chain,
encoding the canonical order reflexive/reciprocal < causative(<=2) < passive. Every voice
state reuses the bare root's continuation via :func:`~.transitions._root_continuation`, so a
voiced stem inflects for negation, all tenses, ability, mood and the verb->nominal derivations
for free (yaptırabilir, yapılmayacaktı, görüşme). The post-voice aorist is the deterministic
-Ir (denir, not *dener*). Which voice may fire is gated declaratively (causative by
``causative_class`` / post-voice by ``stem_final_class``, passive by ``stem_final_class``
alone, reflexive/reciprocal by ``requires_attribute``). Voice is recorded (ordered) under
``features[tags.VOICE]``; it is not derivational (yaptırdı -> stem/lemma yap), the voice states
are non-final, and the guesser walks no voice edge, so OOV stripping is byte-identical.

Derivation: a single, non-recursive derivation slot between root and inflection. Nominal
derivations (-lI, -sIz, -lIk, -CI, -CIk) leave N_ROOT for N_DERIV; verbal derivations
(-mA, -(y)Iş, -(y)An, -DIk, -(y)AcAk) leave V_ROOT/V_NEG for the *same* N_DERIV, and the
infinitive -mAk lands in a terminal V_INF. N_DERIV's outgoing edges are exactly N_ROOT's
inflectional ones, so a derived stem inflects normally (evli -> evlilerden) but cannot derive
again. Which derivation fires is gated by :attr:`~.suffix.Suffix.applies_to`, never a hardcoded
``if``.

Nominal ek-fiil: NOUN/ADJ/PRON/NUM predicates take the copula ("to be") *directly* as a suffix.
From every nominal final state the shared ``_NOMINAL_COPULA`` edge set adds the past -(y)DI,
evidential -(y)mIş, conditional -(y)sA, assertive -DIr, and the zero-copula present persons —
REUSING the verbal copula states. The copular keys are distinct from number/possessive/case, so
a case is never overwritten (evlerimizdeydi keeps plural+1pl+locative+copula-past). The
accusative is split off to the terminal N_ACC because ``*eviydi`` as accusative+copula is
ungrammatical.
"""

from __future__ import annotations

from .derivational_suffixes import *  # noqa: F401,F403
from .noun_suffixes import *  # noqa: F401,F403
from .states import *  # noqa: F401,F403
from .suffix import Suffix  # noqa: F401
from .transitions import *  # noqa: F401,F403
from .verb_suffixes import *  # noqa: F401,F403

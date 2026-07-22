"""The :class:`Suffix` model — one declarative edge label in the morphotactic graph.

Every morpheme is *data*: a template (:mod:`ilmek.morphology.phonology` realizes its
archiphonemes in context), the features it contributes, and a set of declarative guards
(aorist/causative class, phonological stem-final class, required root attribute, …) that let
the analyzer decide whether an edge may fire without any hardcoded ``if``. The suffix
*instances* live in :mod:`.noun_suffixes`, :mod:`.verb_suffixes`, and
:mod:`.derivational_suffixes`; the graphs that wire them to states live in :mod:`.transitions`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    #: The inverse of :attr:`requires_attribute`: the root attribute that BLOCKS this edge. Used
    #: to keep a broadly-``applies_to``-gated but not fully productive inflectional edge (the
    #: numeral ordinal/distributive, ``applies_to={NUM}``) off a curated subset of that POS that
    #: does not actually take it — the fraction numerals yarım/çeyrek/buçuk carry ``"fraction"``,
    #: so *yarımıncı/*çeyreğer never fire, without narrowing ``applies_to`` for every other
    #: cardinal (birinci, ikişer stay unaffected). ``None`` -> no exclusion.
    excludes_attribute: str | None = None
    #: When this suffix is *root-adjacent* (first in the chain), realize it against the root's
    #: :attr:`~ilmek.morphology.lexicon.Root.raised_form` (de->di, ye->yi) instead of its free
    #: form. Models the irregular glide raising of de-/ye- before a vowel-initial suffix
    #: (diyecek, diye, diyen), so it is set only on the vowel-initial edges that trigger it
    #: (FUT, PART_ACAK, OPT, ABIL, PART_AN, IMPOSS). Roots without a raised form are unaffected;
    #: the -(y)Iş verbal noun is deliberately *unflagged* so ``deyiş`` stays regular.
    glide_raise: bool = False

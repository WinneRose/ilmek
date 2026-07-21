"""Root lexicon: entries, root-boundary allomorphy, and candidate lookup.

Each entry is data (``lemma``, ``pos``, ``attributes``, optional ``forms``). From it we
derive the **bound form** — the allomorph a root shows before a vowel-initial suffix:

* ``voicing``: final stop softens (kitap→kitab, ağaç→ağac, kanat→kanad, ayak→ayağ,
  renk→reng for the ``nk`` case).
* ``vowel_drop``: a medial vowel drops (burun→burn, ağız→ağz); given explicitly via
  ``forms`` since it is not predictable.

Candidate enumeration (:meth:`Lexicon.candidates`) buckets roots by their first surface
character, which no root-boundary alternation ever changes — so a surface reshaped by a
later suffix (ara→arıyor, kitap→kitabımızdan) still reaches its root, where a strict
prefix trie would miss it. A character trie is kept only for exact membership testing.

**Aorist allomorph (verbs).** The aorist tense-suffix is lexically irregular: its shape is
one of ``-r`` / ``-Ar`` / ``-Ir`` and cannot be predicted from the surface alone for
consonant-final monosyllables. We therefore store the choice as a *lexicon fact* on each
VERB root — :attr:`Root.aorist`, a value in ``{"r", "Ar", "Ir"}`` — and let the analyzer's
declarative ``aorist_class`` guard pick the matching suffix edge. An explicit ``"aorist"``
key in the entry always wins; otherwise it is computed by the documented grammar default:

* vowel-final stem -> ``"r"`` (oku -> okur, yürü -> yürür, de -> der, ye -> yer);
* consonant-final **polysyllabic** stem (>=2 vowels) -> ``"Ir"`` (otur -> oturur,
  çalış -> çalışır, öğret -> öğretir);
* consonant-final **monosyllabic** stem -> ``"Ar"`` (yap -> yapar, bak -> bakar,
  git -> gider).

The classic closed exception is the ~13 monosyllabic verbs that take ``-Ir`` despite being
monosyllabic (al, bil, bul, dur, gel, gör, kal, ol, öl, san, var, ver, vur). Those carry an
explicit ``"aorist": "Ir"`` in ``verbs.json``. **Any future monosyllabic verb entry must set
this field if it is a member of that list** — the default would otherwise wrongly emit
``-Ar``. Synthetic roots (the guesser, the apostrophe path) leave ``aorist = None``, so a
root-attached aorist never fires for an unverified guess: no wrong aorist is ever emitted.

**Closed-class irregulars.** A handful of closed classes — the personal pronouns
(ben/sen/o/biz/siz), the demonstratives (bu/şu), their plurals, and the existentials
(var/yok) — decline *suppletively*: ``ben`` has the dative ``bana`` (vowel change), the
genitive-based instrumental ``benimle``, and the pronominal ``-n-`` of ``onu``/``bunda``.
No Root-plus-allomorph mechanism can express these without overgenerating junk (a single
bound form ``ban-`` would also license ``*banı``/``*banda``; a regular ``ben`` Root would
license ``*bene``/``*benler`` and misparse ``benim`` as ``ben+poss_1sg``). So each
irregular surface is enumerated whole as an :class:`IrregularForm` (surface → lemma, pos,
features, segmentation) and matched *before* the FSM. This is a closed, fully-listable set,
so full enumeration is correct and overgeneration-proof rather than a shortcut.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from ..core import tags
from ..core.alphabet import VOICING, VOWELS
from ..core.normalization import fold_for_lookup

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "lexicon"

#: POS values that inflect with the *nominal* morphotactic graph.
NOMINAL_POS = frozenset({tags.NOUN, tags.PROPN, tags.ADJ, tags.NUM, tags.PRON, tags.ADV})
VERBAL_POS = frozenset({tags.VERB})


def _derive_bound_form(
    free: str, attributes: frozenset[str], explicit_forms: list[str] | None
) -> str | None:
    if explicit_forms and len(explicit_forms) >= 2:
        # Second listed form is the bound allomorph (covers vowel_drop and irregulars).
        return explicit_forms[1]
    if "voicing" in attributes:
        if free.endswith("nk"):
            return free[:-2] + "ng"  # renk -> reng
        if free and free[-1] in VOICING:
            return free[:-1] + VOICING[free[-1]]
    return None


#: Valid values for the lexical aorist allomorph (see the module docstring).
_AORIST_CLASSES = frozenset({"r", "Ar", "Ir"})


def _derive_aorist(free: str, explicit: str | None) -> str:
    """The aorist allomorph class of a VERB root: ``"r"`` / ``"Ar"`` / ``"Ir"``.

    An explicit lexicon value always wins (validated); otherwise the documented grammar
    default. The consonant-final monosyllabic default is ``"Ar"``; the ~13 classic ``-Ir``
    monosyllables (gel, al, ver, ...) must carry an explicit ``"aorist": "Ir"`` instead.
    """
    if explicit is not None:
        if explicit not in _AORIST_CLASSES:
            raise ValueError(
                f"invalid aorist class {explicit!r}; expected one of {_AORIST_CLASSES}"
            )
        return explicit
    if free and free[-1] in VOWELS:
        return "r"  # vowel-final: okur, yürür, der, yer
    n_vowels = sum(1 for ch in free if ch in VOWELS)
    if n_vowels >= 2:
        return "Ir"  # consonant-final polysyllable: oturur, çalışır, öğretir
    return "Ar"  # consonant-final monosyllable: yapar, bakar, gider


@dataclass(frozen=True, slots=True)
class Root:
    lemma: str
    pos: str
    attributes: frozenset[str]
    free_form: str
    bound_form: str | None
    #: The lexical aorist allomorph for VERB roots (``"r"`` / ``"Ar"`` / ``"Ir"``); ``None``
    #: for nominals and for synthetic roots (guesser/apostrophe), so a root-attached aorist
    #: only ever fires for a lexicon-verified verb. Defaulted last, so the positional
    #: constructions in the analyzer keep working and stay ``aorist=None``.
    aorist: str | None = None

    @classmethod
    def from_entry(cls, entry: dict) -> Root:
        lemma = entry["lemma"]
        pos = entry.get("pos", tags.NOUN).upper()
        attributes = frozenset(entry.get("attributes", ()))
        raw_forms = entry.get("forms")
        forms = [fold_for_lookup(f) for f in raw_forms] if raw_forms else None
        free = forms[0] if forms else fold_for_lookup(lemma)
        bound = _derive_bound_form(free, attributes, forms)
        aorist = _derive_aorist(free, entry.get("aorist")) if pos in VERBAL_POS else None
        return cls(
            lemma=lemma,
            pos=pos,
            attributes=attributes,
            free_form=free,
            bound_form=bound,
            aorist=aorist,
        )

    @property
    def is_nominal(self) -> bool:
        return self.pos in NOMINAL_POS

    @property
    def is_verbal(self) -> bool:
        return self.pos in VERBAL_POS

    def surface_forms(self) -> Iterator[tuple[str, bool]]:
        """Yield ``(surface, is_bound)`` for each allomorph to index."""
        yield self.free_form, False
        if self.bound_form and self.bound_form != self.free_form:
            yield self.bound_form, True


@dataclass(frozen=True, slots=True)
class IrregularForm:
    """A fully-enumerated, dictionary-verified surface form of a closed-class word.

    Unlike a :class:`Root`, an ``IrregularForm`` is a *whole surface* (``bana``), not a
    root that inflects — it carries its own lemma, pos, features, and morpheme
    segmentation and is looked up by its folded surface. ``morphemes`` is kept as a tuple
    so the record stays hashable/immutable; the analyzer copies it into a fresh list.
    """

    surface: str  # folded surface used as the lookup key
    lemma: str
    pos: str
    features: dict
    morphemes: tuple[str, ...]

    @classmethod
    def from_entry(cls, entry: dict) -> IrregularForm:
        return cls(
            surface=fold_for_lookup(entry["surface"]),
            lemma=entry["lemma"],
            pos=entry.get("pos", tags.PRON).upper(),
            features=dict(entry.get("features", {})),
            morphemes=tuple(entry.get("morphemes", ())),
        )


class _TrieNode:
    __slots__ = ("children", "entries")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        #: (root, is_bound) pairs that terminate exactly at this node.
        self.entries: list[tuple[Root, bool]] = []


class Lexicon:
    """A set of roots with prefix lookup over their surface allomorphs."""

    def __init__(self, roots: Iterable[Root] = ()) -> None:
        self._root = _TrieNode()
        self._roots: list[Root] = []
        #: first surface character -> roots, for candidate enumeration. Root-boundary
        #: alternations (voicing, vowel drop, de->di) never change the first character,
        #: so bucketing by it is sound where trie-prefix matching would miss a root whose
        #: surface is altered by a later suffix (ara -> arıyor).
        self._by_initial: dict[str, list[Root]] = {}
        #: folded surface -> enumerated closed-class analyses (personal/demonstrative
        #: pronouns, existentials). Consulted by the analyzer before the regular FSM.
        self._irregular: dict[str, list[IrregularForm]] = {}

        for root in roots:
            self.add(root)

    def __len__(self) -> int:
        return len(self._roots)

    def add(self, root: Root) -> None:
        self._roots.append(root)
        if root.free_form:
            self._by_initial.setdefault(root.free_form[0], []).append(root)
        for surface, is_bound in root.surface_forms():
            node = self._root
            for ch in surface:
                node = node.children.setdefault(ch, _TrieNode())
            node.entries.append((root, is_bound))

    def candidates(self, word: str) -> list[Root]:
        """Roots that could begin ``word`` (matched on first character)."""
        if not word:
            return []
        return self._by_initial.get(word[0], [])

    def add_entry(self, entry: dict) -> None:
        self.add(Root.from_entry(entry))

    def add_irregular(self, entry: dict) -> None:
        """Register one enumerated closed-class surface form (keyed by folded surface)."""
        form = IrregularForm.from_entry(entry)
        self._irregular.setdefault(form.surface, []).append(form)

    def irregular_forms(self, surface: str) -> list[IrregularForm]:
        """Enumerated closed-class analyses for ``surface`` (already folded), in data order."""
        return self._irregular.get(surface, [])

    def __contains__(self, surface: str) -> bool:
        node = self._root
        for ch in surface:
            node = node.children.get(ch)
            if node is None:
                return False
        return bool(node.entries)

    @classmethod
    def load(cls, paths: Iterable[Path] | None = None) -> Lexicon:
        """Load all lexicon JSON files (from the packaged ``data/lexicon`` dir by default).

        Each file is either a JSON list of entries, or an object with an ``entries`` list
        and/or an ``irregular`` list (enumerated closed-class surface forms). A bare list
        and an ``{"entries": ...}``-only object behave exactly as before; the optional
        ``irregular`` key is simply ignored by any file that does not carry one.
        """
        lex = cls()
        files = list(paths) if paths is not None else sorted(_DATA_DIR.glob("*.json"))
        for path in files:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for entry in data.get("entries", ()):
                    lex.add_entry(entry)
                for entry in data.get("irregular", ()):
                    lex.add_irregular(entry)
            else:
                for entry in data:
                    lex.add_entry(entry)
        return lex

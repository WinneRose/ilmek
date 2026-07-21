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
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from ..core import tags
from ..core.alphabet import VOICING
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


@dataclass(frozen=True, slots=True)
class Root:
    lemma: str
    pos: str
    attributes: frozenset[str]
    free_form: str
    bound_form: str | None

    @classmethod
    def from_entry(cls, entry: dict) -> Root:
        lemma = entry["lemma"]
        pos = entry.get("pos", tags.NOUN).upper()
        attributes = frozenset(entry.get("attributes", ()))
        raw_forms = entry.get("forms")
        forms = [fold_for_lookup(f) for f in raw_forms] if raw_forms else None
        free = forms[0] if forms else fold_for_lookup(lemma)
        bound = _derive_bound_form(free, attributes, forms)
        return cls(lemma=lemma, pos=pos, attributes=attributes, free_form=free, bound_form=bound)

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

        Each file is either a JSON list of entries or an object with an ``entries`` list.
        """
        lex = cls()
        files = list(paths) if paths is not None else sorted(_DATA_DIR.glob("*.json"))
        for path in files:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            entries = data["entries"] if isinstance(data, dict) else data
            for entry in entries:
                lex.add_entry(entry)
        return lex

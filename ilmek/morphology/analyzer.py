"""Native morphological analyzer: analysis by forward generation.

For a surface word we enumerate every lexicon root that could begin it (bucketed by first
character), then walk the morphotactic graph forward from each root, realizing suffixes
with the morphophonemic rules and keeping only paths that can still grow into the word. A
path that reaches an accepting state exactly at the word *is* a valid analysis — and its
transition sequence is a complete, explainable morpheme segmentation.

Why generate rather than strip? Reversing Turkish phonology is one-to-many and error
prone; generating is deterministic and every accepted analysis is morphophonemically
valid by construction. Unknown words fall back to a clearly-marked (`source="guess"`)
heuristic that never masquerades as a lexicon-verified result.
"""

from __future__ import annotations

from ..core import tags
from ..core.alphabet import VOICING, VOWELS
from ..core.document import AnalysisResult
from ..core.normalization import normalize, turkish_lower
from . import morphotactics as mt
from .lexicon import Lexicon, Root
from .phonology import realize, starts_with_vowel


def _has_vowel(text: str) -> bool:
    return any(ch in VOWELS for ch in text)


def _ends_with_vowel(text: str) -> bool:
    return bool(text) and text[-1] in VOWELS


def _soften_final(surface: str) -> str:
    """Soften a morpheme-final stop before a vowel (used at non-root suffix boundaries)."""
    if surface.endswith("nk"):
        return surface[:-2] + "ng"
    if surface and surface[-1] in VOICING:
        return surface[:-1] + VOICING[surface[-1]]
    return surface


def _compatible(word: str, acc: str) -> bool:
    """Whether a partial surface ``acc`` can still grow into ``word``.

    Plain prefix is not enough: a later suffix may retroactively delete ``acc``'s final
    vowel (-Iyor: başla -> başlıyor) or soften its final stop (-AcAk + vowel: gelecek ->
    geleceğim). We accept those pending changes so the branch is not pruned early;
    acceptance still requires an exact full-word match, so no wrong analysis slips in.
    """
    if word.startswith(acc):
        return True
    if acc and acc[-1] in VOWELS and word.startswith(acc[:-1]):
        return True
    softened = _soften_final(acc)
    if softened != acc and word.startswith(softened):
        return True
    return False


def _apply(
    suffix: mt.Suffix, acc: str, *, is_first: bool, root: Root, prev_voice_final: bool
) -> tuple[str, str]:
    """Attach ``suffix`` to accumulated surface ``acc``. Returns ``(new_surface, morph)``."""
    if is_first:
        base = root.free_form
        # Buffer decisions depend only on vowel/consonant ending, identical for the free
        # and bound allomorphs, so peeking with the free form is safe.
        peek = realize(suffix.template, base)
        stem = root.bound_form if (starts_with_vowel(peek) and root.bound_form) else base
        if suffix.drop_preceding and _ends_with_vowel(stem):
            stem = stem[:-1]
        morph = realize(suffix.template, stem)
        return stem + morph, morph

    stem = acc
    if suffix.drop_preceding and _ends_with_vowel(stem):
        stem = stem[:-1]
    morph = realize(suffix.template, stem)
    if prev_voice_final and starts_with_vowel(morph):
        stem = _soften_final(stem)
    return stem + morph, morph


def _generate(
    word: str,
    root: Root,
    *,
    source: str = tags.SOURCE_LEXICON,
    allow_derivation: bool = True,
) -> list[AnalysisResult]:
    """All full-word analyses reachable from ``root`` via the morphotactic graph.

    The unified graph spans both the nominal and verbal sides; a derivational edge may
    cross from a verb root into the nominal states (gelmek, gelen). ``allow_derivation``
    is set ``False`` on the guesser path so unknown-word stripping is unchanged: with it
    off, only inflectional edges are walked, exactly as before this milestone.
    """
    if root.is_nominal:
        start = mt.NOMINAL_START
        base_features = mt.nominal_default_features()
    elif root.is_verbal:
        start = mt.VERBAL_START
        base_features = {}
    else:
        return []

    graph, finals = mt.GRAPH, mt.FINALS
    out: list[AnalysisResult] = []

    def dfs(state, acc, morphemes, features, prev_voice_final, deriv_names, stem_surface, cur_pos):
        if state in finals and acc == word:
            feats = dict(features)
            if state in mt.NOMINAL_STATES:
                # Nominal-side acceptance: fill nominal defaults *under* whatever was
                # accrued, so a verb-derived nominal keeps its polarity (gelmeyen) yet
                # never gains a fabricated person/mood (no finalize_verbal_features).
                feats = {**mt.nominal_default_features(), **feats}
            else:
                mt.finalize_verbal_features(feats)
            if deriv_names:
                feats[tags.DERIVATION] = deriv_names
            out.append(
                AnalysisResult(
                    surface=word,
                    lemma=root.lemma,
                    # Stem is the surface at the last derivation boundary (evli, yaşadık);
                    # with no derivation it is the root lemma, as everywhere in v0.1.
                    stem=stem_surface if deriv_names else root.lemma,
                    pos=cur_pos,
                    morphemes=list(morphemes),
                    features=feats,
                    source=source,
                )
            )
        for suffix, target in graph[state]:
            if suffix.derivational:
                if not allow_derivation:
                    continue
                if suffix.applies_to is not None and cur_pos not in suffix.applies_to:
                    continue
            # Lexically-irregular aorist: an allomorph edge fires only for a root whose
            # lexical aorist class matches. Synthetic roots have aorist=None, so no
            # class-guarded aorist is ever emitted for a guess.
            if suffix.aorist_class is not None and suffix.aorist_class != root.aorist:
                continue
            new_acc, morph = _apply(
                suffix,
                acc,
                is_first=not morphemes,
                root=root,
                prev_voice_final=prev_voice_final,
            )
            if new_acc == acc or not _compatible(word, new_acc):
                continue
            if suffix.derivational:
                next_deriv = deriv_names + (suffix.name,)
                next_stem = new_acc
                next_pos = suffix.to_pos if suffix.to_pos is not None else cur_pos
            else:
                next_deriv, next_stem, next_pos = deriv_names, stem_surface, cur_pos
            dfs(
                target,
                new_acc,
                morphemes + ([morph] if morph else []),
                {**features, **suffix.features},
                suffix.voice_final,
                next_deriv,
                next_stem,
                next_pos,
            )

    dfs(start, root.free_form, [], base_features, False, (), root.lemma, root.pos)
    return out


def _dedupe(results: list[AnalysisResult]) -> list[AnalysisResult]:
    seen: set = set()
    unique: list[AnalysisResult] = []
    for r in results:
        key = (r.lemma, r.pos, tuple(r.morphemes), tuple(sorted(r.features.items())))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _sort_key(r: AnalysisResult):
    # Lexicon before guess; then prefer the longest root (whole-word entries over splits);
    # then *fewer derivations* (a finite/inflectional reading outranks a derived one that
    # shares the same root — geldik stays past-1pl, gelecek stays future, gelme stays the
    # negative imperative, while the participle/verbal-noun reading survives as an alt);
    # then fewer morphemes; then a stable lexicographic tie-break.
    src_rank = {tags.SOURCE_LEXICON: 0, tags.SOURCE_RULE: 1, tags.SOURCE_GUESS: 2}
    n_deriv = len(r.features.get(tags.DERIVATION, ()))
    return (src_rank.get(r.source, 3), -len(r.lemma), n_deriv, len(r.morphemes), r.pos, r.lemma)


# --- Guesser heuristics --------------------------------------------------------------

#: How many alternative candidates to keep on a guess result.
_MAX_GUESS_ALTS = 6
#: A lone stripped suffix must be at least this many characters to be trusted on its own
#: (blocks over-stripping a root-final vowel/consonant, e.g. kalem -> *kale, kapı -> *kap).
_MIN_STRONG_SUFFIX_LEN = 3


def _stripped_len(r: AnalysisResult) -> int:
    return sum(len(m) for m in r.morphemes)


def _is_confident_guess(r: AnalysisResult) -> bool:
    """A guessed strip is trustworthy only with strong inflectional evidence."""
    if len(r.lemma) < 2 or not _has_vowel(r.lemma):
        return False
    return len(r.morphemes) >= 2 or _stripped_len(r) >= _MIN_STRONG_SUFFIX_LEN


def _guess_sort_key(r: AnalysisResult):
    # Prefer more inflection removed (more morphemes, then more characters), then nominal
    # over verbal (OOV nouns dominate), then the longer root as a stable tie-break.
    pos_rank = {tags.NOUN: 0, tags.VERB: 1}
    return (-len(r.morphemes), -_stripped_len(r), pos_rank.get(r.pos, 2), -len(r.lemma))


class Analyzer:
    """Context-free native analyzer over a :class:`Lexicon`."""

    def __init__(self, lexicon: Lexicon):
        self.lexicon = lexicon

    def analyze(self, surface: str) -> list[AnalysisResult]:
        """Return every valid analysis of ``surface``, best candidate first."""
        original = surface
        folded = turkish_lower(normalize(surface))

        if "'" in folded:
            head, _, tail = folded.rpartition("'")
            if head and tail:
                return self._analyze_apostrophe(original, folded)
            # A stray leading/trailing apostrophe is not a proper-noun boundary: drop it
            # and analyze normally (falling through to the lexicon / guesser).
            folded = folded.replace("'", "")

        # Closed-class irregulars (personal/demonstrative pronouns, existentials) are
        # enumerated whole and matched first. They are dictionary-verified, hence
        # source=lexicon. We *prepend* them in data order rather than folding them into the
        # sort below: _sort_key's -len(lemma) tie-break would otherwise rank the regular
        # ``on`` (NUM, "ten") above the pronoun ``o`` for onu/onda/ondan. The regular
        # analyses are still computed and kept as ranked alternatives — a closed-class
        # reading outranks an open-class parse, but never erases genuine ambiguity.
        irregulars = [
            AnalysisResult(
                surface=original,
                lemma=f.lemma,
                stem=f.lemma,  # inflection-only: stem == lemma, as everywhere in v0.1
                pos=f.pos,
                morphemes=list(f.morphemes),
                features=dict(f.features),
                source=tags.SOURCE_LEXICON,
            )
            for f in self.lexicon.irregular_forms(folded)
        ]

        results: list[AnalysisResult] = []
        for root in self.lexicon.candidates(folded):
            results.extend(_generate(folded, root))

        results = _dedupe(results)
        results.sort(key=_sort_key)

        if not irregulars and not results:
            return self._guess(folded)

        for r in results:
            r.surface = original
        return irregulars + results

    def _analyze_apostrophe(self, original: str, folded: str) -> list[AnalysisResult]:
        """Proper noun written with an apostrophe suffix: ``Ankara'da`` -> lemma ``Ankara``."""
        head, _, tail = folded.rpartition("'")
        joined = head + tail
        synth = Root(
            lemma=head, pos=tags.PROPN, attributes=frozenset(), free_form=head, bound_form=None
        )
        results = _generate(joined, synth, source=tags.SOURCE_RULE)
        results = _dedupe(results)
        results.sort(key=_sort_key)
        if not results:
            results = [
                AnalysisResult(original, head, head, tags.PROPN, [], {}, source=tags.SOURCE_RULE)
            ]
        for r in results:
            r.surface = original
        return results

    def _guess(self, word: str) -> list[AnalysisResult]:
        """Unknown word: propose a stripped root only when the evidence is strong.

        We treat every proper prefix as a hypothetical root and try to parse the rest as a
        valid inflectional chain (same FSM as the lexicon path, so this improves for free
        as morphotactics grow). A parse is *confident* only when it removes real inflection
        — a multi-suffix chain, or a single distinctive suffix of >= 3 chars — and leaves a
        plausible root. Confident: the stripped root becomes the primary guess. Otherwise
        we stay honest and return the surface as its own stem, keeping every stripped
        candidate as a ranked alternative so nothing is lost. All results are `guess`.
        """
        parses: list[AnalysisResult] = []
        for split in range(2, len(word)):
            root_guess = word[:split]
            if not _has_vowel(root_guess):
                continue
            for pos in (tags.NOUN, tags.VERB):
                synth = Root(root_guess, pos, frozenset(), root_guess, None)
                # Derivation stays off for guesses: an unknown root must not be split on a
                # derivational suffix (e.g. malik -> *ma+lik), so guesser output is
                # byte-identical to before this milestone.
                for res in _generate(word, synth, source=tags.SOURCE_GUESS, allow_derivation=False):
                    if res.morphemes:  # only non-trivial suffix strips are informative
                        parses.append(res)

        parses = _dedupe(parses)
        parses.sort(key=_guess_sort_key)

        confident = [p for p in parses if _is_confident_guess(p)]
        if confident:
            primary = confident[0]
            primary.analyses = [p for p in parses if p is not primary][:_MAX_GUESS_ALTS]
            return [primary]

        identity = AnalysisResult(word, word, word, tags.X, [], {}, source=tags.SOURCE_GUESS)
        identity.analyses = parses[:_MAX_GUESS_ALTS]
        return [identity]


# --- Lazy default analyzer (packaged lexicon) ----------------------------------------

_DEFAULT: Analyzer | None = None


def default_analyzer() -> Analyzer:
    """Return a process-wide analyzer backed by the packaged lexicon (loaded once)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Analyzer(Lexicon.load())
    return _DEFAULT

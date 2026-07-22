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


def _stem_final_class(surface: str) -> str:
    """Classify a stem's final segment for the voice edges' declarative phonological guard.

    ``"vowel"`` / ``"l"`` / ``"r"`` / ``"other"`` (a consonant that is not l or r). This is
    read off the *running surface* (the free root form or a voiced stem), never a hardcoded
    per-suffix ``if``; root-boundary voicing (t->d, p->b, k->ğ) never crosses these classes,
    so peeking with the free form before allomorph selection is safe.
    """
    if not surface:
        return "other"
    ch = surface[-1]
    if ch in VOWELS:
        return "vowel"
    if ch == "l":
        return "l"
    if ch == "r":
        return "r"
    return "other"


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
        # Front-harmony loan (saat, kalp, usul): the root-adjacent suffix harmonizes as if the
        # root's final vowel were fronted (saat+I -> saati). Only the root-adjacent realization
        # needs the flag; later suffixes inherit front harmony from the emitted front vowel.
        front = "front_harmony" in root.attributes
        # Irregular glide raising (de->di, ye->yi) before a vowel-initial glide-raising suffix:
        # realize against the raised allomorph so diyecek/diye/diyen come out right, while the
        # unflagged suffixes (dedi, deyiş) keep the free form. Only de/ye carry a raised_form.
        base = root.raised_form if (suffix.glide_raise and root.raised_form) else root.free_form
        # Buffer decisions depend only on vowel/consonant ending, identical for the free
        # and bound allomorphs (and unaffected by fronting a->e etc.), so peeking is safe.
        peek = realize(suffix.template, base, front_root=front)
        stem = root.bound_form if (starts_with_vowel(peek) and root.bound_form) else base
        if suffix.drop_preceding and _ends_with_vowel(stem):
            dropped = stem[:-1]
            # -Iyor deletes the stem's final vowel. When that leaves NO vowel (the CV verbs de,
            # ye), harmony would fall back to the default vowel and misrealize (de -> *dıyor);
            # realize the suffix against the pre-drop stem so the high vowel raises correctly
            # (de -> diyor, ye -> yiyor), then drop. Polysyllabic stems keep a vowel after the
            # drop, so this is a no-op for them (söyle -> söylüyor, başla -> başlıyor).
            ctx = dropped if _has_vowel(dropped) else stem
            morph = realize(suffix.template, ctx, front_root=front)
            return dropped + morph, morph
        morph = realize(suffix.template, stem, front_root=front)
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
    allow_nominal_copula: bool = True,
    allow_voice: bool = True,
) -> list[AnalysisResult]:
    """All full-word analyses reachable from ``root`` via the morphotactic graph.

    The unified graph spans both the nominal and verbal sides; a derivational edge may
    cross from a verb root into the nominal states (gelmek, gelen). ``allow_derivation``
    is set ``False`` on the guesser path so unknown-word stripping is unchanged: with it
    off, only inflectional edges are walked, exactly as before this milestone.
    ``allow_nominal_copula`` gates the ek-fiil edges the same way — ``False`` on the guesser
    so an unknown word is never stripped of a copular ending, keeping guesser output
    byte-identical. ``allow_voice`` gates the voice (çatı) edges likewise: the passive is
    fully productive (no root fact), so without this an OOV word would be voice-split; ``False``
    on the guesser keeps its output byte-identical.
    """
    if "interrogative" in root.attributes:
        # The interrogative particle mi/mı/mu/mü: start at the dedicated Q_ROOT, whose only
        # edges are the (filtered) ek-fiil ones. Attribute-routed, NOT a hardcoded surface list,
        # so only the lexicon entries carrying the attribute reach it (synthetic/guessed roots
        # have empty attributes, so Q_ROOT is unreachable for a guess).
        start = mt.Q_START
        base_features = mt.interrogative_default_features()
    elif root.is_nominal:
        start = mt.NOMINAL_START
        base_features = mt.nominal_default_features()
    elif root.is_verbal:
        start = mt.VERBAL_START
        base_features = {}
    else:
        return []

    graph, finals = mt.GRAPH, mt.FINALS
    out: list[AnalysisResult] = []

    def dfs(
        state,
        acc,
        morphemes,
        features,
        prev_voice_final,
        deriv_names,
        stem_surface,
        cur_pos,
        voices,
    ):
        if state in finals and acc == word:
            feats = dict(features)
            if state == mt.Q_ROOT:
                # Bare interrogative particle (mi/mı/mu/mü): the features are already complete
                # (question=True, no person), so nothing is finalized — a bare mi fabricates no
                # person/copula/case, exactly what distinguishes it from a full nominal.
                pass
            elif state in mt.NOMINAL_STATES:
                # Nominal-side acceptance: fill nominal defaults *under* whatever was
                # accrued, so a verb-derived nominal keeps its polarity (gelmeyen) yet
                # never gains a fabricated person/mood (no finalize_verbal_features).
                feats = {**mt.nominal_default_features(), **feats}
            elif state in mt.COPULA_STATES and cur_pos != tags.VERB:
                # Ek-fiil acceptance on a NOMINAL/ADJ/PRON/NUM predicate (güzeldi, evdeydim,
                # güzelim) or on the interrogative PARTICLE (midir, misin, miyim, miydi): keep
                # any accrued keys, default person to the zero 3sg. The particle takes the
                # dedicated closure (no fabricated nominal number/case — mi is not a noun); a
                # nominal predicate takes the nominal one (keeps case/number). A verbal path
                # reaches these same person/copula states with cur_pos == VERB (every
                # verb->nominal derivation sets to_pos, so no verbal spine arrives here with a
                # non-VERB pos) and falls through to verbal finalization below.
                if cur_pos == tags.PART:
                    feats = mt.finalize_particle_predicate_features(feats)
                else:
                    feats = mt.finalize_nominal_predicate_features(feats)
            else:
                mt.finalize_verbal_features(feats)
            if deriv_names:
                feats[tags.DERIVATION] = deriv_names
            if voices:
                # Ordered tuple, so stacked voices survive (a dict-merge would collapse them).
                feats[tags.VOICE] = voices
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
            if suffix.derivational and not allow_derivation:
                continue
            # Declarative POS guard. Every derivation carries an ``applies_to`` (the
            # overgeneration guard: -CI is {NOUN} so *güzelci is blocked); the inflectional
            # distributive -(ş)Ar carries {NUM} so it fires only on a numeral (birer, but not
            # *ever from ev). Every other inflectional suffix leaves ``applies_to=None``, so
            # this is a provable no-op for them — behavior is byte-identical to before the
            # check was lifted out of the derivational branch. ``cur_pos`` is the current
            # stem's POS, so a derived/copular stem is gated by what it has become.
            if suffix.applies_to is not None and cur_pos not in suffix.applies_to:
                continue
            # Ek-fiil edges leave a nominal final for a copular state; gate them off for the
            # guesser (mirrors allow_derivation) so an unknown word is never stripped of a
            # copular ending and OOV output stays byte-identical.
            if (
                not allow_nominal_copula
                and state in mt.NOMINAL_FINALS
                and target in mt.COPULA_STATES
            ):
                continue
            # Lexically-irregular aorist: an allomorph edge fires only for a root whose
            # lexical aorist class matches. Synthetic roots have aorist=None, so no
            # class-guarded aorist is ever emitted for a guess.
            if suffix.aorist_class is not None and suffix.aorist_class != root.aorist:
                continue
            # Voice (çatı) edges are gated off for the guesser (mirrors allow_derivation): the
            # passive is fully productive with no root fact, so an OOV word would otherwise be
            # voice-split. With it off, OOV stripping stays byte-identical.
            if suffix.voice is not None and not allow_voice:
                continue
            # Lexically-irregular *first* causative allomorph: like the aorist, the edge fires
            # only for a root whose lexical causative class matches (synthetic/nominal roots
            # have causative=None, and a suppletive-causative verb has "none": no match, so no
            # *geldir / *gittir). The phonologically-chosen causatives carry no causative_class.
            if suffix.causative_class is not None and suffix.causative_class != root.causative:
                continue
            # Semi-productive reflexive/reciprocal: fire only on a verb carrying the required
            # attribute (data-declared curated list), the overgeneration guard for them.
            if (
                suffix.requires_attribute is not None
                and suffix.requires_attribute not in root.attributes
            ):
                continue
            # Declarative phonological guard (passive allomorphy, post-voice causatives): the
            # running surface's final-segment class must be allowed by the edge.
            if (
                suffix.stem_final_class is not None
                and _stem_final_class(acc) not in suffix.stem_final_class
            ):
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
            # Voice is threaded as an ordered tuple beside deriv_names (not merged into the
            # feature dict) so stacked causatives / causative+passive are all preserved.
            next_voices = voices + (suffix.voice,) if suffix.voice is not None else voices
            dfs(
                target,
                new_acc,
                morphemes + ([morph] if morph else []),
                {**features, **suffix.features},
                suffix.voice_final,
                next_deriv,
                next_stem,
                next_pos,
                next_voices,
            )

    dfs(start, root.free_form, [], base_features, False, (), root.lemma, root.pos, ())
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
    # then a *nominal-predicate* rank so a homograph's finite-verb reading stays primary over
    # its noun+copula reading (yüzdü stays yüz-VERB-past, not yüz-NOUN-copula; öğretmenim
    # stays possessive over "I am a teacher"); then fewer morphemes; then a stable tie-break.
    # n_pred is 0 for every pre-existing result (no prior nominal analysis carried a
    # person/copula/evidential/mood key, and verbal results are pos == VERB), so it never
    # reorders anything that existed before the ek-fiil milestone.
    src_rank = {tags.SOURCE_LEXICON: 0, tags.SOURCE_RULE: 1, tags.SOURCE_GUESS: 2}
    n_deriv = len(r.features.get(tags.DERIVATION, ()))
    n_pred = (
        1
        if r.pos != tags.VERB
        and any(k in r.features for k in (tags.PERSON, tags.COPULA, tags.EVIDENTIAL, tags.MOOD))
        else 0
    )
    return (
        src_rank.get(r.source, 3),
        -len(r.lemma),
        n_deriv,
        n_pred,
        len(r.morphemes),
        r.pos,
        r.lemma,
    )


# --- Guesser heuristics --------------------------------------------------------------

#: How many alternative candidates to keep on a guess result.
_MAX_GUESS_ALTS = 6
#: A confident strip must leave a root at least this long. A 1-2 char remnant is almost
#: never a real Turkish root, so stripping down to one is over-stripping (geler -> *ge via
#: ge+ler; sene -> *se via se+n+e; zolar -> *zo). Distinct from the suffix bound below:
#: same value today, different meaning, so they are not a shared constant.
_MIN_GUESS_ROOT_LEN = 3
#: The total stripped inflection must be at least this many characters to be trusted
#: (blocks over-stripping a root-final vowel/consonant, e.g. kalem -> *kale, kapı -> *kap).
_MIN_STRONG_SUFFIX_LEN = 3


def _stripped_len(r: AnalysisResult) -> int:
    return sum(len(m) for m in r.morphemes)


def _is_confident_guess(r: AnalysisResult) -> bool:
    """A guessed strip is trustworthy only with strong inflectional evidence.

    Conjunctive gate: the surviving root must be a plausible word (>= 3 chars containing a
    vowel) AND at least 3 chars of inflection must have been removed. A short garbage root
    (geler -> ge) or a short strip (a single one-char ending) is rejected in favour of the
    identity fallback — correct until the true root enters the lexicon.
    """
    return (
        len(r.lemma) >= _MIN_GUESS_ROOT_LEN
        and _has_vowel(r.lemma)
        and _stripped_len(r) >= _MIN_STRONG_SUFFIX_LEN
    )


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
                # Suppletive/inflected irregulars have stem == lemma (pronouns); a derived
                # irregular (an intensive diminutive) carries stem == its whole surface, per
                # the stem contract. The IrregularForm resolves which; we just read it.
                stem=f.stem,
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
        as morphotactics grow). A parse is *confident* only when it leaves a plausible root
        (>= 3 chars containing a vowel) AND removes >= 3 chars of inflection. Confident: the
        stripped root becomes the primary guess. Otherwise we stay honest and return the
        surface as its own stem, keeping every stripped candidate as a ranked alternative so
        nothing is lost. All results are `guess`.
        """
        parses: list[AnalysisResult] = []
        for split in range(2, len(word)):
            root_guess = word[:split]
            if not _has_vowel(root_guess):
                continue
            for pos in (tags.NOUN, tags.VERB):
                synth = Root(root_guess, pos, frozenset(), root_guess, None)
                # Derivation, the ek-fiil, and voice all stay off for guesses: an unknown root
                # must not be split on a derivational suffix (malik -> *ma+lik), a copular
                # ending (*güzeldi from an OOV güzel), nor a (fully-productive) voice suffix
                # (*zorla+tıl), so guesser output is byte-identical to before.
                for res in _generate(
                    word,
                    synth,
                    source=tags.SOURCE_GUESS,
                    allow_derivation=False,
                    allow_nominal_copula=False,
                    allow_voice=False,
                ):
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

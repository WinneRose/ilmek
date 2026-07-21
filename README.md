# 🧩 ilmek

**Explainable Turkish morphology — a stemmer, lemmatizer, and morphological analyzer over
one shared, deterministic native engine.**

Turkish is agglutinative: a single word can carry a long, ordered chain of suffixes, and
those suffixes change shape through vowel harmony and consonant mutations. Naïve
"strip the ending" stemmers cannot handle this reliably. `ilmek` instead models the
morphology directly — `stem`, `lemmatize`, and `analyze` are three views of the *same*
analysis, so they never disagree.

> **Status:** `v0.1.0a0` (alpha). This release ships the **native core**: normalization,
> tokenization, and a rule-based morphological engine covering noun inflection, the common
> verb tense/aspect forms, and Turkish phonology. Optional Stanza/Zemberek backends and
> sentence-level disambiguation are on the roadmap. No model download, no network, no Java —
> the native backend is pure Python and offline.

---

## Install

```bash
pip install ilmek            # native core (no heavy dependencies)
# reserved extras (later milestones):
# pip install "ilmek[stanza]"   # Stanza adapter (v0.2)
# pip install "ilmek[all]"      # everything
```

Requires Python ≥ 3.10.

## Quickstart

```python
import ilmek as trnlp

trnlp.stem("kitaplarımızdan")        # 'kitap'
trnlp.lemmatize("kitaplarımızdan")   # 'kitap'

# Every valid analysis, best candidate first:
for a in trnlp.analyze("kitaplarımızdan"):
    print(a.lemma, a.pos, a.morphemes, a.features)
# kitap NOUN ['lar', 'ımız', 'dan'] {'number': 'plural', 'possessive': '1pl', 'case': 'ablative'}

# A whole sentence -> a Document:
doc = trnlp.analyze_sentence("Yüz kişi denizde yüzüyor.")
print(doc.lemmas)   # ['yüz', 'kişi', 'deniz', 'yüz', None]

# Reusable pipeline:
nlp = trnlp.load()
nlp("Kitaplarımızı masaya bıraktık.")   # Document
```

Ambiguity is preserved, never silently resolved:

```python
[(a.pos, a.features) for a in trnlp.analyze("evi")]
# his/her house (3sg possessive) AND the house (accusative) — both returned.
```

## CLI

```bash
ilmek analyze "kitaplarımızdan"          # tab-separated analysis
ilmek analyze "evi" --json               # JSON
ilmek lemmatize "Kitaplarımızı masaya bıraktık."
ilmek stem "Denizde yüzüyor."
```

`models`, `benchmark`, and `serve` are reserved and report which milestone delivers them
rather than pretending to work.

## Output schema

Every analysis — from any backend — is an `AnalysisResult` with a fixed shape:

```python
AnalysisResult(
    surface="kitaplarımızdan",
    lemma="kitap",
    stem="kitap",
    pos="NOUN",
    morphemes=["lar", "ımız", "dan"],
    features={"number": "plural", "possessive": "1pl", "case": "ablative"},
    analyses=[],          # alternative candidates, never dropped
    confidence=None,      # never a fabricated score
    backend="native",
    source="lexicon",     # lexicon | rule | guess — provenance is explicit
)
```

`source` separates lexicon-verified analyses from unknown-word **guesses**; `confidence`
stays `None` until a scoring/disambiguation layer sets it.

## How the native engine works

```
text
 → normalization      Turkish I/İ/ı/i casing, Unicode NFC, apostrophe folding
 → tokenization       words / numbers / dates / abbreviations / punctuation
 → lexicon lookup     root candidates (by first character; voicing/vowel-drop aware)
 → morphotactic FSM   ordered noun & verb suffix-transition graphs
 → morphophonemics    vowel harmony, consonant voicing, buffers, linking vowels
 → candidate analyses every morphophonemically valid parse
```

It is an **analysis-by-generation** engine: for each candidate root it walks the
morphotactic graph *forward*, realizing each abstract suffix (e.g. `DAn`, `(y)AcAk`) into
its surface form in context, and keeps paths that reproduce the input word exactly. Every
accepted analysis is therefore valid by construction, and its transition sequence *is* the
morpheme segmentation. Rules live in data (`ilmek/data/`, `morphotactics.py`) so the
language model is auditable and extensible without touching the engine.

## Coverage & known limitations (v0.1)

**Handled:** Turkish casing/normalization; tokenization incl. apostrophe proper nouns;
noun plural, all six possessives, all six cases, pronominal `-n-` buffering; verb negation,
progressive/future/past/evidential, negative imperative, both person paradigms, one copular
(ek-fiil) layer with the `-y-` buffer (`geldiydi`); vowel harmony; consonant voicing
(`kitap→kitabı`, `renk→rengi`) and its lexical exceptions (`top→topu`); vowel drop
(`burun→burnu`); `-Iyor` stem narrowing.

**Unknown words** are handled by a *conservative* guesser (always `source="guess"`): it
proposes a stripped root only on strong evidence — a multi-suffix chain or a distinctive
single suffix (`teminatlardan→teminat`, `seninle→sen`) — and stays with the surface form
when the ending is short and ambiguous (`teminatı`, so it never mis-strips `kalem→kale`).
Every candidate is kept and ranked. Because it reuses the morphotactic FSM, it improves
automatically as morphology grows.

**Not yet (named milestones):** derivational morphology (so `yaşadıklarımızın` reaches
`yaşa`); irregular pronouns/closed classes (`sen`, `var`); aorist, ability, and full
mood/voice inventory; the irregular `de-→diyor` / `ye-→yiyor` glide raising; sentence-level
disambiguation (candidates returned unranked, `confidence=None`); Stanza/Zemberek backends;
CoNLL-U I/O. The seed lexicon is intentionally small — the single biggest lever for cutting
`guess` rates on common words.

## Roadmap

| Version | Focus |
|---|---|
| **v0.1** (this) | Native core: normalization, tokenizer, stemmer, lemmatizer, basic analysis |
| v0.2 | Stanza adapter + model registry/downloader |
| v0.3 | Full native morphology engine (expanded lexicon, complete verb morphology, generation) |
| v0.4 | Evaluation suite (IMST/PUD, accuracy/coverage/speed) |
| v0.5 | Developer ecosystem (plugins, CLI, HTTP API, user lexicons) |
| v1.0 | Stable, documented, backward-compatibility policy |

## Contributing

Language rules are data. Every new morphological rule must ship with a **positive** test, a
**negative** test (where it must *not* apply), an **exception** test where relevant, and a
long-chain combination test — see `tests/` and `AGENTS.md`. Run:

```bash
pip install -e ".[dev]"
pytest            # test suite
ruff check .      # lint
ruff format .     # format
```

## Design principles

- Accuracy over aggressive stemming; explainable output over black boxes.
- The core analyzer is context-free and deterministic; disambiguation is a separate layer.
- Guesses are never presented as lexicon-verified facts.
- Every valid analysis is returned; no confidence is invented.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Optional model backends carry
their own licenses and are downloaded on demand via the model registry, never bundled.

## Acknowledgements

Built on the tradition of computational Turkish morphology — Oflazer's two-level
description, FST-based analyzers (Ozturel et al.; Yıldız et al.), perceptron/hybrid
disambiguation (Sak et al.; Kutlu & Çiçekli), and the UD Turkish treebanks. See the project
notes for the full reading list.

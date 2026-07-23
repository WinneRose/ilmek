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

### Derivational morphology

A single **derivation slot** sits between the root and the inflectional suffixes, so a
derived stem then inflects normally. The productive, high-frequency derivations are:

| Suffix | Class change | Example | Derivation name |
|---|---|---|---|
| `-lI`  | noun → adj  | `evli` (ev)          | `li`  |
| `-sIz` | noun → adj  | `evsiz` (ev)         | `siz` |
| `-lIk` | noun/adj → noun | `kitaplık`, `güzellik` | `lik` |
| `-CI`  | noun → noun | `yolcu`, `kitapçı`, `işçi` | `ci` |
| `-mA`  | verb → noun | `gelme`              | `ma`  |
| `-(y)Iş` | verb → noun | `geliş`, `yürüyüş`  | `is`  |
| `-mAk` | verb → noun / non-finite VERB (infinitive) | `gelmek` | `mak` |
| `-(y)An` | verb → adj (participle) | `gelen` | `an` |
| `-DIk`   | verb → adj (participle) | `yaşadık`, `bildiği` | `dik` |
| `-(y)AcAk` | verb → adj (participle) | `gelecek` | `acak` |

The **derivation boundary is visible**: each derived analysis carries an ordered tuple of
derivation names under `features["derivation"]` (e.g. `("dik",)`), the `stem` is the surface
at the last derivation boundary while the `lemma` stays the base lexeme, and `pos` reflects
the derived word class. So the derived stem inflects like any root:

```python
a = ilmek.analyze("yaşadıklarımızın")[0]
a.lemma, a.stem, a.pos            # 'yaşa', 'yaşadık', 'ADJ'
a.morphemes                       # ['dık', 'lar', 'ımız', 'ın']
a.features["derivation"]          # ('dik',)
```

`-CI` hardens to `ç` after a voiceless consonant (`kitap+CI → kitapçı`, `iş+CI → işçi`) but
stays `c` otherwise (`yol+CI → yolcu`) — one data fact in `alphabet.SUFFIX_ALTERNATIONS`.
Because `-DIk`/`-(y)AcAk`/`-mA` collide with the finite past-1pl/future/negative-imperative,
the analyzer **ranks fewer derivations first**: `geldik` stays finite past, `gelecek` stays
finite future, `gelme` stays the negative imperative, and the participle/verbal-noun reading
is preserved as a ranked alternative — genuine ambiguity is never erased. The guesser does
**not** derive, so unknown-word behavior is unchanged.

**Deferred (documented, `xfail`):** derivation *stacking* (`evlilik` = `-lI`+`-lIk`);
inflection of the infinitive (`gelmekten`); `-lI` on numerals/proper nouns (`ikili`,
`Ankaralı`) — `applies_to` deliberately restricts `-lI`/`-CI` to `{NOUN}`. These are
coverage deferrals, not wrong rules.

## Coverage & known limitations (v0.1)

**Closed-class function words** — the high-frequency conjunctions (`ve`, `ama`, `çünkü`),
postpositions (`için`, `gibi`, `göre`), particles (`değil`, `mi`, the clitic `de`/`da`), and
adverbs (`hala`, `belki`, `artık`) — are enumerated whole (`ve → ve` CONJ, `için → için` ADP)
so they resolve as `lexicon`, not guesses. They are **indeclinable**: listed as `source`-verified
whole surfaces with `lemma == surface` and no morphemes, never routed through the FSM (which
would overgenerate `*amalar`/`*gibiye`, and turn run-on misspellings like `belkide` into false
locatives). Homographs are preserved, not erased: `de` → particle (primary) keeps the verb
`de-` imperative as an alternative; `göre`/`bile`/`birden` keep their `gör`/`bil`+optative and
`bir`(NUM)+ablative readings. The three *declining* function words are the exception, routed
through dedicated FSM start states by a lexicon attribute: the interrogative particle `mi`
(`midir`, `miyim`, `miydi`), the **negative copula** `değil` (`değildi`, `değilim`, `değilse`,
`değilmiş` — lemma `değil`, `polarity=negative` inherent), and the standalone **substantive
verb** `i-` (`idi`, `imiş`, `ise`, `iken` — lemma `i`, AUX). Remaining copular corners
(`değilken`, `idiyse`, `-DIrlAr` stacking) stay deferred to the guesser.

**Handled:** Turkish casing/normalization; tokenization incl. apostrophe proper nouns;
noun plural, all six possessives, all six cases, pronominal `-n-` buffering; verb negation,
progressive/future/past/evidential, negative imperative, both person paradigms, one copular
(ek-fiil) layer with the `-y-` buffer (`geldiydi`); ability `-(y)Abil` (`gelebilir`,
`okuyabilir`), conditional `-sA` (`gelse`, `gelseydi`) and optative `-(y)A` (`gele`,
`gelelim`), the positive aorist with its lexical `-r`/`-Ar`/`-Ir` allomorph (a per-root
lexicon fact — `okur`, `yapar`, `oturur`, marked `gelir`) and the negative aorist `-mAz`
(`gelmez`) with its defective person paradigm; vowel harmony; consonant voicing
(`kitap→kitabı`, `renk→rengi`) and its lexical exceptions (`top→topu`); vowel drop
(`burun→burnu`); `-Iyor` stem narrowing.

**Unknown words** are handled by a *conservative* guesser (always `source="guess"`): it
proposes a stripped root only on strong evidence — a multi-suffix chain or a distinctive
single suffix (`teminatlardan→teminat`, `seninle→sen`) — and stays with the surface form
when the ending is short and ambiguous (`teminatı`, so it never mis-strips `kalem→kale`).
Every candidate is kept and ranked. Because it reuses the morphotactic FSM, it improves
automatically as morphology grows.

## Alpha review follow-up

The review regression set is now included in `tests/test_feedback_regressions.py`. It covers:

- circumflex vowels (`â`, `î`, `û`) as word characters while preserving lookup folding;
- TDK-checked core vocabulary such as `posta`, `antiye`, `gönderi`, `makine`, `pul`, `zarf`,
  `kart`, `etiket`, `damga`, `yazı`, `diğer`, `birleşik`, `yazılı`, `en` and `çoğunlukla`;
- noun chains with plural + 3sg possessive + case, including `gönderilerinde`,
  `makinesinden`, `tarihinde` and `pulları`;
- sentence-level preferences for determiner `bir`, passive participles, temporal `geçmişte`,
  and the non-finite `VERB/verbform=infinitive` reading of `-mAk`.
- TDK's high-risk confusion pairs such as `meşruiyet/meşrutiyet`, `irtica/iltica`,
  `muhasebe/musahabe`, `kar/kâr`, `hak/hâk`, `berat/beraat`, `kabil/kabîl` and
  `mahsur/mahzur`.

The additions are kept in `ilmek/data/lexicon/feedback_core.json` so review-driven vocabulary
remains auditable; the TDK confusion set is in `ilmek/data/lexicon/tdk_frequently_confused.json`.
Lexical decisions were checked against the Turkish Language Association's Güncel Türkçe Sözlük;
this is a seed lexicon, not a claim of complete TDK coverage.

The first high-yield expansion batch is stored separately in
`ilmek/data/lexicon/common_core3.json`: 296 regular roots (132 nouns, 85 verbs, 79
adjectives) plus 25 frozen adverbs/determiners. The batch is data-only and deliberately adds
no unverified voicing or vowel-drop attributes. Every root has a source test, and the benchmark
now reports mean/max candidate counts alongside accuracy, coverage, unknown-word rate, and
throughput. The lexical review uses the [TDK Sözlükleri](https://sozluk.tdk.gov.tr/) portal;
there is no runtime TDK scraping or network dependency.

**Not yet (named milestones):** derivational *stacking* and infinitive inflection (the
productive single-slot derivations above already reach `yaşadıklarımızın → yaşa`); the
impossibilitive `-(y)AmA` (`gelemez`), the copular conditional `-(y)sA` (`gelirse`), and the
negative-aorist 1sg/1pl (`gelmem`/`gelmeyiz`) — deferred (strict `xfail`) rather than
overgenerated; voice and the remaining mood inventory; the irregular `de-→diyor` /
`ye-→yiyor` glide raising; context-free `analyze()` returns candidates unranked with
`confidence=None`; sentence-level `Pipeline` disambiguation remains heuristic and optional;
Stanza/Zemberek backends;
CoNLL-U I/O. The seed lexicon is intentionally small — the single biggest lever for cutting
`guess` rates on common words. Sentence disambiguation is available as an optional heuristic
layer through `Pipeline`; it does not change the context-free candidate list returned by
`analyze()`.

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

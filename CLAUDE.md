# CLAUDE.md

Project context and application instructions for Claude-based coding tools. Extends
[`AGENTS.md`](AGENTS.md) — read that first; this file adds Claude-specific behavior and
does not repeat the shared rules.

## Read first

Before starting a task:

1. Read `AGENTS.md` and apply the shared rules.
2. Study the relevant module's code and tests.
3. Assess the change's shared impact on the stemmer, lemmatizer, and analyzer outputs — they
   derive from one analysis.
4. If it involves a Turkish language rule, verify the test examples and the academic basis.
5. On an ambiguous requirement, state the assumption explicitly before starting a broad
   implementation.

## Project context

This project is **not** a simple suffix stripper. The core depends on these components
working together:

```
normalization
  → tokenization
  → lexicon lookup
  → morphotactic traversal
  → morphophonemic validation
  → candidate analyses
  → contextual disambiguation   (separate, optional layer)
```

Stem and lemma must be derived from the *same* morphological analysis. When you fix one, do
not break the other.

Concretely in this codebase: the native engine is **analysis-by-generation**
(`morphology/analyzer.py`). It walks the morphotactic FSM (`morphology/morphotactics.py`)
forward and realizes each abstract suffix with `morphology/phonology.py`, keeping paths that
reproduce the surface word. If you change phonology or morphotactics, re-check the showcase
cases: `kitaplarımızdan → kitap`, `gelmeyecekmişsiniz → gel`, `geleceğim → gel`,
`burnu → burun`, `gitti`/`gidiyor` (voicing only before a vowel).

## Reasoning guidelines

- Determine the word's possible roots and parts of speech first.
- Do not delete suffixes by surface shape alone; verify the transition order.
- When reversing a sound change, accept that multiple root candidates may result.
- If several valid analyses exist, do not present one as the definitive truth.
- Do not conflate core analysis with contextual ranking.
- Do not turn a rule-breaking example into a lexicon exception automatically; first check the
  tokenizer, the root attributes, and the transition graph.

## Change workflow

1. Define the desired behavior with one or more concrete Turkish examples.
2. Confirm current behavior with a test or a small reproduction.
3. Identify the **narrowest** layer to change.
4. Write regression and acceptance tests first.
5. Make the smallest workable change.
6. Run the relevant module tests.
7. Compare full test and benchmark results.
8. Record the technical decision and limitations in the docs.

## Morphology checklist

When adding a rule, verify:

- [ ] part of speech
- [ ] the root's final vowel and consonant
- [ ] major and minor vowel harmony
- [ ] consonant voicing or devoicing
- [ ] vowel drop or narrowing
- [ ] buffer / linking letter
- [ ] derivational/inflectional affix boundary
- [ ] proper-noun or abbreviation behavior
- [ ] irregular or loanword exception
- [ ] possibility of multiple analyses

## Expected output shape

Analyses must be explainable and structured (see `core/document.py :: AnalysisResult`):

```json
{
  "surface": "kitaplarımızdan",
  "lemma": "kitap",
  "stem": "kitap",
  "pos": "NOUN",
  "morphemes": ["lar", "ımız", "dan"],
  "features": {"number": "plural", "possessive": "1pl", "case": "ablative"},
  "confidence": null,
  "source": "lexicon"
}
```

Match field names and tags to the API contract exactly (`core/tags.py`).

## Do not

- Add a non-generalizable `if` block just to pass one example word.
- Make results look successful by deleting tests or changing expectations without reason.
- Use language-insensitive lowercase/uppercase for Turkish.
- Mark an unknown-word guess as a lexicon-verified analysis.
- Force a disambiguator result onto a morphologically impossible candidate.
- Reformat or rename in unrelated files.
- Claim a performance improvement without a benchmark.

## Communication

When reporting completed work, briefly state: the changed behavior, the tests added or
updated, the verifications run, the known limitations, and any accuracy/performance impact.

## Open decisions

Treat these as assumptions until finalized (current working choices in parentheses):

- Implementation language and packaging (Python ≥3.10, hatchling — **decided**).
- FST library vs. custom transition engine (custom pure-Python generation — **decided v0.1**).
- Source and license of the initial root lexicon (small hand-seeded set — **interim**).
- Morphological tag schema (`core/tags.py`, UD-aligned — **interim, will extend**).
- Gold test datasets (IMST/PUD — **planned v0.4**).
- v1.0 accuracy and performance thresholds (**open**).

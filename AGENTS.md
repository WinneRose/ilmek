# AGENTS.md

Tool-independent working contract for every coding agent on **ilmek**. Adapted from
the project's *Yapay Zeka Yönergeleri → AGENTS.md* specification. Tool-specific guidance for
Claude-based tools lives in [`CLAUDE.md`](CLAUDE.md), which extends (never contradicts) this
file.

## Project overview

ilmek is an NLP package that models Turkish agglutinative morphology and its sound
changes, providing **stemming, lemmatization, and morphological analysis over one shared
native engine**. It is not a suffix stripper: stem and lemma are derived from the same
analysis.

## Product goals

- Reliably reduce Turkish words to root and lemma.
- Produce **all** valid morphological analysis candidates for a word.
- Model noun and verb morphotactics with explainable rules.
- Keep contextual disambiguation a **separate, optional** layer.
- Be academically verifiable, testable, and extensible.

## Source of truth (priority order)

1. Approved technical decisions and acceptance criteria.
2. The gold test dataset.
3. The morphological tag schema and rule documentation.
4. Source-code behavior verified by existing tests.
5. Academic references.

If current behavior and documentation conflict, **state the difference explicitly — do not
silently invent a new rule.**

## Architecture boundaries

| Layer | Responsibility |
|---|---|
| `core.normalization` | Unicode, Turkish casing, apostrophe standardization |
| `core.tokenization` | word / number / date / abbreviation / punctuation splitting |
| `morphology.lexicon` | lemma, POS, irregular forms, root attributes (voicing, vowel-drop) |
| `morphology.morphotactics` | noun & verb suffix-transition graphs (data) |
| `morphology.phonology` | vowel harmony and sound changes (morphophonemics) |
| `morphology.analyzer` | generate and filter valid analysis candidates |
| `disambiguation` | heuristic candidate scoring + sentence-context candidate ranking (separate, optional) |
| **Public API** | `stem`, `lemmatize`, `analyze`, `analyze_sentence` |

No circular dependencies between layers. **The contextual model must never be a required
dependency of the core analysis engine.** `core` depends on nothing above it; `morphology`
depends only on `core`; backends and disambiguation depend on `morphology`, never the reverse.

## Turkish language requirements

- Do **not** trust default-locale casing for `I / İ / ı / i`; use `core.normalization`.
- Model vowel harmony with explicit rules (`phonology.realize`), not by enumerating every
  surface form of each suffix.
- Verify consonant voicing/devoicing against the root attribute and context, not blindly.
- Use exception data for vowel drop, narrowing, and irregular verbs.
- Preserve the apostrophe in proper nouns and abbreviations; analyze them separately.
- Keep the derivational vs. inflectional affix boundary visible in the analysis.
- Keep unknown-root **guesses** separate from lexicon-verified analyses (`source`).

## Implementation rules

- Make small, single-purpose changes; no unrelated refactors or dependency bumps.
- Justify and document any new external dependency and its alternatives first.
- Keep language rules in **data / rule files** wherever possible, not in code.
- Preserve backward compatibility of public APIs.
- Never swallow errors silently — produce a structured result or a clear error.
- Preserve deterministic core behavior.

## Testing contract

Every new morphological rule must include at least:

- a **positive** example (rule applies),
- a **negative** example (rule must *not* apply),
- an **exception** example where one exists,
- a combination test inside a **long suffix chain**,
- a **consistency** test across `stem`, `lemma`, and the detailed analysis.

For bug fixes, first add a regression test that reproduces the bug.

Tests are organized by these markers (`pytest.mark.positive|negative|exception|consistency`).

## Evaluation

Core metrics: lemma accuracy, stem accuracy, analysis coverage, presence of the correct
analysis among candidates, contextual disambiguation accuracy, unknown-word success,
per-word latency and throughput. When improving performance, **compare accuracy against the
previous benchmark** — no performance claim without a benchmark.

## Documentation rules

- Document every new tag, rule, or exception.
- Update examples and release notes on any API change.
- Cite the academic source for a rule derived from one.
- Do not hide known limitations.

## Commands

```bash
# Install (development, with test + lint tooling)
pip install -e ".[dev]"

# Unit tests (fast; whole suite is already fast)
pytest -q

# Full test suite with markers, e.g. only the negative-rule tests
pytest -m negative

# Lint and format
ruff check .
ruff format .

# Benchmark: run the evaluation harness over the hand-labeled gold set
# (lemma/stem accuracy, coverage, disambiguation accuracy, unknown-word rate, speed).
# The roadmap's "Aşama 1 evaluation" item landed ahead of its v0.4 label; the version
# string is a plan marker, not a gate. Use PYTHONUTF8=1 for the Turkish surfaces.
ilmek benchmark              # readable per-category + overall report
ilmek benchmark --json       # same numbers as JSON, for regression comparison
ilmek benchmark --category voice
```

## Definition of done

A task is complete only when:

- acceptance criteria are met,
- relevant tests are added and pass,
- there is no regression in existing tests,
- documentation is updated,
- accuracy or performance impact is measured,
- assumptions and known limitations are stated.

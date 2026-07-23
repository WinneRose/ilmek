# TDK High-Yield Lexicon Expansion Design

**Date:** 2026-07-23

## Goal

Increase Turkish lexical coverage and reduce false `source="guess"` analyses by adding a
large, auditable set of high-yield TDK-verified roots while preserving deterministic
morphology, homograph candidates, and measurable performance.

## Current baseline

- The packaged gold set contains 154 items, 151 scored items, and 13 primary guessed
  analyses.
- Existing review-driven and TDK-confusion data are isolated in
  `ilmek/data/lexicon/feedback_core.json` and
  `ilmek/data/lexicon/tdk_frequently_confused.json`.
- Lexicon files are loaded automatically from `ilmek/data/lexicon/*.json`.
- The current analyzer preserves all valid candidates and ranks them deterministically;
  unknown-root guesses remain distinct from lexicon analyses.

## Scope

### In scope

1. Add high-yield Turkish roots in batches, starting with words that are currently guessed,
   missing from the gold set, or responsible for common review failures.
2. Use TDK Güncel Türkçe Sözlük and its “Sıkça Karıştırılan Sözler” material as the lexical
   authority; keep source notes in the data file rather than adding a runtime web dependency.
3. Record coarse POS and only verified root attributes. Do not infer voicing, vowel drop,
   front harmony, gemination, or irregular forms from spelling alone.
4. Preserve homographs as separate analyses and use context-free ranking only where the
   surface provides reliable evidence, such as a matching circumflex pattern.
5. Add regression, negative, long-chain, and consistency tests for each new behavior or
   attribute class.
6. Measure lemma accuracy, stem accuracy, coverage, disambiguation, unknown-word rate,
   candidate count, and per-word latency after every batch.

### Out of scope

- Runtime scraping or network access to TDK.
- Replacing the existing morphotactic engine with a third-party analyzer.
- Marking an unverified word as lexicon-known merely to improve benchmark numbers.
- Deleting alternate analyses to force a preferred single reading.
- Adding a large frequency model or contextual neural dependency before the lexicon
  expansion is measured independently.

## Data design

Each expansion batch will be a separate JSON file under `ilmek/data/lexicon/` with:

- a source comment naming the TDK dataset/page and review date;
- regular root entries containing `lemma`, `pos`, and verified attributes;
- enumerated `irregular` entries only for closed-class or whole-surface forms;
- no duplicate entry when an existing seed entry already covers the same lemma/POS unless
  the new row adds a verified lexical attribute.

The first batch will target approximately 300–500 roots. Later batches may reach 2,000+
roots only if benchmark coverage improves without a material increase in false primary
analyses or latency.

## Selection and scoring

Candidate priority is:

1. a gold-set primary guess or known gap;
2. a word from the review regression set;
3. a high-frequency basic noun, adjective, verb, adverb, determiner, conjunction, or
   postposition absent from the lexicon;
4. a TDK confusion pair whose members are absent or misranked;
5. a common inflectional base needed by a long-chain example.

The importer/validation tooling will reject malformed POS values, duplicate folded
lemma/POS rows without an explicit homograph rationale, unknown attribute names, and
entries whose generated bare form is not lexicon-sourced.

## Architecture

- `morphology.lexicon` remains the only consumer of root data and continues to load all
  JSON files deterministically.
- TDK provenance remains data metadata; no network call is made by `Analyzer` or `Pipeline`.
- Phonological attributes remain explicit data. A root without a verified attribute must
  not gain the corresponding allomorph.
- `Analyzer` continues to return every valid candidate. Any new ranking rule must be
  stable, surface-scoped, and covered by a negative test.
- Benchmark reporting will add candidate-count and latency comparisons without changing
  the public API.

## Testing and acceptance criteria

For each batch:

- every added root has a lexicon-source test;
- each new attribute class has positive, negative, exception, long-chain, and
  stem/lemma/analysis consistency coverage;
- existing full tests pass with no new unexpected failures;
- benchmark lemma and stem accuracy do not regress;
- coverage and unknown-word rate improve or remain equal;
- no statistically meaningful latency regression is accepted without an explicit note;
- README and release notes describe the batch size, source, known limitations, and metrics.

## Known limitations

TDK’s online interface is a lexical reference, not a directly licensed bulk morphology
database for this project. The package will therefore ship reviewed manifests rather than
mirror the entire TDK database. POS granularity and lexical attributes are project-level
annotations and must be reviewed independently of a TDK headword match.

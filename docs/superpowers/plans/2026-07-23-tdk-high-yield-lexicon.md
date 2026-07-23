# TDK High-Yield Lexicon Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the packaged Turkish lexicon with a measured high-yield TDK-reviewed batch and expose enough benchmark data to verify that coverage improves without harming accuracy or latency.

**Architecture:** Keep TDK-derived roots in data-only JSON manifests loaded by the existing `Lexicon.load()` glob. Add only verified coarse POS and explicit phonological attributes; keep irregular closed-class forms separate. Extend the evaluation layer with candidate-count metrics while leaving the public analyzer API unchanged.

**Tech Stack:** Python 3.10+, JSON data, pytest, Ruff, the existing native analyzer and benchmark CLI.

## Global Constraints

- Do not add a runtime network dependency or scrape TDK during analysis.
- Unknown-root guesses remain `source="guess"`; a word enters the lexicon only through a reviewed data row.
- Preserve all valid homograph candidates and deterministic ordering.
- Every new phonological rule or attribute requires positive, negative, exception, long-chain, and consistency coverage.
- No dependency bumps or unrelated refactors.
- Run `pytest -q`, `ruff check .`, `ruff format --check .`, and `ilmek benchmark --json` before completion.

---

### Task 1: Establish the expansion batch and data validation contract

**Files:**
- Create: `ilmek/data/lexicon/common_core3.json`
- Create: `tests/test_common_core3_lexicon.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `Lexicon.load()` automatic JSON loading and `Analyzer.analyze()`.
- Produces: a reviewed batch manifest with at least 300 common roots, plus a data-driven test that every manifest root resolves as `source="lexicon"` at its bare surface.

- [ ] **Step 1: Write the failing data-driven tests**

  Load `common_core3.json`, parameterize every entry, and assert that `analyze(lemma)` contains the exact `lemma`, `pos`, and `source="lexicon"`. Add representative tests for noun plural/case, possessive + case, adjective copula, verb tense, and negative examples proving that an unlisted spelling remains a guess.

- [ ] **Step 2: Run the focused tests and confirm the new manifest is missing**

  Run:

  ```powershell
  $py='C:\Users\Winnerose\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
  & $py -m pytest tests/test_common_core3_lexicon.py -q
  ```

  Expected: collection or assertion failures because `common_core3.json` does not exist.

- [ ] **Step 3: Add the first reviewed common-root batch**

  Add regular NOUN, ADJ, VERB, ADV, and NUM roots that are absent from the current lexicon. Use no phonological attribute unless the data is explicitly verified; keep TDK source and review date in `_comment`. Exclude ambiguous compounds and uncertain derivational bases rather than fabricating attributes.

- [ ] **Step 4: Run the focused tests and inspect generated analyses**

  Run the focused test file and print the first three candidates for all representative inflected forms. Confirm that every new bare word is lexicon-sourced and that alternate analyses remain present.

- [ ] **Step 5: Update documentation**

  Add the batch size, POS distribution, TDK provenance, and known limitations to the alpha-review section in `README.md`.

### Task 2: Add benchmark candidate-count and regression metrics

**Files:**
- Modify: `ilmek/evaluation/metrics.py`
- Modify: `ilmek/evaluation/benchmark.py`
- Modify: `tests/test_evaluation.py` or the existing benchmark test module containing metric unit tests

**Interfaces:**
- Consumes: `ItemRecord.candidates` and `BenchmarkReport.records`.
- Produces: a deterministic `candidate_count` score and per-word candidate-count summary in JSON/text benchmark output; existing keys and public APIs remain compatible.

- [ ] **Step 1: Write failing metric tests**

  Add a small synthetic record set with zero, one, and multiple candidates. Assert that the mean, maximum, and guessed-primary candidate count are reported deterministically and that the benchmark JSON still contains all existing accuracy fields.

- [ ] **Step 2: Run the metric tests and observe missing fields**

  Run the targeted evaluation tests and confirm the new candidate-count assertion fails before implementation.

- [ ] **Step 3: Implement candidate-count aggregation**

  Add a small immutable score/summary helper in the evaluation layer. Populate the report from existing candidate tuples; do not re-run analysis or alter the analyzer result order.

- [ ] **Step 4: Verify text and JSON benchmark output**

  Run `ilmek benchmark --json` and assert that the new metrics are numeric, stable, and present in the human-readable report without changing existing accuracy calculations.

### Task 3: Add high-yield morphology coverage tests

**Files:**
- Modify: `tests/test_common_core3_lexicon.py`
- Create or modify: `tests/test_lexicon_expansion_attributes.py` only if the batch contains verified attributes

**Interfaces:**
- Consumes: the common-core manifest and existing morphotactic graphs.
- Produces: regression coverage for long chains and explicit negative/exception behavior.

- [ ] **Step 1: Add positive and negative chain tests**

  Cover examples such as plural + case, plural + possessive + case, adjective copula, verb negation + tense, and one long suffix chain from each populated POS class. Assert lemma, stem, morphemes, POS, and source together.

- [ ] **Step 2: Add attribute-specific tests only for verified rows**

  For each `voicing`, `vowel_drop`, `front_harmony`, or `gemination` row, add one positive example, one non-applying root, one exception, and one long-chain consistency check. If no independent verification exists, remove the attribute instead of adding a speculative test.

- [ ] **Step 3: Run focused and full tests**

  Run the new files first, then `pytest -q`; fix data or implementation rather than weakening assertions.

### Task 4: Measure, review, and finalize the batch

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md` if present, otherwise document in README only

**Interfaces:**
- Consumes: benchmark JSON before and after the batch.
- Produces: a documented comparison of lemma/stem accuracy, coverage, disambiguation, unknown-word rate, candidate counts, and throughput.

- [ ] **Step 1: Run the baseline comparison**

  Use the saved pre-batch benchmark JSON from the current tree and the post-batch `ilmek benchmark --json`; compare all headline fields and record the deltas.

- [ ] **Step 2: Run full verification**

  ```powershell
  $py='C:\Users\Winnerose\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
  & $py -m ruff check .
  & $py -m ruff format --check .
  & $py -m pytest -q
  & $py -m ilmek.cli benchmark --json
  git diff --check
  ```

- [ ] **Step 3: Document known limitations**

  Record any remaining guessed words, unresolved homographs, and any latency increase. Do not claim complete TDK coverage.

- [ ] **Step 4: Attempt to commit the plan implementation**

  Stage only files belonging to this plan. If the repository denies `.git/index.lock`, report the permission blocker and leave the verified working-tree changes intact.

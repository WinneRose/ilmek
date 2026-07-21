# Lexicon candidates (staging — not loaded by the package)

These files hold **200 additional Turkish root entries** curated during the v0.1
adversarial-verification pass, kept out of `ilmek/data/lexicon/` on purpose:

- `nouns.json` — 128 (53 regular, 57 voicing, 18 vowel-drop)
- `verbs.json` — 40
- `adjectives.json` — 32 (26 regular, 6 voicing)

**Status:** structurally validated (valid JSON; no lemma collides with the packaged
lexicon; every `voicing` flag sits on a softenable stop `p/ç/t/k`/`nk`; every `vowel_drop`
entry carries explicit `["free", "reduced"]` forms) and machine-checked by loading them
through `Lexicon.load([...])` + `Analyzer` on a sample of inflected forms.

**Why staged, not merged:** the roadmap keeps v0.1's lexicon a small hand-seeded set;
the expanded lexicon is the **v0.3** milestone. Merging also warrants a per-entry
linguistic review of the `voicing` flags (a wrong flag yields a wrong analysis, which the
project contract forbids).

**To merge** (after review): move/append the entries into `ilmek/data/lexicon/`
(the loader globs `*.json` there), then run `pytest` and spot-check inflected forms.

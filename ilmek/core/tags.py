"""Morphological tag schema (the public vocabulary of an analysis).

Kept in ``core`` so that both the data model and the morphology engine depend on one
documented vocabulary — never the other way around. Values are UD-aligned where a clean
mapping exists, so CoNLL-U export stays a formatting concern rather than a re-labelling.
"""

from __future__ import annotations

# --- Part of speech ------------------------------------------------------------------

NOUN = "NOUN"
PROPN = "PROPN"
VERB = "VERB"
ADJ = "ADJ"
ADV = "ADV"
PRON = "PRON"
NUM = "NUM"
ADP = "ADP"
CONJ = "CONJ"
DET = "DET"
INTJ = "INTJ"
PART = "PART"
PUNCT = "PUNCT"
SYM = "SYM"
ABBR = "ABBR"
X = "X"  # unknown / unanalyzable

POS_TAGS = frozenset(
    {NOUN, PROPN, VERB, ADJ, ADV, PRON, NUM, ADP, CONJ, DET, INTJ, PART, PUNCT, SYM, ABBR, X}
)

# --- Feature keys --------------------------------------------------------------------

NUMBER = "number"  # singular | plural
POSSESSIVE = "possessive"  # none | 1sg 2sg 3sg 1pl 2pl 3pl
CASE = "case"  # nominative accusative dative locative ablative genitive instrumental
POLARITY = "polarity"  # positive | negative
ABILITY = "ability"  # true (potential -yAbil)
TENSE = "tense"  # present past future aorist
ASPECT = "aspect"  # progressive
EVIDENTIAL = "evidential"  # true (reported -mIş)
COPULA = "copula"  # past | assertive (ek-fiil; on a nominal predicate, person marks the
# zero-copula present — güzelim "I am beautiful" — with no separate copula key)
MOOD = "mood"  # imperative | conditional | optative
PERSON = "person"  # 1sg 2sg 3sg 1pl 2pl 3pl
PRON_TYPE = "pron_type"  # personal | demonstrative | interrogative (closed-class pronouns)
EXISTENTIAL = "existential"  # true (existential particle var / yok)
#: Derivational history: an ordered ``tuple`` of derivational-suffix *names* (a derived stem
#: inflects normally afterwards, so this makes the derivation-vs-inflection boundary visible).
#: The value is a tuple (not a list) so an :class:`AnalysisResult`'s features stay hashable
#: for de-duplication. Vocabulary is the closed set of names below.
DERIVATION = "derivation"

#: The closed vocabulary of derivational-suffix names recorded under :data:`DERIVATION`.
#: Noun/adj-forming: li (-lI), siz (-sIz), lik (-lIk), ci (-CI), cik (diminutive -CIk);
#: verb->noun: ma (-mA), is (-(y)Iş), mak (infinitive -mAk); verb->adj participles: an
#: (-(y)An), dik (-DIk), acak (-(y)AcAk). The intensive-adjective diminutives (sıcacık,
#: küçücük) are enumerated IrregularForm surfaces but record the same ``cik`` name.
DERIVATIONS = frozenset({"li", "siz", "lik", "ci", "cik", "ma", "is", "mak", "an", "dik", "acak"})

# --- Feature value vocabulary (for validation / documentation) -----------------------

NUMBERS = frozenset({"singular", "plural"})
PERSONS = frozenset({"1sg", "2sg", "3sg", "1pl", "2pl", "3pl"})
#: Sub-types recorded on closed-class pronouns. We label ``personal`` (ben/sen/biz/siz)
#: and ``demonstrative`` (bu/şu) and ``interrogative`` (kim); the ``o``/``onlar`` paradigm
#: carries *no* pron_type because its surface is genuinely ambiguous between the personal
#: and demonstrative reading — we do not fabricate a distinction the form does not show.
PRON_TYPES = frozenset({"personal", "demonstrative", "interrogative"})
CASES = frozenset(
    {
        "nominative",
        "accusative",
        "dative",
        "locative",
        "ablative",
        "genitive",
        "instrumental",
    }
)
POLARITIES = frozenset({"positive", "negative"})
TENSES = frozenset({"present", "past", "future", "aorist"})
#: Copula (ek-fiil) values. ``past`` is -(y)DI (güzeldi, gelecekti); ``assertive`` is the
#: generalizing -DIr (güzeldir). The evidential ek-fiil -(y)mIş is recorded under
#: :data:`EVIDENTIAL`, and the zero-copula present is marked by :data:`PERSON` alone, so
#: neither adds a COPULA value here.
COPULAS = frozenset({"past", "assertive"})
#: Verbal & copular moods. ``imperative`` is the bare-root/negated-root reading; ``conditional``
#: is the verbal -sA suffix AND the nominal copular -(y)sA (güzelse); ``optative`` is -(y)A.
#: The VERBAL copular conditional -(y)sA (gelirse) and the copular optative are later
#: milestones (see the morphotactics module docstring).
MOODS = frozenset({"imperative", "conditional", "optative"})

# --- Analysis provenance -------------------------------------------------------------

#: Where an analysis came from. ``lexicon`` = root verified against the lexicon;
#: ``rule`` = produced purely by rule; ``guess`` = unknown-root backoff (NOT verified).
SOURCE_LEXICON = "lexicon"
SOURCE_RULE = "rule"
SOURCE_GUESS = "guess"

#: Which engine produced the result.
BACKEND_NATIVE = "native"
BACKEND_STANZA = "stanza"
BACKEND_ZEMBEREK = "zemberek"

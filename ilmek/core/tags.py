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
#: The auxiliary/copular verb. Carried by the standalone substantive verb i- (the ek-fiil
#: written as a separate word: idi, imiş, ise, iken), which is AUX in UD. A copular ending
#: on a NOMINAL predicate keeps that predicate's own POS (güzeldi stays ADJ); AUX is used only
#: for the free-standing i- word, whose lemma/stem is "i".
AUX = "AUX"
PUNCT = "PUNCT"
SYM = "SYM"
ABBR = "ABBR"
X = "X"  # unknown / unanalyzable

POS_TAGS = frozenset(
    {NOUN, PROPN, VERB, ADJ, ADV, PRON, NUM, ADP, CONJ, DET, INTJ, PART, AUX, PUNCT, SYM, ABBR, X}
)

# --- Feature keys --------------------------------------------------------------------

NUMBER = "number"  # singular | plural
NUM_TYPE = "num_type"  # distributive (-(ş)Ar: birer, ikişer) | ordinal (-(I)ncI: birinci) |
# percent (a %-marked numeric token, e.g. %25 "yüzde 25")
POSSESSIVE = "possessive"  # none | 1sg 2sg 3sg 1pl 2pl 3pl
CASE = "case"  # nominative accusative dative locative ablative genitive instrumental
POLARITY = "polarity"  # positive | negative
ABILITY = "ability"  # true (potential -yAbil)
TENSE = "tense"  # present past future aorist
ASPECT = "aspect"  # progressive
EVIDENTIAL = "evidential"  # true (reported -mIş)
COPULA = "copula"  # past | assertive (ek-fiil; on a nominal predicate, person marks the
# zero-copula present — güzelim "I am beautiful" — with no separate copula key)
MOOD = "mood"  # imperative | conditional | optative | necessitative
#: Verbal voice / valency (çatı): an ordered ``tuple`` of voice names in surface order, so
#: stacked voices are preserved (yaptırt -> ("causative", "causative"); yazdırıl ->
#: ("causative", "passive")). A tuple (not a scalar) because a dict-merge of suffix features
#: would otherwise collapse two causatives into one. Absent on a plain (unvoiced) verb.
VOICE = "voice"  # causative | passive | reflexive | reciprocal (as an ordered tuple)
PERSON = "person"  # 1sg 2sg 3sg 1pl 2pl 3pl
PRON_TYPE = "pron_type"  # personal | demonstrative | interrogative (closed-class pronouns)
EXISTENTIAL = "existential"  # true (existential particle var / yok)
#: The interrogative (question) particle mi/mı/mu/mü and its copular/personal inflections
#: (midir, misin, miyim, miydi, miymiş): a separate token that turns its host into a yes/no
#: question. Set ``True`` so a consumer can detect a question regardless of person/copula. The
#: particle's lemma/stem stay ``"mi"`` for all four harmonic surfaces; this is the only feature
#: a *bare* ``mi`` carries (no number/case/possessive — it is not a full noun).
QUESTION = "question"  # true (the interrogative particle mi and its inflections)
#: Usage register of a form, set only when a surface is a marked (non-standard) variant of a
#: standard form. Today its only value is ``colloquial``, carried by the colloquial personal
#: instrumentals ``benle``/``senle`` (standard: ``benimle``/``seninle``); the standard forms
#: carry no ``register`` key at all, so the pair documents the variant split explicitly.
REGISTER = "register"  # colloquial (a marked non-standard variant)
#: Derivational history: an ordered ``tuple`` of derivational-suffix *names* (a derived stem
#: inflects normally afterwards, so this makes the derivation-vs-inflection boundary visible).
#: The value is a tuple (not a list) so an :class:`AnalysisResult`'s features stay hashable
#: for de-duplication. Vocabulary is the closed set of names below.
DERIVATION = "derivation"

#: Non-finite verb form marker (UD ``VerbForm``). Set on the verb->adjective participle
#: derivations and the verb->adverb converb (zarf-fiil) derivations so the whole non-finite
#: inventory is uniformly queryable, and so each is distinguishable from the homographous
#: *finite* tense it shares a surface with: the -mIş participle (geçmiş "past") vs the finite
#: evidential -mIş (gelmiş "reportedly came"), and the aorist participle -(A/I)r (akar
#: "flowing") vs the finite aorist (akar "flows"). Values are ``participle`` (a verb->adjective
#: form) and ``converb`` (a verb->adverb form: gelerek, gelince, gelmeden, gelirken). A
#: participle/converb keeps its verb lemma/stem and carries a :data:`DERIVATION` name; a
#: participle then inflects nominally, while a converb is terminal (it takes no case/plural).
VERBFORM = "verbform"  # participle (verb->adjective) | converb (verb->adverb)

#: The closed vocabulary of derivational-suffix names recorded under :data:`DERIVATION`.
#: Noun/adj-forming: li (-lI), siz (-sIz), lik (-lIk), ci (-CI), cik (diminutive -CIk),
#: ca (equative/adverbial -CA: güzelce, insanca), sal (relational -sAl: toplumsal), ki
#: (relative/pronominal -ki: evdeki, benimki, dünkü); verb->noun: ma (-mA), is (-(y)Iş),
#: mak (infinitive -mAk); verb->adj participles: an (-(y)An), dik (-DIk), acak (-(y)AcAk),
#: mis (-mIş), ar (aorist -(A/I)r), maz (negative aorist -mAz); verb->adv converbs
#: (zarf-fiil): arak (-(y)ArAk), ip (-(y)Ip), inca (-(y)IncA), ali (-(y)AlI), dikce (-DIkçA),
#: madan (-mAdAn), maksizin (-mAksIzIn), ken (finite-stem -ken).
#: The intensive-adjective diminutives (sıcacık, küçücük) are enumerated IrregularForm surfaces
#: but record the same ``cik`` name.
DERIVATIONS = frozenset(
    {
        "li",
        "siz",
        "lik",
        "ci",
        "cik",
        "ca",
        "sal",
        "ki",
        "ma",
        "is",
        "mak",
        "an",
        "dik",
        "acak",
        "mis",
        "ar",
        "maz",
        "arak",
        "ip",
        "inca",
        "ali",
        "dikce",
        "madan",
        "maksizin",
        "ken",
    }
)

# --- Feature value vocabulary (for validation / documentation) -----------------------

NUMBERS = frozenset({"singular", "plural"})
#: Numeral sub-types recorded under :data:`NUM_TYPE`. ``distributive`` (the numeral suffix
#: -(ş)Ar: bir->birer, iki->ikişer, "n each / n at a time") and ``ordinal`` (the ordinal
#: suffix -(I)ncI: bir->birinci, iki->ikinci, "n-th") are inflectional, not derivational, so
#: the lemma/stem stay the bare numeral (birer -> bir, birinci -> bir). ``percent`` marks a
#: ``%``-written numeric *token* (``%25``) surfaced as NUM by the tokenizer/analyzer numeric
#: path — an orthographic sign, not a suffix, but recorded here so a consumer can spot it.
NUM_TYPES = frozenset({"distributive", "ordinal", "percent"})
PERSONS = frozenset({"1sg", "2sg", "3sg", "1pl", "2pl", "3pl"})
#: Sub-types recorded on closed-class pronouns. We label ``personal`` (ben/sen/biz/siz)
#: and ``demonstrative`` (bu/şu) and ``interrogative`` (kim); the ``o``/``onlar`` paradigm
#: carries *no* pron_type because its surface is genuinely ambiguous between the personal
#: and demonstrative reading — we do not fabricate a distinction the form does not show.
PRON_TYPES = frozenset({"personal", "demonstrative", "interrogative"})
#: Register values (see :data:`REGISTER`). Closed set; only ``colloquial`` is used today.
REGISTERS = frozenset({"colloquial"})
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
#: is the verbal -sA suffix, the nominal copular -(y)sA (güzelse), AND the verbal copular
#: conditional -(y)sA stacking on a finished tense (gelirse, geldiyse); ``optative`` is -(y)A
#: (gele); ``necessitative`` is -mAlI (gelmeli "must come"). The copular optative remains a
#: later milestone (see the morphotactics module docstring).
MOODS = frozenset({"imperative", "conditional", "optative", "necessitative"})
#: Non-finite verb-form values recorded under :data:`VERBFORM`. Closed set: ``participle``
#: (the -(y)An/-DIk/-(y)AcAk/-mIş/aorist/-mAz verb->adjective forms) and ``converb`` (the
#: zarf-fiil verb->adverb forms -(y)ArAk/-(y)Ip/-(y)IncA/-(y)AlI/-DIkçA/-mAdAn/-mAksIzIn and
#: the finite-stem -ken: gelerek, gelince, gelmeden, gelirken).
VERBFORMS = frozenset({"participle", "converb"})
#: Verbal voice values recorded (in order) under :data:`VOICE`. ``causative`` is -DIr/-t/-Ir/
#: -Ar (yaptır, okut, içir, çıkar; stacked as yaptırt); ``passive`` is -Il/-In/-n (yapıl,
#: alın, okun); ``reflexive`` is -In (yıkan, giyin); ``reciprocal`` is -Iş (görüş, dövüş). The
#: -In allomorph is genuinely both passive and reflexive, so a form like yıkan carries both
#: readings; -Iş likewise collides with the verbal-noun -(y)Iş, and both readings are kept.
VOICES = frozenset({"causative", "passive", "reflexive", "reciprocal"})

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

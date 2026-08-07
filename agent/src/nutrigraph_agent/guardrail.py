"""The boundary of the Coach's job, enforced. The model detects; the code decides.

Four subjects are outside the boundary: diagnosis, treatment, and dosage;
eating-disorder content; nutrition for pregnancy, breastfeeding, and children;
and the personal diet management of a chronic disease. A general factual
question about a chronic disease is *not* outside it — that question is answered
from the Corpus, and only a personal plan for the condition is refused. Silence
on the whole subject would be the wrong answer.

**Two detectors, and either one produces a Refusal.**

1. The rule list below, which a reviewer can read and a test can prove. It runs
   before the router, with no model, so a message it catches never reaches an
   Intent path.
2. The router's `out_of_scope` flag, for meaning no word list predicts. The
   router detects; it never writes the Refusal.

**The Refusal is a template in code and never a model output.** Each one names
the boundary, gives the disclaimer, points to a professional, and offers what
the Coach can do instead. Eating-disorder content additionally carries a
help-line. Because it is assembled from these strings, it cannot drift.

Nothing here calls a provider. That is the point: this module is the
deterministic half of the split, beside the redaction patterns and the schema
validation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .models import CoachReply, ReplyPart

DISCLAIMER = (
    "I am not a doctor, a dietitian, or a therapist, and nothing I say is medical advice."
)

# The National Center for Mental Health crisis line, which is toll-free from a
# Philippine landline. A Refusal about eating carries it; the others do not.
HELPLINE = (
    "If eating itself has become hard, you can reach the NCMH Crisis Hotline on "
    "1553 toll-free, or 0917 899 8727, any hour of the day."
)


def _any(*terms: str) -> re.Pattern[str]:
    return re.compile("|".join(terms), re.IGNORECASE)


# A figure with a unit is not a clinical claim by itself. Public dietary
# guidance is full of milligram figures — the sodium limit is one of its
# headline facts — so a bare `\d+ mg` refuses the Corpus's own vocabulary while
# catching no actual prescription. The signal is the prescriptive framing around
# the figure, not the figure: an amount counts only with a dosage marker beside
# it, in the same sentence, in either order.
#
# The unit list is unchanged on purpose. This narrows a rule; it does not widen
# one, and adding `g` or `iu` would be a separate decision about the boundary.
AMOUNT = r"\d[\d,]*\s*(?:mg|mcg|ml)\b"

DOSAGE_MARKER = (
    r"tak(?:e|es|ing|en)|took|dos(?:e|es|age|ages)|supplements?|"
    r"tablets?|capsules?|pills?|sachets?|injections?|"
    # A counted frequency is a prescription's rhythm. Bare "per day" is not:
    # "less than 2,300 mg of sodium per day" is how guidance states a limit.
    r"(?:once|twice|thrice|three times|four times|\d+ times)"
    r"\s+(?:a\s+|per\s+)?(?:day|daily|week|weekly)"
)

# Either order, within one sentence: "take 500 mg", "500 mg twice daily",
# "one 500 mg tablet". The window stops at sentence punctuation, so a marker in
# one sentence cannot arm a figure in the next.
DOSAGE = _any(
    rf"\b(?:{DOSAGE_MARKER})\b[^.?!]{{0,40}}?{AMOUNT}",
    rf"{AMOUNT}[^.?!]{{0,40}}?\b(?:{DOSAGE_MARKER})\b",
)

# Diagnosis, treatment, and dosage.
CLINICAL = _any(
    r"\bdiagnos(?:e|is|ed|ing)\b",
    r"\bdo i have\b",
    r"\bprescri(?:be|bed|ption)\b",
    r"\bdosages?\b",
    r"\bdose\b",
    DOSAGE.pattern,
    r"\bcure[sd]?\b",
    r"\btreatments?\b",
    r"\btreat (?:my|his|her|this|the)\b",
    r"\bmedications?\b",
    r"\bmedicines?\b",
    r"\binsulin\b",
    r"\bmetformin\b",
    r"\bantibiotics?\b",
    r"\bis it safe to take\b",
    r"\bstop taking\b",
)

# Eating-disorder content. "binge" alone is not on the list: a User who says
# they binged on pandesal last night is logging a Meal, and the Coach logs it.
EATING_DISORDER = _any(
    r"\banorexi(?:a|c)\b",
    r"\bbulimi(?:a|c)\b",
    r"\bpurg(?:e|es|ing)\b",
    r"\blaxatives?\b",
    r"\bbinge[- ]?eat(?:ing)?\b",
    r"\bbinge and purge\b",
    r"\bmake myself (?:throw up|vomit|sick)\b",
    r"\bthrow up (?:after|my)\b",
    r"\bstarv(?:e|ing|ation)\b",
    r"\bthinspo\b",
    r"\bstop eating\b",
    r"\beat as little as\b",
    r"\bfast(?:ing)? for (?:days|a week)\b",
)

# Pregnancy, breastfeeding, and children.
PREGNANCY_AND_CHILDREN = _any(
    r"\bpregnan(?:t|cy)\b",
    r"\bbreast[- ]?feed(?:ing)?\b",
    r"\bnursing (?:mother|mom|mum)\b",
    r"\blactating\b",
    r"\bmy baby\b",
    r"\binfants?\b",
    r"\btoddlers?\b",
    r"\bmy (?:son|daughter|kid|kids|child|children)\b",
    r"\b\d+[- ]year[- ]old\b",
    r"\bweaning\b",
)

DISEASES = (
    "diabetes",
    "diabetic",
    "blood sugar",
    "blood pressure",
    "hypertension",
    "hypertensive",
    "high cholesterol",
    "kidney disease",
    "chronic kidney",
    "heart disease",
    "fatty liver",
    "gout",
    "uric acid",
    "cancer",
    "thyroid",
    "pcos",
    "celiac",
    "crohn",
    "ulcer",
)

# The personal half of a chronic disease, and only the personal half. A marker
# of a person — the User, or somebody the User is asking on behalf of — has to
# stand in the same sentence as the condition. "What is type 2 diabetes" carries
# no such marker and is answered from the Corpus; "my friend has type 2 diabetes,
# what should he eat" carries one, and is refused on the same terms as "I have".
# The bias is deliberate: a general question wrongly refused is a worse answer,
# a personal plan wrongly given is a worse failure.
CHRONIC_DISEASE = re.compile(
    r"\b(?:my|our|his|her|their|i|we|me|mom|mother|dad|father|friend)\b"
    r"[^.?!]{0,60}?\b(?:" + "|".join(DISEASES) + r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Subject:
    """One subject outside the boundary, and the Refusal it produces."""

    name: str
    boundary: str
    professional: str
    offer: str
    helpline: str | None = None


EATING = Subject(
    name="eating_disorder",
    boundary="Eating-disorder territory is outside what I do as a nutrition Coach.",
    professional="a mental-health professional, or your doctor",
    offer="talk with you about ordinary meals and everyday nutrition, whenever you want",
    helpline=HELPLINE,
)
CLINICAL_SUBJECT = Subject(
    name="clinical",
    boundary="Diagnosis, treatment, and dosage are outside what I do as a nutrition Coach.",
    professional="a licensed doctor, or a registered dietitian",
    offer="log what you eat, answer a general nutrition question from the Corpus, "
    "review your day, and suggest what to eat next",
)
PREGNANCY_SUBJECT = Subject(
    name="pregnancy_and_children",
    boundary="Nutrition for pregnancy, breastfeeding, and children is outside what I do "
    "as a nutrition Coach.",
    professional="an obstetrician, a paediatrician, or a registered dietitian",
    offer="help with your own everyday meals and your own Goal",
)
CHRONIC_SUBJECT = Subject(
    name="chronic_disease",
    boundary="Managing a chronic condition through a personal diet is outside what I do "
    "as a nutrition Coach.",
    professional="a licensed doctor, or a registered dietitian",
    offer="answer a general question about that condition from the Corpus, and log and "
    "review what you eat",
)
# The router flagged meaning that no rule predicted. The wording stays general,
# because choosing which of the four subjects it was would be the model deciding
# how the Coach speaks about safety, and that decision belongs to the code.
OUT_OF_SCOPE = Subject(
    name="out_of_scope",
    boundary="That is outside what I do as a nutrition Coach.",
    professional="a licensed professional for anything clinical",
    offer="log what you eat, answer a general nutrition question from the Corpus, "
    "review your day, and suggest what to eat next",
)

# Order matters. A message that is both about eating and about a condition is
# answered with the Refusal that carries the help-line.
RULES: tuple[tuple[re.Pattern[str], Subject], ...] = (
    (EATING_DISORDER, EATING),
    (CLINICAL, CLINICAL_SUBJECT),
    (PREGNANCY_AND_CHILDREN, PREGNANCY_SUBJECT),
    (CHRONIC_DISEASE, CHRONIC_SUBJECT),
)


def match_rule(message: str) -> Subject | None:
    """The deterministic detector. No model, and it runs before the router."""
    for pattern, subject in RULES:
        if pattern.search(message):
            return subject
    return None


def refusal(subject: Subject) -> CoachReply:
    """The template. Four parts, in one order, and a fifth for eating."""
    sentences = [
        subject.boundary,
        DISCLAIMER,
        f"Please bring this to {subject.professional}.",
        f"What I can do instead is {subject.offer}.",
    ]
    if subject.helpline:
        sentences.append(subject.helpline)
    text = " ".join(sentences)
    return CoachReply(
        text=text,
        parts=[ReplyPart(intent="refusal", text=text)],
        disclaimers=[DISCLAIMER],
    )


# A claim only a clinician may make. The scan runs on the finished text of an
# answer, before the answer event is sent, so no unapproved sentence reaches the
# screen. A Refusal is not scanned: it is this file's own words, and it names
# the boundary in exactly the vocabulary the scan looks for.
#
# `DOSAGE` is the same object the rule list uses. One pattern, two detectors, so
# the way in and the way out cannot drift apart: whatever the Coach may not be
# asked for is also what it may not say.
MEDICAL_CLAIM = _any(
    r"\bcure[sd]?\b",
    r"\btreats?\b",
    r"\bheals?\b",
    r"\bdiagnos(?:e|is|ed|es)\b",
    r"\bprescri(?:be|bed|ption)\b",
    DOSAGE.pattern,
    r"\byou (?:have|are suffering from) (?:" + "|".join(DISEASES) + r")\b",
    r"\bmedically proven\b",
    r"\breplaces? your medication\b",
)

SAFE_MESSAGE = (
    "I started an answer I cannot stand behind, so I am not going to give it. "
    + DISCLAIMER
    + " Please bring anything clinical to a licensed doctor, or a registered dietitian. "
    "Ask me what you ate, or what to eat next, and I will help with that."
)


def scan_reply(text: str) -> str | None:
    """What the finished text says that it may not. `None` means it may go out.

    This is the medical-claim half. The allergen half is `allergens_in_prose`
    below, and it is a separate function rather than a second return value from
    this one because the two failures are answered differently: a medical claim
    ends the Turn, while an allergen is struck out and the answer is written
    again. The stream contract above both is unchanged.
    """
    found = MEDICAL_CLAIM.search(text)
    return found.group(0) if found else None


def safe_reply() -> CoachReply:
    """What a Turn whose scan failed ends with. Never a partial answer."""
    return CoachReply(
        text=SAFE_MESSAGE,
        parts=[ReplyPart(intent="blocked", text=SAFE_MESSAGE)],
        disclaimers=[DISCLAIMER],
    )


# --- the allergy check ---------------------------------------------------------
#
# It runs on `recommend` and `log_meal`, and on no other path. The list is
# `graph.ALLERGY_CHECKED_INTENTS`, and `update_profile`, `ask_question` and
# `review_day` are absent from it on purpose: a correct answer on those paths
# may name the allergen as a fact — a confirmation of "I am allergic to shrimp"
# says shrimp, and so does a cited answer about peanut allergy — and the check
# would destroy the very answer the User asked for. The prototype found this.
#
# Two comparisons, and neither of them is a model.
#
#  1. The structured one, `allergens_named`, which the Intent path runs on the
#     food names its own data holds, before the answer is written.
#  2. The prose one, `allergens_in_prose`, which `turn.run_turn` runs on the
#     finished draft for a food the model named only in a sentence.
#
# This is a second line of defence. The recommender removes allergens in SQL
# before the model ever sees a candidate, so a hit here means something upstream
# failed — which is exactly why it exists.


def allergens_named(allergies: Sequence[str], text: str) -> list[str]:
    """The Profile allergies this text names, in the Profile's own order.

    An exact word comparison against `user_profile.allergies`, which is a text
    array on the Profile row the Turn already loaded — so there is no join, no
    second query, no fuzzy match, and no model.

    ponytail: a trailing plural `s` is the only inflection handled. A synonym
    list per allergen wants a source to take one from, and there is not one.
    """
    return [
        allergen
        for allergen in allergies
        if (word := allergen.strip())
        and re.search(rf"\b{re.escape(word)}s?\b", text, re.IGNORECASE)
    ]


# The Coach's own fixed sentences. None of them names a food, all of them are
# this codebase's words rather than a model's, and so none of them is a food
# sentence — the same reason a Refusal is not scanned at all. `test_allergy.py`
# drives a Turn whose answer carries every one that `meal.compose` can write and
# fails if a reworded sentence stops being covered here.
NOT_ABOUT_FOOD = (
    "I am not a doctor",  # DISCLAIMER
    "If eating itself has become hard",  # HELPLINE
    "Tell me if I got any of that wrong",  # meal.TELL_ME
    "Where you did not give a weight",  # the assumed portion, in meal.compose
    "My source prints no",  # the short totals, in meal.compose
)

SENTENCE = re.compile(r"(?<=[.?!])\s+")


def food_sentences(text: str) -> list[str]:
    """The sentences of a draft that are about a food. The prose scan reads
    these and nothing else."""
    return [s for s in SENTENCE.split(text) if s and not s.startswith(NOT_ABOUT_FOOD)]


def allergens_in_prose(
    allergies: Sequence[str], text: str, foods: Sequence[str]
) -> list[str]:
    """The Profile allergies the draft names in prose, and the structured data
    does not — the "try a peanut sauce with that" the first comparison cannot
    see, because that food is in no Item.

    An allergen the Items already hold is excluded, because the first comparison
    dealt with it. On `log_meal` the answer names such an allergen on purpose,
    in the warning, and reading the Coach's own warning as a violation would
    take the warning down with the answer.
    """
    known = allergens_named(allergies, " ".join(foods))
    named = allergens_named(allergies, " ".join(food_sentences(text)))
    return [allergen for allergen in named if allergen not in known]

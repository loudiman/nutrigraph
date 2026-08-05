"""Redaction, immediately before a provider call and nowhere else (ADR 0002).

Gemini runs on the free tier, so Google uses submitted prompts to improve its
products and a person may read them. Identifiers are therefore replaced with
placeholders on the way out and put back on the way in. The database keeps the
raw text; nothing in here touches a database write.

Redacted: person names, email addresses, phone numbers, street addresses, exact
dates of birth, and government identity numbers. Not redacted: weight, height,
allergies, diet pattern, goals, and the food itself — the Coach cannot work
without these, and no redaction could hide them anyway.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)

# A span the scanner found: where it is, and what kind of identifier it is.
Span = tuple[int, int, str]


def _has_enough_digits(match: re.Match[str]) -> bool:
    """A phone number, not a weight, a height, or a quantity of eggs."""
    return sum(c.isdigit() for c in match.group()) >= 9


# Order matters. A date of birth would otherwise be swallowed by the phone
# pattern, and an email address by the name pattern.
PATTERNS: tuple[tuple[str, re.Pattern[str], Callable[[re.Match[str]], bool] | None], ...] = (
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]*\w"), None),
    (
        "DOB",
        re.compile(
            rf"\b(?:\d{{4}}-\d{{2}}-\d{{2}}"
            rf"|\d{{1,2}}[/.]\d{{1,2}}[/.]\d{{4}}"
            rf"|(?:{MONTHS})\s+\d{{1,2}},?\s+\d{{4}})\b"
        ),
        None,
    ),
    (
        "GOV_ID",
        re.compile(
            r"\b(?:SSS|TIN|SSN|PhilHealth|PhilSys|passport)\b\s*"
            r"(?:no\.?|number|#|id)?\s*:?\s*[A-Za-z0-9][A-Za-z0-9-]{5,}"
            r"|\b\d{3}-\d{2}-\d{4}\b"
            r"|\b\d{2}-\d{7}-\d\b"
            r"|\b\d{3}-\d{3}-\d{3}-\d{3}\b",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        "ADDRESS",
        re.compile(
            r"\b\d{1,5}\s+(?:[A-Z][\w.'-]*\s+){0,4}"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Highway|Hwy)"
            r"\b\.?"
        ),
        None,
    ),
    ("PHONE", re.compile(r"\+?\d[\d\s().-]{7,}\d"), _has_enough_digits),
)


def known_name_finder(names: Iterable[str]) -> Callable[[str], list[tuple[int, int]]]:
    """Every name the Coach already holds, and each word of it: a Profile says
    "Lou", and the User writes "Lou" or "Lou Morados"."""
    words = {
        word
        for name in names
        for word in [name, *name.split()]
        if len(word) > 1
    }
    if not words:
        return lambda text: []
    pattern = re.compile(
        r"\b(?:" + "|".join(sorted(map(re.escape, words), key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    return lambda text: [m.span() for m in pattern.finditer(text)]


def entity_name_finder() -> Callable[[str], list[tuple[int, int]]]:
    """Names and addresses that no Profile knows: the names of other people.
    Exact patterns are regular expressions; open-ended entities need a model.

    ponytail: without the `ner` extra installed this finds nothing, so only
    names the Coach already holds are redacted. `pip install -e ".[ner]" &&
    python -m spacy download en_core_web_sm` turns it on; no call site changes.
    """
    try:  # pragma: no cover - exercised only where the extra is installed
        import spacy

        nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])
    except Exception:
        return lambda text: []

    def find(text: str) -> list[tuple[int, int]]:  # pragma: no cover
        return [
            (ent.start_char, ent.end_char)
            for ent in nlp(text).ents
            if ent.label_ in {"PERSON", "FAC", "LOC", "GPE"}
        ]

    return find


@dataclass(frozen=True)
class Redacted:
    """What a provider is allowed to see, and how to read its answer back."""

    texts: tuple[str, ...]
    mapping: dict[str, str]

    @property
    def text(self) -> str:
        return self.texts[0]

    def restore(self, text: str) -> str:
        """Put the identifiers back, so the answer can address the User by name."""
        for placeholder, original in self.mapping.items():
            text = text.replace(placeholder, original)
        return text


@dataclass
class Redactor:
    """Built once per Turn, from the names the Coach already holds."""

    known_names: Sequence[str] = ()
    entities: Callable[[str], list[tuple[int, int]]] = field(default_factory=entity_name_finder)

    def __post_init__(self) -> None:
        self._known = known_name_finder(self.known_names)

    def _spans(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for kind, pattern, valid in PATTERNS:
            spans += [
                (m.start(), m.end(), kind)
                for m in pattern.finditer(text)
                if valid is None or valid(m)
            ]
        spans += [(s, e, "NAME") for s, e in self._known(text)]
        spans += [(s, e, "NAME") for s, e in self.entities(text)]
        # Earliest wins, longest wins on a tie, and nothing overlaps after this.
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        kept: list[Span] = []
        for span in spans:
            if not kept or span[0] >= kept[-1][1]:
                kept.append(span)
        return kept

    def redact(self, *texts: str) -> Redacted:
        """Replace every identifier in every text with a placeholder. One
        original always gets the same placeholder, across all the texts."""
        mapping: dict[str, str] = {}
        assigned: dict[tuple[str, str], str] = {}
        counts: dict[str, int] = {}
        out: list[str] = []
        for text in texts:
            pieces: list[str] = []
            cursor = 0
            for start, end, kind in self._spans(text):
                original = text[start:end]
                key = (kind, original.casefold())
                placeholder = assigned.get(key)
                if placeholder is None:
                    counts[kind] = counts.get(kind, 0) + 1
                    placeholder = f"[{kind}_{counts[kind]}]"
                    assigned[key] = placeholder
                    mapping[placeholder] = original
                pieces.append(text[cursor:start])
                pieces.append(placeholder)
                cursor = end
            pieces.append(text[cursor:])
            out.append("".join(pieces))
        return Redacted(texts=tuple(out), mapping=mapping)

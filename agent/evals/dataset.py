"""The cases, as data. Standard library only.

Three files, one for each group, and a small fixed Corpus the retrieval group is
answered from. Every case is hand-written; nothing here is generated, and
`validate` is what says so out loud — a case carrying a `skip`, `quarantine`,
`expected_failure` or `generated_by` key is a load error, not a warning, because
a quarantined safety case is a silent hole.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CASES_DIR = HERE / "cases"
CORPUS_FILE = HERE / "corpus.json"
BASELINE_FILE = HERE / "baseline.json"

GROUPS = ("safety", "retrieval", "behaviour")

# There is no quarantine list and no mechanism to make one. These are the words
# such a mechanism would arrive under, and the loader refuses all of them.
FORBIDDEN_KEYS = frozenset(
    {"skip", "skipped", "quarantine", "quarantined", "xfail", "expected_failure",
     "generated_by", "generated", "flaky", "disabled"}
)

# What the four ragas metrics are called, everywhere: in the baseline file, in
# the judge's output, and in the report.
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

# How far a mean may fall below the stored baseline before the build stops. A
# fixed pass mark trains people to re-run until it passes; a margin below a
# baseline that ratchets upwards does not.
MARGIN = 0.05


@dataclass(frozen=True)
class Case:
    """One case, as written. `spec` is the JSON object itself: the assertions
    are read by name in `gate`, so adding one is a key here and a check there."""

    group: str
    spec: dict[str, Any]

    @property
    def id(self) -> str:
        return self.spec["id"]

    @property
    def draft(self) -> str | None:
        """A finished Coach draft, for a case about the way out rather than the
        way in. The release gate is deterministic, so these cost nothing and can
        assert things a live Turn cannot be made to produce — an allergen the
        recommender's SQL filter has already removed, for instance."""
        return self.spec.get("draft")

    @property
    def message(self) -> str:
        return self.spec["message"]

    @property
    def given(self) -> list[str]:
        """Messages run before the one under test, on the same Thread. They are
        setup: nothing about them is asserted."""
        return list(self.spec.get("given", []))

    @property
    def profile(self) -> str:
        """Which seeded Profile shape this case's own User is made from."""
        return self.spec.get("profile", "default")

    @property
    def user_id(self) -> str:
        """One User for each case. Cases share a database, and a day total is
        per User, so a shared Profile would make one case's Meal another case's
        arithmetic."""
        return f"eval-{self.id}"


def _read(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases(groups: tuple[str, ...] = GROUPS) -> list[Case]:
    cases = [
        Case(group=group, spec=spec)
        for group in groups
        for spec in _read(CASES_DIR / f"{group}.json")
    ]
    validate(cases)
    return cases


def validate(cases: list[Case]) -> None:
    seen: set[str] = set()
    for case in cases:
        spec = case.spec
        if "id" not in spec:
            raise ValueError(f"a {case.group} case has no id")
        if case.id in seen:
            raise ValueError(f"{case.id} is used twice")
        seen.add(case.id)
        forbidden = FORBIDDEN_KEYS & set(spec)
        if forbidden:
            raise ValueError(
                f"{case.id} carries {sorted(forbidden)}. There is no quarantine or "
                f"skip list: fix the case or fix the behaviour, and say which in "
                f"the pull request"
            )
        if ("message" in spec) == ("draft" in spec):
            raise ValueError(f"{case.id} needs exactly one of 'message' and 'draft'")
        if case.group == "retrieval":
            for needed in ("reference", "reference_contexts", "citations_from"):
                if not spec.get(needed):
                    raise ValueError(f"{case.id} is a retrieval case with no {needed}")


def load_corpus() -> list[dict[str, Any]]:
    """The fixed Corpus the retrieval group is answered from.

    Not the live manifest. Forty documents fetched from forty web servers is the
    slow, flaky step this run may not take on every pull request, and a Citation
    can only be asserted against a passage that is the same one every time.
    """
    return _read(CORPUS_FILE)


def load_baseline() -> dict[str, float | None]:
    stored = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    return {metric: stored.get(metric) for metric in METRICS}


def store_baseline(means: dict[str, float]) -> None:
    BASELINE_FILE.write_text(
        json.dumps({m: means[m] for m in METRICS}, indent=2) + "\n", encoding="utf-8"
    )

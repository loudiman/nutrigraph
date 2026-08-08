"""Everything a Turn reaches outside its own process, in one object.

This is the agent turn seam: a test builds a `Deps` with fakes, and everything
below it is real.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from .db import Database
from .food import FoodSearch
from .meal import MANILA
from .providers import Models


def manila_now() -> datetime:
    return datetime.now(MANILA)


class NotWired:
    """A dependency no node reaches in this slice. Touching it fails loudly,
    so a test can prove a Turn made no provider call."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, item: str) -> object:
        raise RuntimeError(f"{self._name} is not wired yet (asked for {item!r})")


@dataclass
class Deps:
    db: Database
    # The provider seam. A node never reaches a chat model directly: it asks
    # this for a Turn-bound `TurnModels`, which redacts before every call.
    models: Models
    # The food data seam. It stays `NotWired` where no key is configured, so a
    # Turn that reaches for it fails loudly rather than quietly counting nothing
    # — and a test can prove that a local dish was matched without it.
    food: FoodSearch | NotWired = field(
        default_factory=lambda: NotWired("FoodData Central")
    )
    # The clock, in Manila. Every node that needs to know which day it is reads
    # it from here rather than calling `datetime.now` itself, so a test can pin
    # "today" the way it pins the database: a Meal stored on a fixed day and a
    # review of "today" would otherwise disagree the moment the wall clock moved
    # past midnight, and the test would go red having changed nothing.
    now: Callable[[], datetime] = manila_now

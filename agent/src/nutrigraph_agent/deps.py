"""Everything a Turn reaches outside its own process, in one object.

This is the agent turn seam: a test builds a `Deps` with fakes, and everything
below it is real.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .db import Database


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
    models: object = field(default_factory=lambda: NotWired("Gemini"))
    food: object = field(default_factory=lambda: NotWired("FoodData Central"))

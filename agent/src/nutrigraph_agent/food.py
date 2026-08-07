"""FoodData Central: the search endpoint, and the seam a test fakes.

The matcher reaches this only for a food the local Filipino dish table does not
hold, which is what keeps the request count down: the local lookup is one
indexed query, and this is a web request.

FoodData Central is public domain with no redistribution duty, and the free
data.gov key allows 1,000 requests an hour — not the demo key, which allows far
fewer. The key is read from the environment and never written to a log.

This is not a model provider, so it is not a hole in ADR 0002's rule: no prompt
goes out of here, only a food word, which the redaction wrapper sends to a
provider unchanged anyway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger("nutrigraph.agent.food")

SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# The top ten candidates, which the choice call then reads (issue #15).
CANDIDATES = 10

# FoodData Central's own nutrient numbers, which are stable across data types
# and across releases in a way the identifiers and the names are not. Every
# value in `foodNutrients` is per 100 g, the same basis the local table uses.
NUTRIENT_NUMBERS = {
    "208": "kcal",
    "203": "protein_g",
    "204": "fat_g",
    "205": "carb_g",
    "291": "fibre_g",
    "307": "sodium_mg",
}

# The six plain numbers a day total sums.
COLUMNS = ("kcal", "protein_g", "fat_g", "carb_g", "fibre_g", "sodium_mg")


@dataclass(frozen=True)
class FoodCandidate:
    """One food FoodData Central offered, per 100 g.

    A nutrient the response does not carry is absent from `per_100g`, and stays
    absent all the way to the column, which is null. Nothing here writes a zero
    for a value the source did not state.
    """

    fdc_id: str
    description: str
    per_100g: dict[str, float] = field(default_factory=dict)
    # The whole food object, for `meal_item.nutrients`, so nothing the source
    # said is lost to the six columns.
    source: dict[str, Any] = field(default_factory=dict)


class FoodSearch(Protocol):
    """What the matcher is allowed to know about FoodData Central."""

    async def search(self, query: str, *, limit: int = CANDIDATES) -> list[FoodCandidate]: ...


def candidate(food: dict[str, Any]) -> FoodCandidate:
    """One `foods[]` entry as the matcher reads it."""
    values: dict[str, float] = {}
    for nutrient in food.get("foodNutrients") or []:
        column = NUTRIENT_NUMBERS.get(str(nutrient.get("nutrientNumber")))
        value = nutrient.get("value")
        # `is None` and not falsiness: a food with zero fibre states zero, and
        # that is a measurement, not a missing value.
        if column is not None and value is not None:
            values[column] = float(value)
    return FoodCandidate(
        fdc_id=str(food.get("fdcId")),
        description=str(food.get("description", "")).strip(),
        per_100g=values,
        source=food,
    )


class FoodDataCentral:
    """The search endpoint, over httpx, which the Corpus ingest already pulls in."""

    def __init__(self, api_key: str, *, timeout: float = 10.0) -> None:
        self._api_key = api_key
        self._timeout = timeout

    async def search(self, query: str, *, limit: int = CANDIDATES) -> list[FoodCandidate]:
        import httpx

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                SEARCH_URL,
                params={"api_key": self._api_key, "query": query, "pageSize": limit},
            )
            response.raise_for_status()
            body = response.json()
        foods = (body.get("foods") or [])[:limit]
        log.info("FoodData Central returned %d candidates for %r", len(foods), query)
        return [candidate(food) for food in foods]

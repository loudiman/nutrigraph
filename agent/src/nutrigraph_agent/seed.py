"""Insert the demo Profiles and the local Filipino dish table. Safe to run twice.

The dish table is `seeds/filipino_dishes.json`, transcribed from DOST-FNRI
PhilFCT by issue #26 and checked by `tests/test_filipino_dishes.py`. Nothing here
adjusts a value: the loader carries the six columns across as they stand, nulls
included, and keeps the whole row in `local_food.source` so nothing transcribed
is lost to those six.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from .config import Settings
from .food import COLUMNS as NUTRIENT_COLUMNS
from .meal import normalize
from .retention import DEMO_WARNING

SEEDS_DIR = Path(__file__).resolve().parents[2] / "seeds"
PROFILES_FILE = SEEDS_DIR / "profiles.json"
DISHES_FILE = SEEDS_DIR / "filipino_dishes.json"

COLUMNS = (
    "user_id", "name", "sex", "age", "height_cm", "weight_kg", "target_weight_kg",
    "activity_level", "diet_pattern", "units", "allergies", "disliked_foods",
)

UPSERT = f"""
insert into user_profile ({", ".join(COLUMNS)})
values ({", ".join("%s" for _ in COLUMNS)})
on conflict (user_id) do update set
    {", ".join(f"{c} = excluded.{c}" for c in COLUMNS if c != "user_id")},
    updated_at = now()
"""


def seed_profiles(database_url: str, profiles_file: Path = PROFILES_FILE) -> list[str]:
    profiles = json.loads(profiles_file.read_text(encoding="utf-8"))
    with psycopg.connect(database_url) as conn:
        for profile in profiles:
            conn.execute(UPSERT, tuple(profile[c] for c in COLUMNS))
    return [p["user_id"] for p in profiles]


DISH_COLUMNS = (
    "name", "aliases", "tags", "value_kind", "source_note", "philfct_food_id",
    *NUTRIENT_COLUMNS,
)

UPSERT_LOCAL_FOOD = f"""
insert into local_food ({", ".join(DISH_COLUMNS)}, source)
values ({", ".join("%s" for _ in DISH_COLUMNS)}, %s)
on conflict (lower(name)) do update set
    {", ".join(f"{c} = excluded.{c}" for c in DISH_COLUMNS if c != "name")},
    source = excluded.source,
    seeded_at = now()
returning local_food_id
"""


def aliases_for(dish: dict) -> list[str]:
    """Every name the dish answers to, its own among them, in the one spelling
    the matcher looks a name up under."""
    written = [dish["name"], *dish["aliases"]]
    return list(dict.fromkeys(normalize(name) for name in written))


def _one_dish_per_alias(dishes: list[dict]) -> None:
    """'lechon manok' on a pork row would hand roast-chicken figures to someone
    eating roast pig. The seed file already forbids that for the names as
    written; this forbids it for the names as normalized, which is what the
    alias table is keyed on."""
    seen: dict[str, str] = {}
    for dish in dishes:
        for alias in aliases_for(dish):
            if alias in seen and seen[alias] != dish["name"]:
                raise ValueError(
                    f"{alias!r} names both {seen[alias]!r} and {dish['name']!r}"
                )
            seen[alias] = dish["name"]


def seed_local_foods(database_url: str, dishes_file: Path = DISHES_FILE) -> list[str]:
    """The hand-entered dish table, loaded as it stands. Safe to run twice: a
    dish is keyed by its name and its aliases are replaced wholesale."""
    dishes = json.loads(dishes_file.read_text(encoding="utf-8"))["dishes"]
    _one_dish_per_alias(dishes)
    with psycopg.connect(database_url) as conn:
        for dish in dishes:
            local_food_id = conn.execute(
                UPSERT_LOCAL_FOOD,
                (*(dish.get(c) for c in DISH_COLUMNS), Jsonb(dish)),
            ).fetchone()[0]
            conn.execute(
                "delete from local_food_alias where local_food_id = %s", (local_food_id,)
            )
            conn.cursor().executemany(
                "insert into local_food_alias (alias, local_food_id) values (%s, %s)",
                [(alias, local_food_id) for alias in aliases_for(dish)],
            )
    return [dish["name"] for dish in dishes]


def main() -> int:
    settings = Settings.from_env()
    seeded = seed_profiles(settings.database_url)
    print(f"seeded {len(seeded)} demo Profiles: {', '.join(seeded)}")
    dishes = seed_local_foods(settings.database_url)
    print(f"seeded {len(dishes)} local Filipino dishes")
    # The warning, where a developer puts data in (ADR 0002).
    print(DEMO_WARNING)
    return 0


if __name__ == "__main__":
    sys.exit(main())

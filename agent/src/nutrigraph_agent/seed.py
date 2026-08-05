"""Insert the demo Profiles. Safe to run twice."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import psycopg

from .config import Settings

SEEDS_DIR = Path(__file__).resolve().parents[2] / "seeds"
PROFILES_FILE = SEEDS_DIR / "profiles.json"

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


def main() -> int:
    settings = Settings.from_env()
    seeded = seed_profiles(settings.database_url)
    print(f"seeded {len(seeded)} demo Profiles: {', '.join(seeded)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

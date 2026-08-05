"""The 90-day retention rule for raw user text, and the warning it rests on.

ADR 0002 stores the User's message unredacted on purpose: the eval dataset stays
truthful and a wrong redaction can be reviewed. The cost is a store of free text,
and this module pays it twice over.

**The purge.** One statement nulls `message.raw_text` after 90 days and stamps
`purged_at`. It writes those two columns of that one table and nothing else, so
the row, its identifiers, its timestamps, and every structured Meal, Item,
Recommendation and `interaction_event` row survive: the day history and the
harvested eval cases stay intact while free text does not accumulate for ever.
It is safe to run twice — a purged row has no `raw_text` left to match.

**The warning.** ADR 0002 says plainly that the warning, and not the schema, is
what protects the raw text store during those 90 days. So it is a constant, not
a paragraph: the User reads it above the box before typing, and the developer
reads it on the way out of `nutrigraph-migrate` and `nutrigraph-seed`.
"""

from __future__ import annotations

import sys

import psycopg

from .config import Settings

RETENTION_DAYS = 90

DEMO_WARNING = (
    "Demo data only. What you type is stored as written and kept for 90 days, "
    "so do not enter real personal or health details, yours or anyone else's."
)

PURGE = """
update message
   set raw_text = null,
       purged_at = now()
 where raw_text is not null
   and created_at < now() - make_interval(days => %s)
"""


def purge_raw_text(database_url: str, *, days: int = RETENTION_DAYS) -> int:
    """Null the raw text of every message older than `days`. Returns the number
    of rows purged, which is zero on every run after the first."""
    with psycopg.connect(database_url) as conn:
        return conn.execute(PURGE, (days,)).rowcount


def main() -> int:
    settings = Settings.from_env()
    purged = purge_raw_text(settings.database_url)
    print(f"purged the raw text of {purged} messages older than {RETENTION_DAYS} days")
    print(DEMO_WARNING)
    return 0


if __name__ == "__main__":
    sys.exit(main())

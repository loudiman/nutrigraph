"""Numbered SQL files applied in order, each recorded in a version table.

No migration library: the file reads exactly as PostgreSQL will hold it, which
is what later slices need for a `vector(768)` column and an HNSW index. There is
no downgrade — a mistake is corrected by writing the next file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg

from .config import Settings

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
VERSION_TABLE = "schema_migration"


def applied_files(conn: psycopg.Connection) -> set[str]:
    # 001 creates the version table, so on a virgin database it does not exist.
    exists = conn.execute("select to_regclass(%s)", (f"public.{VERSION_TABLE}",)).fetchone()
    if not exists or exists[0] is None:
        return set()
    rows = conn.execute(f"select filename from {VERSION_TABLE}").fetchall()
    return {row[0] for row in rows}


def pending(conn: psycopg.Connection, directory: Path = MIGRATIONS_DIR) -> list[Path]:
    done = applied_files(conn)
    return [p for p in sorted(directory.glob("*.sql")) if p.name not in done]


def migrate(database_url: str, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every unapplied file, in order. Returns the names applied."""
    applied: list[str] = []
    with psycopg.connect(database_url, autocommit=False) as conn:
        for path in pending(conn, directory):
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute(
                f"insert into {VERSION_TABLE} (filename) values (%s)", (path.name,)
            )
            conn.commit()
            applied.append(path.name)
    return applied


def main() -> int:
    settings = Settings.from_env()
    applied = migrate(settings.database_url)
    print("\n".join(f"applied {name}" for name in applied) or "nothing to apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())

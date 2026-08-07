"""The eval gate, first half: run every case and apply the structural gate.

    python -m evals.run --rows rows.json

**It runs the real graph.** A case goes in at the agent turn seam — a `user_id`
and a message — and everything below it is the code that serves a User: the same
router, the same rule list, the same Corpus search, the same composer, the same
release gate. Nothing is faked, because a golden case answered by a fake proves
only that the fake is consistent.

**One User for each case.** A day total is per User and cases share a database,
so a shared Profile would make one case's Meal another case's arithmetic. Each
case gets a Profile of its own, seeded from one of two shapes, and a checkpointer
of its own so a `given` message reaches the Turn under test and nothing else does.

**FoodData Central is deliberately not wired.** A golden case may not depend on
a third party's search endpoint being up, so the behaviour cases name dishes the
local Filipino table holds — which is also the path this project is actually
about — and one food that nothing anywhere holds. `match_food` already treats an
unreachable food source as an uncounted Item, so nothing special happens here.

**The Corpus is the fixture in `corpus.json`, not the live manifest.** Forty
documents from forty web servers is the slow, flaky step a pull request may not
take, and a Citation can only be asserted against a passage that is the same one
every time. The passages are embedded once per database; a second run over the
same database re-embeds nothing.

Only the retrieval group leaves rows behind for the judge, which is why a whole
run costs a few cents: the safety and behaviour groups are plain assertions, and
a rule-list Refusal does not reach a provider at all.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from .dataset import GROUPS, Case, load_cases, load_corpus
from .gate import Failure, check

log = logging.getLogger("nutrigraph.evals")

# The two Profile shapes a case is made from. `default` is the demo Profile the
# rest of the repository uses; the other two exist for the allergen cases.
PROFILES: dict[str, dict[str, Any]] = {
    "default": {
        "name": "Lou", "sex": "M", "age": 24, "height_cm": 172, "weight_kg": 78,
        "target_weight_kg": 72, "activity_level": "light", "diet_pattern": "omnivore",
        "units": "metric", "allergies": [], "disliked_foods": ["bitter gourd"],
    },
    "peanut-allergic": {
        "name": "Lou", "sex": "M", "age": 24, "height_cm": 172, "weight_kg": 78,
        "target_weight_kg": 72, "activity_level": "light", "diet_pattern": "omnivore",
        "units": "metric", "allergies": ["peanut"], "disliked_foods": [],
    },
    "shrimp-allergic": {
        "name": "Ana", "sex": "F", "age": 31, "height_cm": 160, "weight_kg": 55,
        "target_weight_kg": 58, "activity_level": "moderate", "diet_pattern": "omnivore",
        "units": "metric", "allergies": ["shrimp"], "disliked_foods": [],
    },
}


# --- the run --------------------------------------------------------------------


def _settings():
    from nutrigraph_agent.config import Settings

    return Settings.from_env()


def _models(settings):
    from nutrigraph_agent.providers import (
        Models,
        langchain_embedding_factory,
        langchain_factory,
    )

    return Models(
        factory=langchain_factory(settings.model_provider),
        schema_model=settings.schema_model,
        prose_model=settings.prose_model,
        embedding_factory=langchain_embedding_factory(settings.model_provider),
        embedding_model=settings.embedding_model,
    )


def prepare(database_url: str, models, cases: list[Case]) -> None:
    """The schema, the dish table, one Profile for each case, and the Corpus.

    Every step is the repository's own, called as a function rather than copied:
    a migration the eval applied differently from production would be an eval
    that proves nothing about production.
    """
    import psycopg
    from nutrigraph_agent.corpus import CorpusEntry
    from nutrigraph_agent.ingest import ingest
    from nutrigraph_agent.migrate import migrate
    from nutrigraph_agent.seed import UPSERT, COLUMNS, seed_local_foods

    applied = migrate(database_url)
    log.info("applied %d migrations", len(applied))
    seed_local_foods(database_url)

    with psycopg.connect(database_url) as conn:
        # Everything the eval's own Users left behind last time. A run has to be
        # repeatable on a database that is not thrown away — a Meal logged by
        # yesterday's run is in today's day total, and a Profile that already
        # holds the allergy a case is about has nothing left to change.
        for table in ("meal", "recommendation", "interaction_event", "message"):
            conn.execute(f"delete from {table} where user_id like 'eval-%%'")
        for case in cases:
            if case.draft is not None:
                continue
            row = {"user_id": case.user_id, **PROFILES[case.profile]}
            conn.execute(UPSERT, tuple(row[column] for column in COLUMNS))
    log.info("seeded one Profile for each case, and cleared what the last run left")

    documents = load_corpus()
    passages = {
        d["slug"]: [(p["locator"], p["text"]) for p in d["passages"]] for d in documents
    }
    entries = [
        CorpusEntry.model_validate({k: v for k, v in d.items() if k != "passages"})
        for d in documents
    ]
    report = asyncio.run(
        ingest(database_url, models, entries, fetcher=lambda e: passages[e.slug])
    )
    if report.failed:
        raise RuntimeError(f"the eval Corpus did not load: {report.failed}")
    log.info("Corpus: %s", report)


class Recording:
    """The database seam, with the three answers a case needs to read.

    A Turn does not hand back what it retrieved, what Items it wrote or whether
    it wrote a Recommendation — the reply is what it hands back, which is the
    point of the seam. This records them on the way past instead of reaching
    into the graph for them, and delegates everything else untouched.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.passages: list[Any] = []
        self.items: list[Any] = []
        self.recommendations = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def search_corpus(self, embedding, *, limit: int = 5):
        found = await self.inner.search_corpus(embedding, limit=limit)
        self.passages += found
        return found

    async def cached_retrieval(self, embedding):
        found = await self.inner.cached_retrieval(embedding)
        if found:
            self.passages += found
        return found

    async def store_meal(self, *, user_id, turn_id, eaten_at, meal_type, items):
        self.items += list(items)
        return await self.inner.store_meal(
            user_id=user_id, turn_id=turn_id, eaten_at=eaten_at,
            meal_type=meal_type, items=items,
        )

    async def correct_meal_item(self, meal_item_id, item):
        self.items.append(item)
        return await self.inner.correct_meal_item(meal_item_id, item)

    async def store_recommendation(self, **kwargs):
        self.recommendations += 1
        return await self.inner.store_recommendation(**kwargs)


def _subjects() -> dict[str, str]:
    from nutrigraph_agent import guardrail as g

    return {
        s.boundary: s.name
        for s in (g.EATING, g.CLINICAL_SUBJECT, g.PREGNANCY_SUBJECT,
                  g.CHRONIC_SUBJECT, g.OUT_OF_SCOPE)
    }


def gate_a_draft(case: Case) -> dict[str, Any]:
    """The release gate on a finished draft, in the order `run_turn` runs it: the
    medical-claim scan first, and the allergen scan only on a draft that passed
    it. No model, no database, and nothing here can flake."""
    from nutrigraph_agent.guardrail import allergens_in_prose, scan_reply

    draft = case.draft or ""
    claim = scan_reply(draft)
    struck = (
        []
        if claim
        else allergens_in_prose(
            case.spec.get("allergies", []), draft, case.spec.get("structured_foods", [])
        )
    )
    return {
        "id": case.id, "group": case.group, "error": None,
        "claim": claim, "struck": struck, "blocked": bool(claim or struck),
        "text": draft,
    }


async def run_case(case: Case, deps_for, graph_for) -> dict[str, Any]:
    """One case at the agent turn seam. Everything below it is real."""
    from nutrigraph_agent.graph import RELEVANCE_FLOOR
    from nutrigraph_agent.guardrail import DISCLAIMER, HELPLINE, SAFE_MESSAGE
    from nutrigraph_agent.graph import COULD_NOT_COMPOSE
    from nutrigraph_agent.models import INTENTS, AnswerEvent, ErrorEvent, NodeEvent
    from nutrigraph_agent.turn import run_turn

    recorder, deps = deps_for()
    graph = graph_for()

    async def drive(message: str) -> list[Any]:
        return [
            event
            async for event in run_turn(
                graph, deps, user_id=case.user_id, turn_id=uuid4(), message=message
            )
        ]

    outcome: dict[str, Any] = {"id": case.id, "group": case.group, "error": None}
    try:
        for earlier in case.given:
            await drive(earlier)
        # Only the Turn under test is read; the `given` messages are setup.
        recorder.passages.clear()
        recorder.items.clear()
        recorder.recommendations = 0
        events = await drive(case.message)
    except Exception as exc:  # pragma: no cover - a case that could not run at all
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        return outcome

    answer = next((e for e in events if isinstance(e, AnswerEvent)), None)
    failure = next((e for e in events if isinstance(e, ErrorEvent)), None)
    outcome["nodes"] = [e.node for e in events if isinstance(e, NodeEvent)]
    if answer is None:
        outcome["error"] = failure.code if failure else "no answer event"
        return outcome

    reply = answer.reply
    parts = [{"intent": p.intent, "text": p.text} for p in reply.parts]
    subject = next(
        (name for boundary, name in _subjects().items() if boundary in reply.text), None
    )
    profile = await deps.db.load_profile(case.user_id)
    outcome.update(
        answered=True,
        text=reply.text,
        parts=parts,
        intents=[p["intent"] for p in parts if p["intent"] in INTENTS],
        refused=bool(parts) and parts[0]["intent"] == "refusal",
        subject=subject,
        clarification=bool(parts) and parts[0]["intent"] == "clarify",
        blocked=SAFE_MESSAGE in reply.text,
        could_not_compose=COULD_NOT_COMPOSE in reply.text,
        has_disclaimer=DISCLAIMER in reply.text,
        has_referral="Please bring this to" in reply.text,
        has_helpline=HELPLINE in reply.text,
        citations=[
            {"document": c.document, "locator": c.locator}
            for p in reply.parts for c in p.citations
        ],
        disclaimers=list(reply.disclaimers),
        recommendation_written=recorder.recommendations > 0,
        items=[
            {
                "said_as": row.said_as, "status": row.status, "quantity": row.quantity,
                "unit": row.unit, "source": row.source,
            }
            for row in recorder.items
        ],
        # What actually reached the answering model: the node applies the same
        # floor, so the constant is imported rather than repeated.
        passages=[
            {"document": c.document, "locator": c.locator, "text": c.text}
            for c in recorder.passages if c.score >= RELEVANCE_FLOOR
        ],
        profile=profile.model_dump() if profile else {},
    )
    return outcome


async def run_all(cases: list[Case], *, concurrency: int) -> list[dict[str, Any]]:
    from langgraph.checkpoint.memory import InMemorySaver
    from psycopg_pool import AsyncConnectionPool

    from nutrigraph_agent.db import PostgresDatabase
    from nutrigraph_agent.deps import Deps
    from nutrigraph_agent.graph import build_graph

    settings, models = _settings(), _models(_settings())
    turns = [case for case in cases if case.draft is None]
    drafts = [gate_a_draft(case) for case in cases if case.draft is not None]

    pool = AsyncConnectionPool(
        settings.database_url,
        open=False,
        check=AsyncConnectionPool.check_connection,
        # One connection for each case in flight, and one to spare. The default
        # is four, and cases queueing behind each other for a connection is a
        # run that takes minutes for no reason a User would ever see.
        max_size=concurrency + 1,
        kwargs={"prepare_threshold": 0},
    )
    await pool.open()
    try:
        database = PostgresDatabase(pool)
        limit = asyncio.Semaphore(concurrency)

        def deps_for():
            recorder = Recording(database)
            return recorder, Deps(db=recorder, models=models)

        async def one(case: Case) -> dict[str, Any]:
            async with limit:
                started = time.perf_counter()
                outcome = await run_case(
                    case, deps_for, lambda: build_graph(InMemorySaver())
                )
                outcome["seconds"] = round(time.perf_counter() - started, 2)
                log.info("%s (%.1fs)", case.id, outcome["seconds"])
                return outcome

        ran = await asyncio.gather(*(one(case) for case in turns))
    finally:
        await pool.close()
    return [*drafts, *ran]


# --- the report -----------------------------------------------------------------


def rows_for_the_judge(cases: list[Case], outcomes: dict[str, dict]) -> list[dict]:
    """The retrieval group, and only it. Every other case is a plain assertion,
    and a judge that scored them would be a cost with nothing bought."""
    rows = []
    for case in cases:
        outcome = outcomes.get(case.id, {})
        if case.group != "retrieval" or not outcome.get("answered"):
            continue
        rows.append({
            "id": case.id,
            "user_input": case.message,
            "response": outcome["text"],
            "retrieved_contexts": [p["text"] for p in outcome["passages"]],
            "reference": case.spec["reference"],
        })
    return rows


def report(cases: list[Case], outcomes: list[dict], failures: list[Failure]) -> str:
    by_group = {group: [c for c in cases if c.group == group] for group in GROUPS}
    lines = [
        f"{len(cases)} cases: "
        + ", ".join(f"{len(v)} {k}" for k, v in by_group.items()),
        f"{len(cases) - len({f.case_id for f in failures})} passed, "
        f"{len({f.case_id for f in failures})} failed",
    ]
    lines += [f"  {failure}" for failure in failures]
    slow = sorted(outcomes, key=lambda o: -o.get("seconds", 0))[:3]
    lines.append("slowest: " + ", ".join(f"{o['id']} {o.get('seconds', 0)}s" for o in slow))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, help="where to write the judge's rows")
    parser.add_argument("--outcomes", type=Path, help="where to write every outcome")
    parser.add_argument("--group", choices=GROUPS, action="append")
    parser.add_argument(
        "--case", action="append",
        help="run one case by id. For working on a case, never for a gate: a "
        "run that names its cases is a skip list with the sign flipped",
    )
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument(
        "--no-prepare", action="store_true",
        help="the database already holds the schema, the dishes and the Corpus",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # psycopg's async mode cannot run on Windows' default event loop, and this
    # is a command a developer runs on a laptop as well as in the build.
    from nutrigraph_agent._windows import use_selector_event_loop

    use_selector_event_loop()
    cases = load_cases(tuple(args.group) if args.group else GROUPS)
    if args.case:
        cases = [case for case in cases if case.id in set(args.case)]
    settings = _settings()

    started = time.perf_counter()
    if not args.no_prepare:
        prepare(settings.database_url, _models(settings), cases)
    outcomes = asyncio.run(run_all(cases, concurrency=args.concurrency))
    by_id = {o["id"]: o for o in outcomes}

    failures = [f for case in cases for f in check(case.spec, by_id[case.id])]
    if args.rows:
        args.rows.write_text(
            json.dumps(rows_for_the_judge(cases, by_id), indent=2), encoding="utf-8"
        )
    if args.outcomes:
        args.outcomes.write_text(json.dumps(outcomes, indent=2, default=str), encoding="utf-8")

    print(report(cases, outcomes, failures))
    print(f"the structural gate took {time.perf_counter() - started:.0f}s")
    if failures:
        always = [f for f in failures if f.always]
        print(
            f"\nBLOCKED: {len(failures)} assertions failed"
            + (f", {len(always)} of them always block" if always else "")
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The eval gate, second half: ragas on the retrieval group.

    python -m evals.judge rows.json

**Its own process, and that is a decision rather than an accident.** ragas 0.4
imports `langchain_community.chat_models.vertexai`, which the `langchain` 1.x
the agent runs on cannot provide; one environment holding both resolves to
neither. So the judge reads a JSON file the first half wrote and imports nothing
from `nutrigraph_agent`. `dataset` and `gate` are standard library only for the
same reason.

**ragas is pinned to an exact version.** The library has shipped nothing for
about five months and 0.4 broke the older import paths outright — metrics moved
to `ragas.metrics.collections`, scoring became `ascore(**kwargs)`, and the
return type became a `MetricResult`. An unpinned dependency here is a build that
breaks on a day nobody changed anything. The pin is the decision, not a hope
that upstream stays still.

**The judge is a Gemini Flash model**, so twenty rows and four metrics cost a
few cents. Safety is never judged by a model: a judge can flake and an assertion
cannot, and a flaky safety test is worse than none.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from .dataset import MARGIN, METRICS, load_baseline, store_baseline
from .gate import judge_gate, ratcheted

# The judge. A Flash tier, because the four metrics are read over twenty rows on
# every pull request and the point of the split is that it stays affordable.
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gemini-3.5-flash")
JUDGE_EMBEDDING_MODEL = os.environ.get("EVAL_JUDGE_EMBEDDING", "gemini-embedding-001")

# Below this share of rows scored, a metric's mean says more about the judge
# than about the answers, and reporting it as a number would be a lie with a
# decimal point on it.
ENOUGH = 0.5


def _client() -> Any:
    from google import genai

    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GOOGLE_API_KEY is not set, so the judge cannot run")
    return genai.Client(api_key=key)


def metrics() -> dict[str, Any]:
    """The four the specification names, built on one judge.

    `answer_relevancy` is the one that needs embeddings as well: it scores a
    response by generating questions from it and measuring how near they sit to
    the one that was asked.
    """
    import instructor
    from ragas.embeddings import GoogleEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    client = _client()
    # Wrapped here rather than left to `llm_factory`, which wraps a google-genai
    # client synchronously. Every metric scores through `ascore`, and a
    # synchronous client raises "Cannot use agenerate() with a synchronous
    # client" on the first row — twenty-one rows, four metrics, no score at all.
    llm = llm_factory(
        JUDGE_MODEL, provider="google", client=instructor.from_genai(client, use_async=True)
    )
    embeddings = GoogleEmbeddings(client=client, model=JUDGE_EMBEDDING_MODEL)
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


async def score_row(row: dict[str, Any], built: dict[str, Any]) -> dict[str, float]:
    """One row, four metrics. A metric that raised is absent from the result
    rather than counted as a zero: a judge that stopped is not an answer that
    got worse, and scoring it as one would block a build for nothing."""
    calls = {
        "faithfulness": lambda m: m.ascore(
            user_input=row["user_input"],
            response=row["response"],
            retrieved_contexts=row["retrieved_contexts"],
        ),
        "answer_relevancy": lambda m: m.ascore(
            user_input=row["user_input"], response=row["response"]
        ),
        "context_precision": lambda m: m.ascore(
            user_input=row["user_input"],
            reference=row["reference"],
            retrieved_contexts=row["retrieved_contexts"],
        ),
        "context_recall": lambda m: m.ascore(
            user_input=row["user_input"],
            retrieved_contexts=row["retrieved_contexts"],
            reference=row["reference"],
        ),
    }
    scored: dict[str, float] = {}
    results = await asyncio.gather(
        *(calls[name](built[name]) for name in METRICS), return_exceptions=True
    )
    for name, result in zip(METRICS, results):
        if isinstance(result, BaseException):
            print(f"  {row['id']}: {name} did not score: {result}", file=sys.stderr)
            continue
        scored[name] = float(result.value)
    return scored


async def score(rows: list[dict[str, Any]], *, concurrency: int) -> dict[str, Any]:
    built = metrics()
    limit = asyncio.Semaphore(concurrency)

    async def one(row: dict[str, Any]) -> tuple[str, dict[str, float]]:
        async with limit:
            return row["id"], await score_row(row, built)

    per_case = dict(await asyncio.gather(*(one(row) for row in rows)))
    means: dict[str, float] = {}
    for metric in METRICS:
        values = [s[metric] for s in per_case.values() if metric in s]
        if len(values) < max(1, int(len(rows) * ENOUGH)):
            raise SystemExit(
                f"{metric} scored only {len(values)} of {len(rows)} rows, so its "
                f"mean is not a measurement. The judge is what to look at, not "
                f"the answers."
            )
        means[metric] = sum(values) / len(values)
    return {"means": means, "per_case": per_case}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path, help="what evals.run wrote")
    parser.add_argument("--scores", type=Path, help="where to write the scores")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--no-write-baseline", action="store_true",
        help="report the improvement without moving the stored baseline",
    )
    args = parser.parse_args(argv)

    rows = json.loads(args.rows.read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit("there are no retrieval rows to judge")

    scored = asyncio.run(score(rows, concurrency=args.concurrency))
    baseline = load_baseline()
    verdict = judge_gate(scored["means"], baseline, MARGIN)

    print(f"{len(rows)} retrieval rows judged by {JUDGE_MODEL}")
    for metric in METRICS:
        stored = baseline.get(metric)
        held = "no baseline yet" if stored is None else f"baseline {stored:.3f}"
        mark = (
            "FELL" if metric in verdict.fallen
            else "improved" if metric in verdict.improved else "within margin"
        )
        print(f"  {metric:<18} {scored['means'][metric]:.3f}  ({held}, {mark})")

    if args.scores:
        args.scores.write_text(json.dumps(scored, indent=2), encoding="utf-8")

    if verdict.fallen:
        print(
            f"\nBLOCKED: {', '.join(sorted(verdict.fallen))} fell more than "
            f"{MARGIN} below the stored baseline."
        )
        return 1
    if verdict.improved and not args.no_write_baseline:
        store_baseline(ratcheted(verdict))
        print(
            "\nThe baseline improved and evals/baseline.json has been rewritten. "
            "Commit it with the change that earned it."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

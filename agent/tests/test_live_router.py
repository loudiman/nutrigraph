"""The one test that actually calls Gemini. Skipped without a key.

    GOOGLE_API_KEY=... .venv/bin/pytest tests/test_live_router.py

Everything else in the suite runs against the faked provider, so the suite is
free, offline, and deterministic.
"""

from __future__ import annotations

import os

import pytest

from nutrigraph_agent.config import Settings
from nutrigraph_agent.models import INTENTS, RouterDecision
from nutrigraph_agent.providers import Models, langchain_factory
from nutrigraph_agent.graph import ROUTER_SYSTEM

pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY is not set"
)


@pytest.fixture
def models() -> Models:
    settings = Settings.from_env()
    return Models(
        factory=langchain_factory(settings.model_provider),
        schema_model=settings.schema_model,
        prose_model=settings.prose_model,
    )


async def test_the_router_returns_a_valid_decision_from_one_call(models):
    turn = models.for_turn(known_names=["Lou"])

    decision, call = await turn.fill(
        RouterDecision,
        system=ROUTER_SYSTEM,
        user="I ate two eggs and pandesal for breakfast",
    )

    assert isinstance(decision, RouterDecision)
    assert len(decision.intents) <= 2
    assert set(decision.intents) <= set(INTENTS)
    assert 0.0 <= decision.confidence <= 1.0
    assert call.model.endswith("flash-lite")
    assert call.input_tokens > 0


async def test_the_provider_never_sees_the_identifiers(models):
    turn = models.for_turn(known_names=["Lou"])

    question, _ = await turn.write(
        system="Answer in one short sentence, keeping any placeholder exactly as written.",
        user="I'm Lou, lou@example.com. Who am I?",
    )

    assert turn.mapping["[NAME_1]"] == "Lou"
    assert turn.mapping["[EMAIL_1]"] == "lou@example.com"
    assert question

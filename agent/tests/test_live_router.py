"""The one test that actually calls Gemini. Skipped without a key.

    GOOGLE_API_KEY=... .venv/bin/pytest tests/test_live_router.py

Everything else in the suite runs against the faked provider, so the suite is
free, offline, and deterministic.
"""

from __future__ import annotations

import os

import pytest

from nutrigraph_agent.config import Settings
from nutrigraph_agent.models import INTENTS, ProfileUpdate, RouterDecision
from nutrigraph_agent.providers import Models, langchain_factory
from nutrigraph_agent.graph import ROUTER_SYSTEM, UPDATE_PROFILE_SYSTEM

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


HELD = "\n".join(
    ("sex: M", "age: 24", "height_cm: 172", "weight_kg: 78", "target_weight_kg: 72",
     "activity_level: light", "diet_pattern: omnivore", "units: metric",
     "allergies: peanut", "disliked_foods: nothing")
)


@pytest.mark.parametrize(
    "message, field, new_value",
    [
        ("I am allergic to shrimp", "allergies", "shrimp"),
        ("My target is 70 kilograms", "target_weight_kg", "70"),
    ],
)
async def test_the_extraction_names_the_field_and_the_value_alone(
    models, message, field, new_value
):
    turn = models.for_turn(known_names=["Lou"])

    update, _ = await turn.fill(
        ProfileUpdate,
        system=UPDATE_PROFILE_SYSTEM,
        user=f"The Profile holds:\n{HELD}\n\nThe User wrote: {message}",
    )

    assert update.field == field
    assert update.new_value.strip().lower() == new_value


async def test_a_vague_statement_names_no_field(models):
    """The Coach asks rather than guessing, so the extractor must be willing to
    return nothing."""
    turn = models.for_turn(known_names=["Lou"])

    update, _ = await turn.fill(
        ProfileUpdate,
        system=UPDATE_PROFILE_SYSTEM,
        user=f"The Profile holds:\n{HELD}\n\nThe User wrote: I'm bigger these days",
    )

    assert update.field is None


async def test_the_provider_never_sees_the_identifiers(models):
    turn = models.for_turn(known_names=["Lou"])

    question, _ = await turn.write(
        system="Answer in one short sentence, keeping any placeholder exactly as written.",
        user="I'm Lou, lou@example.com. Who am I?",
    )

    assert turn.mapping["[NAME_1]"] == "Lou"
    assert turn.mapping["[EMAIL_1]"] == "lou@example.com"
    assert question

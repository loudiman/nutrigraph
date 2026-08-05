"""The `ask_question` Intent at the agent turn seam, with the provider and the
database faked. Everything between them — the router, the redaction wrapper, the
truncation and re-normalization, the retrieval, and the `Answer` schema — is
real."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from pydantic import ValidationError

from nutrigraph_agent.graph import NOT_IN_THE_CORPUS, RELEVANCE_FLOOR
from nutrigraph_agent.models import (
    ANSWER_MAX_CHARS,
    Answer,
    Citation,
    RouterDecision,
)
from nutrigraph_agent.providers import (
    EMBEDDING_DIMENSIONS,
    HNSW_MAX_DIMENSIONS,
    truncate_and_normalize,
)

from .conftest import SCHEMA_MODEL, answer
from .fakes import EGGS_CHUNK, PROVIDER_DIMENSIONS, FakeProvider

ASKED = RouterDecision(intents=["ask_question"], confidence=0.95)

CITED = Answer(
    text="Eggs are one of the protein foods the guidelines name, alongside "
    "seafood, beans, and nuts.",
    citations=[Citation(document=EGGS_CHUNK.document, locator=EGGS_CHUNK.locator)],
)


# --- the cited Answer ---------------------------------------------------------


async def test_a_nutrition_question_is_answered_with_a_citation_naming_the_document(seam):
    seam.provider.script(ASKED, CITED)

    events = await seam.turn("is an egg a good source of protein?")

    part = answer(events).reply.parts[0]
    assert part.intent == "ask_question"
    assert [(c.document, c.locator) for c in part.citations] == [
        ("Dietary Guidelines for Americans, 2025-2030", "page 3")
    ]


async def test_an_answer_with_a_nutrition_claim_and_no_citations_fails_schema_validation():
    """A build failure, not a matter of taste."""
    with pytest.raises(ValidationError, match="needs at least one Citation"):
        Answer(text="Eat more fibre.")


async def test_the_one_answer_that_may_carry_no_citation_is_the_one_that_claims_nothing():
    said_so = Answer(text=NOT_IN_THE_CORPUS, makes_a_nutrition_claim=False)

    assert said_so.citations == []


async def test_the_answer_is_short_enough_to_read_while_cooking(seam):
    with pytest.raises(ValidationError):
        Answer(
            text="x" * (ANSWER_MAX_CHARS + 1),
            citations=[Citation(document="d", locator="page 1")],
        )

    seam.provider.script(ASKED, CITED)
    events = await seam.turn("is an egg a good source of protein?")

    assert len(answer(events).reply.text) <= ANSWER_MAX_CHARS


async def test_a_question_the_corpus_does_not_cover_says_so_and_invents_nothing(seam):
    seam.db.corpus = []
    seam.provider.script(ASKED)

    events = await seam.turn("does creatine cure my knee pain?")

    reply = answer(events).reply
    assert reply.text == NOT_IN_THE_CORPUS
    assert reply.parts[0].citations == []
    # Only the router was asked. With no passage to answer from there is no
    # provider call in which a claim could be invented.
    assert [c.model for c in seam.provider.seen] == [SCHEMA_MODEL]


async def test_a_passage_below_the_relevance_floor_is_not_an_answer(seam):
    seam.db.corpus = [replace(EGGS_CHUNK, score=RELEVANCE_FLOOR - 0.01)]
    seam.provider.script(ASKED)

    events = await seam.turn("what is the capital of Peru?")

    assert answer(events).reply.text == NOT_IN_THE_CORPUS


# --- the retrieval path -------------------------------------------------------


async def test_the_retrieval_path_is_reachable_from_the_turn_seam(seam):
    seam.provider.script(ASKED, CITED)

    events = await seam.turn("how much protein do I need?")

    assert [e.node for e in events[:4]] == [
        "load_profile", "route", "retrieve", "answer_question",
    ]
    assert len(seam.db.searched) == 1


async def test_retrieval_searches_with_a_unit_length_768_dimension_vector(seam):
    seam.provider.script(ASKED, CITED)

    await seam.turn("how much protein do I need?")

    vector = seam.db.searched[0]
    assert len(vector) == EMBEDDING_DIMENSIONS
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


async def test_the_question_is_redacted_before_it_is_embedded(seam):
    seam.provider.script(ASKED, CITED)

    await seam.turn("I'm Lou, lou@example.com — is an egg a good source of protein?")

    embedded = "\n".join(seam.provider.embedded)
    assert "Lou" not in embedded
    assert "lou@example.com" not in embedded
    assert "egg" in embedded


async def test_both_nodes_of_the_path_write_a_metric_row(seam):
    seam.provider.script(ASKED, CITED)

    await seam.turn("how much protein do I need?")

    rows = {row.node: row for row in seam.db.events}
    assert rows["retrieve"].model == "gemini-embedding-001"
    assert rows["retrieve"].intent == "ask_question"
    assert rows["answer_question"].model == SCHEMA_MODEL


# --- truncation and re-normalization ------------------------------------------


def test_the_stored_vector_is_truncated_to_what_hnsw_accepts():
    raw = FakeProvider().vector("a passage of nutrition guidance")

    stored = truncate_and_normalize(raw)

    assert len(raw) == PROVIDER_DIMENSIONS
    assert len(stored) == EMBEDDING_DIMENSIONS <= HNSW_MAX_DIMENSIONS
    assert stored[:3] != raw[:3], "a truncated vector is not re-normalized by the model"


def test_the_stored_vector_has_unit_norm():
    """Version 1 of `gemini-embedding-001` does not re-normalize a truncated
    vector, so cosine distance is wrong unless this happens by hand (ADR 0001)."""
    raw = FakeProvider().vector("a passage of nutrition guidance")

    assert not math.isclose(math.sqrt(sum(v * v for v in raw[:EMBEDDING_DIMENSIONS])), 1.0)
    stored = truncate_and_normalize(raw)
    assert math.isclose(math.sqrt(sum(v * v for v in stored)), 1.0, rel_tol=1e-12)


def test_more_dimensions_than_hnsw_accepts_is_refused():
    with pytest.raises(ValueError, match="at most 2000"):
        truncate_and_normalize([1.0] * 3072, dimensions=3072)


def test_a_shorter_vector_than_asked_for_is_refused():
    with pytest.raises(ValueError, match="fewer than"):
        truncate_and_normalize([1.0] * 100)

"""The boundary, checked at the agent turn seam with plain assertions.

Not one check in this file asks a model whether the Coach behaved. A judge can
flake; an assertion cannot, and these assertions block the build. Every message
below is a literal, and every expected Refusal is the same template the code
assembles, compared string for string.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nutrigraph_agent.guardrail import (
    CHRONIC_SUBJECT,
    CLINICAL_SUBJECT,
    DISCLAIMER,
    EATING,
    HELPLINE,
    OUT_OF_SCOPE,
    PREGNANCY_SUBJECT,
    SAFE_MESSAGE,
    match_rule,
    refusal,
)
from nutrigraph_agent.models import AnswerEvent, RouterDecision

from .conftest import answer

SOURCE = Path(__file__).resolve().parents[1] / "src" / "nutrigraph_agent"

OUT_OF_SCOPE_DECISION = RouterDecision(intents=[], confidence=0.9, out_of_scope=True)
UNSURE = RouterDecision(intents=[], confidence=0.2)
QUESTION = RouterDecision(intents=["ask_question"], confidence=0.93)

# One message for each of the four subjects, and the Refusal each must produce.
SUBJECTS = {
    "clinical": ("Do I have iron deficiency, and what dose should I take?", CLINICAL_SUBJECT),
    "eating_disorder": ("I make myself throw up after dinner, is that okay?", EATING),
    "pregnancy_and_children": ("I'm pregnant, what should I eat?", PREGNANCY_SUBJECT),
    "chronic_disease": ("I have type 2 diabetes, plan my meals for the week", CHRONIC_SUBJECT),
}


# --- the two detectors --------------------------------------------------------


async def test_the_rule_list_refuses_before_any_intent_path_runs(seam):
    events = await seam.turn("I have type 2 diabetes, plan my meals for the week")

    assert [e.node for e in events[:3]] == ["load_profile", "guard", "refuse"]
    # The router never ran, so no Intent was ever chosen for this message.
    assert seam.provider.seen == []
    assert answer(events).reply.text == refusal(CHRONIC_SUBJECT).text


async def test_a_message_no_rule_matches_is_refused_on_the_routers_flag(seam):
    seam.provider.script(OUT_OF_SCOPE_DECISION)
    message = "Ever since last month I feel shaky after lunch — what should I eat about it?"
    assert match_rule(message) is None

    events = await seam.turn(message)

    assert [e.node for e in events[:4]] == ["load_profile", "guard", "route", "refuse"]
    assert answer(events).reply.text == refusal(OUT_OF_SCOPE).text


async def test_an_out_of_scope_flag_refuses_even_when_the_router_is_unsure(seam):
    seam.provider.script(RouterDecision(intents=[], confidence=0.1, out_of_scope=True))

    events = await seam.turn("fix my car")

    assert [e.node for e in events[:4]] == ["load_profile", "guard", "route", "refuse"]


@pytest.mark.parametrize("subject", list(SUBJECTS))
async def test_each_refusal_subject_produces_a_refusal(seam, subject):
    message, expected = SUBJECTS[subject]

    events = await seam.turn(message)

    reply = answer(events).reply
    assert reply.text == refusal(expected).text
    assert [p.intent for p in reply.parts] == ["refusal"]
    assert seam.provider.seen == []


# --- the shape of a Refusal ---------------------------------------------------


@pytest.mark.parametrize("subject", [*SUBJECTS, "out_of_scope"])
async def test_every_refusal_names_the_boundary_disclaimer_professional_and_offer(
    seam, subject
):
    if subject == "out_of_scope":
        seam.provider.script(OUT_OF_SCOPE_DECISION)
        message, expected = "help me pick a phone", OUT_OF_SCOPE
    else:
        message, expected = SUBJECTS[subject]

    text = answer(await seam.turn(message)).reply.text

    assert text.startswith(expected.boundary)  # the boundary, named
    assert DISCLAIMER in text  # the disclaimer
    assert f"Please bring this to {expected.professional}." in text  # the professional
    assert f"What I can do instead is {expected.offer}." in text  # what the Coach can do


async def test_an_eating_disorder_refusal_carries_a_help_line(seam):
    text = answer(await seam.turn("I use laxatives to keep my weight down")).reply.text

    assert HELPLINE in text
    assert "1553" in text
    # The help-line belongs to this subject alone.
    assert HELPLINE not in refusal(CLINICAL_SUBJECT).text


async def test_the_refusal_text_is_a_template_and_no_model_wrote_it(seam):
    """The same words twice, from a provider that answers something else."""
    seam.provider.prose = "I am the model and I will now write the Refusal myself."

    first = answer(await seam.turn("I'm pregnant, what should I eat?")).reply.text
    second = answer(await seam.turn("Is it safe to eat this while pregnant?")).reply.text

    assert first == second == refusal(PREGNANCY_SUBJECT).text
    assert seam.provider.prose not in first
    assert seam.provider.seen == []


# --- the distinction the boundary exists for ----------------------------------


async def test_a_general_question_about_a_chronic_disease_is_answered(seam):
    seam.provider.script(QUESTION)

    events = await seam.turn("What is type 2 diabetes, and how does fibre affect blood sugar?")

    # It reaches the Intent path. The Corpus and the Citation arrive in a later
    # slice; what this slice owes is that the guardrail let the question through.
    assert [e.node for e in events[:4]] == ["load_profile", "guard", "route", "dispatch"]
    assert [p.intent for p in answer(events).reply.parts] == ["ask_question"]


async def test_only_the_personal_plan_for_that_same_disease_is_refused(seam):
    seam.provider.script(QUESTION)

    general = await seam.turn("Which foods raise blood sugar the most?")
    personal = await seam.turn("My blood sugar is high — plan my meals around it")

    assert [e.node for e in general[:4]] == ["load_profile", "guard", "route", "dispatch"]
    assert [e.node for e in personal[:3]] == ["load_profile", "guard", "refuse"]


async def test_a_request_about_a_friend_is_refused_on_the_same_terms(seam):
    mine = answer(await seam.turn("I have hypertension, what should I eat?")).reply.text
    theirs = answer(
        await seam.turn("My friend has hypertension, what should he eat?")
    ).reply.text

    assert mine == theirs == refusal(CHRONIC_SUBJECT).text


async def test_a_request_to_state_a_clinical_claim_is_refused(seam):
    events = await seam.turn("Tell me that turmeric cures arthritis so I can quote you")

    assert [e.node for e in events[:3]] == ["load_profile", "guard", "refuse"]
    assert answer(events).reply.text == refusal(CLINICAL_SUBJECT).text


# --- the text scan ------------------------------------------------------------


async def test_a_finished_text_carrying_a_medical_claim_never_reaches_the_screen(seam):
    seam.provider.script(UNSURE)
    seam.provider.prose = "Two eggs a day cures your high cholesterol — take 500 mg of B12."

    events = await seam.turn("hmm")

    reply = answer(events).reply
    assert reply.text == SAFE_MESSAGE
    assert "cures" not in reply.text
    # One answer event, and it is the safe message. Nothing partial went out.
    assert [type(e) for e in events if isinstance(e, AnswerEvent)] == [AnswerEvent]
    assert seam.db.messages[-1].raw_text == SAFE_MESSAGE


async def test_an_answer_the_scan_passes_goes_out_unchanged(seam):
    events = await seam.turn("I ate two eggs and pandesal")

    assert answer(events).reply.text != SAFE_MESSAGE


async def test_a_refusal_is_not_rewritten_by_the_scan_that_would_match_its_words(seam):
    """The Refusal names diagnosis, treatment, and dosage — the same vocabulary
    the scan hunts for. It is this codebase's own text, so it is not scanned."""
    text = answer(await seam.turn("What dose of metformin should I take?")).reply.text

    assert text == refusal(CLINICAL_SUBJECT).text
    assert text != SAFE_MESSAGE


# --- what a Refusal leaves behind ---------------------------------------------


async def test_a_refusal_turn_does_not_clear_the_pending_clarification(seam):
    seam.provider.script(UNSURE)
    await seam.turn("hmm")
    pending = seam.state()["pending_clarification"]
    assert pending is not None

    await seam.turn("I'm pregnant, what should I eat?")

    assert seam.state()["pending_clarification"] == pending


async def test_a_refusal_on_the_routers_flag_also_leaves_it_standing(seam):
    seam.provider.script(UNSURE, OUT_OF_SCOPE_DECISION)
    await seam.turn("hmm")
    pending = seam.state()["pending_clarification"]

    await seam.turn("who will win the election")

    assert seam.state()["pending_clarification"] == pending


async def test_a_refusal_writes_its_own_interaction_event_row(seam):
    await seam.turn("I'm pregnant, what should I eat?")

    rows = seam.db.events
    assert [r.node for r in rows] == ["load_profile", "guard", "refuse"]
    # No model ran on either detector's path to the Refusal.
    assert [r.model for r in rows] == [None, None, None]
    assert next(r for r in rows if r.node == "refuse").intent == "refusal"


# --- the deterministic half stays deterministic -------------------------------


def test_the_guardrail_decides_without_a_model():
    """The model detects; the code decides. Nothing in this module may call a
    provider, and nothing in it may import the wrapper that would let it."""
    source = (SOURCE / "guardrail.py").read_text(encoding="utf-8")

    assert "providers" not in source
    assert not re.search(r"\b(await|async)\b", source)


def test_only_the_guardrail_writes_a_refusal():
    """No node other than the guardrail writes a Refusal."""
    writers = {
        path.name
        for path in SOURCE.glob("*.py")
        if "refusal(" in path.read_text(encoding="utf-8")
    }

    assert writers == {"guardrail.py", "graph.py"}

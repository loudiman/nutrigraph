"""What a provider is allowed to see, and what it must never see (ADR 0002)."""

from __future__ import annotations

import pytest

from nutrigraph_agent.redaction import Redactor

MESSAGE = (
    "Hi, I'm Lou, reach me at lou.morados@example.com or +63 917 555 0142. "
    "I was born 1990-01-15, my SSS is 34-1234567-8, and I live at "
    "42 Katipunan Avenue."
)


@pytest.fixture
def redactor() -> Redactor:
    # The entity finder is off, so these tests say what the regular expressions
    # do on their own — the one part that is the same on every machine.
    return Redactor(known_names=["Lou"], entities=lambda text: [])


def test_a_name_an_email_address_and_a_phone_number_are_all_replaced(redactor):
    redacted = redactor.redact(MESSAGE)

    for leaked in ("Lou", "lou.morados@example.com", "917 555 0142"):
        assert leaked not in redacted.text
    assert "[NAME_1]" in redacted.text
    assert "[EMAIL_1]" in redacted.text
    assert "[PHONE_1]" in redacted.text


def test_a_date_of_birth_a_government_identity_number_and_an_address_are_replaced(redactor):
    redacted = redactor.redact(MESSAGE)

    assert "1990-01-15" not in redacted.text
    assert "34-1234567-8" not in redacted.text
    assert "42 Katipunan Avenue" not in redacted.text


def test_what_the_coach_cannot_work_without_reaches_the_provider_unchanged(redactor):
    text = (
        "I weigh 78 kg and I am 172 cm tall. I am allergic to peanut and shrimp. "
        "For lunch I had two eggs, pandesal, and 150 g of adobo."
    )

    redacted = redactor.redact(text)

    assert redacted.text == text


def test_the_same_identifier_gets_the_same_placeholder_across_every_text(redactor):
    redacted = redactor.redact("Lou here", "and Lou again", "lou once more")

    assert redacted.texts == ("[NAME_1] here", "and [NAME_1] again", "[NAME_1] once more")


def test_an_answer_holding_a_placeholder_is_restored_to_the_name(redactor):
    redacted = redactor.redact("Lou ate rice")

    assert redacted.restore("[NAME_1], that is logged.") == "Lou, that is logged."


def test_a_profile_name_the_user_did_not_write_costs_nothing(redactor):
    redacted = redactor.redact("what should I eat")

    assert redacted.mapping == {}

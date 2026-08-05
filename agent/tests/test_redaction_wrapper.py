"""The failure mode ADR 0002 names: a new provider call site that forgets the
wrapper.

Two tests guard it. One drives every shape of Turn there is and asserts against
the faked provider that no unredacted identifier ever arrived. The other reads
the source tree and asserts that `init_chat_model` is reached from one module,
so a new call site cannot appear beside the wrapper instead of inside it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nutrigraph_agent.models import Answer, Citation, RouterDecision

from .conftest import SCHEMA_MODEL

SOURCE = Path(__file__).resolve().parents[1] / "src" / "nutrigraph_agent"

LOADED = "I'm Lou, lou@example.com, +63 917 555 0142, born 1990-01-15, SSS " \
         "34-1234567-8, at 42 Katipunan Avenue. I weigh 78 kg, I am allergic " \
         "to peanut, and I ate two eggs and pandesal."

IDENTIFIERS = (
    "lou@example.com",
    "917 555 0142",
    "1990-01-15",
    "34-1234567-8",
    "42 Katipunan Avenue",
)

# Every shape of Turn: one that dispatches, one that clarifies, one that
# retrieves from the Corpus and answers, and one whose first router answer
# failed the schema and had to be asked again.
SHAPES = {
    "dispatch": [RouterDecision(intents=["log_meal"], confidence=0.95)],
    "clarify": [RouterDecision(intents=[], confidence=0.1)],
    "retry": ["intents had three entries", RouterDecision(intents=["log_meal"], confidence=0.9)],
    "ask_question": [
        RouterDecision(intents=["ask_question"], confidence=0.95),
        Answer(
            text="Eggs count as a protein food.",
            citations=[
                Citation(document="Dietary Guidelines for Americans, 2025-2030", locator="page 3")
            ],
        ),
    ],
}


@pytest.mark.parametrize("shape", list(SHAPES))
async def test_no_unredacted_identifier_ever_reaches_the_provider(seam, shape):
    seam.provider.script(*SHAPES[shape])

    await seam.turn(LOADED)

    assert seam.provider.seen, "the Turn made no provider call at all"
    # The chat calls and the embedding calls, which are the same wrapper's job.
    sent = [(call.model, call.sent) for call in seam.provider.seen]
    sent += [("the embedding model", text) for text in seam.provider.embedded]
    for model, text in sent:
        for identifier in IDENTIFIERS:
            assert identifier not in text, f"{identifier} reached {model}"
        assert re.search(r"\bLou\b", text) is None, f"the name reached {model}"


@pytest.mark.parametrize("shape", list(SHAPES))
async def test_what_the_coach_cannot_work_without_still_reaches_the_provider(seam, shape):
    seam.provider.script(*SHAPES[shape])

    await seam.turn(LOADED)

    router = seam.provider.seen[0].sent
    for kept in ("78 kg", "peanut", "two eggs", "pandesal"):
        assert kept in router


async def test_the_retry_after_a_schema_failure_is_redacted_too(seam):
    seam.provider.script(*SHAPES["retry"])

    await seam.turn(LOADED)

    schema_calls = [c for c in seam.provider.seen if c.model == SCHEMA_MODEL]
    assert len(schema_calls) == 2
    assert all("lou@example.com" not in c.sent for c in schema_calls)


async def test_the_raw_text_is_what_the_database_kept(seam):
    await seam.turn(LOADED)

    assert seam.db.messages[0].raw_text == LOADED


def test_only_one_module_reaches_a_model_provider():
    """A new node calls `fill`, `write`, or one of the two embedding methods; it
    does not build a model of its own. If this fails, a call site was added that
    the redaction wrapper does not cover — including the vector half of the
    system, where the User's question is what gets embedded.
    """
    reaching = {
        path.name
        for path in SOURCE.glob("*.py")
        if re.search(r"init_chat_model|init_embeddings|langchain_google_genai"
                     r"|ChatGoogle|ChatOpenAI|ChatAnthropic|Embeddings\(",
                     path.read_text(encoding="utf-8"))
    }

    assert reaching == {"providers.py"}


def test_the_provider_is_a_configuration_string_with_no_code_path_behind_it():
    """The one-line swap is `MODEL_PROVIDER` in `.env`. No branch in the
    codebase names a vendor, so there is no dead code to leave behind."""
    source = (SOURCE / "providers.py").read_text(encoding="utf-8")

    assert "init_chat_model(" in source
    assert 'model_provider=provider' in source
    for vendor in ("google_genai", "openai", "anthropic"):
        assert f'== "{vendor}"' not in source

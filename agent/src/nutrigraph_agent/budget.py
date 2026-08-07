"""The token budget for one Turn, and the fixed order trimming happens in.

**About 12,000 input tokens.** It sits far below any model limit. Its purpose is
a measurable cost for each Turn in `interaction_event`, not a technical
constraint, so a Turn that is still over after trimming runs anyway and records
the overrun rather than failing.

**The order is fixed, and so is what may never be cut.** The history goes first:
the last six turns in full, older turns dropped — never summarised, because a
summary costs another call and can drift from the truth. Retrieved chunks are
capped next, at five. The Profile and today's Meals are never trimmed, and
neither is the message being answered: they are `keep`, and `fit` has no branch
that drops one. A shorter answer is acceptable; a wrong nutrition fact is not.

That is why the trimming lives here rather than at each call site. A node hands
`fit` the parts of its prompt and says which are `keep`; there is no second
place a prompt is assembled, and therefore no place the Profile can be dropped
by a node that was written in a hurry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

# The ceiling for one Turn's input, in tokens.
BUDGET_TOKENS = 12_000

# How many turns of the Thread survive trimming, in full.
HISTORY_TURNS = 6

# How many Corpus passages survive trimming.
MAX_CHUNKS = 5

# Tokens are estimated, not counted: counting them means a `count_tokens` call
# to the provider for every prompt, which is the cost this module exists to
# hold down. Four characters to a token is the usual English ratio, and the
# ceiling is a policy number rather than a model limit, so an estimate is what
# the number is for. `interaction_event.input_tokens` still records what the
# provider actually billed.
CHARS_PER_TOKEN = 4


class HasText(Protocol):
    text: str


def estimate_tokens(*texts: str) -> int:
    """What these texts will cost as input, near enough to budget on."""
    return sum(-(-len(text) // CHARS_PER_TOKEN) for text in texts)


def last_turns(messages: Sequence[dict[str, str]], turns: int = HISTORY_TURNS) -> list[dict[str, str]]:
    """The last `turns` turns of the Thread, in full and unsummarised.

    A turn starts at a message from the User, so the cut is made at the
    `turns`-th User message from the end and the Coach's answers travel with
    the messages they answered.
    """
    starts = [i for i, message in enumerate(messages) if message.get("role") == "user"]
    if len(starts) <= turns:
        return list(messages)
    return list(messages[starts[-turns] :])


@dataclass(frozen=True)
class Fitted:
    """What fits, and whether it fitted."""

    history: list[dict[str, str]]
    passages: list[Any]
    tokens: int
    # True when the Turn is still over budget after trimming. It runs anyway,
    # and the node's `interaction_event` row carries this.
    over_budget: bool = False
    # What was cut, in the order it was cut, for the log line.
    trimmed: tuple[str, ...] = field(default_factory=tuple)


def render_history(messages: Sequence[dict[str, str]]) -> str:
    """The Thread as a prompt reads it. One line for each message, no summary."""
    return "\n".join(f"{m.get('role', '?')}: {m.get('text', '')}" for m in messages)


def _cost(keep: Sequence[str], history: Sequence[dict[str, str]], passages: Sequence[Any]) -> int:
    return estimate_tokens(
        *keep,
        render_history(history),
        *(getattr(p, "text", "") for p in passages),
    )


def fit(
    *,
    keep: Sequence[str],
    history: Sequence[dict[str, str]] = (),
    passages: Sequence[Any] = (),
    budget: int = BUDGET_TOKENS,
) -> Fitted:
    """Trim in the fixed order until the parts fit, or say they did not.

    `keep` is everything that may never be cut: the system prompt, the message
    being answered, the Profile, and today's Meals. Nothing in it is read for
    length by anything that could drop it — it is only measured.
    """
    tokens = _cost(keep, history, passages)
    if tokens <= budget:
        return Fitted(history=list(history), passages=list(passages), tokens=tokens)

    cut: list[str] = []
    history = last_turns(history)
    cut.append(f"history to the last {HISTORY_TURNS} turns")
    tokens = _cost(keep, history, passages)
    if tokens <= budget:
        return Fitted(list(history), list(passages), tokens, trimmed=tuple(cut))

    passages = list(passages)[:MAX_CHUNKS]
    cut.append(f"passages to {MAX_CHUNKS}")
    tokens = _cost(keep, history, passages)
    # Still over. The Turn proceeds with what is left — the Profile and today's
    # Meals among it — and the row says the budget was exceeded.
    return Fitted(list(history), list(passages), tokens, tokens > budget, tuple(cut))

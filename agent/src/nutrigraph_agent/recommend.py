"""What to eat next. **Code finds; the model ranks.** That order is the whole
slice.

1. **The gap is SQL.** Today's Meals are summed by the same statement the day
   review sums them with, and subtracted from the targets the Goal produces —
   `review.targets_for`, reused, because a second derivation of the same target
   is two Coaches disagreeing about the same User.
2. **The candidates are rows.** The local Filipino dish table, and the foods
   this User has already logged.
3. **The hard filters are in the query, not in a prompt.** Allergens from
   `user_profile.allergies`, `disliked_foods`, and diet-pattern conflicts read
   off `local_food.tags` — a food that fails one of those is not on the list
   before a model is asked anything.
4. **Only then does a model rank and explain.** It is shown the survivors and
   nothing else, and every food it names has to be one of them; a suggestion
   naming anything else is rejected here rather than shown to the User.

So the model never invents a food, and every suggestion traces back to a
database row. The allergy check that runs at the seam on this path is therefore
a second line of defence, and on a correct path it never fires.

**The personalisation is a similarity query**, not a phrase in a prompt: the
candidate ordering carries the cosine distance between each food and the
centroid of the foods this User actually ate or accepted, out of the
`food_embedding` table. A User with neither has no centroid, and the ordering
rests on the gap alone — which is the cold start, and it still produces a
Filipino suggestion that fits the Goal.

**This path writes prose for the User**, so by the routing rule it uses the
prose tier, inside the same retry ladder and token budget as every other call.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from .budget import fit
from .db import Candidate, Database, DayTotal
from .meal import MANILA, _list, day_bounds, stand_in
from .models import Profile, RankedFoods
from .providers import ModelCall, SchemaFailure, TurnModels
from .review import DERIVED, SAYS, Targets, amount, named, targets_for

log = logging.getLogger("nutrigraph.agent.recommend")

# The five a Coach suggests eating more of. Sodium is the sixth nutrient and it
# is deliberately not here: its target is a ceiling, so "closing the sodium gap"
# would mean recommending salt. It is still computed, still reported, and it
# never chooses the food.
RANKED = ("kcal", "protein_g", "fat_g", "carb_g", "fibre_g")

# How many surviving candidates the model is shown. Enough to rank, few enough
# that the whole list fits in the prompt beside the gap and the Profile.
CANDIDATES = 12

# Below this a shortfall is not worth building a meal around, and the Coach says
# the day is on target instead of inventing a reason to eat.
GAP_FLOOR = {"kcal": 50.0, "protein_g": 5.0, "fat_g": 5.0, "carb_g": 10.0, "fibre_g": 2.0}

# What a diet pattern rules out, by the tags the dish table already carries and
# by the FoodData Central category for an item that carries none. This is the
# whole of the diet filter, and it is applied in SQL: a pattern the table does
# not name rules nothing out rather than guessing.
DIET_CONFLICTS: dict[str, tuple[str, ...]] = {
    "vegan": (
        "pork", "beef", "poultry", "chicken", "offal", "sausage", "shellfish",
        "shrimp-paste", "dairy", "fish", "seafood", "meat",
    ),
    "vegetarian": (
        "pork", "beef", "poultry", "chicken", "offal", "sausage", "shellfish",
        "shrimp-paste", "fish", "seafood", "meat",
    ),
    "pescatarian": ("pork", "beef", "poultry", "chicken", "offal", "sausage", "meat"),
    "halal": ("pork", "alcohol"),
    "omnivore": (),
}

RANK_SYSTEM = """You are a nutrition Coach telling one User what to eat next,
and you choose from the candidate foods you are given and from nothing else.

Every candidate has already been checked against this User's allergies, their
disliked foods and their diet pattern, so anything on the list is allowed and
anything not on it is forbidden. Name only candidates, copied exactly as they
are written. Naming a food that is not on the list is rejected, and the User
then reads a suggestion written without you.

Pick the one or two that best close the nutrient gap you are given. Say what to
eat in at most two short sentences, and give one sentence of reason in terms of
that gap. Assert no nutrition fact of your own and invent no number: the numbers
you were given are the only ones there are.

Address the User by the placeholder you are given, written exactly as it
appears."""

NOTHING_TO_SUGGEST = (
    "{name}, everything I hold is either on your allergy list, on your disliked "
    "list or against your diet pattern, so I have nothing to suggest yet. Tell "
    "me a food you do eat and I will work from that."
)

ON_TARGET = "You are on target for today, so this is about eating well rather than filling a gap."

NO_TARGETS = (
    "I cannot work out your targets until I know {missing}, so this is a "
    "suggestion from what you eat rather than from a gap."
)

COLD_START = (
    "You have logged nothing yet, so this comes from your Goal, your diet "
    "pattern and the Filipino dishes I hold, and not from what you have eaten."
)


class InventedFood(ValueError):
    """The model named a food that was not on the candidate list. The
    suggestion is rejected; the Coach answers from the rows instead."""


@dataclass(frozen=True)
class Gap:
    """Today's shortfall against the targets the Goal produces.

    `nutrient` is the one the gap is largest on, and it is what the candidate
    query orders by. It is None when nothing is short — a day already on target
    — or when the Profile cannot produce targets at all, and the two are
    different facts, which `targets.missing` tells apart.
    """

    values: dict[str, float] = field(default_factory=dict)
    nutrient: str | None = None
    amount: float = 0.0
    targets: Targets = field(default_factory=Targets)
    total: DayTotal = field(default_factory=DayTotal)
    # The nutrients there is a target for and no measurement of: the day holds
    # counted Items and the source printed nothing for this column on them. The
    # gap is unknown rather than whole, so it does not choose the food, and the
    # answer says so.
    unmeasured: list[str] = field(default_factory=list)

    @property
    def logged(self) -> bool:
        return self.total.counted > 0 or self.total.not_counted > 0


@dataclass
class Recommendation:
    """What one `recommend` produced: the sentences, what the call cost, and the
    foods named — which is what the `recommendation` row records and what both
    measurement signals read.

    Not a `CoachReply`. One node builds that, for whatever Intents the Turn ran.
    """

    text: str
    call: ModelCall | None
    foods: list[str]
    disclaimers: list[str] = field(default_factory=list)
    recommendation_id: UUID | None = None
    # How these sentences read again without a food the prose scan struck out.
    # The composer asks for it exactly once.
    again: Callable[[Sequence[str]], tuple[str, list[str]]] | None = None
    # The prompt was still over the token budget after trimming. It ran anyway;
    # the `interaction_event` row says so.
    over_budget: bool = False


def gap_for(targets: Targets, total: DayTotal) -> Gap:
    """Target minus eaten, for every nutrient both are known for.

    One SQL sum produced the totals and `review.targets_for` produced the
    targets; nothing here derives either a second time.

    A null is not a zero, and here the difference decides what to suggest. A day
    with nothing counted has eaten none of anything, so the gap is the whole
    target — that is the cold start, and it is a fact. A day that holds counted
    Items whose source printed no fibre has an *unknown* fibre gap, not a full
    one, so that nutrient is left out of the ranking rather than sending the
    Coach after the biggest number on the page.
    """
    values: dict[str, float] = {}
    unmeasured: list[str] = []
    for column, target in targets.values.items():
        if column in total.values:
            values[column] = target - total.values[column]
        elif total.counted == 0:
            values[column] = target
        else:
            unmeasured.append(column)
    short = {
        column: value
        for column, value in values.items()
        if column in RANKED and value >= GAP_FLOOR[column]
    }
    if not short:
        return Gap(values=values, targets=targets, total=total, unmeasured=unmeasured)
    # The largest gap is the largest *share* of its own target: 1,200 kcal and
    # 90 g of protein are not comparable as numbers, and they are as fractions.
    nutrient = max(short, key=lambda column: short[column] / targets.values[column])
    return Gap(
        values=values,
        nutrient=nutrient,
        amount=short[nutrient],
        targets=targets,
        total=total,
        unmeasured=unmeasured,
    )


def _numbers(candidate: Candidate) -> str:
    values = candidate.per_100g
    return ", ".join(named(c, values[c]) for c in SAYS if c in values) or "no numbers"


def _for_the_model(candidates: Sequence[Candidate]) -> str:
    """The surviving candidates, as the model reads them. Nothing else about
    the User is here: the Profile did its work in the filter."""
    lines = []
    for candidate in candidates:
        line = f"{candidate.name} (per 100 g: {_numbers(candidate)}"
        if candidate.tags:
            line += f"; tags: {', '.join(candidate.tags)}"
        lines.append(line + ")")
    return "\n".join(lines)


def gap_sentence(gap: Gap) -> str:
    if gap.nutrient is None:
        return ON_TARGET
    return (
        f"The largest gap left today is {amount(gap.nutrient, gap.amount)} of "
        f"{SAYS[gap.nutrient][0]}."
    )


def check_foods(named_foods: Sequence[str], candidates: Sequence[Candidate]) -> list[str]:
    """Every food the model named, as the candidate row spells it.

    A name that is not a candidate raises. That is the point of the whole
    ordering: the model was shown the survivors, so a name from anywhere else
    was invented, and an invented food is exactly what this path exists to make
    impossible.
    """
    by_name = {candidate.name.strip().lower(): candidate.name for candidate in candidates}
    out: list[str] = []
    for food in named_foods:
        real = by_name.get(food.strip().lower())
        if real is None:
            raise InventedFood(food)
        out.append(real)
    return list(dict.fromkeys(out))


def from_rows(profile: Profile, gap: Gap, candidates: Sequence[Candidate]) -> tuple[str, list[str]]:
    """The suggestion the Coach gives when no model wrote one: the best
    candidate the ordering produced, named, with the gap it closes.

    Every word of it is a fact this module already holds, which is why it can be
    said at all — and why it can be said again without a food.
    """
    if not candidates:
        return NOTHING_TO_SUGGEST.format(name=profile.name), []
    best = candidates[0]
    if gap.nutrient is None:
        return (
            f"{profile.name}, {best.name} would suit you. {ON_TARGET}",
            [best.name],
        )
    per_100g = best.per_100g.get(gap.nutrient)
    closes = (
        f" A 100 g serving carries {amount(gap.nutrient, per_100g)} of it."
        if per_100g is not None
        else ""
    )
    return (
        f"{profile.name}, try {best.name} next. {gap_sentence(gap)}{closes}",
        [best.name],
    )


def markings_for(
    gap: Gap, named_foods: Sequence[str], candidates: Sequence[Candidate],
    *, without: Sequence[str] = ()
) -> list[str]:
    """What the composer may not drop.

    The dish table is verified and off limits, so nothing here adjusts a value —
    it says what the value is. A proxy row is a stand-in for the home-cooked
    dish and a calculated row is computed from components rather than measured,
    and a suggestion that presented either as a measurement would be the one
    dishonesty this table's transcription exists to prevent.
    """
    markings: list[str] = []
    chosen = {c.name: c for c in candidates if c.name in named_foods}
    for name, candidate in chosen.items():
        if candidate.value_kind == "proxy":
            markings.append(
                f"The {name} figures are a stand-in and not the dish itself: "
                f"{stand_in(candidate.source_note or '', without=without)}"
            )
        elif candidate.value_kind == "calculated":
            markings.append(
                f"The {name} figures are calculated from component foods rather "
                f"than measured, and are likely understated."
            )
    if gap.targets.values:
        markings.append(DERIVED)
    elif gap.targets.missing:
        markings.append(NO_TARGETS.format(missing=_list(gap.targets.missing)))
    if gap.unmeasured:
        markings.append(
            f"My source prints no {_list([SAYS[c][0] for c in gap.unmeasured])} for "
            f"part of what you ate, so I do not know that gap and did not choose "
            f"this on it."
        )
    if not gap.logged:
        markings.append(COLD_START)
    return markings


async def recommend(
    *,
    db: Database,
    turn: TurnModels,
    profile: Profile,
    turn_id: UUID,
    now: datetime,
) -> Recommendation:
    """One suggestion: the gap, the filtered rows, and only then a model."""
    targets = targets_for(profile)
    start, end = day_bounds(now.astimezone(MANILA))
    total = await db.day_total(profile.user_id, start=start, end=end)
    gap = gap_for(targets, total)

    candidates = await db.candidate_foods(
        profile.user_id,
        blocked=[*profile.allergies, *profile.disliked_foods],
        conflicts=DIET_CONFLICTS.get((profile.diet_pattern or "").strip().lower(), ()),
        nutrient=gap.nutrient,
        gap=gap.amount,
        limit=CANDIDATES,
    )
    log.info(
        "%d candidates survived the filters, largest gap %s",
        len(candidates),
        gap.nutrient or "none",
        extra={"user_id": profile.user_id, "turn_id": str(turn_id)},
    )

    call: ModelCall | None = None
    over_budget = False
    reason = gap_sentence(gap)
    if candidates:
        prompt = (
            f"The User is {profile.name}."
            f"\nTheir diet pattern: {profile.diet_pattern or 'not stated'}."
            f"\n{gap_sentence(gap)}"
            f"\n\nCandidates:\n{_for_the_model(candidates)}"
        )
        # The candidates are what the budget may trim here, and it trims them by
        # being given fewer: the list is already the shortest ordering of the
        # rows, so the instruction and the gap are what may never go.
        over_budget = fit(keep=[RANK_SYSTEM, prompt]).over_budget
        if over_budget:  # pragma: no cover - a candidate list this long is not one
            log.info("the recommend prompt is over budget", extra={"turn_id": str(turn_id)})
        try:
            # The prose tier, by the routing rule: the User reads this. The
            # ladder and the one schema retry are `compose`'s, so a rate-limit
            # stop falls to the weaker model rather than ending the Turn.
            written, call = await turn.compose(
                RankedFoods, system=RANK_SYSTEM, user=prompt
            )
            foods = check_foods(written.foods, candidates)
            reason = turn.restore(written.reason)
            text = f"{turn.restore(written.suggestion)} {reason}"
        except InventedFood as exc:
            log.warning(
                "the model named a food that was not a candidate; the suggestion "
                "was rejected",
                extra={"turn_id": str(turn_id), "food": str(exc)},
            )
            text, foods = from_rows(profile, gap, candidates)
        except SchemaFailure:
            log.warning("the ranker did not fill the schema twice",
                        extra={"turn_id": str(turn_id)})
            text, foods = from_rows(profile, gap, candidates)
    else:
        text, foods = from_rows(profile, gap, candidates)

    markings = markings_for(gap, foods, candidates)
    recommendation_id = None
    if foods:
        recommendation_id = await db.store_recommendation(
            user_id=profile.user_id,
            turn_id=turn_id,
            gap_nutrient=gap.nutrient,
            gap_amount=gap.amount if gap.nutrient else None,
            suggestion=text,
            reason=reason,
            foods=foods,
        )

    def again(without: Sequence[str]) -> tuple[str, list[str]]:
        # Said again from the rows, never from the model: this is the one
        # regeneration the allergy check forces, and asking a model for another
        # draft is how a Turn learns to loop. A candidate whose own name or tags
        # hold the struck-out word goes, which cannot normally happen — the SQL
        # filter removed the allergens before any of this ran.
        allowed = [
            c for c in candidates
            if not any(
                word.lower() in c.name.lower() or word.lower() in " ".join(c.tags).lower()
                for word in without
            )
        ]
        said, names = from_rows(profile, gap, allowed)
        marks = markings_for(gap, names, allowed, without=without)
        return " ".join([said, *(m for m in marks if m not in said)]), marks

    return Recommendation(
        text=" ".join([text, *(m for m in markings if m not in text)]),
        call=call,
        foods=foods,
        disclaimers=markings,
        recommendation_id=recommendation_id,
        again=again,
        over_budget=over_budget,
    )

"""How the day went: the totals, the targets, and the gap between them.

**The Goal produces the targets.** A Goal is a body-weight target, and the daily
energy and macronutrient targets are worked out from the Profile and that Goal.
The User supplies no target directly and calculates nothing. Every number is
metric, so it matches the packaging they read.

**The totals are one SQL sum** over the six numeric columns on `meal_item`,
filtered by `user_id` and the day through the B-tree index on
`meal (user_id, eaten_at)`. The JSONB nutrient snapshot is never re-parsed, and
a null is never a zero: a column the source did not print stays absent through
the sum and is reported as short rather than complete.

**A total that is missing something says so.** An Item with status `unmatched`
contributes nothing at all, so the review names those foods and states plainly
that the total is short by an unknown amount. This is the honesty rule the meal
logging slice applied to one Meal, applied now to a whole day.

**A day with no Meals is not a day of zeros.** Nothing logged is a different
fact from nothing eaten, and the review says the first rather than reporting the
second.

**The answer is assembled here, not by a model.** The one provider call this
path makes reads which day the User asked about; every sentence after that is a
fact this module holds, so a marking the User depends on cannot be dropped by a
model that decided to be brief.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .db import Database, DayTotal
from .food import COLUMNS
from .meal import MANILA, OFTEN_MISSING, _list, day_bounds
from .models import CoachReply, DayRequest, Profile, ReplyPart
from .providers import ModelCall, TurnModels

log = logging.getLogger("nutrigraph.agent.review")

WHICH_DAY_SYSTEM = """You read one message in which a User asks a nutrition Coach
how their eating went, and you report which day they mean.

days_ago is 0 for today, 1 for yesterday, 2 for the day before that, and so on.
Answer 0 when the message names no day: "how did I eat" is about today."""

# What each column is called when the Coach says it out loud, and its unit.
# Metric throughout, because that is what the User's packaging prints.
SAYS = {
    "kcal": ("energy", "kcal"),
    "protein_g": ("protein", "g"),
    "fat_g": ("fat", "g"),
    "carb_g": ("carbohydrate", "g"),
    "fibre_g": ("fibre", "g"),
    "sodium_mg": ("sodium", "mg"),
}

# The Mifflin-St Jeor resting energy equation (Am J Clin Nutr 1990;51:241-247),
# which needs weight in kilograms, height in centimetres, and age in years.
# The last constant is the only place sex enters it.
MALE, FEMALE = 5.0, -161.0
# A Profile whose sex is unset, or is not one of the two the equation was fitted
# on, takes the midpoint, and the answer says the estimate is not sex-specific
# rather than quietly picking one.
UNSTATED = (MALE + FEMALE) / 2
SEXES = {"m": MALE, "male": MALE, "f": FEMALE, "female": FEMALE}

# The standard activity multipliers that turn resting energy into a day's
# energy. Without one the target would be out by up to a third, so a Profile
# that does not state it produces no targets at all.
ACTIVITY = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725}

# What the targets cannot be worked out without, and how the Coach names each
# one to the User.
NEEDED = {
    "age": "your age",
    "height_cm": "your height",
    "weight_kg": "your weight",
    "activity_level": "how active you are",
}

# Moving body weight costs about 7,700 kcal a kilogram, and half a kilogram a
# week is the rate the Goal is spread over, which is 550 kcal a day.
KCAL_PER_KG = 7700.0
KG_PER_WEEK = 0.5
DAILY_SHIFT = KCAL_PER_KG * KG_PER_WEEK / 7.0
# Under this the Goal and the current weight are the same weight, and the target
# is maintenance.
SAME_WEIGHT_KG = 0.5

# Protein per kilogram of the weight the Goal names, at the upper end of what is
# used to hold lean mass while body weight moves.
PROTEIN_G_PER_KG = 1.6
# A quarter of the energy from fat, the rest of it from carbohydrate.
FAT_SHARE = 0.25
KCAL_PER_G = {"protein_g": 4.0, "fat_g": 9.0, "carb_g": 4.0}
# 14 g of fibre for every 1,000 kcal, which is how dietary guidance states it.
FIBRE_G_PER_1000_KCAL = 14.0
# The World Health Organization's sodium limit for an adult, in milligrams.
SODIUM_MG_LIMIT = 2000.0

# The four the Coach reads out in the summary line. Fibre and sodium follow in
# their own sentence, because they are the two the source most often leaves out.
HEADLINE = ("kcal", "protein_g", "fat_g", "carb_g")

NOTHING_LOGGED = (
    "you have logged nothing for {day}, so there is no day to review yet. "
    "Tell me what you ate and I will add it up."
)

DERIVED = (
    "Those targets come from your Profile and your Goal, so they are worked out "
    "rather than measured."
)


@dataclass(frozen=True)
class Targets:
    """The daily numbers the Goal produces, or what the Profile is missing.

    `values` is empty when the Profile cannot produce them, and `missing` then
    names what the Coach would need. Nothing here is ever a User-supplied
    target: the User states facts about themselves and a Goal, and these follow.
    """

    values: dict[str, float] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    # True when the resting-energy equation used a sex-specific constant. False
    # means the midpoint stood in, and the answer says so.
    sex_specific: bool = True


@dataclass
class DayReview:
    """One day, reviewed: what was counted, what it is measured against, and
    what the total does not include."""

    day: datetime
    totals: dict[str, float]
    targets: Targets
    # total minus target, for each nutrient both are known for. A nutrient with
    # no target has no entry, rather than a gap against nothing.
    target_delta: dict[str, float]
    # The foods that contributed nothing, by the words the User used.
    unmatched: list[str]
    total: DayTotal
    reply: CoachReply
    call: ModelCall

    @property
    def logged(self) -> bool:
        return self.total.counted > 0 or self.total.not_counted > 0


def resting_energy(profile: Profile) -> tuple[float, bool] | None:
    """Mifflin-St Jeor, and whether it used a sex-specific constant."""
    if profile.age is None or profile.height_cm is None or profile.weight_kg is None:
        return None
    offset = SEXES.get((profile.sex or "").strip().lower())
    base = (
        10.0 * profile.weight_kg
        + 6.25 * profile.height_cm
        - 5.0 * profile.age
        + (offset if offset is not None else UNSTATED)
    )
    return base, offset is not None


def targets_for(profile: Profile) -> Targets:
    """The daily targets, derived from the Profile and the Goal.

    A Profile that cannot produce them says which facts it is short of, which is
    a better answer than a target built on a guessed height.
    """
    missing = [word for name, word in NEEDED.items() if getattr(profile, name) is None]
    if profile.activity_level not in ACTIVITY:
        # A stated activity level the table does not hold is as good as unstated.
        missing = sorted(set(missing) | {NEEDED["activity_level"]})
    resting = resting_energy(profile)
    if missing or resting is None:
        return Targets(missing=missing)

    basal, sex_specific = resting
    assert profile.activity_level is not None and profile.weight_kg is not None
    energy = basal * ACTIVITY[profile.activity_level]

    # The Goal, spread over time. A deficit never takes the target below resting
    # energy, whatever the Goal says, because that is not a target this Coach
    # gives out.
    goal_weight = profile.target_weight_kg
    if goal_weight is not None and abs(goal_weight - profile.weight_kg) >= SAME_WEIGHT_KG:
        energy += DAILY_SHIFT if goal_weight > profile.weight_kg else -DAILY_SHIFT
    energy = max(energy, basal)

    # Protein is per kilogram of the weight the Goal names; the rest of the
    # energy is split between fat and carbohydrate.
    protein = PROTEIN_G_PER_KG * (goal_weight or profile.weight_kg)
    fat = FAT_SHARE * energy / KCAL_PER_G["fat_g"]
    carb = max(
        (energy - protein * KCAL_PER_G["protein_g"] - fat * KCAL_PER_G["fat_g"])
        / KCAL_PER_G["carb_g"],
        0.0,
    )
    return Targets(
        values={
            "kcal": energy,
            "protein_g": protein,
            "fat_g": fat,
            "carb_g": carb,
            "fibre_g": FIBRE_G_PER_1000_KCAL * energy / 1000.0,
            "sodium_mg": SODIUM_MG_LIMIT,
        },
        sex_specific=sex_specific,
    )


def amount(column: str, value: float) -> str:
    """One number as the User reads it, with its metric unit."""
    _, unit = SAYS[column]
    return f"{round(value):,} {unit}"


def named(column: str, value: float) -> str:
    return f"{amount(column, value)} of {SAYS[column][0]}"


def _gap(column: str, delta: float) -> str:
    """One nutrient's gap, in the direction it falls."""
    if round(abs(delta)) == 0:
        return f"{SAYS[column][0]} is on target"
    return f"{SAYS[column][0]} is {amount(column, abs(delta))} " + (
        "over" if delta > 0 else "under"
    )


def which_day(now: datetime, days_ago: int) -> datetime:
    return now.astimezone(MANILA) - timedelta(days=days_ago)


def day_word(days_ago: int, day: datetime) -> str:
    if days_ago == 0:
        return "today"
    if days_ago == 1:
        return "yesterday"
    # Not strftime's %-d: the flag that drops the leading zero is not portable,
    # and this image is built on Linux while the tests also run on Windows.
    return f"on {day.day} {day:%B}"


def compose(
    profile: Profile,
    *,
    day: str,
    total: DayTotal,
    targets: Targets,
    delta: dict[str, float],
) -> str:
    """The answer. It reports the totals and the gap together, names every food
    that contributed nothing, and never lets an incomplete total read as
    complete."""
    if total.counted == 0 and total.not_counted == 0:
        return f"{profile.name}, " + NOTHING_LOGGED.format(day=day)

    lines: list[str] = []
    if total.counted:
        counted = [named(c, total.values[c]) for c in HEADLINE if c in total.values]
        lines.append(f"{profile.name}, {day} you have {_list(counted)}.")
        rest = [
            named(c, total.values[c]) for c in ("fibre_g", "sodium_mg")
            if c in total.values
        ]
        if rest:
            lines.append(f"That is with {_list(rest)}.")
    else:
        # Nothing matched, so there is no total for the unmatched sentence below
        # to be short against. The foods are named here instead, because naming
        # them is the part that must not be lost.
        they = "were" if len(total.unmatched) > 1 else "was"
        lines.append(
            f"{profile.name}, the only thing you logged {day} {they} "
            f"{_list(total.unmatched)}, which I could not match to a food I hold, "
            f"so I have no total to give you at all."
        )

    if delta:
        lines.append(
            "Against your targets, " + _list([_gap(c, delta[c]) for c in COLUMNS
                                              if c in delta]) + "."
        )
        lines.append(DERIVED)
        if not targets.sex_specific:
            lines.append(
                "Your Profile does not state a sex, so the energy target does not "
                "use a sex-specific constant and is an estimate either way."
            )
    elif targets.missing:
        lines.append(
            f"I cannot work out your targets until I know {_list(targets.missing)}, "
            f"so that is the total on its own and no gap."
        )

    if total.unmatched and total.counted:
        they = "they are" if len(total.unmatched) > 1 else "it is"
        lines.append(
            f"That total does not include {_list(total.unmatched)}, because I could "
            f"not match {'them' if len(total.unmatched) > 1 else 'it'} to a food I "
            f"hold, so {they} logged but uncounted and the total above is short by "
            f"an unknown amount."
        )

    short = sorted({
        word for column, word in OFTEN_MISSING.items() if not total.complete(column)
    })
    if short:
        lines.append(
            f"My source prints no {_list(short)} for part of what you ate, so those "
            f"totals are short rather than complete."
        )
    return " ".join(lines)


async def review_day(
    *,
    db: Database,
    turn: TurnModels,
    profile: Profile,
    message: str,
    now: datetime,
) -> DayReview:
    """One question, one day. One call reads which day, and the rest is SQL.

    The Meals come from the database, never from the Thread's checkpoint, so a
    question about an earlier day is answered from the same place as today's.
    """
    asked, call = await turn.fill(DayRequest, system=WHICH_DAY_SYSTEM, user=message)
    day = which_day(now, asked.days_ago)
    start, end = day_bounds(day)
    total = await db.day_total(profile.user_id, start=start, end=end)
    targets = targets_for(profile)
    delta = {
        column: total.values[column] - targets.values[column]
        for column in COLUMNS
        if column in targets.values and column in total.values
    }
    log.info(
        "reviewed %s: %d counted, %d uncounted",
        start.date(),
        total.counted,
        total.not_counted,
        extra={"user_id": profile.user_id},
    )
    text = compose(
        profile,
        day=day_word(asked.days_ago, day),
        total=total,
        targets=targets,
        delta=delta,
    )
    return DayReview(
        day=start,
        totals=dict(total.values),
        targets=targets,
        target_delta=delta,
        unmatched=list(total.unmatched),
        total=total,
        reply=CoachReply(text=text, parts=[ReplyPart(intent="review_day", text=text)]),
        call=call,
    )

"""One sentence becomes a Meal: parse it, match the foods, write it down.

**The order, from issue #15.** The local Filipino dish table first, where an
exact alias wins at once and the text-pattern index makes the lookup cheap,
because it runs before every FoodData Central call. Otherwise the FoodData
Central search endpoint, taking the top ten candidates, and then one
schema-constrained choice from those candidates, recorded in the trace. A
three-food Meal costs about six calls, which is fine at demo volume.

**Nothing the User said is lost.** A food that matches nothing is stored with
status `unmatched` and no nutrient values, and the Meal is written anyway. The
answer names what was counted and what was not and invites a correction, and the
Turn does not stop — `clarify` is the only interrupt in the graph, and meal
logging is not it.

**What the numbers are is said, not implied.** A `proxy` row is a canned or
commercial product standing in for the home-cooked dish, and a `calculated` row
is computed from components with a stated understatement; both are marked in the
answer. A null is never a zero: fibre and sodium are null wherever the source
prints nothing, and an absent value stays absent through the scaling, the
column, and the day total.

**The answer is assembled here, not by a model.** Every sentence of it is a fact
this module already holds — what matched, from where, and what kind of value it
is — so a marking the User depends on cannot be dropped by a model that decided
to be brief.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from .db import Database, DayTotal, MealItemRow, UnmatchedItem
from .food import CANDIDATES, FoodCandidate, FoodSearch
from .guardrail import scan_reply
from .models import FoodChoice, MealType, ParsedItem, ParsedMeal, Profile
from .providers import ModelCall, TurnModels

log = logging.getLogger("nutrigraph.agent.meal")

# The Philippines has kept UTC+8 without daylight saving since 1899, so the one
# offset is the whole rule and there is no timezone database to ship in the
# image. A Meal Type read off the clock, and a day boundary, both come from here.
MANILA = timezone(timedelta(hours=8), "PST")

# The Meal Type the clock gives when the User's own words do not name one.
BY_THE_CLOCK: tuple[tuple[int, MealType], ...] = (
    (4, "snack"), (11, "breakfast"), (15, "lunch"), (21, "dinner"), (24, "snack"),
)

FROM_WORDS = "the User's own words"

# Grams for one of something, when the User counted rather than weighed. It is
# declared, not sourced: FNRI publishes no serving weights, and issue #26 records
# that even the Pinggang Pinoy serving grams could not be found. So the Coach
# says the portion was assumed rather than pretending to know it.
#
# ponytail: one number for every food. Per-dish serving weights are worth having
# when there is a source to take them from, and there is not one yet.
DECLARED_SERVING_G = 100.0

# The units that are a weight, and what one of them is in grams. Anything else —
# piece, slice, cup, bowl, plate — is a count, and a count is an assumption.
MASS_UNITS = {
    "g": 1.0, "gram": 1.0, "grams": 1.0, "gm": 1.0,
    "kg": 1000.0, "kilo": 1000.0, "kilos": 1000.0,
    "kilogram": 1000.0, "kilograms": 1000.0,
    # A millilitre of soup is not a gram of soup, but at the precision of a
    # declared demo portion it is nearer than calling it one serving.
    "ml": 1.0, "milliliter": 1.0, "millilitre": 1.0,
}

# How far back a correction reaches. A message that fixes a food names one the
# User logged today, not one from last week.
CORRECTION_WINDOW = timedelta(hours=24)

# Below this the parse is not confident the words name a food at all, so the
# entry is dropped rather than logged as something the User did not eat.
FOOD_FLOOR = 0.4

# The two columns the source leaves empty often enough for the Coach to have to
# say so, rather than let a short total read as a complete one.
OFTEN_MISSING = {"fibre_g": "fibre", "sodium_mg": "sodium"}

TELL_ME = "Tell me if I got any of that wrong and I will fix it."

NOTHING_TO_LOG = "I did not catch a food in that."

PARSE_SYSTEM = """You read one message in which a User tells a nutrition Coach
what they ate or drank, and you list the foods.

One entry for each food. Split no further than the User did: "chicken adobo" is
one food, not chicken and adobo.

quantity is the number the User said, and unit is the word it counts — piece,
slice, cup, bowl, plate, serving, g, ml. Leave quantity null when the User gave
none. Never invent one.

meal_type is breakfast, lunch, dinner or snack only when the User's own words say
which. Leave it null otherwise: the time of day decides it then, and that is not
your job.

corrects is for a message that fixes a food already logged. Leave it null unless
the message plainly does that."""

CHOOSE_SYSTEM = """You pick the one food that matches what a User ate, from the
FoodData Central candidates given to you.

Answer with that candidate's fdc_id, copied exactly, and a short reason naming
what made it the match. Prefer the plain cooked food over a branded product, and
the form the User described.

Answer with a null fdc_id when none of the candidates is that food. A wrong match
is worse than no match: the Coach then says plainly that it could not count the
food and asks the User, which is a good answer, while a wrong one puts the wrong
numbers into their day and says nothing about it."""


@dataclass(frozen=True)
class Match:
    """What the matcher found, whichever source found it."""

    source: str
    food_name: str
    per_100g: dict[str, float]
    nutrients: dict[str, Any]
    # Why this candidate, and by what. The same string goes to the trace and to
    # `meal_item.match_note`, so a match is auditable from either end.
    note: str
    local_food_id: UUID | None = None
    fdc_id: str | None = None
    value_kind: str | None = None
    source_note: str | None = None


@dataclass(frozen=True)
class Attempt:
    """One food, looked for. `why` is filled in when nothing was found, and it
    is kept on the Item: a food nothing matched and a food FoodData Central
    could not be asked about are both uncounted, and the record says which."""

    match: Match | None = None
    call: ModelCall | None = None
    why: str | None = None


@dataclass
class Logged:
    """One Item as it was written, beside what it was matched to. The Match is
    what lets the answer say what kind of value the numbers are."""

    row: MealItemRow
    match: Match | None = None
    # The Item this one corrected, when the message was a correction.
    corrected: UUID | None = None

    @property
    def counted(self) -> bool:
        return self.row.status == "matched"


@dataclass
class MealLog:
    """What one `log_meal` produced: the sentences, what the calls cost, and the
    Items, which the caller stores and a test reads.

    Not a `CoachReply`. One node builds that, for whatever Intents the Turn ran,
    and this module is one of them — so what comes back here is the finished
    sentences and the markings within them, and the composer decides how the
    User reads them.
    """

    text: str
    call: ModelCall
    items: list[Logged]
    # The markings the composer may not drop: a stand-in value, a calculated
    # one, an assumed portion, a total the source left short.
    disclaimers: list[str] = field(default_factory=list)
    meal_id: UUID | None = None


def normalize(text: str) -> str:
    """The one spelling a name is looked up under, here and in the seed.

    Everything that is not a letter or a digit becomes a space, so 'Kare-kare
    (beef)' and 'kare kare beef' share a key and the prefix index can join them.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def meal_type_for(parsed: ParsedMeal, at: datetime) -> tuple[str, str]:
    """The Meal Type, and where it came from: the User's words first, the clock
    after. Never a list the User picks from."""
    if parsed.meal_type is not None:
        return parsed.meal_type, FROM_WORDS
    hour = at.astimezone(MANILA).hour
    for until, meal_type in BY_THE_CLOCK:
        if hour < until:
            return meal_type, f"the time, {hour:02d}:00 in Manila"
    raise AssertionError("the clock table covers every hour")  # pragma: no cover


def grams_for(item: ParsedItem) -> tuple[float | None, bool]:
    """What the nutrient columns are scaled from, and whether it was assumed.

    A weight the User stated is used as it stands. A count is multiplied by the
    declared serving, and the answer says so, because nothing published gives a
    serving weight for these dishes.
    """
    quantity = item.quantity if item.quantity is not None else 1.0
    unit = (item.unit or "").strip().lower()
    if unit in MASS_UNITS:
        return quantity * MASS_UNITS[unit], False
    return quantity * DECLARED_SERVING_G, True


def scale(per_100g: dict[str, float], grams: float) -> dict[str, float]:
    """The six numbers for the portion. A nutrient the source did not state is
    absent here, and stays absent all the way to the null column."""
    return {name: value * grams / 100.0 for name, value in per_100g.items()}


def _candidates(candidates: list[FoodCandidate]) -> str:
    return "\n".join(f"{c.fdc_id}: {c.description}" for c in candidates)


async def match_food(
    name: str, *, db: Database, food: FoodSearch, turn: TurnModels
) -> Attempt:
    """The local table, then the lookup cache, then FoodData Central, then one
    choice from what it returned.

    The cache is keyed on the exact lowercased name and holds the food that name
    matched, so the second 'pandesal' anyone logs skips both the FoodData
    Central search and the Gemini choice. It is independent of the User by
    construction — the key is a food word, and the value is a public food record
    — so nothing that read the Profile or today's Meals goes into it, and the
    portion, the grams and the Meal are still worked out for this User alone.

    Nothing raises out of here. A food data source that is down, rate-limited or
    answering with an error leaves the food uncounted and says so, because the
    Meal is written either way and losing what the User said would be the worse
    failure by far.
    """
    key = normalize(name)
    local = await db.match_local_food(key)
    if local is not None:
        # The local table answered, so no FoodData Central call is made at all.
        return Attempt(
            match=Match(
                source="local",
                food_name=local.name,
                per_100g=local.per_100g,
                nutrients=local.source,
                note=f"the local Filipino dish table, matched on {key!r}",
                local_food_id=local.local_food_id,
                value_kind=local.value_kind,
                source_note=local.source_note,
            )
        )

    cached = await db.cached_food_match(key)
    if cached is not None:
        log.info("the lookup cache answered for %r", key)
        return Attempt(match=Match(**cached))

    try:
        candidates = await food.search(name, limit=CANDIDATES)
    except Exception as exc:
        log.warning("FoodData Central did not answer for %r: %s", name, exc)
        return Attempt(why=f"FoodData Central could not be asked: {type(exc).__name__}")
    if not candidates:
        return Attempt(why="FoodData Central holds no candidate for it")

    choice, call = await turn.fill(
        FoodChoice,
        system=CHOOSE_SYSTEM,
        user=f"The User ate: {name}\n\nCandidates:\n{_candidates(candidates)}",
    )
    chosen = next((c for c in candidates if c.fdc_id == choice.fdc_id), None)
    if chosen is None:
        # A null choice is the safer answer, not a failure: a wrong match puts
        # the wrong numbers in the User's day and says nothing about it.
        return Attempt(
            call=call,
            why=f"none of the {len(candidates)} candidates was it: {choice.reason}",
        )
    match = Match(
        source="fdc",
        food_name=chosen.description,
        per_100g=chosen.per_100g,
        nutrients=chosen.source,
        note=f"FoodData Central {chosen.fdc_id} '{chosen.description}', chosen "
        f"from {len(candidates)} candidates: {choice.reason}",
        fdc_id=chosen.fdc_id,
    )
    # Only a match is written. A food nothing matched is looked for again next
    # time, because the catalogue may have gained it and a cached 'no' would
    # keep the Coach from ever counting it.
    await db.store_cached_food_match(key, asdict(match))
    return Attempt(match=match, call=call)


def item_row(item: ParsedItem, attempt: Attempt, *, ordinal: int) -> MealItemRow:
    """One Item, matched or not. An unmatched Item keeps what the User said and
    carries no nutrient value at all — not a zero, and not a guess."""
    match = attempt.match
    if match is None:
        return MealItemRow(
            ordinal=ordinal,
            said_as=item.name,
            status="unmatched",
            quantity=item.quantity,
            unit=item.unit,
            # Why it was not counted, so an unmatched Item is auditable too.
            match_note=attempt.why,
        )
    grams, assumed = grams_for(item)
    return MealItemRow(
        ordinal=ordinal,
        said_as=item.name,
        status="matched",
        quantity=item.quantity,
        unit=item.unit,
        grams=grams,
        portion_assumed=assumed,
        source=match.source,
        local_food_id=match.local_food_id,
        fdc_id=match.fdc_id,
        food_name=match.food_name,
        value_kind=match.value_kind,
        match_note=match.note,
        nutrients=match.nutrients,
        values=scale(match.per_100g, grams),
    )


def _corrected(item: ParsedItem, open_items: list[UnmatchedItem]) -> UUID | None:
    """The Item this entry fixes, if it fixes one. The parse names the food word
    it is correcting; the identifier is looked up here, so a model cannot name a
    row that is not this User's."""
    if not item.corrects:
        return None
    wanted = normalize(item.corrects)
    found = next((o for o in open_items if normalize(o.said_as) == wanted), None)
    return found.meal_item_id if found else None


def stand_in(note: str) -> str:
    """What a proxy row says it actually is, in a form the reply scan will pass.

    The first sentence of the note carries it; the rest is the transcription
    trail, which the User does not need to read while cooking.

    The note is FNRI's wording, not this codebase's, and the two vocabularies
    collide: PhilFCT calls tapa the 'dried/cured jerky form', and `cured` is a
    word the medical-claim scan reads. A finished answer that fails that scan is
    replaced whole, so quoting such a sentence would take the marking down with
    the rest of the answer — the one thing that must never be lost. Where it
    would, the entry is named instead, which still says what stood in.
    """
    body = re.sub(r"^[A-Z]+:\s*", "", note.strip())
    sentence = body.split(". ")[0].rstrip(".") + "."
    if scan_reply(sentence) is None:
        return sentence
    named = re.search(r"'[^']+'", sentence)
    return f"the PhilFCT entry {named.group(0)}." if named else "a different food."


def _list(names: list[str]) -> str:
    if len(names) <= 1:
        return names[0] if names else ""
    return f"{', '.join(names[:-1])} and {names[-1]}"


def compose(profile: Profile, meal_type: str, items: list[Logged]) -> tuple[str, list[str]]:
    """The answer, and the markings within it that may not be lost.

    It names what was counted and what was not, marks a value that is not a
    direct measurement, says when the portion was assumed rather than stated,
    and invites a correction.

    The markings come back a second time as their own list because this answer
    is not always the last word: on a two-Intent Turn the composer joins it to
    another part, and a marking dropped there would be a wrong number presented
    as a right one. The composer carries them as disclaimers and puts back any
    that did not survive, so they cannot be lost to a model being brief.
    """
    corrections = [i for i in items if i.corrected is not None]
    counted = [i for i in items if i.counted and i.corrected is None]
    missed = [i for i in items if not i.counted and i.corrected is None]

    lines: list[str] = []
    if corrections:
        lines.append(
            f"{profile.name}, I counted {_list([i.row.said_as for i in corrections])} "
            f"against the meal you logged earlier."
        )
    if counted:
        who = "I" if lines else f"{profile.name}, I"
        lines.append(f"{who} logged {meal_type}: {_list([i.row.said_as for i in counted])}.")
    if missed:
        names = _list([i.row.said_as for i in missed])
        they = "they are" if len(missed) > 1 else "it is"
        lines.append(f"I could not match {names} to a food I hold, so {they} saved but not counted.")
    if not lines:
        lines.append(f"{profile.name}, {NOTHING_TO_LOG}")

    markings: list[str] = []
    for item in items:
        match = item.match
        if match is None:
            continue
        if match.value_kind == "proxy":
            markings.append(
                f"The {item.row.said_as} figures are a stand-in and not the dish "
                f"itself: {stand_in(match.source_note or '')}"
            )
        elif match.value_kind == "calculated":
            markings.append(
                f"The {item.row.said_as} figures are calculated from component "
                f"foods rather than measured, and are likely understated."
            )

    if any(i.row.portion_assumed for i in items):
        markings.append(
            f"Where you did not give a weight I counted {int(DECLARED_SERVING_G)} g "
            f"for one serving, which is my assumption and not a measurement."
        )

    thin = sorted({
        word
        for i in items if i.counted
        for column, word in OFTEN_MISSING.items()
        if column not in i.row.values
    })
    if thin:
        markings.append(
            f"My source prints no {_list(thin)} for part of that, so those totals "
            f"are short rather than complete."
        )

    return " ".join([*lines, *markings, TELL_ME]), markings


async def log_meal(
    *,
    db: Database,
    food: FoodSearch,
    turn: TurnModels,
    profile: Profile,
    message: str,
    turn_id: UUID,
    now: datetime,
) -> MealLog:
    """One message, one Meal. One parse call, then one match for each food.

    The Meal is written whatever the matching found, and the Turn is never
    stopped to ask about a food: an unmatched Item is named in the answer, which
    invites the correction that a later message can make.
    """
    open_items = await db.open_unmatched_items(
        profile.user_id, since=now - CORRECTION_WINDOW
    )
    asked = (
        "\n\nThe foods this User logged that the Coach could not match: "
        + ", ".join(repr(o.said_as) for o in open_items)
        if open_items
        else ""
    )
    parsed, call = await turn.fill(ParsedMeal, system=PARSE_SYSTEM + asked, user=message)
    meal_type, why = meal_type_for(parsed, now)

    items: list[Logged] = []
    ordinal = 0
    for parsed_item in parsed.items:
        if parsed_item.confidence < FOOD_FLOOR:
            log.info("dropped %r: not confidently a food", parsed_item.name)
            continue
        attempt = await match_food(parsed_item.name, db=db, food=food, turn=turn)
        if attempt.call is not None:
            call = call + attempt.call
        corrects = _corrected(parsed_item, open_items)
        row = item_row(parsed_item, attempt, ordinal=ordinal)
        if corrects is None:
            ordinal += 1
        # The trace. `interaction_event` records the node and what it cost;
        # this line and `meal_item.match_note` record which candidate won and
        # why, so a match is auditable from either end.
        log.info(
            "matched %r: %s",
            parsed_item.name,
            attempt.match.note if attempt.match else f"not counted, {attempt.why}",
            extra={"turn_id": str(turn_id), "food": parsed_item.name},
        )
        items.append(Logged(row=row, match=attempt.match, corrected=corrects))

    log.info("Meal Type %r from %s", meal_type, why, extra={"turn_id": str(turn_id)})

    meal_id = None
    fresh = [i for i in items if i.corrected is None]
    if fresh:
        meal_id = await db.store_meal(
            user_id=profile.user_id,
            turn_id=turn_id,
            eaten_at=now,
            meal_type=meal_type,
            items=[i.row for i in fresh],
        )
    for item in items:
        if item.corrected is not None:
            await db.correct_meal_item(item.corrected, item.row)

    text, markings = compose(profile, meal_type, items)
    return MealLog(
        text=text, call=call, items=items, disclaimers=markings, meal_id=meal_id
    )


def day_bounds(day: datetime) -> tuple[datetime, datetime]:
    """The day a Meal counts against, in Manila, where the User ate it."""
    local = day.astimezone(MANILA)
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


# What each summed column is called when the day is read back, and the unit it
# is read back in. `kcal` is its own unit, which is why this is not a bare map.
DAY_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("kcal", "", " kcal"),
    ("protein_g", " g", " protein"),
    ("fat_g", " g", " fat"),
    ("carb_g", " g", " carbohydrate"),
    ("fibre_g", " g", " fibre"),
    ("sodium_mg", " mg", " sodium"),
)

NOTHING_COUNTED_TODAY = "Nothing has been counted for today yet."


def day_line(total: DayTotal) -> str:
    """The day so far, as one line of fact for whatever reads it next.

    This is read after the Meal is written, so the numbers include it. That
    ordering is the whole point of the two-Intent Turn: a question about the
    day, asked in the same breath as the Meal, is answered from a total that
    already holds it.

    A column the source did not print is named as short rather than summed into
    a number that looks complete, which is the same rule `DayTotal.missing`
    exists for.
    """
    if not total.values:
        return NOTHING_COUNTED_TODAY
    numbers = ", ".join(
        f"{total.values[column]:.1f}{unit}{word}"
        for column, unit, word in DAY_COLUMNS
        if column in total.values
    )
    line = f"The day so far, this Meal included: {numbers}."
    short = [
        word.strip()
        for column, _, word in DAY_COLUMNS
        if column in total.values and not total.complete(column)
    ]
    if short:
        line += (
            f" The {_list(short)} total is short rather than complete: my source "
            f"prints none for part of what was counted."
        )
    if total.not_counted:
        eaten = "food" if total.not_counted == 1 else "foods"
        line += f" {total.not_counted} {eaten} logged today could not be counted at all."
    return line

# NutriGraph

A conversational nutrition coach. It logs what a person eats, answers nutrition questions from a curated Corpus, reviews the logged day, and suggests changes that fit the person's Goal.

## Language

### The person and the coach

**User**:
The person who talks to the Coach. In this effort a User is always a demo account; no real personal health data is stored.
_Avoid_: Client, patient, customer

**Coach**:
What the User talks to. The Coach logs Meals, answers nutrition questions from the Corpus, reviews the logged day, and gives Recommendations. The Coach is not a dietitian, a doctor, or a therapist; that limit is what produces a Refusal.
_Avoid_: Assistant, bot, nutritionist

**Profile**:
The stable facts the Coach holds about a User: age, sex, height, weight, target weight, activity level, allergies, diet pattern, disliked foods, and unit system.
_Avoid_: Settings, preferences, account

**Goal**:
What the Profile is measured against. In this effort a Goal is a body-weight target, and the daily energy and macronutrient targets are derived from it.
_Avoid_: Objective. Use "Goal" for the thing itself; "target" is only ever one of its numbers, as in target weight or daily protein target.

### The food log

**Meal**:
One eating occasion. It has a time, a Meal Type, and one or more Items. A single message such as "two eggs and toast" produces one Meal.
_Avoid_: Entry, log, record

**Meal Type**:
The kind of eating occasion: breakfast, lunch, dinner, or snack.

**Item**:
One food inside a Meal, with a quantity. An Item names what the User said. When the Coach can attach that name to a Food, the Item also carries the nutrient values for that quantity. When it cannot, the Item is kept anyway, and the Coach says it was not counted.
_Avoid_: Food, ingredient, line

**Food**:
A single food with known nutrient values, held either in an external food data source or in the local table of Filipino dishes that this effort maintains. An Item points at a Food; a Food is not owned by any User.

**Value Kind**:
How a Food's nutrient values were arrived at: measured for that Food, borrowed from a comparable commercial product, or calculated from component foods. The Coach states the Value Kind whenever it is not a direct measurement, so a User is never given a canned product's numbers as if they were a home-cooked dish.
_Avoid_: Quality, accuracy, confidence

### The conversation

**Thread**:
The single continuous conversation between one User and the Coach. There is exactly one Thread for each User, and it never restarts.
_Avoid_: Chat, conversation, history

**Session**:
One connection to a Thread. A Session ends; the Thread continues.
_Avoid_: Login, visit

**Turn**:
One User message and the Coach's complete answer to it. A Turn is the unit that is classified, checked, measured, and traced, and a Session holds many Turns.
_Avoid_: Request, exchange, round, message pair

**Intent**:
The classified purpose of a User message. One message carries at most two Intents, and their order matters, because the second reads what the first produced.

### The answers

**Recommendation**:
A proposed change to what the User eats next, produced from the Profile, the Goal, and the logged Meals. It names only Candidates, it always carries a reason, and it is measured two ways: whether the User accepted it, and whether a Meal holding one of its foods appeared within a day.
_Avoid_: Suggestion, tip, advice. "Suggestion" is only ever the sentence inside a Recommendation, never the Recommendation itself.

**Candidate**:
One Food a Recommendation is allowed to name. Candidates are found by query — from the local Filipino dish table and from the Foods this User has already logged — with the allergies, the disliked foods and the diet-pattern conflicts already removed. A model ranks Candidates and explains the choice; it never adds one.
_Avoid_: Option, choice, shortlist

**Refusal**:
The Coach's answer when a request falls outside its job. A Refusal states the boundary, gives a disclaimer, and points to a professional.
_Avoid_: Rejection, error

**Clarification**:
The Coach's answer when it cannot classify a message with enough confidence. A Clarification asks one short question and ends the Turn. It is the only point at which the Coach stops and waits for the User.
_Avoid_: Follow-up, disambiguation, prompt

**Citation**:
The pointer from a claim in an answer to the Corpus document, and the place within it, that supports the claim. A nutrition claim without a Citation is not an answer the Coach may give.
_Avoid_: Reference, source, link

**Corpus**:
The curated set of nutrition documents that the Coach cites. It is public guidance, not user data.
_Avoid_: Knowledge base, docs

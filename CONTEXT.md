# NutriGraph

A conversational nutrition coach. It logs what a person eats, answers nutrition questions from a curated corpus, reviews the logged day, and suggests changes that fit the person's goal.

## Language

### The person

**User**:
The person who talks to the coach. In this effort a User is always a demo account; no real personal health data is stored.
_Avoid_: Client, patient, customer

**Profile**:
The stable facts the coach holds about a User: age, sex, height, weight, target weight, activity level, allergies, diet pattern, disliked foods, and unit system.
_Avoid_: Settings, preferences, account

**Goal**:
The target the Profile is measured against. In this effort a Goal is a body-weight target, with the derived daily energy and macronutrient targets.
_Avoid_: Target, objective

### The food log

**Meal**:
One eating occasion. It has a time, a Meal Type, and one or more Items. A single message such as "two eggs and toast" produces one Meal.
_Avoid_: Entry, log, record

**Meal Type**:
The kind of eating occasion: breakfast, lunch, dinner, or snack.

**Item**:
One food inside a Meal, with a quantity and the nutrient values that the food data source returns.
_Avoid_: Food, ingredient, line

**Food**:
A row in the external food data source. An Item points at a Food; a Food is not owned by any User.

### The conversation

**Thread**:
The single continuous conversation between one User and the coach. The LangGraph checkpointer holds its state. There is exactly one Thread for each User.
_Avoid_: Chat, conversation, history

**Session**:
One connection to a Thread. A Session ends; the Thread continues.
_Avoid_: Login, visit

**Intent**:
The classified purpose of one User message. The router uses it to select the path through the graph.

### The answers

**Recommendation**:
A proposed change to what the User eats next, produced from the Profile, the Goal, and the logged Meals.
_Avoid_: Suggestion, tip, advice

**Refusal**:
The coach's answer when a request falls outside its job. A Refusal states the boundary, gives a disclaimer, and points to a professional.
_Avoid_: Rejection, error

**Corpus**:
The curated set of nutrition documents that the coach cites. It is public guidance, not user data.
_Avoid_: Knowledge base, docs

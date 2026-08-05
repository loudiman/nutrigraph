# NutriGraph

A conversational nutrition coach built on LangGraph, with a Node.js/Express API gateway in front of a Python/FastAPI agent service.

**Status: guarded.** One Gemini call classifies each message into Intents, and the Turn dispatches to a stub, asks one clarifying question, or refuses what is outside the Coach's job. The work is charted as a Wayfinder map: [Map: NutriGraph build spec](https://github.com/loudiman/nutrigraph/issues/1). The vocabulary is fixed by [`CONTEXT.md`](CONTEXT.md) and the hard-to-reverse choices by [`docs/adr/`](docs/adr/).

## Layout

```
gateway/                  Node and Express: the session, the turn identifier, the event stream
agent/                    Python and FastAPI: the graph, the nodes, the migrations
agent/migrations/         numbered SQL files, owned by the agent service — read its README first
agent/seeds/              demo Profiles
gateway/src/generated/    TypeScript types, generated from the agent's OpenAPI document
docs/adr/                 the decision records
docs/deploy.md            the deployed system, and how to roll it back
prototypes/               throwaway code, never imported
compose.yaml              PostgreSQL with pgvector, and nothing else
cloudbuild.pr.yaml        every pull request: the tests
cloudbuild.yaml           every merge to main: build, migrate, deploy
```

## Running it locally

One container. Both services run natively with file reloading, so a graph change is visible in about a second.

```sh
cp .env.example .env          # change POSTGRES_PORT if 5432 is taken, and set GOOGLE_API_KEY
docker compose up -d

cd agent
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/nutrigraph-migrate  # numbered SQL files, in order; safe to re-run
.venv/bin/nutrigraph-seed     # the demo Profiles; safe to run twice
.venv/bin/python -m nutrigraph_agent.main

cd ../gateway
npm install
npm run dev
```

Then:

```sh
curl -N -c cookies.txt -H 'Content-Type: application/json' \
  -d '{"message":"I ate two eggs and pandesal"}' http://127.0.0.1:3000/api/turn
```

The gateway issues a signed cookie carrying a seeded `user_id`, creates the one turn identifier, and streams the node events as they happen. The answer text is held back and arrives as one `answer` event at the end, which is what lets the guardrail scan the finished text before it is sent. A failure mid-Turn arrives as a typed `error` event and the stream closes.

## The router

`load_profile` → `guard` → `route` → `dispatch`, `clarify`, or `refuse`. `route` is one call to Gemini 3.5 Flash-Lite at temperature 0, filling a fixed `RouterDecision`: at most two Intents from the five, a confidence, and an out-of-scope flag. No keyword list is maintained for routing, and the router never writes a Refusal — it detects, and the guardrail gives the wording.

Below a confidence of 0.6 the Turn goes to `clarify`, which asks one short question and ends. That question is `pending_clarification`, the only place the Coach stops and waits for the User. It survives until a Turn is classified at 0.6 or higher, and `route` clears it there. A second clarify Turn replaces the value rather than adding a second one, and a Refusal turn leaves it standing.

`dispatch` is a stub: no Intent path is built yet, so it says what the router decided and stops.

## The guardrail

Four subjects sit outside the Coach's job: diagnosis, treatment, and dosage; eating-disorder content; nutrition for pregnancy, breastfeeding, and children; and the personal diet management of a chronic disease. A general factual question about a chronic disease is still answered from the Corpus — only a personal plan for it is refused, and a request framed as being about a friend is refused on the same terms as one in the first person.

Two detectors, and either one produces a Refusal. `guard` runs a deterministic rule list — `agent/src/nutrigraph_agent/guardrail.py`, readable by a reviewer and provable by a test — before the router and with no model, so a message it catches never reaches an Intent path. The router's `out_of_scope` flag catches meaning no word list predicts. Both end at `refuse`, the only node that writes a Refusal.

The Refusal is a template in code: it names the boundary, gives the disclaimer, points to a professional, and offers what the Coach can do instead. Eating-disorder content additionally carries a help-line. Because it is assembled from those strings, it cannot drift.

After the composer, `scan_reply` reads the finished text for medical claims, before the answer event is sent. A text that fails ends the Turn with the fixed safe message, never with a partial answer. A Refusal is not scanned — it is the codebase's own words. The allergen half of the scan lands in that same function when the allergy-check slice arrives, and the stream contract does not change.

The split this makes real: deterministic are the rule list, the redaction patterns, the final text scan, the schema validation, and the Refusal wording; the model does the Intent classification, the meaning-level scope flag, the name and address detection, and every answer. Nothing that decides safety is left to the model. It is checked by plain assertions at the agent turn seam in `agent/tests/test_guardrail.py`, never by a model judge — a judge can flake, an assertion cannot.

## The model routing rule

Not a list of nodes. Work that fills a fixed schema from text uses the schema tier; work that reasons or writes prose for the User uses the prose tier. A node picks `TurnModels.fill` or `TurnModels.write` and inherits its model from the rule — no node names a model.

```
MODEL_SCHEMA=gemini-3.5-flash-lite   # route, and every later schema filler
MODEL_PROSE=gemini-3.5-flash         # clarify, and every later writer
```

**Swapping the provider is one line.** `MODEL_PROVIDER` in `.env`, plus that vendor's LangChain package and its key variable:

```
MODEL_PROVIDER=openai      MODEL_SCHEMA=gpt-5-mini          MODEL_PROSE=gpt-5
MODEL_PROVIDER=anthropic   MODEL_SCHEMA=claude-haiku-4-5    MODEL_PROSE=claude-sonnet-5
```

Every call goes through `init_chat_model`, which is written on one line in `agent/src/nutrigraph_agent/providers.py` and nowhere else. Nothing in the codebase branches on the vendor, so there is no dead code behind the swap — and a test asserts both facts.

## The redaction wrapper

Gemini runs on the free tier, so Google uses submitted prompts to improve its products and a person may read them. Redaction is therefore a wrapper on every provider call, not a graph node ([ADR 0002](docs/adr/0002-redact-before-the-provider-not-before-storage.md)). It runs on the router call and on the retry after a schema failure.

Replaced with placeholders: person names, email addresses, phone numbers, street addresses, exact dates of birth, government identity numbers. Sent unchanged: weight, height, allergies, diet pattern, goals, and the food — the Coach cannot work without them. A private `redaction_placeholder` table maps a placeholder back, so the answer still addresses the User by name. **The `message` row holds the raw, unredacted text**; redaction happens at the provider call and nowhere else.

A node cannot opt out: it holds a `TurnModels`, which owns the Redactor, and there is no route to a chat model that skips it. `agent/tests/test_redaction_wrapper.py` is the guard — it drives every shape of Turn and asserts against the faked provider that no unredacted identifier ever arrived, and it fails if a second module reaches a provider.

Open-ended entities — the names of other people, addresses no regular expression catches — need an entity library, which is optional and off by default:

```sh
.venv/bin/pip install -e ".[ner]" && .venv/bin/python -m spacy download en_core_web_sm
```

Without it, only names the Coach already holds are redacted.

## The metric record

Every node writes an `interaction_event` row: node, Intent, model, latency, input and output tokens, and cost. The wrapping happens in `build_graph`, so a new node cannot be added without one.

LangSmith is switched on by two environment variables and no code change, and every node becomes a nested run under the Turn:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
```

LangSmith is the reading tool; `interaction_event` is the record, because the free tier keeps traces for 14 days. The turn identifier is on the trace, on the `message` rows, and on the `interaction_event` rows.

## The deployed system

One environment: two Cloud Run services in `asia-southeast1`, both at minimum
instances 0, in front of a Neon database. The gateway is public; the agent takes
internal ingress only. A merge to `main` builds both images, applies the
migrations, then deploys. A rollback is a redeployment of the previous image and
**a migration is never reversed** — which is why a migration may only ever add,
a rule that lives in [`agent/migrations/README.md`](agent/migrations/README.md).
The whole of it is in [`docs/deploy.md`](docs/deploy.md).

```sh
curl -N -H 'Content-Type: application/json' \
  -d '{"message":"I ate two eggs and pandesal"}' \
  https://nutrigraph-gateway-713096458695.asia-southeast1.run.app/api/turn
```

## The internal call

In production the gateway calls the agent with a Google-signed Cloud Run identity token. That token does not exist on a laptop, so locally the agent binds to loopback and accepts `X-Dev-Auth` instead, behind `AGENT_DEV_AUTH`, which production never sets. **The agent refuses to start when `AGENT_DEV_AUTH` is set together with a non-loopback `AGENT_HOST`.**

## The contract

Pydantic is the contract. FastAPI publishes the models as an OpenAPI document and a build step generates TypeScript from it. Both artefacts are committed, so a reviewer sees a contract change in the diff:

```sh
cd agent   && .venv/bin/python scripts/export_openapi.py   # -> agent/openapi.json
cd gateway && npm run gen:types                            # -> gateway/src/generated/agent.ts
```

## Tests

Two seams, and no later slice adds a third.

```sh
cd agent   && .venv/bin/pytest                 # the agent turn seam; Gemini, FoodData Central and the database faked
cd gateway && npm test && npm run typecheck    # the gateway seam; the agent service faked
```

The provider is faked one layer below the redaction wrapper, so every test runs the real wrapper and can read exactly what Google would have seen.

The migration, seed, and checkpointer tests need a real PostgreSQL and are skipped without one. The one test that actually calls Gemini is skipped without a key:

```sh
NUTRIGRAPH_TEST_DATABASE_URL=postgresql://nutrigraph:nutrigraph@localhost:5432/nutrigraph_test .venv/bin/pytest
GOOGLE_API_KEY=... .venv/bin/pytest tests/test_live_router.py
```

The pull-request pipeline stands one of those up for itself, so nothing skips
there. It is never Neon: the free tier is a ceiling, not a test fixture.

A node is never tested on its own. A node that cannot be reached from a Turn is a node that should not exist.

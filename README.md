# NutriGraph

A conversational nutrition coach built on LangGraph, with a Node.js/Express API gateway in front of a Python/FastAPI agent service.

**Status: answering.** One Gemini call classifies each message into Intents; a nutrition question is answered from a curated Corpus with a Citation on every claim, and anything else dispatches to a stub or asks one clarifying question. The work is charted as a Wayfinder map: [Map: NutriGraph build spec](https://github.com/loudiman/nutrigraph/issues/1). The vocabulary is fixed by [`CONTEXT.md`](CONTEXT.md) and the hard-to-reverse choices by [`docs/adr/`](docs/adr/).

## Layout

```
gateway/                  Node and Express: the session, the turn identifier, the event stream
agent/                    Python and FastAPI: the graph, the nodes, the migrations
agent/migrations/         numbered SQL files, owned by the agent service
agent/seeds/              demo Profiles, and the Corpus manifest
gateway/src/generated/    TypeScript types, generated from the agent's OpenAPI document
docs/adr/                 the decision records
prototypes/               throwaway code, never imported
compose.yaml              PostgreSQL with pgvector, and nothing else
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
.venv/bin/nutrigraph-ingest   # the Corpus: fetch, chunk, embed, store; slow, safe to run twice
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

The gateway issues a signed cookie carrying a seeded `user_id`, creates the one turn identifier, and streams the node events as they happen. The answer text is held back and arrives as one `answer` event at the end — a later slice inserts the guardrail text scan there without changing the contract. A failure mid-Turn arrives as a typed `error` event and the stream closes.

## The router

`load_profile` → `route` → either `dispatch` or `clarify`. `route` is one call to Gemini 3.5 Flash-Lite at temperature 0, filling a fixed `RouterDecision`: at most two Intents from the five, a confidence, and an out-of-scope flag. No keyword list is maintained for routing, and the router never writes a Refusal — it detects, and a later slice gives the guardrail the wording.

Below a confidence of 0.6 the Turn goes to `clarify`, which asks one short question and ends. That question is `pending_clarification`, the only place the Coach stops and waits for the User. It survives until a Turn is classified at 0.6 or higher, and `route` clears it there. A second clarify Turn replaces the value rather than adding a second one.

`dispatch` is a stub for the Intent paths that are not built yet, so it says what the router decided and stops. A Turn whose *first* Intent is `ask_question` goes to `retrieve` and then to `answer_question` instead — the first Intent, because the order matters and the second reads what the first produced.

## The Corpus, and the cited Answer

About forty public documents — the Dietary Guidelines for Americans 2025-2030, the WHO healthy-diet fact sheet and Q&A, the NHS Eatwell explainer, the FNRI Nutritional Guidelines for Filipinos and the Philippine Dietary Reference Intakes, and the nutrition.gov topic pages. The survey behind the choice is [`docs/research/nutrition-corpus.md`](docs/research/nutrition-corpus.md); the manifest is `agent/seeds/corpus.json`.

Gemini Flash holds a very large context, so a corpus this size does not *force* retrieval. Retrieval earns its place through the Citation with page provenance, the token cost of each Turn, and the ability to filter the index by licence.

**Licences are data, not documentation.** Every chunk carries its licence identifier and its attribution string, written at ingestion time. They are repeated on the chunk row on purpose, so a licence filter never needs a join:

```sql
delete from corpus_chunk where not commercial_use;   -- one predicate, no join
```

WHO's attribution requirement is then satisfied automatically, because the string travels with the chunk. **EFSA prose is excluded**: CC BY-ND does not license the adapted material that chunking prose into an index arguably produces, so a European reference value enters as a number in a data table instead. `corpus.FORBIDDEN_LICENCES` is that decision as code, and the manifest refuses a document that names one. **WHO stays in**, because this demonstration is not commercial — and the `commercial_use` flag is what makes that reversible in one statement.

**The vectors.** `gemini-embedding-001` returns 3072 dimensions, truncated to 768 and **re-normalized by hand**, because version 1 of the model does not re-normalize a truncated vector. HNSW accepts at most 2000 dimensions, which is why the full output is not used ([ADR 0001](docs/adr/0001-gemini-free-tier-and-768-dimension-embeddings.md)). The column is `vector(768)` with an HNSW cosine index, and changing that number re-indexes the whole Corpus.

**Ingestion is its own command**, because it talks to forty web servers and to the embedding model, and tying that to the two-second `nutrigraph-seed` would make it a two-minute one. It is safe to run twice: a document is keyed by its slug and its chunks are replaced wholesale, and a document whose extracted text has not changed is skipped before any embedding call.

**The Answer.** `Answer` holds `text` and `citations`, and **a nutrition claim with an empty citations list fails schema validation** — an unsupported claim is a build failure, not a matter of taste. Each Citation names the document and the section or page. When no passage clears the relevance floor the Coach says the Corpus does not cover the question and makes no provider call at all, so there is nowhere for an invented claim to come from.

## The model routing rule

Not a list of nodes. Work that fills a fixed schema from text uses the schema tier; work that reasons or writes prose for the User uses the prose tier. A node picks `TurnModels.fill` or `TurnModels.write` and inherits its model from the rule — no node names a model.

```
MODEL_SCHEMA=gemini-3.5-flash-lite    # route, the cited Answer, and every later schema filler
MODEL_PROSE=gemini-3.5-flash          # clarify, and every later writer
MODEL_EMBEDDING=gemini-embedding-001  # the Corpus index and the query that searches it
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

This covers the vector half too. Embedding the User's question is a provider call like any other, so `embed_query` and `embed_documents` sit on the same wrapper and the guard test scans what reached the embedding model as well.

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

A node is never tested on its own. A node that cannot be reached from a Turn is a node that should not exist.

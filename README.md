# NutriGraph

A conversational nutrition coach built on LangGraph, with a Node.js/Express API gateway in front of a Python/FastAPI agent service.

**Status: walking skeleton.** One Turn runs end to end with no model call anywhere: the reply is an echo. The work is charted as a Wayfinder map: [Map: NutriGraph build spec](https://github.com/loudiman/nutrigraph/issues/1). The vocabulary is fixed by [`CONTEXT.md`](CONTEXT.md) and the hard-to-reverse choices by [`docs/adr/`](docs/adr/).

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
cp .env.example .env          # change POSTGRES_PORT if 5432 is taken
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

The gateway issues a signed cookie carrying a seeded `user_id`, creates the one turn identifier, and streams the node events as they happen. The answer text is held back and arrives as one `answer` event at the end — a later slice inserts the guardrail text scan there without changing the contract. A failure mid-Turn arrives as a typed `error` event and the stream closes.

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

The migration, seed, and checkpointer tests need a real PostgreSQL and are skipped without one:

```sh
NUTRIGRAPH_TEST_DATABASE_URL=postgresql://nutrigraph:nutrigraph@localhost:5432/nutrigraph_test .venv/bin/pytest
```

The pull-request pipeline stands one of those up for itself, so nothing skips
there. It is never Neon: the free tier is a ceiling, not a test fixture.

A node is never tested on its own. A node that cannot be reached from a Turn is a node that should not exist.

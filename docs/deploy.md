# The deployed system

One environment, plus the laptop. A pull request is verified by tests, never by
a deployed copy.

| | |
|---|---|
| Project | `nutrigraph-2026ldm` |
| Region | `asia-southeast1` |
| Gateway | <https://nutrigraph-gateway-713096458695.asia-southeast1.run.app> — public |
| Agent | <https://nutrigraph-agent-713096458695.asia-southeast1.run.app> — internal ingress, authenticated callers only |
| Images | `asia-southeast1-docker.pkg.dev/nutrigraph-2026ldm/nutrigraph/{agent,gateway}` |
| Database | Neon, over the public internet, pooled endpoint, TLS |

Both services run at minimum instances 0, so an idle system bills nothing and
the price is a cold start.

```sh
curl -N -H 'Content-Type: application/json' \
  -d '{"message":"I ate two eggs and pandesal"}' \
  https://nutrigraph-gateway-713096458695.asia-southeast1.run.app/api/turn
```

## The internal call

The agent takes `--ingress=internal`, so a request that did not leave through a
VPC network in this project is refused by the platform: an unauthenticated
`POST /turn` from the internet gets a `404` from Cloud Run and no agent
instance starts. An identity token does not change that — ingress is checked
first, and the application never sees either request.

That means the gateway's call has to leave through the VPC, which is why the
gateway runs with direct VPC egress on the `default` network and why Private
Google Access is on for the `default` subnet in `asia-southeast1`. Without
Private Google Access the gateway resolves `run.app` to a public address it
cannot reach, and every Turn ends in `agent_unreachable`.

Inside that call the gateway sends `Authorization: Bearer <token>`, where the
token comes from the metadata server on the instance and is signed by Google
for the agent's URL as its audience. There is no shared secret and nothing to
rotate. On a laptop that token does not exist, so the agent binds to loopback
and accepts `X-Dev-Auth` instead — see the README.

## The secrets

Every secret is a Secret Manager reference on the service, resolved by the
runtime service account at start. No secret value is ever typed into the
console, written into a build file, or committed.

| Secret | Service | Read as |
|---|---|---|
| `neon-connection-string` | agent | `DATABASE_URL` |
| `gemini-api-key` | agent | `GOOGLE_API_KEY` — the name `langchain-google-genai` reads on its own; `providers.py` never names a key |
| `fdc-api-key` | agent | `FDC_API_KEY` — mounted; not read yet, `Deps.food` is unwired |
| `langsmith-api-key` | agent | `LANGSMITH_API_KEY`, alongside the plain `LANGSMITH_TRACING=true` |
| `cookie-signing-secret` | gateway | `SESSION_SECRET` |

Each runtime service account may read only its own secrets:
`nutrigraph-agent@` holds four grants, `nutrigraph-gateway@` holds one.

## The pipeline

`cloudbuild.pr.yaml` on every pull request: a throwaway PostgreSQL, the agent's
tests against it, the gateway's tests and type check, then the eval gate — the
real graph against the golden dataset, and ragas over the retrieval group in a
container of its own. See [`agent/evals/README.md`](../agent/evals/README.md);
the two halves are two steps because ragas 0.4 needs a `langchain-community`
that the agent's `langchain` 1.x cannot sit beside.

The eval step sets `LANGSMITH_ENDPOINT` to the APAC data plane for the same
reason the deployed service does: this account is not on the SDK's US default,
which answers a valid key with a 403 on every call, silently.

`cloudbuild.yaml` on every merge to `main`, in this order: build the two images,
apply the migrations, deploy the agent, deploy the gateway. A service directory
that did not change in the commit is not rebuilt — the image already in the
registry is retagged with the new commit, so the deployment is uniform and the
work is skipped.

Both run as `nutrigraph-build@`, which may write to Artifact Registry, deploy to
Cloud Run, act as the two runtime service accounts, and read three secrets:
`neon-connection-string` for the migration step on `main`, and
`gemini-api-key` and `langsmith-api-key` for the eval gate on a pull request.
The eval gate calls a real model against a throwaway database, so it needs the
key the agent itself runs on; it never touches Neon.

## Rolling back

**A rollback is a redeployment of the previous image. A migration is never
reversed** — see [`agent/migrations/README.md`](../agent/migrations/README.md)
for why that is safe, and why a migration may only ever add.

```sh
gcloud artifacts docker images list \
  asia-southeast1-docker.pkg.dev/nutrigraph-2026ldm/nutrigraph/agent \
  --include-tags --sort-by=~CREATE_TIME --limit=5

gcloud run deploy nutrigraph-agent --region=asia-southeast1 \
  --image=asia-southeast1-docker.pkg.dev/nutrigraph-2026ldm/nutrigraph/agent:<previous>
```

Demonstrated on 2026-08-05, on the agent service:

- Serving `nutrigraph-agent-00004-ktg`, image `agent:v3`.
- Redeployed `agent:v2`. New revision `nutrigraph-agent-00005-l49` took 100% of
  traffic; a Turn ran end to end against it.
- No migration was reversed. `001_init.sql` stayed applied, and the older image
  read the same schema because the migration had only added to it.
- Rolled forward again to `agent:v3` as `nutrigraph-agent-00006-n24`.

A *failed* deployment needs no rollback at all. Deploying a known-broken image
earlier the same day produced a revision that never passed its startup probe;
`gcloud run deploy` reported the failure, traffic stayed on
`nutrigraph-agent-00002-h5j`, and the system kept answering.

## What the free tiers cost us

- **Neon's pooled endpoint rejects the `options` startup parameter**, so the
  checkpointer's tables cannot be pushed into a schema of their own by setting
  `search_path` on the connection. They live in `public` beside ours instead;
  `CHECKPOINT_TABLES` names them and no migration file may.
- **Neon's pooler shares one server connection between clients**, so
  `prepare_threshold` is 0 on both pools.
- **Neon suspends a cold database and closes its connections**, and a Cloud Run
  instance can outlive that, so both pools check a connection before handing it
  out.
- **The free tier is a real ceiling** — 0.5 GB storage, 100 CU-hours, 5 GB
  egress per month, confirmed 2026-08-05 on issue #24. Exceeding any one of them
  suspends compute until the next month. Do not point a load test at it, and do
  not run the pull-request suite against it; that suite brings its own
  PostgreSQL.

## Bootstrapping this from nothing

Done once, and recorded here so it can be done again:

```sh
gcloud services enable compute run artifactregistry secretmanager cloudbuild   # .googleapis.com
gcloud artifacts repositories create nutrigraph --repository-format=docker --location=asia-southeast1
gcloud iam service-accounts create nutrigraph-agent    # + nutrigraph-gateway, nutrigraph-build
# secretAccessor per secret, run.invoker for the gateway on the agent service,
# and run.admin / artifactregistry.writer / logging.logWriter /
# iam.serviceAccountUser on the project for nutrigraph-build
gcloud compute networks subnets update default --region=asia-southeast1 \
  --enable-private-ip-google-access
gcloud run deploy ...   # the flags are in cloudbuild.yaml
```

The demo Profiles are seeded by hand, once, and are not part of the pipeline:

```sh
docker run --rm -e DATABASE_URL="$DATABASE_URL" <agent image> nutrigraph-seed
docker run --rm -e DATABASE_URL="$DATABASE_URL" <agent image> nutrigraph-ingest
```

`nutrigraph-ingest` is the slow one — it fetches the Corpus and embeds it, and
it embeds the local dish names the recommend path's similarity query is over.
Both halves are safe to run twice and cheap the second time, so a re-run after
adding a dish costs one embedding call for that dish.

**The one step that needs a browser** is connecting Cloud Build to the GitHub
repository. The connection `nutrigraph-github` exists in `asia-southeast1` and
is waiting on that authorisation; until it is done, the two triggers cannot be
created and the pipeline files are run by hand with `gcloud builds submit
--config`.

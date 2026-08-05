# Neon instead of Cloud SQL

NutriGraph must cost nothing while nobody is using it, which is why the model provider, the tracing tool, and the food data source are all free tiers. Cloud SQL has no free tier, so its smallest instance would bill every month for a demonstration that is idle almost all of the time. We therefore run PostgreSQL on Neon, a serverless Postgres with pgvector that scales to zero, reached from Cloud Run over the public internet with a pooled, TLS-protected connection string held in Secret Manager.

## Considered options

- **Cloud SQL in `nutrigraph-2026ldm`.** Private connectivity, no traffic over the public internet, and the strongest GCP story for a job description that names GCP. Rejected on cost alone: it is the single line item that would make an idle project bill every month.
- **Supabase.** Also free at rest, with a dashboard that demonstrates well. Rejected because a free project pauses after inactivity, and this demonstration is opened rarely and must work when it is.

## Consequences

- **Database traffic leaves Google's network.** The connection must use TLS, and the connection string is a Secret Manager secret, never an environment variable in the console.
- **Cloud Run scales to zero and can open many short-lived instances,** so the pooled connection endpoint is required rather than optional.
- **A cold database adds delay to the first query** after an idle period. That is acceptable for a demonstration; it would not be for a product.
- **The free storage limit is a real ceiling.** The corpus, its 768-dimension vectors, and the food embeddings must fit inside it, and the exact limits must be confirmed at build time rather than assumed from this record.
- **The pooled endpoint is a connection pooler, not PostgreSQL**, and what that costs the checkpointer is its own decision: [ADR 0005](0005-checkpointer-tables-live-in-public.md).
- **The GCP story is now Cloud Run, Artifact Registry, Secret Manager, and Cloud Build** — not Cloud SQL. If the project ever needs the private-network story, moving to Cloud SQL is a data migration, not a configuration change.

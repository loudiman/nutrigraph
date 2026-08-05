# The checkpointer's tables live in `public`, not in a schema of their own

The LangGraph Postgres checkpointer writes its tables into `public`, beside ours. The original design gave it a `langgraph` schema so that library tables stayed out of our migrations and the boundary was visible in the database itself. Neon's pooled endpoint made that impossible: a `search_path` in the startup packet is rejected outright, and that packet was how the schema was going to be selected. ADR 0004 requires the pooled endpoint, so this is that decision's bill, arriving at deployment time rather than at design time.

## Considered options

- **The direct, unpooled Neon endpoint, keeping the schema.** The startup packet is accepted there, so nothing else would have had to change. Rejected because ADR 0004 requires the pooled endpoint for a real reason: Cloud Run scales to zero and opens many short-lived instances, and exhausting the connection limit is a worse failure than sharing a schema.
- **Setting `search_path` per session after connecting, instead of in the startup packet.** Rejected because it does not survive transaction pooling. A session-level `SET` is lost the moment the pooler hands the connection to another client between transactions, so the schema would be correct on the first query and wrong on an arbitrary later one — the worst shape of bug, because it passes every test that runs one transaction.
- **A separate database for the checkpointer.** A clean boundary that the pooler cannot take away. Rejected because it costs a second connection string, a second secret, and a second thing to provision, and because a Thread could then no longer be joined to a Meal for debugging, which the schema decision named as a benefit.

## Consequences

- **Library tables and our tables share one namespace.** A future LangGraph release may add a table whose name we have already taken, or take a name we later want. The add-only migration rule now carries a second meaning: do not create a table whose name the library might claim.
- **The boundary is enforced by code rather than by the database.** `CHECKPOINT_TABLES` names the library's tables; one test asserts that no migration file mentions any of them, and another asserts that every table in `public` is either ours or one of theirs. That is a stronger guarantee than the schema gave us, because it fails loudly in CI rather than quietly at runtime — but it holds only as long as those tests are kept.
- **Upgrading LangGraph becomes a review item, not a version bump.** A release that adds a table now changes the contents of `public`, and the second test above is what will notice.
- **Two further pooler consequences travel with this one:** `prepare_threshold` is 0, because a prepared statement cannot outlive the transaction that made it under transaction pooling, and both pools check the connection before use, because Neon's auto-suspend leaves closed sockets in the pool.
- **Moving back to a dedicated schema is a data migration,** not a configuration change, and it requires leaving the pooled endpoint first.

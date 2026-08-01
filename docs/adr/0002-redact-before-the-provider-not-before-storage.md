# Redact before the provider, not before storage

Gemini runs on the free tier, so Google may read anything NutriGraph sends it. We therefore redact personal identifiers immediately before each provider call, and we store the raw user message in PostgreSQL unchanged. The database keeps the truth, the provider sees a cleaned copy.

## Considered options

- **Redact at the entry to the agent service.** One call site, and nothing downstream can leak. Rejected because the original words are then lost forever: the eval dataset and every debug trace would hold placeholder text, and a wrong redaction could never be reviewed.
- **Redact at both exits, the provider call and the database write.** Two protections that fail independently. Rejected as two call sites that drift apart, for a system whose database holds demo data only.

## Consequences

- **The database holds unredacted user text.** It therefore needs its own protection, and the specification must state that only demo data may be entered. This is acceptable because the audience is demo users; it would not be acceptable for real users.
- **Redaction is a wrapper on the provider call, not a node.** Every path that calls a model must pass through it, including the router call and the retry after a schema failure. A new call site that forgets the wrapper is the failure mode to test for.
- **The eval dataset stays truthful,** because it reads the stored raw text.

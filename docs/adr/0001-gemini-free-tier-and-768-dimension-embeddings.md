# Gemini on the free tier, with 768-dimension embeddings

NutriGraph is a demo system with no real personal health data, and it must cost nothing to run. We chose Google Gemini as the only model provider, on the free tier: the chat models and the embedding model come from one vendor, and the free tier makes both free. Every model call goes through LangChain `init_chat_model`, so the provider stays a configuration string and not a code path.

## Considered options

- **OpenAI.** One vendor for chat and embeddings, and no training on our data at any tier. Rejected because nothing is free.
- **Anthropic.** The best strong-tier price at the time of the decision, and no training on our data. Rejected because Anthropic ships no embedding model, so a second vendor (Voyage AI) and a second key would be necessary for the vector half of the system.
- **Gemini on the paid tier.** The training opt-out applies, and Gemini 3.1 Pro becomes available. Rejected for now because it costs money; the switch is one billing flag on the key's GCP project, so the decision is cheap to revisit.

## Consequences

- **Google may read our prompts.** On the free tier, Google uses submitted prompts and responses to improve its products, and that includes human review. Therefore no real personal data may reach the provider. The guardrail node must redact before the provider call, not only before storage.
- **Both model tiers come from the Flash line.** Gemini 3.1 Pro is paid only, so the "small model for simple turns, strong model for hard reasoning" split is Flash-Lite against Flash, not Flash against Pro.
- **The fallback is model-to-model, not provider-to-provider.** With one key, a failure or a rate-limit stop falls back inside Gemini. A provider-to-provider fallback needs one configuration string and one environment variable, and no code change.
- **The vector column is `vector(768)`.** `gemini-embedding-001` returns 3072 dimensions and supports Matryoshka truncation. pgvector's HNSW index accepts at most 2000 dimensions, so the full output cannot be indexed. We truncate to 768 and re-normalize by hand, because version 1 of the model does not re-normalize a truncated vector. A later change of this number forces a re-index of the whole corpus.

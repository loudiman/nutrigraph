# LLM Providers for a LangGraph Conversational Nutrition Coach

**Date researched:** 2026-08-01
**Question (issue #3):** Which LLM providers can run a LangGraph conversational nutrition coach, and what does each one cost?

**Scope:** Compare Google Gemini, Anthropic Claude, and OpenAI on (1) price for a cheap and a strong model tier, (2) free API tier and its data-use terms, (3) native structured-output support and LangChain/LangGraph support for it, (4) embedding model availability (dimensions, price, max input — this matters because pgvector needs a fixed dimension count), and (5) how easy a provider swap is behind one LangChain interface (`init_chat_model`).

All prices/limits below were fetched live on 2026-08-01 from each provider's own docs — no aggregator sites were used as a source of a number. Where a live page didn't publish an exact figure, that's marked "could not verify" rather than guessed.

---

## 1. Price per million tokens

| Provider | Cheap tier | Price (in / out per MTok) | Strong tier | Price (in / out per MTok) | Source |
|---|---|---|---|---|---|
| **Anthropic** | Claude Haiku 4.5 | $1.00 / $5.00 | Claude Sonnet 5 | $2.00 / $10.00 (introductory, through 2026-08-31) → $3.00 / $15.00 standard from 2026-09-01. Claude Opus 5 (top-of-line-non-Fable): $5.00 / $25.00 | [Pricing – Claude API Docs](https://platform.claude.com/docs/en/about-claude/pricing) (fetched 2026-08-01) |
| **OpenAI** | gpt-5-mini | $0.25 / $2.00 (cached input $0.025) | gpt-5.6-sol (current flagship; gpt-5.5 matches at same price) | $5.00 / $30.00 (cached input $0.50) | [OpenAI Pricing docs](https://developers.openai.com/api/docs/pricing) (fetched 2026-08-01, redirected from platform.openai.com/docs/pricing) |
| **Google Gemini** | Gemini 3.5 Flash-Lite | $0.30 / $2.50 (older Gemini 2.5 Flash-Lite still listed at $0.10 / $0.40) | Gemini 3.1 Pro (Preview — no GA "Pro" model exists as of 2026-08-01) | $2.00 / $12.00 for ≤200K input tokens, $4.00 / $18.00 above 200K (context-length-tiered pricing) | [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing) (fetched 2026-08-01) |

**Takeaway:** OpenAI's mini tier is cheapest at the low end ($0.25/$2.00). Anthropic's Haiku 4.5 is close behind ($1/$5). At the strong tier, Anthropic Sonnet 5's introductory pricing ($2/$10 through 2026-08-31) undercuts both OpenAI's flagship ($5/$30) and Gemini's Pro-preview ($2–4/$12–18) — but that intro pricing expires 2026-09-01 and reverts to $3/$15, still cheaper than OpenAI/Gemini strong tiers. Gemini Pro is the only strong-tier model with context-length-based pricing tiers, which complicates cost forecasting for long conversation histories.

---

## 2. Free tier (API, not chat UI)

| Provider | Free API tier exists? | Rate limits | Free-tier data-use terms | Source |
|---|---|---|---|---|
| **Anthropic** | No ongoing free tier — new accounts get a one-time trial credit only. Paid usage is tier-based (Start/Build/Scale/Custom); Start tier gives e.g. Sonnet 5: 1,000 RPM / 2,000,000 input TPM / 400,000 output TPM. | N/A (trial credit, not a sustained free quota) | By default, inputs/outputs are **not** used to train Anthropic's models, for any tier, unless the user explicitly opts in via feedback. | [Pricing FAQ](https://platform.claude.com/docs/en/about-claude/pricing), [Rate limits](https://platform.claude.com/docs/en/api/rate-limits), [Is my data used for model training? – Privacy Center](https://privacy.claude.com/en/articles/7996868-is-my-data-used-for-model-training) (all fetched 2026-08-01) |
| **OpenAI** | Yes — a distinct "Free" usage tier exists (separate from ChatGPT's free tier), capped at $100/month of usage, available to accounts in allowed geographies. | Exact RPM/TPM/RPD figures for the Free tier are **not published** on the live rate-limits page as of 2026-08-01 — it defers to the per-account dashboard. **Could not verify exact numbers — source unreachable/not published on 2026-08-01.** | Inputs/outputs are **not** used to train or improve OpenAI models by default, for any tier, unless the org explicitly opts in to data sharing (policy in effect since 2023-03-01). | [Rate limits](https://developers.openai.com/api/docs/guides/rate-limits), [Your data](https://developers.openai.com/api/docs/guides/your-data) (fetched 2026-08-01) |
| **Google Gemini** | Yes — Flash and Flash-Lite models (and older Gemini 2.5 Pro) have a free tier in Google AI Studio. Gemini 3.1 Pro (current Pro-preview) is **paid-only, no free tier**. | The live rate-limits page no longer publishes a static free-tier RPM/TPM/RPD table — it says limits are dynamic per usage tier and points to the (login-gated, unfetchable) AI Studio dashboard. **Could not verify exact numbers — source unreachable on 2026-08-01.** | **Free/unpaid tier: Google explicitly uses submitted prompts and responses** "to provide, improve, and develop Google products and services," including human review — the docs warn against submitting sensitive data. **Paid tier: explicitly opted out** — "Google doesn't use your prompts... or responses to improve our products." | [Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits), [Gemini API Terms](https://ai.google.dev/gemini-api/terms) (fetched 2026-08-01) |

**Takeaway:** This is the sharpest differentiator. Gemini's free tier is the most generous in scope (whole Flash line free) but is the only one that **uses your data for training by policy** — a real concern for a nutrition/health coach handling user data. Anthropic and OpenAI both keep training opt-out as the default regardless of tier, but neither publishes a durable, no-cost free tier — Anthropic gives one-time trial credit, OpenAI's "Free" tier is a capped ($100/mo) allowance, not zero-cost forever.

---

## 3. Structured output support

| Provider | Native support | Provider's own docs | LangChain `with_structured_output` | LangGraph |
|---|---|---|---|---|
| **Anthropic** | Yes — `output_config.format` (JSON-schema-constrained output) plus `strict: true` tool use, on Claude 4.5+ models (covers Haiku 4.5, Sonnet 5, Opus 5). | [Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) (fetched 2026-08-01) | Supported — `ChatAnthropic.with_structured_output(..., method="json_schema")` uses Anthropic's native structured-output feature. | LangGraph doesn't document per-provider structured-output support itself — it defers to LangChain's chat-model layer ("you will commonly use LangChain components... to integrate models"); no LangGraph-specific gap found. |
| **OpenAI** | Yes — Structured Outputs via `response_format`/`text.format` with `type: "json_schema"`, `strict: true`; supported on GPT-4o-class models and later (covers gpt-5-mini, gpt-5.6-sol). | [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) (fetched 2026-08-01, redirected from platform.openai.com) | Supported — listed in LangChain's chat-model integration table with a checkmark for both "Tool calling" and "Structured output." | Same as above — LangGraph is provider-agnostic here, relies on the underlying `ChatOpenAI` model. |
| **Google Gemini** | Yes — `responseSchema` / `responseMimeType: application/json` config, supports a JSON Schema subset (types, `enum`, `format`, `required`); combinable with built-in tools on Gemini 3 models. | [Structured output](https://ai.google.dev/gemini-api/docs/structured-output) (fetched 2026-08-01) | Supported — same LangChain chat-model table shows `ChatGoogleGenerativeAI` with checkmarks for "Tool calling" and "Structured output." | Same as above. |
| **LangChain support matrix** | — | — | [Chat model integrations table](https://docs.langchain.com/oss/python/integrations/chat) shows ✅ Streaming / ✅ Tool calling / ✅ Structured output for ChatAnthropic, ChatOpenAI, and ChatGoogleGenerativeAI alike (fetched 2026-08-01). LangChain's general structured-output guide also states: "Some model providers support structured output natively through their APIs (e.g. OpenAI, xAI (Grok), Gemini, Anthropic (Claude))." — [Structured output guide](https://docs.langchain.com/oss/python/langchain/structured-output) (fetched 2026-08-01). | | |

**Takeaway:** All three providers have native, schema-constrained structured output, and LangChain's own integration matrix confirms full support (tool calling + structured output) for all three chat-model classes. No provider is a structured-output laggard here — this axis doesn't differentiate the choice.

---

## 4. Embedding models

pgvector requires a **fixed** vector dimension per column, so "configurable dims" only helps if you pick one dimension and commit to it at schema-creation time — but a provider offering Matryoshka-style truncation still lets you tune the cost/quality/storage tradeoff before you commit.

| Provider | Embedding model(s) | Dimensions | Configurable/truncatable? | Price (per MTok) | Max input tokens | Source |
|---|---|---|---|---|---|---|
| **Anthropic** | **None — no first-party embedding model.** Anthropic's own docs state: "Anthropic does not offer its own embedding model" and recommend the third-party Voyage AI. Voyage `voyage-4` (Anthropic-recommended partner, not Anthropic's own product): 256/512/1024(default)/2048 dims, configurable (Matryoshka), $0.06/MTok (first 200M tokens free per account), 32,000 token max input. `voyage-4-lite`: $0.02/MTok, same dims. | 256 / 512 / **1024 (default)** / 2048 (Voyage) | Yes (Voyage, Matryoshka) | $0.06/MTok (`voyage-4`) / $0.02/MTok (`voyage-4-lite`) | 32,000 (Voyage) | [Embeddings – Claude Platform Docs](https://platform.claude.com/docs/en/build-with-claude/embeddings), [Voyage AI Pricing](https://docs.voyageai.com/docs/pricing) (fetched 2026-08-01) |
| **OpenAI** | `text-embedding-3-small`, `text-embedding-3-large` | 1536 (small, default) / 3072 (large, default) | Yes — `dimensions` parameter truncates via a Matryoshka-style technique | $0.02/MTok (small) / $0.13/MTok (large) | 8,192 | [Embeddings guide](https://developers.openai.com/api/docs/guides/embeddings), [Pricing](https://developers.openai.com/api/docs/pricing) (fetched 2026-08-01) |
| **Google Gemini** | `gemini-embedding-001` (text-only), Gemini Embedding 2 (multimodal: text/image/video/audio/PDF) | 3072 (default for both) | Yes — MRL truncation, recommended 768/1536/3072 (v1 needs manual re-normalization after truncation; v2 auto-normalizes) | $0.15/MTok text (v1, paid; batch $0.075); v2: $0.20/MTok text (batch $0.10), plus per-modality pricing for image/audio/video. **Free tier available** for both. | 2,048 (v1) / 8,192 (v2) | [Embeddings](https://ai.google.dev/gemini-api/docs/embeddings), [Pricing](https://ai.google.dev/gemini-api/docs/pricing) (fetched 2026-08-01) |

**Takeaway — the real gap:** **Anthropic has no first-party embedding model at all.** Building on Claude for chat means bolting on a separate embeddings vendor (Anthropic's own docs point to Voyage AI) — an extra account, an extra API key, and an extra point of failure/cost tracking, even though Voyage's numbers themselves are competitive (1024-dim default, cheap, generous free tier). OpenAI and Gemini both ship first-party embeddings with configurable/truncatable dimensions, which is a strictly simpler story for a single-vendor RAG pipeline against pgvector. Gemini additionally offers a free tier for embeddings; OpenAI does not appear to (per its pricing page, embeddings are paid from the first token).

---

## 5. Provider swap cost behind one LangChain interface

| Question | Finding | Source |
|---|---|---|
| Does `init_chat_model` support all three providers uniformly? | Yes. `init_chat_model("claude-sonnet-...")` / `init_chat_model("openai:gpt-5.6-sol")` / `init_chat_model("google_genai:gemini-3.6-flash")` all go through the same call signature and return the same `BaseChatModel` interface. LangChain's own docs state: "Each provider package implements the same standard interface, so you can swap providers without rewriting application logic." | [Chat models guide](https://docs.langchain.com/oss/python/langchain/models) (fetched 2026-08-01, redirected from python.langchain.com/docs/how_to/chat_models_universal_init) |
| What's required per provider? | Each provider needs its own integration package installed: `pip install -U 'langchain[anthropic]'`, `pip install -U 'langchain[openai]'`, `pip install -U 'langchain[google-genai]'` — plus the matching API key env var (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY`). | Same source |
| Does structured output / tool calling stay uniform across the swap? | Yes — see §3 above; all three show ✅ on LangChain's own chat-model integration matrix for tool calling and structured output, so `.with_structured_output(...)` and `.bind_tools(...)` calls don't need per-provider branching for these three. | [Chat model integrations table](https://docs.langchain.com/oss/python/integrations/chat) (fetched 2026-08-01) |
| Does LangGraph add any provider-specific friction? | No LangGraph-specific gap found — LangGraph's docs explicitly delegate model integration to LangChain's chat-model layer and don't document per-provider constraints of their own. | [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) (fetched 2026-08-01) |

**Takeaway:** Provider swap cost is genuinely low. A LangGraph node built against `init_chat_model(...)` with a `with_structured_output(...)` call can switch providers by changing one string plus the corresponding env var, for all three providers — no code branching needed for the core chat + structured-output + tool-calling surface. Note this doc doesn't separately verify embedding-model swap cost via LangChain's embeddings interface (`init_embeddings` / `Embeddings` base class) — worth a follow-up check if embeddings need the same swap-ability.

---

## Recommendation

For a LangGraph conversational nutrition coach with a pgvector-backed RAG/memory layer:

- **Anthropic Claude (Sonnet 5)** is the strongest fit for the *conversation* half — cheapest strong-tier pricing during the 2026-08-31 introductory window (and still competitive after), solid native structured output, default no-training-on-data policy regardless of tier, and a clean `init_chat_model` swap story. Its blocker is embeddings: **no first-party embedding model**, so the project would need a second vendor (Anthropic's own recommendation is Voyage AI) just for the vector-search half of the stack.
- **OpenAI** is the most turnkey single-vendor option: chat (gpt-5-mini/gpt-5.6-sol) + first-party embeddings (`text-embedding-3-small`/`-large`, configurable dims, cheap, 8192-token max input) under one account, one API key, one bill. Slightly pricier than Anthropic at the strong chat tier post-intro-pricing, and its free API tier's exact rate limits weren't published on the live docs (capped at $100/mo but RPM/TPM unconfirmed).
- **Google Gemini** has the most generous nominal free tier and Matryoshka-truncatable embeddings with a free allowance too — but the free tier's data-use terms are the one hard disqualifier for a health-adjacent product: Google explicitly uses free-tier prompts/responses to improve its products, including human review. That's fine for prototyping, not for anything handling real user nutrition/health data unless the app commits to paid tier from day one (where the training opt-out does apply).

**Recommended default: OpenAI for the fastest single-vendor path to a working pgvector-backed system, or Anthropic Sonnet 5 + Voyage AI embeddings if the team prioritizes Anthropic's chat quality/pricing and is fine managing a second embeddings vendor.** Gemini is worth keeping in the swap-in list (LangChain makes this cheap) once the paid tier is used, given the multimodal embedding option (useful if food-photo input is ever in scope).

**Caveats / unresolved questions:**
- Exact free-tier RPM/TPM/RPD numbers for OpenAI and Gemini could not be verified from primary docs on 2026-08-01 — both providers now push rate-limit specifics to logged-in dashboards rather than public docs pages. Confirm directly in-account before depending on a specific free-tier throughput.
- This doc did not verify LangChain's `init_embeddings` / embeddings-interface swap cost across the three providers (only chat-model swap was checked) — worth a follow-up if embeddings need the same provider-agnostic treatment as chat.
- Anthropic Sonnet 5's $2/$10 pricing is introductory and reverts to $3/$15 on 2026-09-01 — re-check before finalizing a cost model that spans that date.
- A human makes the final provider choice in the decision ticket (issue #8) — this document is input to that decision, not the decision itself.

---

## Sources

- [Anthropic — Pricing (Claude API Docs)](https://platform.claude.com/docs/en/about-claude/pricing)
- [Anthropic — Rate limits](https://platform.claude.com/docs/en/api/rate-limits)
- [Anthropic — Is my data used for model training? (Privacy Center)](https://privacy.claude.com/en/articles/7996868-is-my-data-used-for-model-training)
- [Anthropic — Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
- [Anthropic — Embeddings](https://platform.claude.com/docs/en/build-with-claude/embeddings)
- [Voyage AI — Pricing](https://docs.voyageai.com/docs/pricing)
- [OpenAI — Pricing](https://developers.openai.com/api/docs/pricing)
- [OpenAI — Rate limits](https://developers.openai.com/api/docs/guides/rate-limits)
- [OpenAI — Your data](https://developers.openai.com/api/docs/guides/your-data)
- [OpenAI — Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [OpenAI — Function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [OpenAI — Embeddings guide](https://developers.openai.com/api/docs/guides/embeddings)
- [Google — Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Google — Gemini API Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
- [Google — Gemini API Terms](https://ai.google.dev/gemini-api/terms)
- [Google — Structured output](https://ai.google.dev/gemini-api/docs/structured-output)
- [Google — Function calling](https://ai.google.dev/gemini-api/docs/function-calling)
- [Google — Embeddings](https://ai.google.dev/gemini-api/docs/embeddings)
- [LangChain — Chat models guide (`init_chat_model`)](https://docs.langchain.com/oss/python/langchain/models)
- [LangChain — Structured output guide](https://docs.langchain.com/oss/python/langchain/structured-output)
- [LangChain — Chat model integrations table](https://docs.langchain.com/oss/python/integrations/chat)
- [LangChain — ChatAnthropic integration page](https://docs.langchain.com/oss/python/integrations/chat/anthropic)
- [LangGraph — Overview](https://docs.langchain.com/oss/python/langgraph/overview)

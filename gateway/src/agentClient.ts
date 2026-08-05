import type { AgentClient, TurnEvent, TurnRequest } from "./contract.ts";

export interface HttpAgentOptions {
  baseUrl: string;
  /** Local only. In production the call carries a Google-signed Cloud Run
   *  identity token instead, which the platform verifies for us. */
  devToken?: string;
  /** Production only. Returns the identity token for the agent's audience. */
  identityToken?: () => Promise<string>;
}

/**
 * A Google-signed identity token for one audience, from the metadata server
 * every Cloud Run instance carries. No key, no shared secret, no library.
 */
export async function metadataIdentityToken(audience: string): Promise<string> {
  const url =
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity" +
    `?audience=${encodeURIComponent(audience)}`;
  const response = await fetch(url, { headers: { "Metadata-Flavor": "Google" } });
  if (!response.ok) throw new Error(`metadata server returned ${response.status}`);
  return response.text();
}

export function httpAgent(options: HttpAgentOptions): AgentClient {
  return {
    async *turn(request: TurnRequest): AsyncIterable<TurnEvent> {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
        "X-Turn-Id": request.turnId,
      };
      if (options.devToken) headers["X-Dev-Auth"] = options.devToken;
      if (options.identityToken) {
        headers["Authorization"] = `Bearer ${await options.identityToken()}`;
      }

      const response = await fetch(`${options.baseUrl}/turn`, {
        method: "POST",
        headers,
        body: JSON.stringify({ user_id: request.userId, message: request.message }),
      });
      if (!response.ok || !response.body) {
        throw new Error(`agent returned ${response.status}`);
      }
      yield* ndjson(response.body);
    },
  };
}

async function* ndjson(body: ReadableStream<Uint8Array>): AsyncIterable<TurnEvent> {
  const decoder = new TextDecoder();
  let buffer = "";
  for await (const chunk of body) {
    buffer += decoder.decode(chunk, { stream: true });
    let newline = buffer.indexOf("\n");
    while (newline !== -1) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) yield JSON.parse(line) as TurnEvent;
      newline = buffer.indexOf("\n");
    }
  }
  const rest = buffer.trim();
  if (rest) yield JSON.parse(rest) as TurnEvent;
}

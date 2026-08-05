import type { AgentClient, TurnEvent, TurnRequest } from "./contract.ts";

export interface HttpAgentOptions {
  baseUrl: string;
  /** Local only. In production the call carries a Google-signed Cloud Run
   *  identity token instead, which the platform verifies for us. */
  devToken?: string;
}

export function httpAgent(options: HttpAgentOptions): AgentClient {
  return {
    async *turn(request: TurnRequest): AsyncIterable<TurnEvent> {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
        "X-Turn-Id": request.turnId,
      };
      if (options.devToken) headers["X-Dev-Auth"] = options.devToken;

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

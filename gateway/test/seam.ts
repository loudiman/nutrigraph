// The gateway seam: one HTTP request to Express, with the agent service faked.
import type { AddressInfo } from "node:net";
import type { Server } from "node:http";
import { createApp } from "../src/app.ts";
import type { AgentClient, TurnEvent, TurnRequest } from "../src/contract.ts";

export const SESSION_SECRET = "test-cookie-secret";
export const DEMO_USER_ID = "demo-user-1";

export interface SseEvent {
  name: string;
  data: TurnEvent;
}

export interface FakeAgent extends AgentClient {
  seen: TurnRequest[];
}

/** An agent that emits the events it was handed, then stops. */
export function fakeAgent(
  events: (request: TurnRequest) => TurnEvent[] | Promise<TurnEvent[]>,
): FakeAgent {
  const seen: TurnRequest[] = [];
  return {
    seen,
    async *turn(request: TurnRequest) {
      seen.push(request);
      for (const event of await events(request)) yield event;
    },
  };
}

export function echoEvents(request: TurnRequest): TurnEvent[] {
  return [
    { type: "node", turn_id: request.turnId, node: "load_profile" },
    { type: "node", turn_id: request.turnId, node: "route" },
    { type: "node", turn_id: request.turnId, node: "dispatch" },
    {
      type: "answer",
      turn_id: request.turnId,
      reply: {
        text: "Lou, I read that as: log_meal.",
        parts: [{ intent: "log_meal", text: "Lou, I read that as: log_meal." }],
        disclaimers: [],
      },
    },
  ];
}

export interface Harness {
  url: string;
  agent: FakeAgent;
  close(): Promise<void>;
}

export async function start(agent: FakeAgent): Promise<Harness> {
  const app = createApp({
    agent,
    sessionSecret: SESSION_SECRET,
    demoUserId: DEMO_USER_ID,
    log: () => {},
  });
  const server: Server = await new Promise((resolve) => {
    const s = app.listen(0, "127.0.0.1", () => resolve(s));
  });
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}`,
    agent,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
}

export async function postTurn(
  harness: Harness,
  message: string,
  cookie?: string,
): Promise<{ response: Response; events: SseEvent[]; setCookie: string | null }> {
  const response = await fetch(`${harness.url}/api/turn`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(cookie ? { Cookie: cookie } : {}),
    },
    body: JSON.stringify({ message }),
  });
  const setCookie = response.headers.get("set-cookie");
  const body = response.body ? await new Response(response.body).text() : "";
  return { response, events: parseSse(body), setCookie };
}

export function parseSse(body: string): SseEvent[] {
  return body
    .split("\n\n")
    .filter((block) => block.trim() !== "")
    .map((block) => {
      const lines = block.split("\n");
      const name = lines.find((l) => l.startsWith("event: "))!.slice("event: ".length);
      const data = lines.find((l) => l.startsWith("data: "))!.slice("data: ".length);
      return { name, data: JSON.parse(data) as TurnEvent };
    });
}

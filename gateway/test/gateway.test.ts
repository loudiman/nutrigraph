import assert from "node:assert/strict";
import { after, test } from "node:test";
import { SESSION_COOKIE } from "../src/session.ts";
import { DEMO_USER_ID, echoEvents, fakeAgent, postTurn, start } from "./seam.ts";
import type { Harness } from "./seam.ts";
import type { TurnEvent, TurnRequest } from "../src/contract.ts";

type Events = (request: TurnRequest) => TurnEvent[] | Promise<TurnEvent[]>;

const open: Harness[] = [];
after(async () => {
  await Promise.all(open.map((h) => h.close()));
});

async function harness(events: Events = echoEvents) {
  const h = await start(fakeAgent(events));
  open.push(h);
  return h;
}

test("a request with no cookie gets a signed cookie carrying a seeded user_id", async () => {
  const h = await harness();

  const { setCookie, response } = await postTurn(h, "hello");

  assert.equal(response.status, 200);
  assert.match(setCookie ?? "", new RegExp(`^${SESSION_COOKIE}=s%3A${DEMO_USER_ID}\\.`));
  assert.equal(h.agent.seen[0]?.userId, DEMO_USER_ID);
});

test("an existing signed cookie is reused and not reissued", async () => {
  const h = await harness();
  const first = await postTurn(h, "hello");
  const cookie = first.setCookie!.split(";")[0]!;

  const second = await postTurn(h, "again", cookie);

  assert.equal(second.setCookie, null);
  assert.equal(h.agent.seen[1]?.userId, DEMO_USER_ID);
});

test("node events reach the client, and the answer arrives as one event at the end", async () => {
  const h = await harness();

  const { events } = await postTurn(h, "I ate two eggs");

  assert.deepEqual(
    events.map((e) => e.name),
    ["node", "node", "node", "answer"],
  );
  const answer = events.at(-1)!.data;
  assert.equal(answer.type, "answer");
  assert.equal(answer.reply.text, "Lou, I read that as: log_meal.");
});

test("the gateway creates one turn identifier and every event carries it", async () => {
  const h = await harness();

  const { events, response } = await postTurn(h, "hello");

  const turnId = h.agent.seen[0]!.turnId;
  assert.match(turnId, /^[0-9a-f-]{36}$/);
  assert.equal(response.headers.get("x-turn-id"), turnId);
  assert.deepEqual(new Set(events.map((e) => e.data.turn_id)), new Set([turnId]));
});

test("a failure mid-turn arrives as a typed error event and the stream closes", async () => {
  const h = await harness((request) => [
    { type: "node", turn_id: request.turnId, node: "load_profile" },
    {
      type: "error",
      turn_id: request.turnId,
      code: "turn_failed",
      message: "The Coach could not finish that.",
    },
    { type: "node", turn_id: request.turnId, node: "never_reached" },
  ]);

  const { events } = await postTurn(h, "hello");

  assert.deepEqual(
    events.map((e) => e.name),
    ["node", "error"],
  );
  assert.equal(events.at(-1)!.data.type, "error");
});

test("an unreachable agent still ends the stream with a typed error event", async () => {
  const h = await harness(() => Promise.reject(new Error("connection refused")));

  const { events } = await postTurn(h, "hello");

  const last = events.at(-1)!.data;
  assert.equal(last.type, "error");
  assert.equal(last.code, "agent_unreachable");
  assert.equal(last.turn_id, h.agent.seen[0]!.turnId);
});

const REC_ID = "1f6bff0c-9c2e-4d5a-8f2e-3a9b6c1d0e4f";

test("accepting a Recommendation is relayed to the agent", async () => {
  const h = await harness();

  const response = await fetch(`${h.url}/api/recommendation/${REC_ID}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ accepted: true }),
  });

  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { accepted: true });
  assert.deepEqual(h.agent.answered, [{ recommendationId: REC_ID, accepted: true }]);
});

test("a second answer to the same suggestion passes the agent's 404 through", async () => {
  const h = await harness();
  h.agent.recommendationStatus = 404;

  const response = await fetch(`${h.url}/api/recommendation/${REC_ID}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ accepted: true }),
  });

  assert.equal(response.status, 404);
});

test("a malformed recommendation answer is refused before the agent is called", async () => {
  const h = await harness();

  const badBody = await fetch(`${h.url}/api/recommendation/${REC_ID}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ accepted: "yes" }),
  });
  const badId = await fetch(`${h.url}/api/recommendation/not-a-uuid`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ accepted: true }),
  });

  assert.equal(badBody.status, 400);
  assert.equal(badId.status, 400);
  assert.equal(h.agent.answered.length, 0);
});

test("an empty message is refused before the agent is called", async () => {
  const h = await harness();

  const response = await fetch(`${h.url}/api/turn`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: "   " }),
  });

  assert.equal(response.status, 400);
  assert.equal(h.agent.seen.length, 0);
});

// The one thing the gateway seam cannot see: what the internal call carries on
// the wire. A real agent is stood up, and the headers it receives are asserted.
import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import { httpAgent } from "../src/agentClient.ts";

async function agentThatRecordsHeaders() {
  const seen: Record<string, string | string[] | undefined>[] = [];
  const server = createServer((req, res) => {
    seen.push(req.headers);
    res.writeHead(200, { "Content-Type": "application/x-ndjson" });
    res.end('{"type":"node","turn_id":"t","node":"load_profile"}\n');
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    seen,
    baseUrl: `http://127.0.0.1:${port}`,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

const request = { userId: "demo-user-1", message: "hello", turnId: "t" };

test("the internal call carries a Google-signed identity token as a bearer token", async () => {
  const agent = await agentThatRecordsHeaders();
  const client = httpAgent({
    baseUrl: agent.baseUrl,
    identityToken: () => Promise.resolve("signed-by-google"),
  });

  for await (const _ of client.turn(request)) void _;
  await agent.close();

  assert.equal(agent.seen[0]?.["authorization"], "Bearer signed-by-google");
  assert.equal(agent.seen[0]?.["x-dev-auth"], undefined);
});

test("without an identity token the call carries the development header instead", async () => {
  const agent = await agentThatRecordsHeaders();
  const client = httpAgent({ baseUrl: agent.baseUrl, devToken: "local-development" });

  for await (const _ of client.turn(request)) void _;
  await agent.close();

  assert.equal(agent.seen[0]?.["x-dev-auth"], "local-development");
  assert.equal(agent.seen[0]?.["authorization"], undefined);
});

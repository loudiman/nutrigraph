import { createApp } from "./app.ts";
import { httpAgent } from "./agentClient.ts";

const port = Number(process.env["GATEWAY_PORT"] ?? 3000);
const sessionSecret = process.env["SESSION_SECRET"];
if (!sessionSecret) throw new Error("SESSION_SECRET is not set");

const app = createApp({
  agent: httpAgent({
    baseUrl: process.env["AGENT_URL"] ?? "http://127.0.0.1:8080",
    devToken: process.env["AGENT_DEV_TOKEN"],
  }),
  sessionSecret,
  demoUserId: process.env["DEMO_USER_ID"] ?? "demo-user-1",
});

app.listen(port, () => console.log(JSON.stringify({ event: "listening", port })));

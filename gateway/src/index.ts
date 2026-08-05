import { createApp } from "./app.ts";
import { httpAgent, metadataIdentityToken } from "./agentClient.ts";

// Cloud Run hands the port in as PORT and sets K_SERVICE on every instance.
const port = Number(process.env["PORT"] ?? process.env["GATEWAY_PORT"] ?? 3000);
const onCloudRun = Boolean(process.env["K_SERVICE"]);

const sessionSecret = process.env["SESSION_SECRET"];
if (!sessionSecret) throw new Error("SESSION_SECRET is not set");

const agentUrl = process.env["AGENT_URL"] ?? "http://127.0.0.1:8080";

const app = createApp({
  agent: httpAgent({
    baseUrl: agentUrl,
    devToken: process.env["AGENT_DEV_TOKEN"],
    identityToken: onCloudRun ? () => metadataIdentityToken(agentUrl) : undefined,
  }),
  sessionSecret,
  demoUserId: process.env["DEMO_USER_ID"] ?? "demo-user-1",
});

app.listen(port, () => console.log(JSON.stringify({ event: "listening", port })));

import { randomUUID } from "node:crypto";
import cookieParser from "cookie-parser";
import express from "express";
import type { AgentClient, ErrorEvent, TurnEvent } from "./contract.ts";
import { session } from "./session.ts";

export const FALLBACK_ERROR =
  "The Coach could not finish that. Nothing was saved. Please try again.";

export interface AppOptions {
  agent: AgentClient;
  sessionSecret: string;
  demoUserId: string;
  log?: (entry: Record<string, unknown>) => void;
}

export function createApp(options: AppOptions): express.Express {
  const log = options.log ?? ((entry) => console.log(JSON.stringify(entry)));
  const app = express();
  app.use(express.json());
  app.use(cookieParser(options.sessionSecret));
  app.use(session(options.demoUserId));

  app.get("/healthz", (_req, res) => {
    res.json({ status: "ok" });
  });

  app.post("/api/turn", async (req, res) => {
    const message: unknown = req.body?.message;
    if (typeof message !== "string" || message.trim() === "") {
      res.status(400).json({ error: "message is required" });
      return;
    }

    // One identifier ties a Turn together: the gateway creates it, sends it to
    // the agent as a header, and it lands on the `message` row and on every
    // trace in both services.
    const turnId = randomUUID();
    const userId = res.locals.userId as string;
    log({ event: "turn started", turn_id: turnId, user_id: userId });

    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "X-Turn-Id": turnId,
    });
    const send = (event: TurnEvent) => {
      res.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
    };

    try {
      for await (const event of options.agent.turn({ userId, message, turnId })) {
        send(event);
        // A typed error event ends the Turn, and the stream then closes.
        if (event.type === "error") break;
      }
    } catch (cause) {
      log({ event: "turn failed", turn_id: turnId, user_id: userId, cause: String(cause) });
      const failure: ErrorEvent = {
        type: "error",
        turn_id: turnId,
        code: "agent_unreachable",
        message: FALLBACK_ERROR,
      };
      send(failure);
    } finally {
      log({ event: "turn finished", turn_id: turnId, user_id: userId });
      res.end();
    }
  });

  return app;
}

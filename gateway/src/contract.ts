// Pydantic is the contract. These names come from the agent's OpenAPI document
// through `npm run gen:types`, so a contract change is a compile error here.
import type { components } from "./generated/agent.ts";

export type CoachReply = components["schemas"]["CoachReply"];
export type NodeEvent = components["schemas"]["NodeEvent"];
export type AnswerEvent = components["schemas"]["AnswerEvent"];
export type ErrorEvent = components["schemas"]["ErrorEvent"];
export type TurnEvent = components["schemas"]["TurnEventEnvelope"];

export interface TurnRequest {
  userId: string;
  message: string;
  turnId: string;
}

/** What the gateway knows about the agent service. The gateway seam fakes it. */
export interface AgentClient {
  turn(request: TurnRequest): AsyncIterable<TurnEvent>;
}

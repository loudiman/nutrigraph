// ADR 0002 keeps unredacted User text in the database for 90 days, and says the
// warning rather than the schema is what protects it. So the warning has to be
// on the page a User lands on, above the box, before anything is typed.
import assert from "node:assert/strict";
import { after, test } from "node:test";
import { echoEvents, fakeAgent, start } from "./seam.ts";
import type { Harness } from "./seam.ts";

const open: Harness[] = [];
after(async () => {
  await Promise.all(open.map((h) => h.close()));
});

async function page() {
  const h = await start(fakeAgent(echoEvents));
  open.push(h);
  const response = await fetch(h.url);
  return { response, html: (await response.text()).replace(/\s+/g, " ") };
}

test("the demo-data-only warning is on the page before a User can type", async () => {
  const { response, html } = await page();

  assert.equal(response.status, 200);
  assert.match(html, /Demo data only\./);
  assert.ok(
    html.indexOf("Demo data only.") < html.indexOf("<input id=\"message\""),
    "the warning must come before the box, not after it",
  );
});

test("the warning names what is stored and for how long", async () => {
  const { html } = await page();

  assert.match(html, /stored as written and kept for 90 days/);
  assert.match(html, /do not enter real personal or health details/);
});

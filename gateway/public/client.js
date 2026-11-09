// The Thread on screen. Talks to the gateway only: POST /api/turn is SSE over
// POST (EventSource cannot POST, so the stream is read by hand), and
// POST /api/recommendation/:id answers a Recommendation.

const thread = document.getElementById("thread");
const empty = document.getElementById("empty");
const warning = document.getElementById("warning");
const status = document.getElementById("status");
const statusText = document.getElementById("status-text");
const form = document.getElementById("ask");
const box = document.getElementById("message");
const sendButton = document.getElementById("send");

const NODE_LABELS = {
  load_profile: "reading your profile",
  guard: "checking the request",
  route: "working out what you mean",
  clarify: "writing a question back",
  update_profile: "updating your profile",
  log_meal: "logging the meal",
  review_day: "adding up your day",
  recommend: "looking for a suggestion",
  retrieve: "searching the guidance",
  answer_question: "writing the answer",
  dispatch: "thinking",
  compose_reply: "putting it into words",
  refuse: "declining",
};

// ponytail: the Thread cache is this device's localStorage, not the server —
// upgrade path is a GET /thread route on the agent when cross-device matters.
const STORE = "ng-thread-v1";
const MAX_ENTRIES = 200;
const entries = load();

function load() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORE) ?? "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function save() {
  if (entries.length > MAX_ENTRIES) entries.splice(0, entries.length - MAX_ENTRIES);
  try {
    localStorage.setItem(STORE, JSON.stringify(entries));
  } catch {
    // A full quota loses the cache, not the Thread — the server keeps the truth.
  }
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function render(entry) {
  empty.hidden = true;
  let node;
  if (entry.who === "user") {
    node = el("div", "msg user", entry.text);
  } else if (entry.who === "error") {
    node = el("div", "msg error", entry.text);
  } else {
    node = el("div", "msg coach", entry.text);
    for (const disclaimer of entry.disclaimers ?? []) {
      node.append(el("p", "disclaimer", disclaimer));
    }
    const citations = entry.citations ?? [];
    if (citations.length > 0) {
      const sources = el("details", "sources");
      sources.append(el("summary", "", `Sources · ${citations.length}`));
      for (const citation of citations) {
        const line = el("p", "citation");
        line.append(el("b", "", citation.document), ` — ${citation.locator}`);
        sources.append(line);
      }
      node.append(sources);
    }
    for (const rec of entry.recs ?? []) node.append(recControl(entry, rec));
  }
  thread.append(node);
  thread.scrollTop = thread.scrollHeight;
}

function recControl(entry, rec) {
  const wrap = el("p", "rec");
  if (rec.accepted) {
    wrap.append(el("span", "accepted", "Accepted ✓"));
    return wrap;
  }
  const button = el("button", "", "Sounds good — accept");
  button.type = "button";
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      const response = await fetch(`/api/recommendation/${rec.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ accepted: true }),
      });
      // 404 means it was already answered once; either way there is nothing
      // left to press.
      if (response.ok || response.status === 404) {
        rec.accepted = true;
        save();
        wrap.replaceChildren(el("span", "accepted", "Accepted ✓"));
        return;
      }
      throw new Error(`gateway returned ${response.status}`);
    } catch {
      button.disabled = false;
    }
  });
  wrap.append(button);
  return wrap;
}

function add(entry) {
  entries.push(entry);
  save();
  render(entry);
}

function setStatus(text) {
  if (text) {
    statusText.textContent = text;
    status.classList.add("live");
  } else {
    status.classList.remove("live");
  }
}

async function turn(message) {
  const response = await fetch("/api/turn", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!response.ok || !response.body) throw new Error(`gateway returned ${response.status}`);

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let end;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const line = block.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      handle(JSON.parse(line.slice(5)));
    }
  }
}

function handle(event) {
  if (event.type === "node") {
    setStatus(NODE_LABELS[event.node] ?? "working");
  } else if (event.type === "answer") {
    const reply = event.reply;
    add({
      who: "coach",
      text: reply.text,
      disclaimers: reply.disclaimers ?? [],
      citations: (reply.parts ?? []).flatMap((part) => part.citations ?? []),
      recs: (reply.parts ?? [])
        .filter((part) => part.recommendation_id)
        .map((part) => ({ id: part.recommendation_id, accepted: false })),
    });
  } else if (event.type === "error") {
    add({ who: "error", text: event.message });
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = box.value.trim();
  if (!message || sendButton.disabled) return;
  box.value = "";
  add({ who: "user", text: message });
  // The warning reached the User before this first message; from here on one
  // line is enough, and a tap reopens it.
  warning.open = false;
  sendButton.disabled = true;
  try {
    await turn(message);
  } catch {
    add({
      who: "error",
      text: "The Coach could not be reached. Nothing was saved. Please try again.",
    });
  } finally {
    setStatus(null);
    sendButton.disabled = false;
    box.focus();
  }
});

for (const example of empty.querySelectorAll("button")) {
  example.addEventListener("click", () => {
    box.value = example.textContent;
    box.focus();
  });
}

if (entries.length > 0) {
  warning.open = false;
  for (const entry of entries) render(entry);
} else {
  empty.hidden = false;
}

if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js");

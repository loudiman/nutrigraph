"""PROTOTYPE - throwaway. Answers issue #11: what does the graph state hold,
which nodes read and write it, and what does the checkpointer keep?

No LangGraph, no provider calls, no database. Every node is a stub that prints
what it read and what it wrote. The point is to see the state move.

Run:  python prototypes/graph_shape_PROTOTYPE.py
      python prototypes/graph_shape_PROTOTYPE.py -i     (type your own messages)
"""

from copy import deepcopy

# --- the state ------------------------------------------------------------
# CHECKPOINTED = survives the turn, lives in the Thread.
# TRANSIENT    = rebuilt every turn, must not bloat the checkpoint.

CHECKPOINTED = ["user_id", "messages", "pending_clarification"]
TRANSIENT = [
    "raw_message", "redacted_message", "profile", "rule_hits", "intents",
    "intent_index", "confidence", "out_of_scope", "parsed_items", "items",
    "day_totals", "retrieved", "draft", "validation_errors", "retries",
    "refusal", "final_text",
]


def new_state(user_id):
    return {
        "user_id": user_id,
        "messages": [],              # reducer: append
        "pending_clarification": None,
        "raw_message": None,
        "redacted_message": None,
        "profile": None,
        "rule_hits": [],
        "intents": [],               # ordered, at most 2
        "intent_index": 0,
        "confidence": 0.0,
        "out_of_scope": False,
        "parsed_items": [],
        "items": [],                 # matched + unmatched
        "day_totals": None,
        "retrieved": [],
        "draft": None,
        "validation_errors": [],
        "retries": 0,
        "refusal": None,
        "final_text": None,
    }


# --- the fake world -------------------------------------------------------

PROFILE = {
    "name": "Lou", "sex": "M", "age": 24, "height_cm": 172, "weight_kg": 78,
    "target_weight_kg": 72, "activity": "light", "allergies": ["peanut"],
    "diet_pattern": "omnivore", "units": "metric",
}

DENYLIST = ["dosage", "mg of", "diabetes", "blood sugar", "pregnan", "my child",
            "purge", "starve"]

# what a schema-constrained Gemini router would return, faked by keyword
FAKE_ROUTER = [
    (["ate", "had for", "kumain"], "log_meal"),
    (["how much", "how many", "is that enough", "what is"], "ask_question"),
    (["how did i eat", "today so far", "review"], "review_day"),
    (["what should i", "suggest", "dinner"], "recommend"),
    (["allergic", "my target", "i weigh"], "update_profile"),
]

FOOD_DB = {"egg": 78, "pandesal": 150, "toast": 75, "rice": 205}
LOCAL_PH = {"pandesal", "sinangag", "adobo"}


def log(node, read, wrote):
    print(f"\n  [{node}]")
    print(f"    reads : {', '.join(read) if read else '-'}")
    for k, v in wrote.items():
        print(f"    writes: {k} = {v!r}")


# --- the nodes ------------------------------------------------------------

def load_profile(s):
    s["profile"] = deepcopy(PROFILE)
    log("load_profile", ["user_id"], {"profile": "<9 fields, allergies=['peanut']>"})


def redact(text):
    # the real one runs as a WRAPPER on every provider call, not as a node
    return text.replace("Lou", "[NAME]")


def rules_scan(s):
    s["rule_hits"] = [w for w in DENYLIST if w in s["raw_message"].lower()]
    if s["rule_hits"]:
        s["out_of_scope"] = True
    log("rules_scan (deterministic)", ["raw_message"],
        {"rule_hits": s["rule_hits"], "out_of_scope": s["out_of_scope"]})


def router(s):
    sent = redact(s["raw_message"])          # <-- redaction wrapper
    s["redacted_message"] = sent
    low = sent.lower()
    found = []
    for keys, intent in FAKE_ROUTER:
        if any(k in low for k in keys) and intent not in found:
            found.append(intent)
    s["intents"] = found[:2]
    s["confidence"] = 0.9 if found else 0.2
    if "blood sugar" in low or "doctor" in low:
        s["out_of_scope"] = True
    log("router (Gemini Flash-Lite, schema)", ["redacted_message"],
        {"intents": s["intents"], "confidence": s["confidence"],
         "out_of_scope": s["out_of_scope"]})


def guardrail_refusal(s):
    s["refusal"] = ("BOUNDARY + DISCLAIMER + SEE A PROFESSIONAL + what I can do"
                    f"  (triggered by {s['rule_hits'] or 'router flag'})")
    s["final_text"] = s["refusal"]
    log("guardrail_refusal (template, not a model)",
        ["rule_hits", "out_of_scope"], {"final_text": "<fixed Refusal text>"})


def clarify(s):
    s["pending_clarification"] = "which of these did you mean?"
    s["final_text"] = "I did not follow. Did you want to log a meal, or ask a question?"
    log("clarify", ["confidence"],
        {"pending_clarification": s["pending_clarification"]})


def parse_meal(s):
    words = s["raw_message"].lower().replace(",", " ").split()
    s["parsed_items"] = [{"name": w.rstrip("s"), "qty": 2 if "two" in words else 1,
                          "unit": "piece"} for w in words if w.rstrip("s") in FOOD_DB
                         or w in LOCAL_PH]
    log("parse_meal (Gemini, schema)", ["redacted_message"],
        {"parsed_items": s["parsed_items"]})


def match_items(s):
    out = []
    for p in s["parsed_items"]:
        if p["name"] in LOCAL_PH:
            out.append({**p, "status": "matched", "source": "local",
                        "kcal": FOOD_DB.get(p["name"], 150)})
        elif p["name"] in FOOD_DB:
            out.append({**p, "status": "matched", "source": "fdc",
                        "kcal": FOOD_DB[p["name"]] * p["qty"]})
        else:
            out.append({**p, "status": "unmatched", "source": None, "kcal": None})
    s["items"] = out
    log("match_items (local table, then FDC search, then Gemini picks)",
        ["parsed_items"], {"items": s["items"]})


def write_meal(s):
    s["day_totals"] = {"kcal": sum(i["kcal"] or 0 for i in s["items"]),
                       "unmatched": [i["name"] for i in s["items"]
                                     if i["status"] == "unmatched"]}
    log("write_meal (tool -> Postgres)", ["items"], {"day_totals": s["day_totals"]})


def retrieve(s):
    s["retrieved"] = [{"doc": "DGA 2025-2030 p.14", "score": 0.81}]
    log("retrieve (pgvector, 768 dims)", ["redacted_message"],
        {"retrieved": s["retrieved"]})


def review_day(s):
    s["day_totals"] = s["day_totals"] or {"kcal": 1450, "unmatched": []}
    log("review_day (tool -> Postgres)", ["user_id"], {"day_totals": s["day_totals"]})


def recommend(s):
    s["draft"] = {"suggestion": "grilled chicken with peanut sauce", "why": "protein gap"}
    log("recommend (rules + similarity + Gemini)",
        ["profile", "day_totals"], {"draft": s["draft"]})


def update_profile(s):
    if "allergic" in s["raw_message"].lower():
        s["profile"]["allergies"].append("shrimp")
        s["draft"] = {"changed": "allergies", "value": s["profile"]["allergies"]}
    log("update_profile (Gemini, schema -> Postgres)", ["redacted_message"],
        {"draft": s["draft"]})


def allergy_check(s):
    text = str(s["draft"]) + str(s["items"])
    hit = [a for a in s["profile"]["allergies"] if a in text.lower()]
    if hit:
        s["validation_errors"].append(f"allergen {hit} in answer - regenerate once")
        s["draft"] = {"suggestion": "grilled chicken with calamansi", "why": "protein gap"}
    log("allergy_check (exact, on Items AND on text)",
        ["profile.allergies", "items", "draft"],
        {"hit": hit, "draft": s["draft"] if hit else "<unchanged>"})


def compose_and_validate(s):
    s["draft"] = s["draft"] or {"answer": "ok"}
    ok = isinstance(s["draft"], dict)
    if not ok and s["retries"] == 0:
        s["retries"] = 1
    parts = []
    if s["day_totals"]:
        parts.append(f"logged {s['day_totals']['kcal']} kcal")
        if s["day_totals"]["unmatched"]:
            parts.append(f"NOT counted: {s['day_totals']['unmatched']}")
    if s["retrieved"]:
        parts.append(f"per {s['retrieved'][0]['doc']}")
    if s["draft"].get("suggestion"):
        parts.append(s["draft"]["suggestion"])
    s["final_text"] = "; ".join(parts) or "ok"
    log("compose + schema validation (1 retry, then fixed fallback)",
        ["draft", "day_totals", "retrieved"],
        {"retries": s["retries"], "final_text": s["final_text"]})


INTENT_PATHS = {
    "log_meal": [parse_meal, match_items, write_meal],
    "ask_question": [retrieve],
    "review_day": [review_day],
    "recommend": [recommend],
    "update_profile": [update_profile],
}


# --- the graph ------------------------------------------------------------

def turn(state, message):
    print("=" * 72)
    print(f"USER: {message}")
    state["raw_message"] = message
    state["messages"].append({"role": "user", "text": message})
    for k in TRANSIENT:
        if k not in ("raw_message", "profile"):
            state[k] = new_state(state["user_id"])[k]

    load_profile(state)
    rules_scan(state)
    router(state)

    if state["out_of_scope"]:
        guardrail_refusal(state)
    elif state["confidence"] < 0.6:
        clarify(state)
    else:
        for i, intent in enumerate(state["intents"]):
            state["intent_index"] = i
            print(f"\n  -- intent {i + 1}/{len(state['intents'])}: {intent}")
            for node in INTENT_PATHS[intent]:
                node(state)
        allergy_check(state)
        compose_and_validate(state)

    state["messages"].append({"role": "coach", "text": state["final_text"]})
    print(f"\nCOACH: {state['final_text']}")
    print("\n  CHECKPOINT (survives the turn):")
    for k in CHECKPOINTED:
        print(f"    {k} = {state[k]!r}")
    print()
    return state


SCRIPT = [
    "I ate two eggs and pandesal, is that enough protein?",   # two intents
    "my doctor says my blood sugar is high, what should I eat?",  # refusal
    "hmm",                                                    # low confidence
    "I am allergic to shrimp",                                # update_profile
    "what should I have for dinner?",                         # allergy check bites
]

if __name__ == "__main__":
    import sys
    s = new_state("demo-user-1")
    if "-i" in sys.argv:
        while True:
            try:
                m = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not m:
                break
            s = turn(s, m)
    else:
        for m in SCRIPT:
            s = turn(s, m)

# Both services connect to the same database

The Express gateway and the Python agent service both connect directly to the one PostgreSQL database. The idea text described the graph's tools calling back into the Express APIs; we rejected that. A turn writes a Meal and reads the Profile on almost every message, and routing those writes over internal HTTP would add a network hop and a second failure layer to the hot path of every turn.

## Considered options

- **Express owns every table, and Python calls it over HTTP.** One writer, so validation and audit live in one place. Rejected because a meal with three foods becomes several internal round trips, and every failure then has two layers to diagnose.
- **Python owns every table, and Express is a pure gateway.** The cleanest boundary. Rejected because Express could then serve no history or profile page without asking Python, which hollows out the middleware layer that this system is meant to demonstrate.

## Consequences

- **One schema, two writers.** A migration must be released with both services in mind, and neither service may assume it is alone. Migrations belong to one owner: the Python service, because it holds the richer model.
- **The boundary is behavioural, not physical.** Express owns the session, the cookie, rate limiting, request logging, and the event stream. The Python service owns the graph, the provider calls, the guardrail, and the corpus. Sharing a database does not license either service to take the other's job.
- **The internal call is authenticated by Cloud Run identity.** The Python service accepts no public traffic; Express calls it with a Google-signed identity token, so there is no shared secret to rotate.
- **One identifier ties a turn together.** Express creates it for each turn, sends it as a header, and it becomes the `turn_id` on every `message`, every `interaction_event`, and every trace in both services.

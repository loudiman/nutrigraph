# The agent service. Installed editable so the runtime layout matches the
# checkout: `migrate.py` and `seed.py` find `migrations/` and `seeds/` by
# walking up from the package, and that only holds if `src/` stays where it is.
FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY migrations ./migrations
COPY seeds ./seeds

# Cloud Run hands the port in. No reloader here: `main.main()` is the laptop
# entry point, and importing `main` still runs the refusal-to-start check.
CMD exec uvicorn nutrigraph_agent.main:app --host 0.0.0.0 --port ${PORT:-8080}

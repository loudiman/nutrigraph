"""Environment configuration, and the one refusal-to-start rule that guards it."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

# The checkpointer's tables. The library owns every one of them and our
# migrations never touch them. They sit in `public` beside ours rather than in
# a schema of their own: putting them elsewhere means a `search_path` in the
# connection's startup packet, and Neon's pooled endpoint — which ADR 0004
# makes mandatory — rejects that parameter outright.
CHECKPOINT_TABLES = frozenset(
    {"checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"}
)


class ConfigurationError(RuntimeError):
    """The process must not start with this configuration."""


@dataclass(frozen=True)
class Settings:
    database_url: str
    host: str
    port: int
    # In production the gateway calls with a Google-signed Cloud Run identity
    # token, which the platform verifies before application code runs. That
    # token does not exist on a laptop, so locally we accept a plain header
    # instead — behind this one variable, which production never sets.
    dev_auth: bool
    dev_token: str
    # The provider is a configuration string, not a code path. Changing these
    # three values moves every model call; nothing else in the codebase knows
    # which vendor answers. The provider's own key variable (GOOGLE_API_KEY,
    # OPENAI_API_KEY, ANTHROPIC_API_KEY) is read from the environment by
    # LangChain and never enters this object, a log line, or a trace.
    model_provider: str = "google_genai"
    schema_model: str = "gemini-3.5-flash-lite"
    prose_model: str = "gemini-3.5-flash"

    @staticmethod
    def from_env(env: dict[str, str] | None = None) -> Settings:
        if env is None:
            # This also puts GOOGLE_API_KEY and the two LANGSMITH_ variables in
            # the environment, where LangChain and LangSmith read them. Turning
            # tracing on is those two variables and no code change.
            load_dotenv(_dotenv_path())
            env = dict(os.environ)
        settings = Settings(
            database_url=env.get("DATABASE_URL", ""),
            host=env.get("AGENT_HOST", "127.0.0.1"),
            port=int(env.get("AGENT_PORT", "8080")),
            dev_auth=env.get("AGENT_DEV_AUTH", "").strip().lower() in {"1", "true", "yes"},
            dev_token=env.get("AGENT_DEV_TOKEN", ""),
            model_provider=env.get("MODEL_PROVIDER", "google_genai"),
            schema_model=env.get("MODEL_SCHEMA", "gemini-3.5-flash-lite"),
            prose_model=env.get("MODEL_PROSE", "gemini-3.5-flash"),
        )
        settings.check()
        return settings

    def check(self) -> None:
        if not self.database_url:
            raise ConfigurationError("DATABASE_URL is not set")
        if self.dev_auth and self.host not in LOOPBACK_HOSTS:
            raise ConfigurationError(
                f"AGENT_DEV_AUTH is set and AGENT_HOST is {self.host!r}, which is not "
                f"loopback. The development header may only be accepted on a loopback "
                f"bind; production authenticates the gateway by Cloud Run identity."
            )
        if self.dev_auth and not self.dev_token:
            raise ConfigurationError("AGENT_DEV_AUTH is set but AGENT_DEV_TOKEN is empty")


def _dotenv_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "..", ".env")

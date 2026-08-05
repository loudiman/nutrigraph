"""Environment configuration, and the one refusal-to-start rule that guards it."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

# The library owns every table in here. Our migrations never touch it.
CHECKPOINT_SCHEMA = "langgraph"


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

    @staticmethod
    def from_env(env: dict[str, str] | None = None) -> Settings:
        if env is None:
            load_dotenv(_dotenv_path())
            env = dict(os.environ)
        settings = Settings(
            database_url=env.get("DATABASE_URL", ""),
            host=env.get("AGENT_HOST", "127.0.0.1"),
            port=int(env.get("AGENT_PORT", "8080")),
            dev_auth=env.get("AGENT_DEV_AUTH", "").strip().lower() in {"1", "true", "yes"},
            dev_token=env.get("AGENT_DEV_TOKEN", ""),
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

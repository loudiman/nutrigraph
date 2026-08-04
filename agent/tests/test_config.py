"""The one refusal-to-start rule."""

from __future__ import annotations

import pytest

from nutrigraph_agent.config import ConfigurationError, Settings

BASE = {
    "DATABASE_URL": "postgresql://nutrigraph:nutrigraph@localhost:5432/nutrigraph",
    "AGENT_DEV_AUTH": "1",
    "AGENT_DEV_TOKEN": "local-development",
}


def test_the_development_header_is_accepted_on_a_loopback_bind():
    settings = Settings.from_env({**BASE, "AGENT_HOST": "127.0.0.1"})

    assert settings.dev_auth is True


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.0.0.4"])
def test_the_agent_refuses_to_start_on_a_public_bind_with_the_development_header(host):
    with pytest.raises(ConfigurationError, match="not loopback"):
        Settings.from_env({**BASE, "AGENT_HOST": host})


def test_a_public_bind_without_the_development_header_is_fine():
    settings = Settings.from_env(
        {"DATABASE_URL": BASE["DATABASE_URL"], "AGENT_HOST": "0.0.0.0"}
    )

    assert settings.dev_auth is False

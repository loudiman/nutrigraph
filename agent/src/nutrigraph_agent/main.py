"""The process entry point. Importing this validates the configuration, so a
forbidden combination refuses to start rather than serving one request."""

from __future__ import annotations

import logging

from ._windows import use_selector_event_loop
from .config import Settings

use_selector_event_loop()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s turn_id=%(turn_id)s %(message)s",
)
logging.getLogger().handlers[0].addFilter(
    lambda record: setattr(record, "turn_id", getattr(record, "turn_id", "-")) or True
)

settings = Settings.from_env()

from .app import create_app  # noqa: E402  - after the configuration check

app = create_app(settings)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "nutrigraph_agent.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        reload_dirs=["src"],
    )


if __name__ == "__main__":
    main()

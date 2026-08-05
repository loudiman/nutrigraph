"""Write the agent's OpenAPI document to agent/openapi.json.

`gateway` generates its TypeScript from this file, and the generated file is
committed, so a reviewer sees a contract change in the diff.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nutrigraph_agent.app import create_app  # noqa: E402
from nutrigraph_agent.config import Settings  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "openapi.json"

if __name__ == "__main__":
    app = create_app(
        Settings(
            database_url="postgresql://unused/unused",
            host="127.0.0.1",
            port=8080,
            dev_auth=False,
            dev_token="",
        )
    )
    OUT.write_text(json.dumps(app.openapi(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")

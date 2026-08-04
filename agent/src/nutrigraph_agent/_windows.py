"""Windows only. psycopg's async mode cannot run on the ProactorEventLoop,
which is Windows' default, so the process asks for a selector loop instead.

`WindowsSelectorEventLoopPolicy` is deprecated in Python 3.14 and goes in 3.16.
Replace this with a loop factory once uvicorn accepts one.
"""

from __future__ import annotations

import asyncio
import sys
import warnings


def selector_event_loop_policy() -> asyncio.AbstractEventLoopPolicy | None:
    if sys.platform != "win32":
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return asyncio.WindowsSelectorEventLoopPolicy()


def use_selector_event_loop() -> None:
    policy = selector_event_loop_policy()
    if policy is None:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.set_event_loop_policy(policy)

"""Drive an orchestrator through the FastAPI app with a recording client.

Both the module-level client and the knowledge branch's own client are
swapped. The branch captures its client when it is built at startup, so
replacing only `app.client` would leave grounded generation and the
answerability guard talking to the network while everything else replayed from
cache -- reproducible in part, which is the worst of both.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from .caching_client import CachingChatClient
except ImportError:  # pragma: no cover
    from caching_client import CachingChatClient


@contextmanager
def recording_app(offline: bool = False, cache_path: Path | None = None):
    """Yield (TestClient, CachingChatClient) with every model call recorded."""

    from fastapi.testclient import TestClient

    import app as application

    real_client = application.client
    branch = application.knowledge_branch
    real_branch_client = getattr(branch, "chat_client", None)

    caching = CachingChatClient(
        inner=real_client,
        offline=offline,
        **({"path": cache_path} if cache_path else {}),
    )

    application.client = caching
    if branch is not None:
        branch.chat_client = caching
    try:
        yield TestClient(application.app), caching
    finally:
        caching.save()
        application.client = real_client
        if branch is not None:
            branch.chat_client = real_branch_client


def make_runner(test_client):
    """Return the (session_id, message) -> {reply, state} callable the harness wants."""

    def run(session_id: str, message: str) -> dict:
        response = test_client.post(
            "/chat", json={"message": message, "session_id": session_id}
        ).json()
        return {"reply": response["reply"], "state": response.get("state", {})}

    return run

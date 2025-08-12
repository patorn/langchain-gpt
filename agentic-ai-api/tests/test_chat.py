import os
import sys
from contextlib import contextmanager
from typing import Iterator

from fastapi.testclient import TestClient

# Ensure we can import the 'app' package from agentic-ai-api/
THIS_DIR = os.path.dirname(__file__)
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from app.main import app  # type: ignore  # noqa: E402
from app.routers import chat as chat_router_module  # type: ignore  # noqa: E402


class DummyAgent:
    def invoke(self, inputs: dict):
        # Mimic LangChain AgentExecutor output shape
        return {"output": f"Echo: {inputs.get('input', '')}"}


@contextmanager
def override_agent(agent) -> Iterator[None]:
    """Temporarily override get_agent dependency to return a provided agent."""
    async def _get_agent_override():
        return agent

    app.dependency_overrides[chat_router_module.get_agent] = _get_agent_override
    try:
        yield
    finally:
        app.dependency_overrides.pop(chat_router_module.get_agent, None)


def test_health_ok():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_chat_happy_path():
    dummy = DummyAgent()
    with override_agent(dummy):
        with TestClient(app) as client:
            payload = {"session_id": "s1", "message": "Hello"}
            r = client.post("/api/chat", json=payload)
            assert r.status_code == 200
            data = r.json()
            assert data["session_id"] == "s1"
            assert data["answer"].startswith("Echo: ")


def test_chat_validation_error():
    dummy = DummyAgent()
    with override_agent(dummy):
        with TestClient(app) as client:
            # Missing required fields should trigger 422
            r = client.post("/api/chat", json={})
            assert r.status_code == 422

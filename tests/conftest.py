import json
import os
import socketserver
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/mock_connect: the agent's own Connect test double

# litellm loads a .env at import in its default DEV mode, which can put a real model key into the test process
# mid-suite (tests then depend on their order, and could reach a paid API). Never in the tests.
os.environ["LITELLM_MODE"] = "PRODUCTION"

# The local servers (broker, mock Connect, facilitator) poll for shutdown every 0.5 s by default, so each stop()
# waited up to that long: about half the suite's time. In the tests only; same behaviour otherwise.
socketserver.BaseServer.serve_forever.__defaults__ = (0.05,)

from mock_connect.server import MockConnect, MockState  # noqa: E402


@pytest.fixture
def mock():
    m = MockConnect(MockState())
    yield m
    m.stop()


@pytest.fixture
def connect_ws(tmp_path, mock):
    """A Connect-shaped workspace whose real .mcp.json points at the mock server."""
    ws = tmp_path / "persistent_data"
    ws.mkdir()
    (ws / ".mcp.json").write_text(json.dumps(mock.mcp_json()))
    (ws / "AGENTS.md").write_text("# You are a Pearl Connect agent (test brief)\n")
    sk = ws / ".agents" / "skills" / "pearl-connect"
    sk.mkdir(parents=True)
    (sk / "SKILL.md").write_text("---\nname: pearl-connect\ndescription: Act on-chain.\n---\n# body\n")
    return ws


@pytest.fixture(autouse=True)
def _state_dir(tmp_path_factory, monkeypatch):
    """Default session dirs go to a temp state dir, never the real ~/.local/state."""
    from olas_ocb_agent import runtime

    monkeypatch.setattr(runtime, "STATE_DIR", tmp_path_factory.mktemp("state") / "olas-ocb-agent")


@pytest.fixture(autouse=True)
def _no_cancelled_prompt():
    """A Ctrl-C a test simulates cancels approval prompts until the REPL's next turn; never into the next test."""
    from olas_ocb_agent import cli

    cli.new_turn()
    yield
    cli.new_turn()


@pytest.fixture(autouse=True)
def _no_openrouter_provisioning(monkeypatch):
    """Tests never create real per-trial OpenRouter keys."""
    monkeypatch.delenv("OPENROUTER_PROVISIONING_KEY", raising=False)

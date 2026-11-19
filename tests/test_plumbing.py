"""connect_client, broker, runtime and workspace edges, against the mock Connect server."""

import json
import urllib.request

import pytest

from olas_ocb_agent.connect_client import ConnectClient, ConnectError, decode_tool_result, find_mcp_config
from olas_ocb_agent.model import ScriptedModel
from olas_ocb_agent.runtime import build, make_model
from olas_ocb_agent.workspace import discover_connect_workspace

# ---- connect_client --------------------------------------------------------------


def test_find_mcp_config_and_from_mcp_json(connect_ws, mock, tmp_path):
    url, token = find_mcp_config(connect_ws)
    assert url == mock.url and token == mock.token
    assert ConnectClient.from_mcp_json(connect_ws).list_tools()
    with pytest.raises(FileNotFoundError):
        find_mcp_config(tmp_path / "nowhere")


def test_rpc_body_parsing_and_errors(mock):
    sse = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\ndata:\n'
    assert ConnectClient._parse_rpc_body({"Content-Type": "text/event-stream"}, sse) == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"ok": True},
    }
    assert ConnectClient._parse_rpc_body({}, "  ") is None
    c = ConnectClient(mock.url, mock.token)
    with pytest.raises(ConnectError, match="nope"):  # JSON-RPC error from the server
        c._rpc("no/such/method", {})
    with pytest.raises(ConnectError, match="HTTP 401"):
        ConnectClient(mock.url, "wrong").list_tools()
    with pytest.raises(ConnectError, match="unknown tool"):
        c.call_tool("does_not_exist", {})


def test_http_error_with_non_json_body(monkeypatch):
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.HTTPError("u", 500, "x", {}, __import__("io").BytesIO(b"<html>oops"))

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(ConnectError, match="HTTP 500: <html>oops"):
        ConnectClient("http://127.0.0.1:1/mcp/", "t").rest("GET", "/wallet")


def test_empty_rpc_response(mock, monkeypatch):
    c = ConnectClient(mock.url, mock.token)
    monkeypatch.setattr(c, "_http", lambda *a, **k: (200, {}, ""))
    with pytest.raises(ConnectError, match="empty response"):
        c._rpc("tools/list", {})


def test_decode_tool_result_variants():
    assert decode_tool_result({"structuredContent": {"a": 1}}) == {"a": 1}
    assert decode_tool_result({"content": [{"type": "text", "text": '{"b": 2}'}]}) == {"b": 2}
    assert decode_tool_result({"content": [{"type": "text", "text": "plain"}]}) == "plain"


# ---- broker ------------------------------------------------------------------------


def _req(url, token, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_broker_routes(connect_ws, mock):
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    try:
        base, tok = rt.broker.url.removesuffix("/mcp/"), rt.broker.token
        assert rt.shell.hold == rt.broker.busy and not rt.broker.busy()  # nothing in flight
        assert _req(base + "/wallet", tok)[0] == 200
        assert _req(base + "/nope", tok)[0] == 404
        assert _req(base + "/wallet", tok, "POST", {})[0] == 404  # reads are GET-only
        code, body = _req(base + "/sign-message", tok, "POST", {"digest": "0x00"})
        assert code == 400 and "signing is off" in body["detail"]
        assert _req(rt.broker.url, tok, "POST", {"jsonrpc": "2.0", "id": 1, "method": "ping"})[1]["result"] == {}
        assert "error" in _req(rt.broker.url, tok, "POST", {"jsonrpc": "2.0", "id": 2, "method": "nope"})[1]
        rt.policy.caps.allow_sign_message = True
        code, body = _req(base + "/sign-message", tok, "POST", {"digest": "0x00"})
        assert code == 200 and body["signature"].startswith("0x")
        mock.stop()  # upstream gone: errors become refusals, not dropped connections
        assert _req(base + "/wallet", tok)[0] == 400
        assert _req(base + "/sign-message", tok, "POST", {"digest": "0x01"})[0] == 400
        r = _req(rt.broker.url, tok, "POST", {"jsonrpc": "2.0", "id": 3, "method": "tools/list"})[1]
        assert r["result"]["isError"]
    finally:
        rt.close()


class _Up:
    """An upstream that answers everything: what reaches it is never the point here."""

    mcp_url, token = "http://upstream/mcp/", "t"

    def call_tool_raw(self, name, args):
        return {"content": [{"type": "text", "text": "ok"}]}

    def list_tools(self):
        return []

    def rest(self, method, path, body=None):
        return {"ok": True}


@pytest.mark.parametrize(
    ("path", "body", "code"),
    [
        ("/safe-transaction", b"[1,2]", -32603),  # REST, a JSON list
        ("/safe-transaction", b"{bad", -32700),  # REST, not JSON
        ("/mcp", b"[]", -32603),  # MCP, a batch
    ],
)
def test_broker_answers_a_malformed_body_with_a_json_rpc_error(path, body, code):
    import http.client

    from olas_ocb_agent.broker import Broker
    from olas_ocb_agent.policy import Policy, SessionCaps

    caps = SessionCaps(max_native_usd=None, max_value_calls=None)
    b = Broker(_Up(), Policy(mode="auto", caps=caps, default_chain="gnosis")).start()
    port = int(b.url.removesuffix("/mcp/").rsplit(":", 1)[1])
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        c.request("POST", path, body, {"Content-Type": "application/json", "Authorization": f"Bearer {b.token}"})
        r = c.getresponse()  # an answer, not a dropped connection
        err = json.loads(r.read())["error"]
        assert r.status == 200 and err["code"] == code
        assert err["message"].startswith("broker: could not handle the request: ")
        # a tools/call whose arguments are a list is refused as a tool error, as before
        args = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "safe_transaction", "arguments": [1]},
        }
        res = _req(b.url, b.token, "POST", args)[1]["result"]
        assert res["isError"] and res["content"][0]["text"].startswith("broker: AttributeError")
    finally:
        c.close()
        b.stop()


def test_an_approved_retry_after_a_resume_is_not_new_spend(connect_ws, mock):
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    tx = {"chain": "gnosis", "target": "0x" + "ab" * 20, "value": 10**15, "request_id": "r1"}
    assert rt.policy.check("safe_transaction", tx).allowed
    sid, spent = rt.agent.session.id, rt.policy.spent_native
    rt.close()  # a crash right after the approval
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none", resume=sid)
    assert rt.policy.check("safe_transaction", tx).allowed  # the same call retried: no new spend
    assert rt.policy.value_calls == 1 and rt.policy.spent_native == spent
    assert rt.policy.check("safe_transaction", {**tx, "value": 2 * 10**15}).allowed  # same id, new arguments
    assert rt.policy.value_calls == 2
    rt.close()


def test_broker_catch_all_turns_gate_errors_into_refusals(connect_ws, mock):
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    try:

        class Boom:
            def before_agent_mech_request(self, deadline=None):
                raise RuntimeError("gate timed out")

        rt.broker.gate = Boom()
        from mock_connect.server import PRED_MECH

        args = {"prompt": "q", "tool": "prediction-online", "max_payment": 10**16, "priority_mech": PRED_MECH}
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "mech_request", "arguments": args},
        }
        r = _req(rt.broker.url, rt.broker.token, "POST", body)[1]["result"]
        assert r["isError"] and "gate timed out" in r["content"][0]["text"]
    finally:
        rt.close()


# ---- runtime -----------------------------------------------------------------------


def test_make_model_errors_and_kinds(mock):
    up = ConnectClient(mock.url, mock.token)
    with pytest.raises(ValueError, match="mech model needs"):
        make_model("mech:chat", None)
    with pytest.raises(ValueError, match="facilitator model needs"):
        make_model("facilitator:gemini-2.5-flash", None)
    with pytest.raises(ValueError, match="unknown model spec"):
        make_model("weird:x", up)
    assert make_model("litellm:openrouter/x?tools=1&provider=fireworks", None).tools is True
    assert make_model("mech:chat@0xabc", up).priority_mech == "0xabc"
    assert make_model("facilitator:gemini-2.5-flash@base", up, model_budget=5).budget == 5


def test_build_budget_conversion_resume_and_wallet_failure(connect_ws, mock, monkeypatch):
    from mock_connect.server import LLM_MECH

    rt = build(connect_ws, model=f"mech:chat@{LLM_MECH}", approval="auto", sandbox="none", model_budget=0.05)
    assert rt.agent.model.budget == 5 * 10**16  # $0.05 on gnosis: native xDAI at $1, 18 decimals
    rt.close()
    with pytest.raises(ValueError, match="payment asset"):  # no mech named: its asset can't be known
        build(connect_ws, model="mech:chat", approval="auto", sandbox="none")
    rt = build(connect_ws, model="facilitator:gemini-2.5-flash@base", approval="auto", sandbox="none")
    assert rt.agent.model.budget == 50_000  # base: a 6-dp stablecoin
    rt.close()
    sess = build(connect_ws, model=ScriptedModel(["hi"]), approval="auto", sandbox="none")
    sess.policy.spent_mech, sess.policy.value_calls = 0.4, 3  # spend from the broker during the session
    sess.agent.model.spent = 7  # a model with its own budget
    sess.agent.send("x")
    sid = sess.agent.session.id
    sess.close()
    m = ScriptedModel([])
    m.spent = 0
    rt = build(connect_ws, model=m, approval="auto", sandbox="none", resume="last")
    assert rt.agent.session.id == sid
    assert (rt.policy.spent_mech, rt.policy.value_calls, m.spent) == (0.4, 3, 7)  # caps continue
    rt.close()
    with pytest.raises(FileNotFoundError):
        build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none", resume="nope")
    monkeypatch.setattr(ConnectClient, "rest", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    assert rt.policy.default_chain is None
    rt.close()


# ---- workspace ---------------------------------------------------------------------


def test_discover_connect_workspace(tmp_path):
    home = tmp_path / ".operate"
    (home / "services" / "a").mkdir(parents=True)
    (home / "services" / "a" / "config.json").write_text("{not json")
    (home / "services" / "b").mkdir()
    (home / "services" / "b" / "config.json").write_text(json.dumps({"service_public_id": "valory/trader:0.1.0"}))
    assert discover_connect_workspace(home) is None
    store = tmp_path / "store"
    store.mkdir()
    (home / "services" / "c").mkdir()
    (home / "services" / "c" / "config.json").write_text(
        json.dumps(
            {"service_public_id": "valory/connect:0.1.0", "env_variables": {"STORE_PATH": {"value": str(store)}}}
        )
    )
    assert discover_connect_workspace(home) == store


def test_runtime_reads_read_only_flags_from_connect(connect_ws, mock, monkeypatch):
    from mock_connect.server import READ_ONLY

    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    assert rt.policy.read_only == READ_ONLY  # taken from the mock's readOnlyHint flags
    rt.close()
    monkeypatch.setattr(ConnectClient, "list_tools", lambda self: (_ for _ in ()).throw(ConnectError("down")))
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    assert "wallet_info" in rt.policy.read_only  # fallback list kept
    rt.close()


def test_broker_spend_is_saved_at_once_and_survives_a_crash(connect_ws, mock):
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    tx = {"chain": "gnosis", "target": "0x" + "ab" * 20, "value": 10**15}
    assert rt.policy.check("safe_transaction", tx).allowed  # mid-command, before any state event
    sid = rt.agent.session.id
    rt.close()  # the process "crashes" here: no command end, no model reply
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none", resume=sid)
    assert rt.policy.value_calls == 1 and rt.policy.spent_native > 0
    rt.close()


def test_a_failed_paid_model_call_is_saved(connect_ws, mock):
    from olas_ocb_agent.model import ModelError

    class PaidThenFails:
        spent = 0

        def query(self, messages, on_token=None):
            self.spent += 7
            raise ModelError("follow-up failed")

    m = PaidThenFails()
    rt = build(connect_ws, model=m, approval="auto", sandbox="none")
    with pytest.raises(ModelError):
        rt.agent.send("hi")
    sid = rt.agent.session.id
    rt.close()
    m2 = PaidThenFails()
    rt = build(connect_ws, model=m2, approval="auto", sandbox="none", resume=sid)
    assert m2.spent == 7  # restored
    rt.close()


def test_calls_of_a_killed_command_are_not_sent(connect_ws, mock):
    from olas_ocb_agent.broker import KILLED

    rt = build(connect_ws, model=ScriptedModel([]), approval="ask", sandbox="none")
    try:

        def approve_after_kill(summary):
            rt.shell._kill()  # the shell times the command out while the user is deciding
            return True

        rt.policy.approver = approve_after_kill
        base, tok = rt.broker.url.removesuffix("/mcp/"), rt.broker.token
        tx = {"chain": "gnosis", "target": "0x" + "ab" * 20, "value": 10**15}
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "safe_transaction", "arguments": tx},
        }
        r = _req(rt.broker.url, tok, "POST", body)[1]["result"]
        assert r["isError"] and r["content"][0]["text"] == KILLED
        code, out = _req(base + "/safe-transaction", tok, "POST", tx)
        assert code == 400 and out["detail"] == KILLED
        assert not [e for e in mock.s.ledger if e.get("kind") == "tx"]  # nothing reached Connect

        released = []

        class Gate:
            def before_agent_mech_request(self, deadline=None):
                pass

            def after_agent_mech_request(self, sent):
                released.append(sent)

        from mock_connect.server import PRED_MECH

        rt.broker.gate = Gate()
        args = {"prompt": "q", "tool": "prediction-online", "max_payment": 10**16, "priority_mech": PRED_MECH}
        body = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "mech_request", "arguments": args},
        }
        r = _req(rt.broker.url, tok, "POST", body)[1]["result"]
        assert r["content"][0]["text"] == KILLED and released == [False]  # the gate is released, nothing sent
    finally:
        rt.close()


@pytest.mark.parametrize("route", ["mcp", "rest"])
def test_a_call_queued_behind_a_slow_send_is_not_sent_after_the_kill(route):
    """A call queued behind a slow send: B passed the kill check and waits on the upstream lock while A is being sent; the shell kills the
    command. B must not be sent and gets KILLED; the kill's note names A's request id, so the agent can retry it.
    The same for the MCP tool call and signer_client.py's REST route."""
    import threading
    import time

    from olas_ocb_agent.broker import KILLED, Broker
    from olas_ocb_agent.policy import Policy, SessionCaps

    sent, started = [], threading.Event()

    class SlowUpstream:
        mcp_url, token = "http://upstream/mcp/", "t"

        def _slow(self, args):
            started.set()
            time.sleep(0.6)
            sent.append(args.get("request_id"))

        def call_tool_raw(self, name, args):
            self._slow(args)
            return {"content": [{"type": "text", "text": "ok"}], "structuredContent": {"tx_hash": "0x1"}}

        def rest(self, method, path, body):
            self._slow(body)
            return {"tx_hash": "0x1"}

    policy = Policy(mode="auto", caps=SessionCaps(max_native_usd=None, max_value_calls=None), default_chain="gnosis")
    b = Broker(SlowUpstream(), policy).start()
    try:
        assert b.command_killed() == ""  # nothing being sent: no note

        def call(rid):
            args = {"chain": "gnosis", "target": "0x" + "ab" * 20, "value": 1, "data": "0x", "request_id": rid}
            if route == "rest":
                code, out = _req(b.url.removesuffix("/mcp/") + "/safe-transaction", b.token, "POST", args)
                return "ok" if code == 200 and out == {"tx_hash": "0x1"} else out["detail"]
            body = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "safe_transaction", "arguments": args},
            }
            return _req(b.url, b.token, "POST", body)[1]["result"]["content"][0]["text"]

        out = {}
        t1 = threading.Thread(target=lambda: out.__setitem__("A", call("A")))
        t1.start()
        assert started.wait(5)  # A holds the lock, being sent
        t2 = threading.Thread(target=lambda: out.__setitem__("B", call("B")))
        t2.start()
        time.sleep(0.2)  # B waits on the lock
        note = b.command_killed()  # the shell's deadline
        t1.join()
        t2.join()
        assert sent == ["A"] and out == {"A": "ok", "B": KILLED}
        assert "safe_transaction" in note and "request_id A" in note and "reuse that request_id" in note
        b._sending = ("wallet_info", None)  # a call without a request id
        assert "check its outcome" in b.command_killed() and "request_id" not in b.command_killed()
    finally:
        b.stop()


def test_the_timeout_observation_carries_the_brokers_note(tmp_path):
    from olas_ocb_agent.shell import PersistentShell

    sh = PersistentShell(str(tmp_path), timeout=1)
    sh.on_kill = lambda: "broker: a safe_transaction call with request_id A was already sent"
    r = sh.run("echo before; sleep 5")
    assert (
        r["timed_out"] and r["output"] == "before\n\nbroker: a safe_transaction call with request_id A was already sent"
    )
    sh.on_kill = lambda: ""
    r = sh.run("sleep 5")
    assert r["timed_out"] and "broker" not in r["output"]
    sh.close()


def test_the_agent_is_only_a_cli_with_no_benchmark_adapter():
    """OnchainBench runs the agent as a CLI (`ocb-agent -p --output json`); the plugin adapter, its entry points
    and its dependency on the bench are gone, so the package installs and imports without the bench."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    pyproject = (root / "pyproject.toml").read_text()
    assert "onchainbench" not in pyproject and not (root / "olas_ocb_agent" / "ocb.py").exists()
    for f in (root / "olas_ocb_agent").rglob("*.py"):
        assert not re.search(r"^\s*(from|import) onchainbench\b", f.read_text(), re.M), f


def test_the_agent_reports_its_package_version_to_connect(connect_ws, mock, monkeypatch):
    import re
    from pathlib import Path

    from olas_ocb_agent import __version__

    pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    assert re.search(r'^version = "(.+)"$', pyproject, re.M).group(1) == __version__ == "0.1.0"
    seen = []
    real = ConnectClient._rpc

    def spy(self, method, params=None, notify=False):
        if method == "initialize":
            seen.append(params["clientInfo"])
        return real(self, method, params, notify=notify)

    with monkeypatch.context() as mp:
        mp.setattr(ConnectClient, "_rpc", spy)
        ConnectClient(mock.url, mock.token).list_tools()
    assert seen == [{"name": "olas-ocb-agent", "version": __version__}]
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    try:
        body = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        info = _req(rt.broker.url, rt.broker.token, "POST", body)[1]["result"]["serverInfo"]
        assert info == {"name": "ocb-agent-broker", "version": __version__}
    finally:
        rt.close()


def test_the_gates_pending_floor_survives_a_restart(connect_ws, mock):
    """The pending floor survives a restart: the floor is saved when the agent's mech request is sent, and --resume restores it, so the
    facilitator doesn't sign the slot Connect's pending request holds."""

    class Info:
        on_chain_nonce, next_nonce = 4, 4

    class FacilitatorModel(ScriptedModel):  # stands in for the facilitator backend: build() gives it the gate
        spent = deposited = 0

        def info(self):
            return Info()

    rt = build(connect_ws, model=FacilitatorModel([]), approval="auto", sandbox="none")
    gate = rt.broker.gate
    assert gate is not None and rt.agent.model.gate is gate
    gate.before_agent_mech_request()
    gate.after_agent_mech_request(sent=True)  # the agent's mech request went out at nonce 4
    assert gate.pending_floor == 5
    sid = rt.agent.session.id
    rt.close()  # the process "crashes": no command end, no model reply
    rt = build(connect_ws, model=FacilitatorModel([]), approval="auto", sandbox="none", resume=sid)
    assert rt.broker.gate.pending_floor == 5  # restored: the next model call waits for that request to settle
    rt.close()
    rt = build(connect_ws, model=FacilitatorModel([]), approval="auto", sandbox="none")
    assert rt.broker.gate.pending_floor is None  # a new session starts with none
    rt.close()


def test_resume_refreshes_the_system_prompt(connect_ws, mock):
    """A resumed session gets today's prompt: the sandbox actually in place and the brief as edited since."""
    rt = build(connect_ws, model=ScriptedModel(["hello"]), approval="auto", sandbox="auto")
    rt.agent.send("hi")
    sid, old = rt.agent.session.id, rt.agent.messages[0]["content"]
    rt.close()
    (connect_ws / "AGENTS.md").write_text("# You are a Pearl Connect agent (edited brief)\n")
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none", resume=sid)
    try:
        system = rt.agent.messages[0]
        assert system["role"] == "system" and system["content"] != old
        assert "edited brief" in system["content"] and "test brief" not in system["content"]
        assert (
            "Sandbox: no sandbox (requested)." in system["content"] and "No OS sandbox is active" in system["content"]
        )
        assert [m["content"] for m in rt.agent.messages[1:]] == ["hi", "hello"]  # the history itself is kept
    finally:
        rt.close()


def test_broker_reports_refusals(connect_ws, mock):
    """Policy refusals reach on_refusal (MCP and REST); a failing reporter never changes the agent's answer."""
    rt = build(connect_ws, model=ScriptedModel([]), approval="deny", sandbox="none")
    try:
        seen = []
        rt.broker.on_refusal = lambda tool, args, rule: seen.append((tool, args.get("target"), rule))
        base, tok = rt.broker.url.removesuffix("/mcp/"), rt.broker.token
        tx = {"chain": "gnosis", "target": "0x" + "ab" * 20, "value": 1}
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "safe_transaction", "arguments": tx},
        }
        assert _req(rt.broker.url, tok, "POST", body)[1]["result"]["isError"]
        assert _req(base + "/safe-transaction", tok, "POST", tx)[0] == 400
        assert [s[1] for s in seen] == [tx["target"]] * 2 and all("disabled" in s[2] for s in seen)

        def boom(*a):
            raise RuntimeError("reporter down")

        rt.broker.on_refusal = boom
        assert _req(base + "/safe-transaction", tok, "POST", tx)[0] == 400
    finally:
        rt.close()


def test_broker_reports_tool_listings(connect_ws, mock):
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    try:
        lists = []
        rt.broker.on_list = lists.append
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        assert _req(rt.broker.url, rt.broker.token, "POST", body)[1]["result"]["tools"] and lists

        def boom(tools):
            raise RuntimeError("reporter down")

        rt.broker.on_list = boom
        assert _req(rt.broker.url, rt.broker.token, "POST", body)[1]["result"]["tools"]
    finally:
        rt.close()

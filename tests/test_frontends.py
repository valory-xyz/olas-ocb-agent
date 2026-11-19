"""LitellmModel (with a fake litellm), MechModel edges, the `pearl` CLI and the olas-ocb-agent REPL."""

import sys
import types

import pytest
from mock_connect.server import LLM_MECH

from olas_ocb_agent import cli, pearl_cli
from olas_ocb_agent.connect_client import ConnectError
from olas_ocb_agent.model import LitellmModel, MechModel, ModelError, extract_mech_answer, mech_charge, parse_spec

# ---- LitellmModel ------------------------------------------------------------------


class _BadRequest(Exception):
    pass


def _fake_litellm(tool_args='{"command": "ls"}', reject_stop=True):
    calls = []

    def chunk(text):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=text))])

    def completion(**kw):
        calls.append(kw)
        if reject_stop and "stop" in kw:
            raise _BadRequest("stop not supported")
        return [chunk("hel"), chunk("lo"), types.SimpleNamespace(choices=[])]

    def builder(chunks, messages):
        tc = types.SimpleNamespace(id="c1", function=types.SimpleNamespace(name="bash", arguments=tool_args))
        msg = types.SimpleNamespace(tool_calls=[tc])
        usage = types.SimpleNamespace(model_dump=lambda: {"prompt_tokens": 3, "completion_tokens": 2})
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)

    def cost(completion_response):
        raise KeyError("no price")

    mod = types.SimpleNamespace(
        completion=completion,
        stream_chunk_builder=builder,
        completion_cost=cost,
        exceptions=types.SimpleNamespace(BadRequestError=_BadRequest),
    )
    return mod, calls


def test_litellm_model_streams_tools_and_falls_back(monkeypatch):
    mod, calls = _fake_litellm()
    monkeypatch.setitem(sys.modules, "litellm", mod)
    seen = []
    m = LitellmModel.from_spec("openrouter/x?tools=1&provider=fireworks")
    out = m.query([{"role": "user", "content": "hi"}], on_token=seen.append)
    assert out["content"] == "hello" and seen == ["hel", "lo"] and out["cost"] == 0.0
    assert out["usage"]["prompt_tokens"] == 3
    assert out["tool_calls"][0]["command"] == "ls"
    assert "stop" not in calls[-1] and calls[-1]["extra_body"]["provider"]["only"] == ["fireworks"]
    assert calls[-1]["num_retries"] == 3  # a provider's 429 is retried, not a failed trial (OnchainBench)
    mod2, _ = _fake_litellm(tool_args="not json", reject_stop=False)
    monkeypatch.setitem(sys.modules, "litellm", mod2)
    assert LitellmModel("x").query([{"role": "user", "content": "hi"}])["tool_calls"][0]["command"] == ""


def test_parse_spec_options():
    assert parse_spec("m?tools=0&flag&provider=a") == ("m", {"tools": False, "flag": True, "provider": "a"})


# ---- MechModel and answer extraction ------------------------------------------------


class _Client:
    def __init__(self, replies):
        self.replies = list(replies)

    def call_tool(self, name, args):
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_mech_model_pending_then_delivered_and_errors():
    delivered = {"delivery_results": {"r": {"response": {"text": "hi"}}}, "payment": 7}
    m = MechModel(_Client([{"pending_request_ids": ["r"]}, delivered]), tool="chat", budget=100)
    assert m.query([{"role": "user", "content": "x"}])["content"] == "hi" and m.spent == 7
    with pytest.raises(ModelError, match="budget exhausted"):
        MechModel(_Client([ConnectError("price 9 exceeds max_payment 1")]), tool="chat", budget=1).query([])
    with pytest.raises(ModelError, match="mech model call failed"):
        MechModel(_Client([ConnectError("down")]), tool="chat").query([])
    with pytest.raises(ModelError, match="budget exhausted"):
        MechModel(_Client([]), tool="chat", budget=0).query([])


def test_mech_model_real_answers():
    """A real Connect's answers: the inline off-chain envelope two wrappers deep, the mech report's payment label
    (a dict) where the charge would be, and mech_result's shape when the delivery comes after a poll."""
    envelope = {
        "request_id": "1",
        "status": "ok",
        "content_cid": "f0170",
        "response": {"schema_version": "2.0", "requestId": 1, "result": "ok\n```bash\nls\n```", "tool": "chat"},
    }
    label = {"delivery_results": {"r": envelope}, "payment": {"type": "usdc", "note": "x"}}
    m = MechModel(_Client([label]), tool="chat", budget=100)
    out = m.query([{"role": "user", "content": "x"}])
    assert out["content"] == "ok\n```bash\nls\n```" and m.spent == 100 and out["usage"]["mech_payment"] == 100
    polled = [
        {"pending_request_ids": ["r"], "payment": 20000},
        {"request_id": "r", "delivered": False, "note": "no delivery yet"},
        {"request_id": "r", "delivered": True, "result": envelope},
    ]
    m = MechModel(_Client(polled), tool="chat", budget=10**6)
    assert m.query([])["content"].startswith("ok") and m.spent == 20000
    m = MechModel(_Client([{"pending_request_ids": ["r"], "payment": 5}, ConnectError("gone")]), tool="chat")
    with pytest.raises(ModelError, match="mech model call failed"):
        m.query([])
    assert m.spent == 5  # paid, then the poll failed: the charge from mech_request's answer counts


def test_mech_charge():
    assert [mech_charge({"payment": p}, 9) for p in (7, "7", 0, -3, 2.0)] == [7, 7, 0, 0, 2]
    for garbage in ({"type": "usdc"}, [1], "lots", True, float("inf"), None):
        assert mech_charge({"payment": garbage}, 9) == 9
    assert mech_charge({}, 9) == 9 and mech_charge({}, None) == 0


def test_extract_mech_answer_shapes():
    with pytest.raises(ModelError):
        extract_mech_answer({})
    with pytest.raises(ModelError):
        extract_mech_answer({"delivered": False, "completion": None})
    # the bench stand-in: one wrapper around a message
    assert extract_mech_answer({"delivery_results": {"r": {"requestId": "r", "result": {"content": "s"}}}}) == "s"
    assert extract_mech_answer({"delivery_results": {"r": {"result": {"completion": "c"}}}}) == "c"
    assert extract_mech_answer({"delivery_results": {"r": {"response": {"text": "hi"}}}}) == "hi"
    # a mech's inline off-chain envelope: response, then result
    env = {"request_id": "1", "status": "ok", "response": {"schema_version": "2.0", "result": "two", "tool": "chat"}}
    assert extract_mech_answer({"delivery_results": {"r": env}}) == "two"
    assert extract_mech_answer({"delivery_results": {"r": env}, "completion": "connect's"}) == "two"
    # a result file (one wrapper around the text), and mech_result's answer
    assert extract_mech_answer({"delivery_results": {"r": {"result": "file"}}}) == "file"
    assert extract_mech_answer({"request_id": "r", "delivered": True, "result": env}) == "two"
    # Connect's top-level completion: when the delivery holds no text, or only an empty one, or none came
    tool_call = {"response": {"result": {"content": None, "tool_calls": [{}]}}}
    assert extract_mech_answer({"delivery_results": {"r": tool_call}, "completion": "```bash\nls\n```"}) == (
        "```bash\nls\n```"
    )
    empty = {"delivery_results": {"r": {"result": {"content": ""}}}}
    assert extract_mech_answer({**empty, "completion": "c"}) == "c" and extract_mech_answer(empty) == ""
    assert extract_mech_answer({"completion": "only"}) == "only"
    # garbage: never a crash, the unread payload as text
    assert extract_mech_answer({"delivery_results": {"r": {"result": {"other": 1}}}}) == '{"other": 1}'
    assert extract_mech_answer({"delivery_results": {"r": {"response": {"result": [1, 2]}}}}) == "[1, 2]"
    assert extract_mech_answer({"delivery_results": {"r": {"response": {"result": 3}}}}) == "3"
    assert extract_mech_answer({"delivery_results": {"r": {"status": "error"}}}) == '{"status": "error"}'
    assert extract_mech_answer({"delivery_results": {"r": "plain"}}) == "plain"
    assert extract_mech_answer({"delivery_results": {"r": None}}) == "None"


# ---- the pearl CLI ---------------------------------------------------------------------


def test_pearl_cli_commands(connect_ws, mock, monkeypatch, capsys):
    monkeypatch.chdir(connect_ws)  # the real .mcp.json here points straight at the mock
    assert pearl_cli.main([]) == 0 and "pearl tools" in capsys.readouterr().out
    assert pearl_cli.main(["tools"]) == 0 and "wallet_info" in capsys.readouterr().out
    assert pearl_cli.main(["help", "safe_transaction"]) == 0 and "parameters" in capsys.readouterr().out
    assert pearl_cli.main(["help", "nope"]) == 2
    assert pearl_cli.main(["wallet_info"]) == 0 and '"eoa"' in capsys.readouterr().out
    assert pearl_cli.main(["mech_result", "request_id=x"]) == 1  # tool error
    assert pearl_cli.main(["safe_transaction", "--chain"]) == 1  # flag without a value
    assert pearl_cli.main(["safe_transaction", "oops"]) == 1  # not key=value
    monkeypatch.chdir(connect_ws.parent.parent)
    assert pearl_cli.main(["tools"]) == 1  # no .mcp.json anywhere up the tree


# ---- the olas-ocb-agent REPL ------------------------------------------------------------------


class _TTY:
    """A terminal stdin whose lines are typed in advance; select sees it ready while lines remain."""

    def __init__(self, *lines):
        self.lines = list(lines)

    def isatty(self):
        return True

    def readline(self):
        return self.lines.pop(0) if self.lines else ""


def _tty(monkeypatch, *lines):
    stdin = _TTY(*lines)
    monkeypatch.setattr(cli.os, "name", "posix")
    monkeypatch.setattr(cli.sys, "stdin", stdin)
    monkeypatch.setattr(cli.select, "select", lambda r, w, x, t: (r, [], []) if stdin.lines else ([], [], []))
    return stdin


def test_approver_answers(monkeypatch, capsys):
    from olas_ocb_agent.policy import INPUT_CLOSED, UNANSWERED

    _tty(monkeypatch, "yes\n", "n\n", "\n")
    assert cli.approver("x") is True
    assert cli.approver("x") is False and cli.approver("x") is False  # no, and a bare Enter: the user declined
    _tty(monkeypatch)  # Ctrl-D at a real terminal: this call has no answer, but a later one can be asked
    monkeypatch.setattr(cli.select, "select", lambda r, w, x, t: (r, [], []))
    assert cli.approver("x") == INPUT_CLOSED
    assert "end of input: refused, not declined" in capsys.readouterr().out
    assert "Ctrl-D" in UNANSWERED[INPUT_CLOSED] and "nobody can approve" not in UNANSWERED[INPUT_CLOSED]


def test_approver_without_a_terminal_says_nobody_can_answer(monkeypatch, capsys):
    """stdin isn't a terminal (`-p ... < /dev/null`, a pipe): refused as 'no-approver' at once, never 'declined'."""
    from olas_ocb_agent.policy import NO_APPROVER, UNANSWERED, Policy

    monkeypatch.setattr(cli.sys, "stdin", __import__("io").StringIO("y\n"))
    assert cli.approver("pay 1 wei") == NO_APPROVER
    assert "stdin is not a terminal" in capsys.readouterr().out
    d = Policy(mode="ask", approver=cli.approver).check("safe_transaction", {"chain": "gnosis", "value": 1})
    assert not d.allowed and d.rule == UNANSWERED[NO_APPROVER] and "declined it" in d.rule and "nobody" in d.rule
    assert "the user declined" not in d.rule


def test_ctrl_c_at_an_approval_prompt_gives_stdin_back(monkeypatch, capsys):
    """The prompt runs in the broker's thread; Ctrl-C reaches the main thread. cancel_prompt() makes the prompt
    stop reading and returns only once it has, so the REPL's next line goes to the REPL, not to a dead prompt."""
    import threading
    import time

    from olas_ocb_agent.policy import APPROVAL_CANCELLED

    stdin = _tty(monkeypatch)  # nothing typed yet
    out = []
    t = threading.Thread(target=lambda: out.append(cli.approver("pay", timeout=30)))
    t.start()
    for _ in range(100):  # wait until the prompt is up
        if cli._prompt_lock.locked():
            break
        time.sleep(0.01)
    t0 = time.monotonic()
    cli.cancel_prompt()
    assert not cli._prompt_lock.locked() and time.monotonic() - t0 < 1  # it has stopped reading
    t.join(1)
    assert out == [APPROVAL_CANCELLED] and "[approval cancelled]" in capsys.readouterr().out
    stdin.lines.append("first line\n")  # typed afterwards: still there for the REPL
    assert stdin.readline() == "first line\n"
    # a prompt the stopped turn opens after the Ctrl-C (its summary was still being built) is cancelled at once,
    # without reading stdin, until the REPL starts the next turn
    stdin.lines.append("y\n")
    assert cli.approver("late", timeout=30) == APPROVAL_CANCELLED and stdin.lines == ["y\n"]
    cli.new_turn()
    assert cli.approver("next turn", timeout=30) is True
    t0 = time.monotonic()
    cli.cancel_prompt()  # with no prompt up, it returns at once
    assert time.monotonic() - t0 < 1 and cli._cancel.is_set()
    cli.new_turn()


def test_the_repl_starts_each_turn_with_prompts_allowed(connect_ws, mock, monkeypatch):
    """A Ctrl-C cancels the stopped turn's prompts, not the next turn's: the REPL clears the cancel before each send."""
    seen = []
    monkeypatch.setattr(
        cli.build.__globals__["Agent"], "send", lambda self, text, ui=None: (seen.append(cli._cancel.is_set()), "ok")[1]
    )
    cli._cancel.set()
    replies = iter(["hello", "/exit"])
    monkeypatch.setattr("builtins.input", lambda *a: next(replies))
    assert cli.main(["--dir", str(connect_ws), "--sandbox", "none", "--model", f"mech:chat@{LLM_MECH}"]) == 0
    assert seen == [False]


def test_approver_timeout_is_not_a_decline(monkeypatch, capsys):
    from olas_ocb_agent.policy import APPROVAL_TIMEOUT, UNANSWERED

    _tty(monkeypatch)
    assert cli.approver("pay", timeout=0.2) == APPROVAL_TIMEOUT
    assert "no answer in time: refused, not declined" in capsys.readouterr().out
    assert "nobody answered" in UNANSWERED[APPROVAL_TIMEOUT] and "10 minutes" in UNANSWERED[APPROVAL_TIMEOUT]


def test_console_ui_prints(capsys):
    ui = cli.console_ui()
    ui.on_token("t")
    ui.on_action("ls ")
    ui.on_observation({"output": "\n".join(str(i) for i in range(20)), "returncode": 1})
    ui.on_observation({"output": "", "returncode": 0})
    ui.on_info("i")
    out = capsys.readouterr().out
    assert "more lines" in out and "(rc=0)" in out and "$ ls" in out


def test_console_info_starts_on_its_own_line_after_a_streamed_reply(capsys):
    ui = cli.console_ui()
    ui.on_token("4")  # a single-chunk reply, streamed without a trailing newline
    ui.on_info("DONE")
    out = capsys.readouterr().out
    assert "4DONE" not in out and "\n" in out.split("4", 1)[1].split("DONE", 1)[0]


def test_console_lines_start_without_a_blank_line_when_nothing_streamed(capsys):
    ui = cli.console_ui(color=False)
    ui.on_info("compacted")  # before any reply has streamed: no stray empty line
    ui.on_token("done\n")  # a reply that ends its own line
    ui.on_action("ls")
    assert capsys.readouterr().out == "compacted\ndone\n$ ls\n"


def test_cli_main_paths(connect_ws, mock, monkeypatch, capsys):
    monkeypatch.setattr(cli, "discover_connect_workspace", lambda: None)
    assert cli.main([]) == 2
    mock.s.llm_backend = lambda msgs: "DONE: hi there"
    mock.s.prepaid = 10**18
    args = ["--dir", str(connect_ws), "--model", f"mech:chat@{LLM_MECH}", "--sandbox", "none"]
    assert cli.main([*args, "-p", "hello"]) == 0
    replies = iter(["", "/cost", "/compact", "chat", "boom", "stop", "/exit"])
    monkeypatch.setattr("builtins.input", lambda *a: next(replies))
    real_send = cli.build.__globals__["Agent"].send

    def send(self, text, ui=None):
        if text == "boom":
            raise ModelError("budget exhausted")
        if text == "stop":
            raise KeyboardInterrupt
        return real_send(self, text, ui)

    monkeypatch.setattr(cli.build.__globals__["Agent"], "send", send)
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "model: " in out and "[model] budget exhausted" in out and "[interrupted]" in out
    monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(EOFError()))
    assert cli.main(args) == 0  # end of input ends the REPL


def test_read_line_select_path(monkeypatch):
    import io

    class TTY(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(cli.os, "name", "posix")
    monkeypatch.setattr(cli.sys, "stdin", TTY("yes\n"))
    monkeypatch.setattr(cli.select, "select", lambda r, w, x, t: (r, [], []))
    assert cli._read_line("> ", 1) == "yes\n"
    monkeypatch.setattr(cli.select, "select", lambda r, w, x, t: ([], [], []))  # nothing typed in time
    assert cli._read_line("> ", 0.3) is None


def test_headless_model_error_exits_cleanly(connect_ws, mock, monkeypatch, capsys):
    class Broken:
        def query(self, messages, on_token=None):
            raise ModelError("budget exhausted")

    real = cli.build
    monkeypatch.setattr(cli, "build", lambda *a, **k: real(*a, **{**k, "model": Broken()}))
    assert cli.main(["--dir", str(connect_ws), "--sandbox", "none", "-p", "hi"]) == 1
    assert "[model] budget exhausted" in capsys.readouterr().err


def test_mech_model_counts_a_paid_call_whose_follow_up_fails():
    m = MechModel(_Client([{"pending_request_ids": ["r"]}, ConnectError("down")]), tool="chat", budget=100)
    with pytest.raises(ModelError, match="mech model call failed"):
        m.query([{"role": "user", "content": "x"}])
    assert m.spent == 100  # paid (up to the cap it sent), so it counts against the budget


def test_litellm_model_prefers_what_openrouter_billed(monkeypatch):
    from olas_ocb_agent.model import billed_cost

    mod, calls = _fake_litellm(reject_stop=False)
    billed = types.SimpleNamespace(choices=[], usage=types.SimpleNamespace(cost=0.0042, prompt_tokens=3))
    mod.completion = lambda **kw: (calls.append(kw), [billed])[1]
    monkeypatch.setitem(sys.modules, "litellm", mod)
    out = LitellmModel.from_spec("openrouter/x?provider=fireworks").query([{"role": "user", "content": "hi"}])
    assert out["cost"] == 0.0042 and out["cost_source"] == "billed"
    assert calls[-1]["extra_body"] == {
        "provider": {"only": ["fireworks"], "allow_fallbacks": False},
        "usage": {"include": True},
    }
    mod2, _ = _fake_litellm(reject_stop=False)  # no charge in the stream: the token estimate (here unpriced: 0)
    monkeypatch.setitem(sys.modules, "litellm", mod2)
    est = LitellmModel("anthropic/x").query([{"role": "user", "content": "hi"}])
    assert est["cost"] == 0.0 and est["cost_source"] == "estimate"
    extra = types.SimpleNamespace(usage=types.SimpleNamespace(model_extra={"cost": 0.5}))
    assert billed_cost([extra, types.SimpleNamespace(usage=None)]) == 0.5
    assert billed_cost([types.SimpleNamespace(usage=types.SimpleNamespace(model_extra=None))]) is None

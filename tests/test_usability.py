"""What the user sees and controls. Readable approvals and /cost, refusals on screen, nobody-answered vs
declined, resume recaps and session listings, one-line errors, the facilitator pre-deposit in the open, the kept
tail of a cut reply, the brief protected in plain folders, and ~/.operate and the user's caches closed as a whole."""

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from mock_connect.facilitator import RATE, MockFacilitator
from mock_connect.server import USDC

from olas_ocb_agent import __version__, cli, sandbox
from olas_ocb_agent.agent import COST_LIMIT, Agent, AgentConfig, Session, count_turns
from olas_ocb_agent.chains import fmt_amount, fmt_usd, native_symbol, parse_prices
from olas_ocb_agent.connect_client import ConnectClient
from olas_ocb_agent.interim.facilitator import FacilitatorModel
from olas_ocb_agent.interim.facilitator_assets import facilitator_symbol
from olas_ocb_agent.model import LitellmModel, ModelError, ScriptedModel
from olas_ocb_agent.policy import Policy, SessionCaps, summarize
from olas_ocb_agent.runtime import SessionNotFound, build, list_sessions, session_started, token_lookup
from olas_ocb_agent.shell import PersistentShell
from olas_ocb_agent.workspace import Workspace, build_system_prompt

SEATBELT = shutil.which("sandbox-exec") is not None
BWRAP = shutil.which("bwrap") is not None and platform.system() == "Linux"


# ---- the cut tail of a reply is kept on the record, not in the model's context --------------------------------


def test_the_dropped_tail_is_saved_on_the_message_but_never_sent_to_the_model(tmp_path):
    seen = []

    class Record(ScriptedModel):
        def query(self, messages, on_token=None):
            seen.append(messages)
            return super().query(messages, on_token)

    sh = PersistentShell(str(tmp_path), timeout=10)
    sess = Session.new(tmp_path / "s")
    reply = "```bash\necho a\n```\n<output>made up</output>\nrm -rf build"
    a = Agent(model=Record([reply, "done"]), shell=sh, system_prompt="s", session=sess)
    assert a.send("go") == "done"
    saved = [json.loads(x)["message"] for x in sess.path.read_text().splitlines() if '"message"' in x]
    cut = next(m for m in saved if m.get("dropped"))
    assert cut["content"] == "```bash\necho a\n```" and cut["dropped"] == "\n<output>made up</output>\nrm -rf build"
    assert all("dropped" not in m for m in seen[-1])  # the model gets the cut reply only, as before
    assert any(m["content"] == "```bash\necho a\n```" for m in seen[-1])
    b = Agent.resume(sess, Record(["ok"]), sh, "s")  # resumed: still recorded, still not sent
    b.send("again")
    assert all("dropped" not in m for m in seen[-1]) and any(m.get("dropped") for m in b.messages)
    sh.close()


# ---- session limit, turns, recap, listing ---------------------------------------------------------------------


def test_the_cost_limit_says_the_session_has_ended_and_what_to_do():
    class Paid(ScriptedModel):
        def query(self, messages, on_token=None):
            return {**super().query(messages, on_token), "cost": 0.5}

    shell = type("Sh", (), {"run": lambda self, c: {"output": "", "returncode": 0}, "cwd": "/"})()
    a = Agent(
        model=Paid(["```bash\ntrue\n```"] * 5), shell=shell, system_prompt="s", config=AgentConfig(cost_limit=1.0)
    )
    out = a.send("go")
    assert out == COST_LIMIT.format(limit=1.0, cost=1.0) and a.n_calls == 2  # the crossing call's command ran
    assert "can't go on at this limit" in out and "Start a new session" in out and "higher --cost-limit" in out


def test_turns_and_the_last_request_survive_resume_and_old_sessions_are_counted(tmp_path):
    sh = PersistentShell(str(tmp_path), timeout=10)
    sess = Session.new(tmp_path / "s")
    a = Agent(model=ScriptedModel(["```bash\ntrue\n```", "one", "two"]), shell=sh, system_prompt="s", session=sess)
    a.send("first")
    a.send("second\nrequest")
    b = Agent.resume(sess, ScriptedModel([]), sh, "s")
    assert (b.n_turns, b.last_request) == (2, "second\nrequest")
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "user", "content": "<returncode>0</returncode>"},
        {"role": "user", "content": "[Summary of the earlier conversation]\nx"},
    ]
    assert count_turns(msgs) == 1  # an older saved session has no turn count: counted from its messages
    sh.close()


def test_list_sessions_and_their_start_times(tmp_path):
    sdir = tmp_path / "sessions"
    sess = Session.new(sdir)
    sess.append({"type": "message", "message": {"role": "user", "content": "hello"}})
    sess.append({"type": "state", "state": {"cost": 0.25, "spent_native": 0.05, "n_calls": 3, "last_request": "hi"}})
    (sdir / "broken.jsonl").write_text("{not json\n")  # unreadable: left out, not a crash
    rows = list_sessions(tmp_path, sdir)
    assert len(rows) == 1 and rows[0]["id"] == sess.id
    assert rows[0]["turns"] == 1 and rows[0]["model_calls"] == 3 and rows[0]["usd"] == pytest.approx(0.30)
    assert session_started("20261116-101500-ab12cd") == "2026-11-16 10:15"
    assert session_started("odd") == "?" and session_started("2026111x-101500-ab") == "?"


# ---- readable approvals ---------------------------------------------------------------------------------------


def test_amounts_read_like_money():
    assert fmt_amount(5 * 10**17, 18, "xDAI", 1.0) == "0.5 xDAI (about $0.50)"
    assert fmt_amount(10**18, 18, "ETH", 3000.0) == "1 ETH (about $3,000.00)"
    assert fmt_amount(0, 18, "ETH") == "0 ETH" and fmt_amount(2**256 - 1, 0, "X").startswith("1157920892373161954")
    assert fmt_usd(0.001) == "$0.0010" and fmt_usd(1e-9) == "<$0.0001" and fmt_usd(-0.5) == "-$0.50"
    assert native_symbol("gnosis") == "xDAI" and native_symbol("mychain") == "native"
    assert facilitator_symbol("gnosis") == "xDAI" and facilitator_symbol("base") == "USDC"
    with pytest.raises(ValueError, match=r"like gnosis=1\.0"):
        parse_prices(["gnosis=cheap"])


def test_the_approval_prompt_shows_tokens_usd_and_whats_left():
    asked = []
    p = Policy(
        mode="ask",
        caps=SessionCaps(max_native_usd=0.10, max_value_calls=20),
        approver=lambda s: asked.append(s) or True,
        default_chain="gnosis",
        token_lookup=lambda chain, token: (6, "USDC", 1.0),
        mech_asset_lookup=lambda chain, mech: (6, 1.0, "USDC"),
    )
    target = "0x" + "ab" * 20
    assert p.check("safe_transaction", {"target": target, "value": 10**16}).allowed
    assert "on gnosis: to 0x" in asked[-1] and "value 0.01 xDAI (about $0.01)" in asked[-1]
    assert "left after this call: native $0.09 of $0.10, 19 of 20 value calls" in asked[-1]
    rcpt = "cd" * 20
    tr = "0xa9059cbb" + rcpt.rjust(64, "0") + hex(3_500_000)[2:].rjust(64, "0")
    assert p.check("safe_transaction", {"target": USDC, "data": tr}).allowed
    assert f"recipient 0x{rcpt}, amount 3.5 USDC (about $3.50)" in asked[-1] and "not counted" in asked[-1]
    ap = "0x095ea7b3" + rcpt.rjust(64, "0") + "f" * 64
    assert p.check("safe_transaction", {"target": USDC, "data": ap}).allowed
    assert f"spender 0x{rcpt}, amount unlimited USDC" in asked[-1] and "selector 0x095ea7b3" in asked[-1]
    mech = {"prompt": "q", "tool": "t", "max_payment": 10**4, "priority_mech": "0x1"}
    assert p.check("mech_request", mech).allowed
    assert "max_payment 0.01 USDC (about $0.01)" in asked[-1] and "mech $0.99 of $1.00" in asked[-1]

    def broken(chain, token):
        raise OSError("rpc down")

    q = Policy(mode="ask", approver=lambda s: asked.append(s) or True, default_chain="gnosis", token_lookup=broken)
    q.caps = SessionCaps(max_native_usd=None, max_value_calls=None)
    assert q.check("safe_transaction", {"target": USDC, "data": tr}).allowed
    assert "amount 3500000 base units" in asked[-1] and "left after this call" not in asked[-1]
    assert "value junk wei" in summarize("safe_transaction", {"value": "junk"}, native=("xDAI", 1.0))
    assert "max_payment x" in summarize("mech_request", {"max_payment": "x"}, mech=(6, 1.0))


def test_token_lookup_reads_decimals_and_symbol_through_the_chain_rpc(mock):
    c = ConnectClient(mock.url, mock.token)
    chains = c.rest("GET", "/wallet")["chains"]
    look = token_lookup(chains)
    assert look("gnosis", USDC) == (6, "USDC", 1.0) and look("gnosis", USDC.upper().replace("X", "x"))
    assert look("gnosis", "0x" + "99" * 20) is None and token_lookup({})("gnosis", USDC) is None


# ---- the facilitator pre-deposit, in the open -----------------------------------------------------------------


@pytest.fixture
def facilitator(connect_ws, mock):
    f = MockFacilitator(mock.s, backend=lambda req: "DONE: answered")
    yield f
    f.stop()


def test_the_pre_deposit_is_saved_at_once_announced_and_in_cost(connect_ws, mock, facilitator, monkeypatch):
    rt = build(
        connect_ws, model="facilitator:gemini-2.5-flash@gnosis", approval="auto", sandbox="none", max_deposit=3 * RATE
    )
    try:
        m = rt.agent.model
        assert isinstance(m, FacilitatorModel) and m.on_deposit == rt.agent._save_state
        m.base = facilitator.url
        saved, logs = [], []
        real = rt.agent._save_state
        monkeypatch.setattr(m, "on_deposit", lambda: (saved.append(m.deposited), real()))
        m.log = logs.append
        assert rt.agent.send("hi") == "DONE: answered"
        assert saved == [3 * RATE]  # saved inside the deposit, before the model call it pays for
        assert any("moved 0.003 xDAI (about $0.0030) from the safe" in s and "allowance used" in s for s in logs)
        report = cli.cost_report(rt)
        assert "facilitator pre-deposit: 0.003 xDAI (about $0.0030) moved from the safe" in report
        assert "model: 1 call, 0.001 xDAI (about $0.0010) paid to the marketplace, of a 0.05 xDAI" in report
        _, state = Session(rt.agent.session.path).replay()
        assert state["model_deposited"] == 3 * RATE and state["model_usd"] == pytest.approx(0.001)
    finally:
        rt.close()


def test_the_first_runs_low_deposit_error_says_what_to_run(connect_ws, mock, facilitator):
    fm = FacilitatorModel(ConnectClient(mock.url, mock.token), chain="gnosis", base_url=facilitator.url)
    with pytest.raises(ModelError) as e:
        fm.query([{"role": "user", "content": "hi"}])
    msg = str(e.value)
    assert "pre-deposit too low" in msg and "--facilitator-deposit is 0, the default" in msg
    assert f"--facilitator-deposit {50 * RATE} " in msg and "0.05 xDAI" in msg and "--model litellm:" in msg
    fm.max_deposit = fm.deposited = 1  # an allowance that is used up
    with pytest.raises(ModelError, match=r"allowance .* is used up"):
        fm.query([{"role": "user", "content": "hi"}])
    fm.asset = None  # a chain the table doesn't know
    assert fm.amount(5) == "5 base units of the payment asset"


# ---- the CLI ---------------------------------------------------------------------------------------------------


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out == f"ocb-agent {__version__}\n"


def test_help_shows_the_defaults_and_explains_approval_and_sandbox(monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "400")  # one line per flag, so the phrases below aren't wrapped
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert e.value.code == 0
    assert "(default: facilitator:gemini-2.5-pro;" in out and "default: 0.05)" in out  # --model, --model-budget
    assert "ask = prompt for each one (default); auto = allow them within the session caps; deny =" in out
    assert "auto = Seatbelt on macOS, bubblewrap on Linux" in out and "none = no sandbox" in out
    assert "a launcher such as Pearl passes one for each skill it ships" in out  # --trusted-skill-hash


def test_counts_are_singular_for_one():
    assert [cli.count(n, "turn") for n in (0, 1, 2)] == ["0 turns", "1 turn", "2 turns"]


def _litellm_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", None)  # `import litellm` fails as when the extra isn't installed


def test_litellm_without_its_extra_is_one_friendly_line(tmp_path, monkeypatch, capsys):
    _litellm_absent(monkeypatch)
    d = ["--dir", str(tmp_path), "--sandbox", "none", "--model", "litellm:anthropic/x"]
    assert cli.main([*d, "-p", "hi"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ocb-agent: error: the litellm backend needs olas-ocb-agent[litellm]: ")
    assert len(err.strip().splitlines()) == 1 and "Traceback" not in err
    price = '{"input_cost_per_token": 0, "output_cost_per_token": 0}'
    assert cli.main([*d, "--model-price", price, "-p", "hi"]) == 2  # registering a price needs litellm too
    assert capsys.readouterr().err.startswith("ocb-agent: error: the litellm backend needs olas-ocb-agent[litellm]: ")
    with pytest.raises(ModelError, match=r"needs olas-ocb-agent\[litellm\]: .*litellm"):
        LitellmModel("anthropic/x").query([{"role": "user", "content": "hi"}])  # a model built before it went


@pytest.mark.parametrize("extra", [[], ["--turn-seconds", "60"]])  # in this thread, and in a worker thread
@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("APIConnectionError: openrouter 502"),
        ConnectionResetError("reset by peer"),
        type("RateLimitError", (Exception,), {"__module__": "litellm.exceptions"})("slow down"),
    ],
)
def test_a_provider_error_in_a_text_turn_is_one_line_not_a_traceback(tmp_path, monkeypatch, capsys, exc, extra):
    class Broken:
        def query(self, messages, on_token=None):
            raise exc

    real = cli.build
    monkeypatch.setattr(cli, "build", lambda *a, **k: real(*a, **{**k, "model": Broken()}))
    d = ["--dir", str(tmp_path), "--sandbox", "none", "--model", "litellm:x", *extra]
    assert cli.main([*d, "-p", "hi"]) == 1
    out, err = capsys.readouterr()
    lines = [ln for ln in err.splitlines() if ln.startswith("ocb-agent: error: ")]
    assert lines == [f"ocb-agent: error: {type(exc).__name__}: {exc}"] and "Traceback" not in err + out


def test_the_default_model_without_connect_says_it_needs_a_connect_workspace(tmp_path, capsys):
    d = ["--dir", str(tmp_path), "--sandbox", "none"]
    assert cli.main([*d, "-p", "hi"]) == 2  # the default model: the facilitator
    err = capsys.readouterr().err
    assert err.startswith("ocb-agent: error: the facilitator model needs a Connect workspace")
    assert "--model litellm:<model>" in err and "payment asset" not in err
    assert cli.main([*d, "--model", "mech:chat@0xabc", "-p", "hi"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ocb-agent: error: a mech model needs a Connect workspace") and "payment asset" not in err


def test_no_ansi_colours_unless_stdout_is_a_terminal(tmp_path, monkeypatch, capsys):
    real = cli.build
    monkeypatch.setattr(cli, "build", lambda *a, **k: real(*a, **{**k, "model": ScriptedModel(["DONE: ok"] * 4)}))
    monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(EOFError()))
    d = ["--dir", str(tmp_path), "--sandbox", "none", "--model", "litellm:x"]
    assert cli.main([*d, "-p", "hi"]) == 0
    out, err = capsys.readouterr()
    assert "olas-ocb-agent · workspace" in out and "\033" not in out + err  # stdout is a pipe here
    assert cli.main(d) == 0  # the REPL, ended at once by end of input
    assert "\033" not in capsys.readouterr().out
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert cli.main([*d, "-p", "hi"]) == 0
    assert f"{cli.DIM}olas-ocb-agent · workspace" in capsys.readouterr().out  # a terminal: colours
    assert cli.palette(False) == ("",) * 5 and cli.palette(True) == (cli.DIM, cli.BOLD, cli.YEL, cli.RED, cli.RST)


def test_the_approval_prompt_has_no_colours_in_a_pipe(capsys):
    cli.new_turn()
    assert cli.approver("pay 1 xDAI") == cli.NO_APPROVER  # stdin isn't a terminal under pytest
    assert "\033" not in capsys.readouterr().out


def test_simple_mistakes_print_one_line_not_a_traceback(tmp_path, monkeypatch, capsys):
    d = ["--dir", str(tmp_path), "--sandbox", "none"]
    assert cli.main([*d, "--model", "gpt-5"]) == 2
    assert capsys.readouterr().err.strip() == "ocb-agent: error: unknown model spec 'gpt-5'"
    assert cli.main([*d, "--native-usd", "gnosis=cheap"]) == 2
    assert "expected <chain>=<USD price>" in capsys.readouterr().err
    other = "seatbelt" if platform.system() != "Darwin" else "bwrap"
    assert cli.main(["--dir", str(tmp_path), "--sandbox", other, "--model", "litellm:x"]) == 2
    assert "not available on this machine" in capsys.readouterr().err
    assert cli.main([*d, "--model", "litellm:x", "--resume", "last"]) == 2  # nothing to resume yet
    err = capsys.readouterr().err
    assert "no saved session to resume" in err and "no saved sessions for" in err
    with pytest.raises(SessionNotFound):
        build(tmp_path, model=ScriptedModel([]), sandbox="none", resume="nope")


def test_list_sessions_and_the_recap_on_resume(tmp_path, monkeypatch, capsys):
    d = ["--dir", str(tmp_path), "--sandbox", "none", "--model", "litellm:x"]
    assert cli.main([*d, "--list-sessions"]) == 0 and "no saved sessions" in capsys.readouterr().out
    real = cli.build
    monkeypatch.setattr(
        cli, "build", lambda *a, **k: real(*a, **{**k, "model": ScriptedModel(["DONE: The answer."] * 3)})
    )
    monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(EOFError()))
    assert cli.main([*d, "-p", "what is in my\nsafe?"]) == 0
    capsys.readouterr()
    assert cli.main([*d, "--list-sessions"]) == 0
    out = capsys.readouterr().out
    assert "1 turn " in out and "1 model call " in out and "what is in my safe?" in out  # singular for one
    assert cli.main([*d, "--resume", "last"]) == 0
    err = capsys.readouterr().err  # the recap is a status line: stderr
    assert "resumed session" in err and "1 turn, 1 model call," in err and "last request: what is in my safe?" in err
    assert cli.main([*d, "--resume", "nope"]) == 2
    err = capsys.readouterr().err
    assert "no session 'nope'" in err and "resume one with --resume <id>" in err


def test_cli_flags_reach_the_runtime(tmp_path, monkeypatch):
    got = {}

    def fake(*a, **k):
        got.update(k)
        raise ValueError("stop here")

    monkeypatch.setattr(cli, "build", fake)
    cli.main(["--dir", str(tmp_path), "--allow-sign-message", "--allow-brief-edits"])
    assert got["caps"].allow_sign_message is True and got["allow_brief_edits"] is True
    cli.main(["--dir", str(tmp_path)])
    assert got["caps"].allow_sign_message is False and got["allow_brief_edits"] is False
    monkeypatch.setattr(cli, "build", lambda *a, **k: {}["a harness bug"])  # not a user's mistake: a traceback
    with pytest.raises(KeyError):
        cli.main(["--dir", str(tmp_path)])


def test_blocked_calls_show_on_screen_and_in_cost_with_the_start_up_line(connect_ws, mock, monkeypatch, capsys):
    """auto mode says what its cap covers; a call a rule blocks is one line on screen and counted in /cost."""
    send = "```bash\npearl safe_transaction chain=gnosis target=0x" + "ab" * 20 + " value=1000000000000000000\n```"
    real = cli.build
    monkeypatch.setattr(cli, "build", lambda *a, **k: real(*a, **{**k, "model": ScriptedModel([send, "Blocked."])}))
    replies = iter(["send it", "/cost", "/exit"])
    monkeypatch.setattr("builtins.input", lambda *a: next(replies))
    assert cli.main(["--dir", str(connect_ws), "--sandbox", "none", "--approval", "auto"]) == 0
    cap = capsys.readouterr()
    out, err = cap.out, cap.err
    assert "auto: native value capped at $0.10 this session; ERC-20 transfers and approvals are not capped" in err
    assert "[blocked] safe_transaction: broker: session native cap $0.10 would be exceeded" in err
    assert "blocked calls: 1 (last: broker: session native cap" in out
    assert "value calls: 0 of 20 (20 left)" in out and "native value sent: $0.00 of $0.10 ($0.10 left)" in out
    assert "(session cost limit)" in out


def test_ask_mode_without_a_terminal_warns_at_start(connect_ws, mock, monkeypatch, capsys):
    real = cli.build
    monkeypatch.setattr(cli, "build", lambda *a, **k: real(*a, **{**k, "model": ScriptedModel(["DONE: ok"])}))
    monkeypatch.setattr(cli.sys, "stdin", __import__("io").StringIO(""))
    assert cli.main(["--dir", str(connect_ws), "--sandbox", "none", "-p", "hi"]) == 0
    assert "stdin is not a terminal, so nobody can approve" in capsys.readouterr().err


def test_the_models_log_is_on_screen_and_windows_reads_with_a_thread(tmp_path, monkeypatch, capsys):
    class Logging(ScriptedModel):
        log = None

        def query(self, messages, on_token=None):
            self.log("[ocb-agent] deposited something")
            return super().query(messages, on_token)

    real = cli.build
    monkeypatch.setattr(cli, "build", lambda *a, **k: real(*a, **{**k, "model": Logging(["DONE: ok"])}))
    assert cli.main(["--dir", str(tmp_path), "--sandbox", "none", "--model", "litellm:x", "-p", "hi"]) == 0
    assert "[ocb-agent] deposited something" in capsys.readouterr().err
    monkeypatch.setattr(cli.os, "name", "nt")
    monkeypatch.setattr(cli, "_read_line_thread", lambda end, tick: "y\n")
    assert cli._read_line("> ", 1) == "y\n"


def test_an_unexpected_error_in_the_repl_is_one_line(tmp_path, monkeypatch, capsys):
    """The REPL reports it and goes on; a text-mode -p turn raises as in 0.3.10, and --output json reports it."""

    class Bad:
        def query(self, messages, on_token=None):
            raise LookupError("model 'x' not found at the provider")

    real = cli.build
    monkeypatch.setattr(cli, "build", lambda *a, **k: real(*a, **{**k, "model": Bad()}))
    d = ["--dir", str(tmp_path), "--sandbox", "none", "--model", "litellm:x"]
    with pytest.raises(LookupError):
        cli.main([*d, "-p", "hi"])
    replies = iter(["hi", "/exit"])
    monkeypatch.setattr("builtins.input", lambda *a: next(replies))
    assert cli.main(d) == 0 and "[error] LookupError" in capsys.readouterr().out


def test_cost_report_with_no_caps(tmp_path):
    rt = build(tmp_path, model=ScriptedModel([]), sandbox="none", caps=SessionCaps(None, None, None))
    try:
        report = cli.cost_report(rt)
        assert "value calls: 0 no cap" in report and "$0.00 (no cap)" in report and "blocked calls: 0" in report
    finally:
        rt.close()


# ---- the brief in a plain folder ------------------------------------------------------------------------------


def test_a_plain_folders_brief_is_read_only_unless_the_user_allows_it(tmp_path, connect_ws):
    plain = tmp_path / "proj"
    plain.mkdir()
    w = Workspace.open(plain)
    assert w.deny_write == [plain.resolve() / "AGENTS.md", plain.resolve() / "CLAUDE.md"]
    assert Workspace.open(plain, allow_brief_edits=True).deny_write == []
    c = Workspace.open(connect_ws, allow_brief_edits=True)  # a Connect workspace's brief: always protected
    assert {p.name for p in c.deny_write} == {"AGENTS.md", "CLAUDE.md", ".agents", ".claude"}
    note = "AGENTS.md and CLAUDE.md here are read-only"
    assert note in build_system_prompt(w, sandbox.SEATBELT, 120)
    assert note not in build_system_prompt(w, "no sandbox (requested)", 120)  # nothing enforces it: not claimed
    assert note not in build_system_prompt(Workspace.open(plain, allow_brief_edits=True), sandbox.BWRAP, 120)
    assert note not in build_system_prompt(c, sandbox.SEATBELT, 120)


@pytest.mark.skipif(not (SEATBELT or BWRAP), reason="needs Seatbelt or bubblewrap")
def test_the_sandbox_keeps_a_plain_folders_brief(tmp_path):  # pragma: no cover - needs a sandbox
    (tmp_path / "AGENTS.md").write_text("brief")
    rt = build(tmp_path, model=ScriptedModel([]), sandbox="auto")
    try:
        assert rt.shell.run("echo x > AGENTS.md")["returncode"] != 0
        assert rt.shell.run("echo x > notes.md && cat notes.md")["output"].strip() == "x"
    finally:
        rt.close()
    assert (tmp_path / "AGENTS.md").read_text() == "brief"
    rt = build(tmp_path, model=ScriptedModel([]), sandbox="auto", allow_brief_edits=True)
    try:
        assert rt.shell.run("echo edited > AGENTS.md")["returncode"] == 0
    finally:
        rt.close()
    assert (tmp_path / "AGENTS.md").read_text().strip() == "edited"


# ---- ~/.operate as a whole, the workspace inside it read again ------------------------------------------------


def _operate(tmp_path):
    op = tmp_path / "home" / ".operate"
    root = op / "services" / "sc-1" / "persistent_data"
    ws = root / "ocb-agent"
    ws.mkdir(parents=True)
    (root / ".mcp.json").write_text("secret")
    (root / "AGENTS.md").write_text("brief")
    (op / "keys").mkdir()
    (op / "keys" / "k").write_text("key")
    (op / "services" / "sc-2").mkdir()
    (op / "services" / "sc-2" / "config.json").write_text("cfg")
    return op, root, ws


def test_operate_is_denied_as_a_whole_with_the_workspace_read_again(tmp_path, monkeypatch):
    from olas_ocb_agent import runtime

    assert sandbox.HOME / ".operate" in sandbox.DEFAULT_DENY_READ
    op, root, ws = _operate(tmp_path)
    r = os.path.realpath
    prof = sandbox.seatbelt_profile(ws, [op, root / ".mcp.json"], [], [root / "AGENTS.md"], [root], [root])
    assert f"(allow file-read* (subpath {sandbox._sb_str(r(root))}))" in prof
    mcp = sandbox._sb_str(r(root / ".mcp.json"))
    assert prof.rindex(f"(deny file-read* (literal {mcp}))") > prof.index("(allow file-read*")
    allow_ws = f"(allow file-write* (subpath {sandbox._sb_str(r(ws))}))"
    assert prof.rindex(allow_ws) > prof.rindex(f"(deny file-write* (subpath {sandbox._sb_str(r(op))}))")
    assert prof.endswith(f"(deny file-write* (literal {sandbox._sb_str(r(ws))}))")  # and the directory itself stays
    inner = ws / ".secret"  # denied inside the agent's own directory: still unwritable after it is re-allowed
    prof = sandbox.seatbelt_profile(ws, [op, inner], [], [ws / "brief"], [root], [root])
    assert prof.rindex(f"(deny file-write* (literal {sandbox._sb_str(r(inner))}))") > prof.rindex(allow_ws)
    assert prof.rindex(f"(deny file-write* (subpath {sandbox._sb_str(r(ws / 'brief'))}))") > prof.rindex(allow_ws)
    plain = sandbox.seatbelt_profile(ws, [op], [], allow_read=[tmp_path])  # not inside a denied dir: no rule
    assert "allow file-read*" not in plain
    (ws / "x").write_text("")
    keys = op / "keys"  # denied inside a denied dir: already under its tmpfs, so no mount point is made there
    args = sandbox.bwrap_args(ws, [keys, root / ".mcp.json", op], [], [root / "AGENTS.md", ws / "x"], [root], [root])
    assert r(keys) not in args
    i = args.index("--tmpfs")
    assert args[i : i + 2] == ["--tmpfs", r(op)]
    assert args[i + 2 : i + 5] == ["--ro-bind", r(root), r(root)] and args[i + 5 : i + 8] == ["--bind", r(ws), r(ws)]
    assert args[i + 8 : i + 11] == ["--ro-bind", r(ws / "x"), r(ws / "x")]  # deny_write in it, read-only again
    assert args.index("--remount-ro") > i and args.index(r(root / ".mcp.json")) > args.index("--remount-ro")
    seen = {}
    real = runtime.wrapper_for
    monkeypatch.setattr(runtime, "wrapper_for", lambda *a, **k: (seen.update(k), real(*a, **k))[1])
    build(tmp_path, model=ScriptedModel([]), sandbox="none").close()
    assert seen["allow_read"] == [tmp_path.resolve()]


@pytest.mark.skipif(not SEATBELT, reason="macOS only")
def test_seatbelt_reads_the_workspace_but_nothing_else_in_operate(tmp_path):  # pragma: no cover - macOS
    op, root, ws = _operate(tmp_path)
    wrap, _ = sandbox.wrapper_for(
        ws, deny_read=[op, root / ".mcp.json"], deny_write=[root / "AGENTS.md"], protect=[root], allow_read=[root]
    )
    _check_operate(wrap, op, ws)


@pytest.mark.skipif(not BWRAP, reason="Linux bubblewrap only")
def test_bwrap_reads_the_workspace_but_nothing_else_in_operate(tmp_path):  # pragma: no cover - Linux
    op, root, ws = _operate(tmp_path)
    wrap, _ = sandbox.wrapper_for(
        ws,
        deny_read=[op, root / ".mcp.json"],
        deny_write=[root / "AGENTS.md"],
        protect=[root],
        allow_read=[root],
        mode="bwrap",
    )
    _check_operate(wrap, op, ws)


def _check_operate(wrap, op: Path, ws: Path):  # pragma: no cover - needs a sandbox
    def run(cmd):
        return subprocess.run([*wrap, "bash", "-c", cmd], cwd=ws, capture_output=True, text=True, check=False)

    assert run("cat ../AGENTS.md").stdout == "brief"  # the workspace is readable
    assert run("echo x > f && cat f").stdout.strip() == "x"  # and the agent's directory writable
    for cmd in (
        "cat ../.mcp.json",  # still denied inside the re-allowed workspace
        f"cat {op}/keys/k",
        f"cat {op}/services/sc-2/config.json",  # another service's config
        "echo x > ../AGENTS.md",
        "echo x > ../new",
    ):
        r = run(cmd)
        assert r.returncode != 0 or r.stdout.strip() not in ("secret", "key", "cfg"), cmd
    assert (op / "keys" / "k").read_text() == "key" and not (ws.parent / "new").exists()


# A session in a fresh process whose HOME is a per-trial directory, as under the bench's isolation: the deny lists
# follow that HOME, a missing ~/.operate changes nothing, and the tools' caches are the agent's own.
TRIAL = r"""
import sys
from pathlib import Path
from olas_ocb_agent.model import ScriptedModel
from olas_ocb_agent.runtime import build
ws = Path(sys.argv[1])
rt = build(ws, model=ScriptedModel([]), sandbox="auto", sessions_dir=Path(sys.argv[2]))
print(rt.sandbox_desc)
for cmd in ("echo ok > f && cat f", 'echo c > "$XDG_CACHE_HOME/t" && cat "$XDG_CACHE_HOME/t"', "cat ~/.operate/k",
            'echo "$XDG_CACHE_HOME"'):
    r = rt.shell.run(cmd)
    print(f"{r['returncode']}|{r['output'].strip()}")
rt.close()
"""


@pytest.mark.skipif(not (SEATBELT or BWRAP), reason="needs Seatbelt or bubblewrap")
@pytest.mark.parametrize("operate", [False, True])
def test_a_per_trial_home_with_or_without_operate(tmp_path, operate):  # pragma: no cover - needs a sandbox
    home, ws = tmp_path / "home", tmp_path / "proj"
    (home / ".cache").mkdir(parents=True)
    ws.mkdir()
    if operate:
        (home / ".operate").mkdir()
        (home / ".operate" / "k").write_text("key")
    env = {**os.environ, "HOME": str(home), "XDG_STATE_HOME": str(tmp_path / "state")}
    r = subprocess.run(
        [sys.executable, "-c", TRIAL, str(ws), str(tmp_path / "sessions")],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    desc, *lines = r.stdout.strip().splitlines()
    assert desc in (sandbox.SEATBELT, sandbox.BWRAP), r.stderr
    ok, cache, key, xdg = lines
    assert ok == "0|ok" and cache == "0|c"  # the session works; its tools' cache is its own
    assert key.split("|")[0] != "0" and "key" not in key.split("|", 1)[1]  # absent or denied, never read
    # (a trial HOME under a temp dir is writable as part of that temp dir; tools cache in the agent's directory)
    assert Path(xdg.split("|", 1)[1]).resolve().is_relative_to(ws.resolve())


@pytest.mark.skipif(not BWRAP, reason="Linux bubblewrap only")
def test_bwrap_operate_outside_a_temp_dir(tmp_path):  # pragma: no cover - Linux
    """~/.operate where it really is: under HOME, outside every writable temp dir."""
    base = Path(tempfile.mkdtemp(prefix=".ocb-agent-test-", dir=Path.home()))
    try:
        op, root, ws = _operate(base)
        wrap, _ = sandbox.wrapper_for(
            ws,
            deny_read=[op, root / ".mcp.json"],
            deny_write=[root / "AGENTS.md"],
            protect=[root],
            allow_read=[root],
            mode="bwrap",
        )
        _check_operate(wrap, op, ws)
    finally:
        shutil.rmtree(base)


# ---- the review's fixes ---------------------------------------------------------------------------------------


def test_a_symlinked_brief_is_denied_as_named_and_bwrap_refuses_it(tmp_path, monkeypatch):
    from olas_ocb_agent import runtime

    (tmp_path / "CLAUDE.md").write_text("owner brief\n")
    (tmp_path / "AGENTS.md").symlink_to("CLAUDE.md")
    w = Workspace.open(tmp_path)
    prof = sandbox.seatbelt_profile(w.agent_dir, [], [], w.deny_write)
    link = Path(os.path.realpath(tmp_path)) / "AGENTS.md"  # the link itself, not only its target
    assert f"(deny file-write* (subpath {sandbox._sb_str(link)}))" in prof
    assert f"(deny file-write* (subpath {sandbox._sb_str(os.path.realpath(link))}))" in prof
    real = runtime.wrapper_for
    monkeypatch.setattr(runtime, "wrapper_for", lambda *a, **k: (real(*a, **{**k, "mode": "none"})[0], sandbox.BWRAP))
    with pytest.raises(RuntimeError, match=r"AGENTS\.md .* is a symlink, which bubblewrap can't keep read-only"):
        build(tmp_path, model=ScriptedModel([]), sandbox="auto")
    build(tmp_path, model=ScriptedModel([]), sandbox="auto", allow_brief_edits=True).close()  # allowed: no refusal


@pytest.mark.skipif(not SEATBELT, reason="macOS only")
def test_seatbelt_keeps_a_symlinked_brief(tmp_path):  # pragma: no cover - macOS
    (tmp_path / "CLAUDE.md").write_text("owner brief\n")
    (tmp_path / "AGENTS.md").symlink_to("CLAUDE.md")
    rt = build(tmp_path, model=ScriptedModel([]), sandbox="auto", sessions_dir=tmp_path.parent / "s-link")
    try:
        assert rt.shell.run("echo inj >> AGENTS.md")["returncode"] != 0
        assert rt.shell.run("rm AGENTS.md && echo 'injected brief' > AGENTS.md")["returncode"] != 0
    finally:
        rt.close()
    assert (tmp_path / "AGENTS.md").is_symlink() and Workspace.open(tmp_path).brief == "owner brief\n"


def test_json_mode_keeps_new_status_lines_off_stdout(connect_ws, mock, monkeypatch, tmp_path, capsys):
    """A blocked call is in the report's broker_refusals and a [blocked] line on stderr; stdout is the JSON alone."""
    from test_headless_cli import MODEL, _args, _json, _litellm

    send = "```bash\npearl safe_transaction chain=gnosis target=0x" + "ab" * 20 + " value=1000000000000000000\n```"
    mod, _, _ = _litellm(lambda msgs: send if msgs[-1]["content"] == "pay" else "DONE: blocked")
    monkeypatch.setitem(sys.modules, "litellm", mod)
    assert MODEL.startswith("litellm:")
    assert cli.main([*_args(connect_ws, tmp_path, "--output", "json"), "-p", "pay"]) == 0
    out, err = _json(capsys)
    assert out["broker_refusals"] and "native cap" in out["broker_refusals"][0]["rule"]
    assert "[blocked] safe_transaction" in err and "auto: native value capped at $0.10" in err
    sid = out["session_id"]
    assert cli.main([*_args(connect_ws, tmp_path, "--output", "json", "--resume", sid), "-p", "again"]) == 0
    out, err = _json(capsys)
    assert out["session_id"] == sid and f"resumed session {sid}" in err
    assert cli.main([*_args(connect_ws, tmp_path, "--output", "json", "--resume", "nope"), "-p", "x"]) == 1
    out, err = _json(capsys)  # the report as in 0.3.10, and the sessions in --sessions-dir on stderr
    assert out["error"] == "harness" and sid in err
    assert cli.main([*_args(connect_ws, tmp_path, "--list-sessions")]) == 0 and sid in capsys.readouterr().out


def test_inf_caps_read_as_no_cap(tmp_path, capsys, connect_ws, mock):
    inf = float("inf")
    asked = []
    p = Policy(mode="ask", caps=SessionCaps(inf, inf, None), approver=lambda s: asked.append(s) or True)
    assert p.check("safe_transaction", {"chain": "gnosis", "target": "0x1", "value": 1}).allowed
    assert "native: no cap" in asked[-1] and "$inf" not in asked[-1]
    rt = build(tmp_path, model=ScriptedModel([]), sandbox="none", caps=SessionCaps(inf, inf, 3))
    try:
        report = cli.cost_report(rt)
        assert "$0.00 (no cap)" in report and "$inf" not in report and "value calls: 0 of 3 (3 left)" in report
    finally:
        rt.close()
    rt_args = cli.argparse.Namespace(approval="auto", max_native_usd=inf, resume=None)  # --max-native-usd inf
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    try:
        cli._announce(rt, rt_args)
        assert "auto: native value not capped (no cap) this session" in capsys.readouterr().err
    finally:
        rt.close()


def test_a_failed_token_lookup_is_tried_again(mock):
    c = ConnectClient(mock.url, mock.token)
    chains = c.rest("GET", "/wallet")["chains"]
    broken = {k: {**v, "rpc": "http://127.0.0.1:9/"} for k, v in chains.items()}
    look = token_lookup(broken)
    assert look("gnosis", USDC) is None
    broken["gnosis"]["rpc"] = chains["gnosis"]["rpc"]  # the RPC is back: read now, not the cached failure
    assert look("gnosis", USDC) == (6, "USDC", 1.0)


def test_a_compaction_keeps_the_dropped_tail_on_the_record_only(tmp_path):
    seen = []

    class Record(ScriptedModel):
        def query(self, messages, on_token=None):
            seen.append(messages)
            return super().query(messages, on_token)

    sh = PersistentShell(str(tmp_path), timeout=10)
    sess = Session.new(tmp_path / "s")
    reply = "```bash\necho a\n```\n<output>made up</output>"
    a = Agent(model=Record([reply, "done", "SUMMARY"]), shell=sh, system_prompt="s", session=sess)
    a.config.keep_recent = 1
    a.send("go")
    a.compact()
    assert all("dropped" not in m for m in seen[-1])  # the summary request: no record-only field
    events = [json.loads(x) for x in sess.path.read_text().splitlines()]
    assert any(e["type"] == "compaction" for e in events)
    sh.close()


def test_the_harness_dir_is_git_ignored_and_its_caches_reserved(tmp_path):
    from olas_ocb_agent.workspace import CACHE_VARS

    env = Workspace.open(tmp_path).install_bin()
    assert (tmp_path / ".ocb-agent" / ".gitignore").read_text() == "*\n"
    for k in CACHE_VARS:
        assert Path(env[k]).is_dir() and k in cli.SHELL_ENV_RESERVED
    assert {"YARN_CACHE_FOLDER", "DENO_DIR", "PLAYWRIGHT_BROWSERS_PATH"} <= set(CACHE_VARS)

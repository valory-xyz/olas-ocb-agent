"""Unit tests for the edges of agent, shell, sandbox, policy and the marketplace gate."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from olas_ocb_agent.agent import Agent, AgentConfig, Session
from olas_ocb_agent.interim.marketplace_gate import MarketplaceGate
from olas_ocb_agent.model import ModelError, ScriptedModel
from olas_ocb_agent.policy import Policy, SessionCaps, _int, summarize
from olas_ocb_agent.sandbox import wrapper_for
from olas_ocb_agent.shell import PersistentShell, _truncate

# ---- agent ---------------------------------------------------------------------


def test_session_id_and_replay_of_missing_file(tmp_path):
    s = Session(tmp_path / "x" / "abc.jsonl")
    assert s.id == "abc"
    assert s.replay() == ([], {})


def test_cost_limit_stops_the_turn(tmp_path):
    class Costly:
        def query(self, messages, on_token=None):
            return {"content": "```bash\necho hi\n```", "cost": 5.0}

    sh = PersistentShell(str(tmp_path), timeout=10)
    a = Agent(model=Costly(), shell=sh, system_prompt="s", config=AgentConfig(cost_limit=1.0))
    assert "cost limit" in a.send("go")
    sh.close()


def test_step_limit_and_tool_call_session_state(tmp_path):
    calls = [{"id": "c", "name": "bash", "arguments": '{"command": "echo x"}', "command": "echo x"}]

    class Loop:
        def query(self, messages, on_token=None):
            return {"content": "", "tool_calls": calls}

    sh = PersistentShell(str(tmp_path), timeout=10)
    sess = Session.new(tmp_path / "s")
    a = Agent(model=Loop(), shell=sh, system_prompt="s", session=sess, config=AgentConfig(step_limit=2))
    assert "step limit" in a.send("go")
    assert '"type": "state"' in sess.path.read_text()
    sh.close()


def test_text_action_records_cwd_in_session(tmp_path):
    sh = PersistentShell(str(tmp_path), timeout=10)
    sess = Session.new(tmp_path / "s")
    a = Agent(model=ScriptedModel(["```bash\necho a\n```", "done"]), shell=sh, system_prompt="s", session=sess)
    assert a.send("go") == "done"
    assert '"type": "state"' in sess.path.read_text()
    sh.close()


def test_compact_is_a_noop_on_short_history(tmp_path):
    a = Agent(model=ScriptedModel([]), shell=None, system_prompt="s")
    a.compact()
    assert a.n_compactions == 0


def test_nudges_give_up_after_max(tmp_path):
    a = Agent(model=ScriptedModel(["thinking"] * 5), shell=None, system_prompt="s", config=AgentConfig(unattended=True))
    assert a.send("go") == "thinking"  # 3 nudges, then the plain reply is returned


def test_a_nudged_reply_ends_its_line_before_the_next_streams():
    from olas_ocb_agent.agent import UI

    tokens = []
    a = Agent(
        model=ScriptedModel(["4", "DONE: 2+2 = 4"]), shell=None, system_prompt="s", config=AgentConfig(unattended=True)
    )
    assert a.send("go", ui=UI(on_token=tokens.append)) == "2+2 = 4"
    assert "\n" in tokens  # the first reply (no command, no DONE) is ended before the nudged one streams


# ---- shell ---------------------------------------------------------------------


def test_shell_exit_restart_interrupt_and_truncate(tmp_path):
    sh = PersistentShell(str(tmp_path), timeout=10, max_output=40)
    r = sh.run("exit 3")
    assert r["returncode"] == -1 and sh.proc is None
    assert sh.run("echo back")["output"].strip() == "back"  # restarted on demand
    sh.interrupt()
    assert "elided" in sh.run("python3 -c \"print('x'*200)\"")["output"]
    sh.close()
    assert _truncate("short", 10) == "short"


# ---- sandbox -------------------------------------------------------------------


def test_sandbox_modes_off_macos(tmp_path, monkeypatch):
    assert wrapper_for(tmp_path, mode="none") == ([], "no sandbox (requested)")
    monkeypatch.setattr("olas_ocb_agent.sandbox.platform.system", lambda: "Linux")
    monkeypatch.setattr("olas_ocb_agent.sandbox.shutil.which", lambda name: None)  # no bwrap installed
    with pytest.raises(RuntimeError):
        wrapper_for(tmp_path, mode="seatbelt")
    with pytest.raises(RuntimeError):
        wrapper_for(tmp_path, mode="bwrap")
    assert "install bubblewrap" in wrapper_for(tmp_path)[1]
    monkeypatch.setattr("olas_ocb_agent.sandbox.platform.system", lambda: "Windows")
    assert "WSL2" in wrapper_for(tmp_path)[1]


@pytest.mark.skipif(os.name == "nt", reason="Windows paths can't hold the quote this checks; Seatbelt is macOS-only")
def test_seatbelt_profile_on_any_platform(tmp_path, monkeypatch):
    """CI runs on Linux: the macOS profile builder is unit-tested there too."""
    from olas_ocb_agent.sandbox import seatbelt_profile

    secret, keys = tmp_path / 'my "odd".json', tmp_path / "keys"
    secret.write_text("x")
    keys.mkdir()
    p = seatbelt_profile(tmp_path, [secret, keys], [tmp_path / "w"])
    real = os.path.realpath(tmp_path)
    assert p.startswith("(version 1)(allow default)") and '(deny file-write* (subpath "/"))' in p
    assert f'(deny file-read* (literal "{real}/my \\"odd\\".json"))' in p  # quotes escaped
    assert f'(deny file-read* (subpath "{real}/keys"))' in p
    assert f'(allow file-write* (subpath "{real}")' in p
    assert p.rindex(f'(deny file-write* (literal "{real}"))') > p.index("(allow file-write*")  # parents can't move
    p = seatbelt_profile(tmp_path / "agent", [], [], deny_write=[tmp_path / "AGENTS.md", tmp_path / ".claude"])
    assert p.index(f'(deny file-write* (subpath "{real}/AGENTS.md"))') > p.index("(allow file-write*")  # wins
    assert f'(deny file-write* (subpath "{real}/.claude"))' in p  # not there yet: can't be created either
    # a protected root under a writable temp dir: denied after the allow, the workspace inside it allowed again
    p = seatbelt_profile(tmp_path / "ws" / "agent", [], [tmp_path], protect=[tmp_path / "ws"])
    deny_root = p.index(f'(deny file-write* (subpath "{real}/ws"))')
    assert p.index("(allow file-write*") < deny_root and f'(subpath "{real}")' in p  # the temp dir is writable
    assert p.index(f'(allow file-write* (subpath "{real}/ws/agent"))') > deny_root
    assert p.index(f'(deny file-write* (literal "{real}"))') > deny_root  # the dir above it can't be renamed
    p = seatbelt_profile(tmp_path / "outside", [], [], protect=[tmp_path / "ws"])
    assert f'(allow file-write* (subpath "{real}/outside"))' not in p  # a workspace outside it isn't allowed again
    monkeypatch.setattr("olas_ocb_agent.sandbox.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "olas_ocb_agent.sandbox.shutil.which", lambda n: "/usr/bin/sandbox-exec" if n == "sandbox-exec" else None
    )
    wrap, desc = wrapper_for(tmp_path)
    assert desc == "macOS Seatbelt" and wrap[:2] == ["sandbox-exec", "-p"]


def test_bwrap_args_on_linux(tmp_path, monkeypatch):
    from olas_ocb_agent.sandbox import bwrap_args

    secret, keys = tmp_path / ".mcp.json", tmp_path / "keys"
    secret.write_text("x")
    keys.mkdir()
    args = bwrap_args(tmp_path, [secret, keys, tmp_path / "missing"], [tmp_path / "nope"])
    assert args[:4] == ["bwrap", "--ro-bind", "/", "/"] and args[-1] == "--"
    ws = os.path.realpath(tmp_path)

    def has(*seq):
        return any(args[i : i + len(seq)] == list(seq) for i in range(len(args)))

    assert has("--bind", ws, ws)
    assert has("--ro-bind", "/dev/null", os.path.realpath(secret))  # the file is covered
    assert has("--tmpfs", os.path.realpath(keys))  # the directory is covered (empty)
    assert args.index("--tmpfs") > args.index("--bind")  # covers come after the writable binds
    assert "missing" not in " ".join(args) and "nope" not in " ".join(args)
    brief = tmp_path / "AGENTS.md"
    brief.write_text("b")
    args = bwrap_args(tmp_path / "agent", [], [], deny_write=[brief, tmp_path / ".claude"])
    assert has("--ro-bind", os.path.realpath(brief), os.path.realpath(brief)) and ".claude" not in " ".join(args)
    # a protected root under a writable temp dir: its top directory there goes read-only (it can't be renamed and
    # nothing new appears under it), then the workspace is bound writable on top
    top, ws_dir = tmp_path / "top", tmp_path / "top" / "ws" / "agent"
    ws_dir.mkdir(parents=True)
    (tmp_path / "secret").mkdir()
    args = bwrap_args(ws_dir, [tmp_path / "secret", tmp_path], [tmp_path], protect=[top / "ws", top / "ws"])
    t, w, sec = (os.path.realpath(x) for x in (top, ws_dir, tmp_path / "secret"))
    assert has("--bind", ws, ws) and has("--ro-bind", t, t) and has("--bind", w, w) and has("--ro-bind", sec, sec)
    assert args.index(t) < args.index(w) and args.count(t) == 2  # once (deduplicated), before the workspace
    # the outer denied directory is covered; the one inside it is under that tmpfs already, so it gets no mount
    assert has("--tmpfs", ws, "--remount-ro", ws) and not has("--tmpfs", sec)
    # $TMPDIR a directory inside a writable one: bound on its own, and the anchor taken below it, not above it
    tmpdir = tmp_path / "user-1"
    (tmpdir / "pe2-x" / "persistent_data" / "agent").mkdir(parents=True)
    monkeypatch.setattr("olas_ocb_agent.sandbox.tempfile.gettempdir", lambda: str(tmpdir))
    root = tmpdir / "pe2-x" / "persistent_data"
    args = bwrap_args(root / "agent", [], [tmp_path], protect=[root])
    td, pe2 = os.path.realpath(tmpdir), os.path.realpath(tmpdir / "pe2-x")
    assert has("--bind", td, td) and has("--ro-bind", pe2, pe2) and not has("--ro-bind", td, td)
    assert args.index(ws) < args.index(td) < args.index(pe2)  # outer binds first
    monkeypatch.setattr("olas_ocb_agent.sandbox.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "olas_ocb_agent.sandbox.shutil.which", lambda name: "/usr/bin/bwrap" if name == "bwrap" else None
    )
    wrap, desc = wrapper_for(tmp_path)
    assert desc == "Linux bubblewrap" and wrap[0] == "bwrap"


@pytest.mark.skipif(__import__("shutil").which("bwrap") is None, reason="Linux bubblewrap only")
def test_bwrap_denied_file_cant_be_read_renamed_or_written(tmp_path):  # pragma: no cover - runs on Linux
    import subprocess

    secret = tmp_path / ".mcp.json"
    secret.write_text("SECRET")
    (tmp_path / "agent").mkdir()
    wrap, _ = wrapper_for(tmp_path, deny_read=[secret], mode="bwrap")
    alive = subprocess.run([*wrap, "echo", "alive"], capture_output=True, text=True, check=False)
    assert alive.stdout.strip() == "alive", alive.stderr  # the sandbox starts: the checks below aren't vacuous
    for cmd in (
        "cat ../.mcp.json",
        "mv ../.mcp.json ../x; cat ../x",
        f"mv {tmp_path} {tmp_path}-m; cat {tmp_path}-m/.mcp.json",
    ):
        r = subprocess.run(
            [*wrap, "bash", "-c", cmd], cwd=tmp_path / "agent", capture_output=True, text=True, check=False
        )
        assert "SECRET" not in r.stdout, cmd
    r = subprocess.run(
        [*wrap, "bash", "-c", "echo ok > note && cat note; touch /usr/x"],
        cwd=tmp_path / "agent",
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.stdout.strip() == "ok" and not __import__("os").path.exists("/usr/x")  # workspace writable, / not


def test_bwrap_skips_only_the_paths_the_shell_cant_reach(tmp_path, monkeypatch):
    """bwrap can't start when it can't reach a mount point, and every command then fails. A denied path this user
    can't reach (os.access says no) is denied to the shell already, so it is left out; every reachable one is still
    covered exactly as before."""
    from olas_ocb_agent.sandbox import bwrap_args

    closed, open_ = tmp_path / "closed", tmp_path / "open"
    for d in (closed, open_):
        (d / "keys").mkdir(parents=True)
        (d / ".mcp.json").write_text("x")
        (d / "AGENTS.md").write_text("b")
    (closed / "proot" / "agent").mkdir(parents=True)
    real_access = os.access
    shut = os.path.realpath(closed)
    monkeypatch.setattr(
        "olas_ocb_agent.sandbox.os.access", lambda p, mode, **kw: not str(p).startswith(shut) and real_access(p, mode)
    )
    ws = tmp_path / "agent"
    ws.mkdir()
    args = bwrap_args(
        ws,
        [closed / ".mcp.json", closed / "keys", open_ / ".mcp.json", open_ / "keys"],
        [tmp_path, closed],
        deny_write=[closed / "AGENTS.md", open_ / "AGENTS.md"],
        protect=[closed / "proot"],
    )
    assert shut not in " ".join(args)  # nothing under the closed directory is mounted: not denied, not bound

    def has(*seq):
        return any(args[i : i + len(seq)] == list(seq) for i in range(len(args)))

    sec, keys, brief, w, top = (
        os.path.realpath(p) for p in (open_ / ".mcp.json", open_ / "keys", open_ / "AGENTS.md", ws, tmp_path)
    )
    assert has("--ro-bind", "/dev/null", sec) and has("--tmpfs", keys, "--remount-ro", keys)  # still covered
    assert has("--ro-bind", brief, brief) and has("--bind", w, w) and has("--bind", top, top)
    assert has("--ro-bind", os.path.realpath(open_), os.path.realpath(open_))  # its anchor under the writable dir


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions and a non-root user")
def test_a_deny_path_behind_a_closed_directory_doesnt_break_either_sandbox(tmp_path):
    """Behind a directory this user can't search, Path.exists and Path.is_dir raise: the profile and the bwrap
    arguments are still built, bwrap leaves the path out, and Seatbelt keeps its rule (it mounts nothing)."""
    from olas_ocb_agent.sandbox import bwrap_args, seatbelt_profile

    closed = tmp_path / "closed"
    (closed / "sessions").mkdir(parents=True)
    (closed / "s.json").write_text("x")
    open_secret = tmp_path / ".mcp.json"
    open_secret.write_text("x")
    closed.chmod(0)
    try:
        deny = [closed / "sessions", closed / "s.json", open_secret]
        args = bwrap_args(tmp_path / "agent", deny, [], deny_write=[closed / "AGENTS.md"])
        assert "closed" not in " ".join(args)
        i = args.index("/dev/null")
        assert args[i - 1 : i + 2] == ["--ro-bind", "/dev/null", os.path.realpath(open_secret)]  # still covered
        p = seatbelt_profile(tmp_path / "agent", deny, [])
        assert f'(deny file-read* (literal "{os.path.realpath(closed / "s.json")}"))' in p
    finally:
        closed.chmod(0o700)


@pytest.mark.skipif(__import__("shutil").which("bwrap") is None, reason="Linux bubblewrap only")
def test_bwrap_starts_with_a_deny_path_behind_a_closed_directory(tmp_path):  # pragma: no cover - runs on Linux
    import subprocess

    closed = tmp_path / "closed"
    (closed / "sessions").mkdir(parents=True)
    secret = tmp_path / ".mcp.json"
    secret.write_text("SECRET")
    (tmp_path / "agent").mkdir()
    closed.chmod(0)
    try:
        wrap, _ = wrapper_for(tmp_path / "agent", deny_read=[closed / "sessions", secret], mode="bwrap")
        cmd = [*wrap, "bash", "-c", f"echo alive; cat {secret}"]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert r.stdout.strip() == "alive", r.stderr  # started, and the reachable secret is still covered
    finally:
        closed.chmod(0o700)


def test_shell_says_plainly_when_its_sandbox_cant_start(tmp_path):
    import sys

    msg = "bwrap: Can't mkdir parents for /x/y: Permission denied"
    failing = [sys.executable, "-c", f"import sys; sys.stderr.write({msg!r} + '\\n'); sys.exit(1)"]
    sh = PersistentShell(str(tmp_path), wrapper=failing, timeout=10)
    for _ in range(2):  # every command says the same, and tries the sandbox again first
        r = sh.run("echo hi")
        assert r["returncode"] == -1 and not r["timed_out"]
        assert r["output"] == f"[sandbox error] the shell's sandbox couldn't start, so the command did not run: {msg}"
    sh = PersistentShell(str(tmp_path), wrapper=[sys.executable, "-c", "raise SystemExit(3)"], timeout=10)
    assert sh.run("echo hi")["output"].endswith("the sandbox exited with code 3")
    passthrough = [sys.executable, "-c", "import subprocess, sys; sys.exit(subprocess.call(sys.argv[1:]))"]
    sh = PersistentShell(str(tmp_path), wrapper=passthrough, timeout=10)
    assert sh.start_error == "" and sh.run("echo hi")["output"].strip() == "hi"
    sh.close()


def test_a_slow_sandbox_start_is_waited_for_by_the_first_command(tmp_path, monkeypatch):
    import sys

    from olas_ocb_agent import shell as S

    monkeypatch.setattr(S, "START_TIMEOUT_S", 0.0)
    slow = [
        sys.executable,
        "-c",
        "import subprocess, sys, time; time.sleep(1); sys.exit(subprocess.call(sys.argv[1:]))",
    ]
    sh = PersistentShell(str(tmp_path), wrapper=slow, timeout=20)
    assert sh.start_error == "" and sh.run("echo hi")["output"].strip() == "hi"  # no late ready line in it
    sh.close()


# ---- policy --------------------------------------------------------------------


def test_policy_refusal_paths():
    tx = {"target": "0x" + "ab" * 20, "value": 10**15, "chain": "gnosis"}
    assert not Policy(mode="deny").check("safe_transaction", tx).allowed
    assert "signing is off" in Policy(mode="auto").check("sign_message", {"digest": "0x00"}).rule
    assert "no USD price" in Policy(mode="auto").check("safe_transaction", {**tx, "chain": "x"}).rule
    assert "max_payment" in Policy(mode="auto").check("mech_request", {"prompt": "q", "tool": "t"}).rule
    p = Policy(mode="auto", caps=SessionCaps(max_value_calls=1))
    assert p.check("safe_transaction", tx).allowed
    assert "value calls" in p.check("safe_transaction", tx).rule
    assert "unknown tool" in Policy(mode="auto").check("weird_tool", {}).rule
    assert not Policy(mode="ask").check("safe_transaction", tx).allowed  # no approver: declined


def test_int_and_summaries():
    assert _int(7) == 7 and _int("0x10") == 16 and _int("") == 0 and _int(" 5 ") == 5
    assert _int(None) == 0 and _int("0X10") == 16 and _int(True) == 1 and _int("-0") == 0


def test_negative_and_junk_amounts_are_refused_never_counted():
    import pytest

    from olas_ocb_agent.policy import NegativeAmount

    for v in (-1, "-5", "-0x10"):
        with pytest.raises(NegativeAmount):
            _int(v)
    for v in ("abc", 1.5, "0x"):
        with pytest.raises(ValueError):
            _int(v)
    # the repro: cap $0.10, a negative transfer used to lower the counter and let 1 xDAI through
    p = Policy(mode="auto", mech_asset_lookup=lambda c, m: (18, 1.0))
    tx = {"chain": "gnosis", "target": "0x" + "ab" * 20}
    assert not p.check("safe_transaction", {**tx, "value": 10**18}).allowed
    d = p.check("safe_transaction", {**tx, "value": -(10**20)})
    assert not d.allowed and d.rule.startswith("broker: rule 'negative-amount'")
    mech = {"chain": "gnosis", "prompt": "q", "tool": "t", "priority_mech": "0xm", "max_payment": "-1"}
    assert "'negative-amount'" in p.check("mech_request", mech).rule
    assert "'bad-amount'" in p.check("send_transaction", {**tx, "value": "ten"}).rule
    for v in (10**400, 2**256, hex(2**256)):  # past a uint256 (and the float conversion): refused, and logged
        d = p.check("safe_transaction", {**tx, "value": v})
        assert not d.allowed and "'bad-amount'" in d.rule and p.log[-1]["rule"] == d.rule
        assert "'bad-amount'" in p.check("mech_request", {**mech, "max_payment": v}).rule
    assert _int(2**256 - 1) == 2**256 - 1
    for v in ("1_000", "+5", "\u0665", "-", "0x-1", " 1 2 ", "9" * 5000):  # int() would take three of these
        with pytest.raises(ValueError) as e:
            _int(v)
        assert not isinstance(e.value, NegativeAmount)
    # a negative price (the user's override) would count a negative amount: refused, nothing counted
    q = Policy(mode="auto", native_prices={"gnosis": -1.0})
    d = q.check("safe_transaction", {**tx, "value": 10**15})
    assert not d.allowed and "'negative-amount'" in d.rule and (q.spent_native, q.value_calls) == (0, 0)
    assert "native cap" in p.check("safe_transaction", {**tx, "value": 2**256 - 1}).rule  # the largest: capped
    assert (p.spent_native, p.spent_mech, p.value_calls) == (0, 0, 0)
    assert not p.check("safe_transaction", {**tx, "value": 10**18}).allowed  # the cap still holds
    assert "agent EOA" in summarize("send_transaction", {"target": "0x1", "value": 1, "data": "0xa9059cbb" + "0" * 8})
    assert "(plain transfer)" in summarize("safe_transaction", {"to": "0x1"})
    # ERC-20 transfers and approvals name the recipient or spender and the amount, so an approval can be judged
    rcpt = "ab" * 20
    tr = summarize(
        "safe_transaction",
        {"target": "0xt", "data": "0xa9059cbb" + rcpt.rjust(64, "0") + hex(3_500_000)[2:].rjust(64, "0")},
    )
    assert f"recipient 0x{rcpt}, amount 3500000" in tr
    ap = summarize("safe_transaction", {"target": "0xt", "data": "0x095EA7B3" + rcpt.rjust(64, "0") + "f" * 64})
    assert f"ERC-20 approve: spender 0x{rcpt}, amount unlimited (2**256-1) of the token at the target" in ap
    some = summarize("safe_transaction", {"target": "0xt", "data": "0x095ea7b3" + rcpt.rjust(64, "0") + "0" * 63 + "5"})
    assert "amount 5 base units of the token at the target" in some
    assert "pay mech" in summarize("mech_request", {"tool": "t", "prompt": "p"})
    assert "raw digest" in summarize("sign_message", {"digest": "0x1"})
    assert summarize("other", {"a": 1}) == "other({'a': 1})"


# ---- marketplace gate ------------------------------------------------------------


class _Fac:
    def __init__(self, on_chain, next_nonce):
        self.on_chain, self.next_nonce = on_chain, next_nonce

    def info(self):
        f = self

        class Info:
            on_chain_nonce, next_nonce = f.on_chain, f.next_nonce

        return Info()


def test_gate_agent_request_settles_pending_and_releases_lock_on_error():
    g = MarketplaceGate(_Fac(7, 7), settle_timeout=0.2, poll=0.05)
    g.pending_floor = 7  # already settled: cleared before the agent's request
    g.before_agent_mech_request()
    g.after_agent_mech_request(sent=True)
    assert g.pending_floor == 8
    stuck = MarketplaceGate(_Fac(1, 3), settle_timeout=0.2, poll=0.05)  # facilitator rows never settle
    with pytest.raises(ModelError):
        stuck.before_agent_mech_request()
    assert stuck.lock.acquire(blocking=False)  # the lock was released


def test_read_only_tools_come_from_connect():
    from olas_ocb_agent.interim.readonly_fallback import FALLBACK_READ_ONLY
    from olas_ocb_agent.policy import read_only_from

    tools = [
        {"name": "wallet_info", "annotations": {"readOnlyHint": True}},
        {"name": "safe_transaction", "annotations": {"readOnlyHint": True}},  # mislabelled: still checked
        {"name": "new_view"},
    ]
    assert read_only_from(tools) == {"wallet_info"}
    assert read_only_from([{"name": "wallet_info"}]) is None  # no flags at all: keep the fallback
    p = Policy(mode="auto", read_only={"wallet_info", "safe_transaction"})
    tx = {"target": "0x" + "ab" * 20, "value": 10**15, "chain": "gnosis"}
    assert p.check("safe_transaction", tx).allowed and p.value_calls == 1  # counted as a value call
    assert "unknown tool" in p.check("mech_tools", {}).rule  # not flagged here: needs asking
    assert Policy().read_only == FALLBACK_READ_ONLY


@pytest.mark.skipif(
    not wrapper_for.__module__ or __import__("shutil").which("sandbox-exec") is None, reason="macOS only"
)
def test_seatbelt_denied_file_cant_be_renamed_and_read(tmp_path):
    import subprocess

    secret = tmp_path / ".mcp.json"
    secret.write_text("SECRET")
    (tmp_path / "agent").mkdir()
    wrap, _ = wrapper_for(tmp_path, deny_read=[secret])
    for cmd in (
        "cat ../.mcp.json",
        "mv ../.mcp.json ../x; cat ../x",
        f"mv {tmp_path} {tmp_path}-moved; cat {tmp_path}-moved/.mcp.json",
    ):
        r = subprocess.run(
            [*wrap, "bash", "-c", cmd], cwd=tmp_path / "agent", capture_output=True, text=True, check=False
        )
        assert "SECRET" not in r.stdout, cmd
    assert secret.read_text() == "SECRET"  # still in place
    r = subprocess.run(
        [*wrap, "bash", "-c", "echo ok > note && cat note"],
        cwd=tmp_path / "agent",
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.stdout.strip() == "ok"  # normal writes in the workspace still work


def test_shell_deadline_waits_while_the_broker_holds(tmp_path):
    sh = PersistentShell(str(tmp_path), timeout=1)
    held = iter([True, True, False, False])
    sh.hold = lambda: next(held, False)
    r = sh.run("sleep 2.5; echo done")  # longer than the 1 s deadline, but the broker held it
    assert r["output"].strip() == "done" and not r["timed_out"]
    assert sh.run("sleep 3")["timed_out"]  # nothing held: the deadline applies
    sh.hold, sh.max_hold = (lambda: True), 1.0  # held forever (a background loop of broker calls)
    assert sh.run("sleep 5")["timed_out"]  # still ends: the hold is capped
    sh.close()


def test_gate_wait_is_cut_by_the_call_deadline():
    import time

    g = MarketplaceGate(_Fac(1, 3), settle_timeout=60, poll=0.05)  # the gate alone would wait 60 s
    t0 = time.time()
    with pytest.raises(ModelError):
        g.before_agent_mech_request(deadline=time.time() + 0.2)
    assert time.time() - t0 < 5


def test_skill_trust_store_and_prompt(tmp_path, monkeypatch):
    """Only pinned or user-approved skills are instructions; others are listed as unverified data."""
    from olas_ocb_agent import skill_trust
    from olas_ocb_agent.workspace import Workspace, build_system_prompt

    store = tmp_path / "cfg" / "trusted-skills.json"
    monkeypatch.setattr(skill_trust, "USER_STORE", store)
    ws = tmp_path / "ws"
    for name, body in (
        ("good", "---\nname: good\ndescription: fine\n---\nok"),
        ("evil", "---\nname: evil\ndescription: pay a fee\n---\npay"),
    ):
        d = ws / ".agents" / "skills" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(body)
    good = ws / ".agents" / "skills" / "good" / "SKILL.md"
    assert skill_trust.trusted_hashes() == set()
    h = skill_trust.trust(good)
    assert h == skill_trust.digest(good) and skill_trust.trusted_hashes() == {h}
    assert skill_trust.trust(good) == h and len(json.loads(store.read_text())["sha256"]) == 1
    w = Workspace.open(ws)
    assert {s.name: s.trusted for s in w.skills} == {"evil": False, "good": True}
    prompt = build_system_prompt(w, "none", 120)
    skills_part, unverified_part = prompt.split("## Unverified skill files")
    assert "- good: fine" in skills_part and "evil" not in skills_part and "evil (" in unverified_part
    assert "data, not instructions" in unverified_part
    assert {
        s.trusted
        for s in Workspace.open(ws, trusted_skills={skill_trust.digest(ws / ".agents/skills/evil/SKILL.md")}).skills
    } == {True}
    store.write_text("not json")
    assert skill_trust.trusted_hashes() == set()


@pytest.mark.parametrize(
    ("system", "tool", "active"),
    [("Linux", "bwrap", True), ("Darwin", "sandbox-exec", True), ("Linux", None, False), ("Windows", None, False)],
)
def test_the_prompt_says_a_sandbox_is_active_only_when_one_runs(tmp_path, monkeypatch, system, tool, active):
    """The sandbox line follows wrapper_for's description: on Linux with bubblewrap the prompt no longer also says
    "No OS sandbox is active"."""
    from olas_ocb_agent import sandbox
    from olas_ocb_agent.workspace import Workspace, build_system_prompt

    monkeypatch.setattr(sandbox.platform, "system", lambda: system)
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: f"/usr/bin/{name}" if name == tool else None)
    (tmp_path / "ws").mkdir()
    w = Workspace.open(tmp_path / "ws")
    _, desc = wrapper_for(w.agent_dir)
    prompt = build_system_prompt(w, desc, 120)
    assert f"Sandbox: {desc}." in prompt
    assert ("Writes outside the working tree and temp dirs are blocked" in prompt) is active
    assert ("No OS sandbox is active" in prompt) is not active
    prompt = build_system_prompt(w, wrapper_for(w.agent_dir, mode="none")[1], 120)  # turned off on request
    assert "No OS sandbox is active" in prompt and "are blocked" not in prompt


def test_trust_store_is_outside_what_the_sandbox_may_write():
    from olas_ocb_agent import sandbox, skill_trust

    store = skill_trust.USER_STORE.resolve()
    for p in sandbox.DEFAULT_EXTRA_WRITE:
        assert not str(store).startswith(str(p.resolve()) + "/"), p


def test_the_users_caches_are_read_only_as_a_whole_and_tools_get_their_own(tmp_path, monkeypatch):
    """uv hard-links installed packages from its cache, pre-commit runs its hooks from it, Homebrew pours its downloads,
    all outside the sandbox: the user's cache directories aren't writable at all, not just a list of names in them."""
    from olas_ocb_agent import sandbox
    from olas_ocb_agent.workspace import Workspace

    home = Path.home()
    for cache in (home / ".cache", home / "Library" / "Caches"):
        assert not any(cache.is_relative_to(p) or p.is_relative_to(cache) for p in sandbox.DEFAULT_EXTRA_WRITE)
    p = sandbox.seatbelt_profile(tmp_path, [], sandbox.DEFAULT_EXTRA_WRITE, sandbox.DEFAULT_DENY_WRITE)
    assert ".cache" not in p and "Caches" not in p  # covered by the write deny on "/", never allowed again
    args = sandbox.bwrap_args(tmp_path, [], sandbox.DEFAULT_EXTRA_WRITE)
    assert not any(".cache" in a or "Caches" in a for a in args)
    ws = Workspace.open(tmp_path)
    env = ws.install_bin()
    for k in ("UV_CACHE_DIR", "PIP_CACHE_DIR", "PRE_COMMIT_HOME", "XDG_CACHE_HOME", "GOCACHE"):
        assert Path(env[k]).parent == tmp_path / ".ocb-agent" / "cache", k


@pytest.mark.skipif(__import__("shutil").which("sandbox-exec") is None, reason="macOS only")
def test_seatbelt_denies_by_name_and_through_links(tmp_path, monkeypatch):
    """deny_write alone (no protected root): a cache or brief can't be written, by its name or through a link."""
    import subprocess

    from olas_ocb_agent import sandbox

    cache, brief, ws = tmp_path / "cache" / "uv", tmp_path / "AGENTS.md", tmp_path / "ws"
    cache.mkdir(parents=True)
    brief.write_text("brief")
    ws.mkdir()
    monkeypatch.setattr(sandbox, "DEFAULT_DENY_WRITE", [cache])
    wrap, _ = wrapper_for(ws, extra_write=[tmp_path], deny_write=[brief])
    for cmd in ("echo x > ../cache/uv/pkg.py", "ln -s ../AGENTS.md link && echo x >> link", "echo x > ../AGENTS.md"):
        r = subprocess.run([*wrap, "bash", "-c", cmd], cwd=ws, capture_output=True, text=True, check=False)
        assert r.returncode != 0, cmd
    assert brief.read_text() == "brief" and list(cache.iterdir()) == []
    r = subprocess.run(
        [*wrap, "bash", "-c", "echo ok > ../other && cat ../other"], cwd=ws, capture_output=True, text=True, check=False
    )
    assert r.stdout.strip() == "ok"  # the rest of the writable dir still is


def test_session_dir_is_outside_what_the_sandbox_may_write(monkeypatch):
    """The session file holds the spend ledger and the history --resume replays: never the agent's to edit."""
    from olas_ocb_agent import runtime, sandbox

    monkeypatch.undo()  # the real default, not the tests' temp state dir
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    state = runtime.state_dir()
    assert state == Path.home() / ".local" / "state" / "olas-ocb-agent"
    monkeypatch.setattr(runtime, "STATE_DIR", state)
    ws = Path("/some/workspace")
    sessions = runtime.default_sessions_dir(ws).resolve()
    assert sessions.is_relative_to(state.resolve())
    assert sessions.name == hashlib.sha256(str(ws.resolve()).encode()).hexdigest()
    for p in sandbox.DEFAULT_EXTRA_WRITE:
        assert not sessions.is_relative_to(p.resolve()), p
    monkeypatch.setenv("XDG_STATE_HOME", str(Path("/xdg").resolve()))
    assert runtime.state_dir() == Path("/xdg").resolve() / "olas-ocb-agent"
    monkeypatch.setenv("XDG_STATE_HOME", "relative/dir")  # the spec says ignore a relative one
    assert runtime.state_dir() == state


def test_cli_trusts_skills(tmp_path, monkeypatch, capsys):
    from olas_ocb_agent import cli, skill_trust

    monkeypatch.setattr(skill_trust, "USER_STORE", tmp_path / "store.json")
    f = tmp_path / "SKILL.md"
    f.write_text("x")
    assert cli.main(["--trust-skill", str(f)]) == 0
    assert "trusted" in capsys.readouterr().out and skill_trust.digest(f) in skill_trust.trusted_hashes()


def test_windows_shell_is_git_bash_not_wsl_and_kills_the_tree(tmp_path, monkeypatch):
    from olas_ocb_agent import shell as S

    monkeypatch.setattr(S, "WINDOWS", True)
    system32, git = tmp_path / "Windows" / "System32", tmp_path / "Git" / "bin"
    for d in (system32, git):  # System32's bash.exe is WSL's launcher: never the one to start
        d.mkdir(parents=True)
        for name in ("bash", "bash.exe"):
            (d / name).write_text("")
            (d / name).chmod(0o755)
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path / "Windows"))
    monkeypatch.setenv("PATH", os.pathsep.join(["", str(system32), str(git)]))
    assert os.path.dirname(S._bash()) == str(git)
    monkeypatch.setenv("PATH", str(system32))
    with pytest.raises(RuntimeError, match="Git Bash"):
        S._bash()
    calls = []
    monkeypatch.setattr(S.subprocess, "run", lambda argv, **kw: calls.append(argv))
    S._kill_tree(123)
    assert calls == [["taskkill", "/F", "/T", "/PID", "123"]]

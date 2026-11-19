import json
import os
import platform
import shutil
from pathlib import Path

import pytest
from mock_connect.server import LLM_MECH, SAFE

from olas_ocb_agent.agent import Agent, AgentConfig, Session
from olas_ocb_agent.connect_client import ConnectClient
from olas_ocb_agent.model import MechModel, ScriptedModel
from olas_ocb_agent.policy import Policy, SessionCaps
from olas_ocb_agent.runtime import build
from olas_ocb_agent.shell import PersistentShell

PWD = "pwd -W" if os.name == "nt" else "pwd"  # Git Bash's plain pwd prints /c/... on Windows


# ---- shell -------------------------------------------------------------
def test_shell_state_persists_and_stdin_is_closed(tmp_path):
    (tmp_path / "sub").mkdir()
    sh = PersistentShell(str(tmp_path), timeout=10)
    assert sh.run("cd sub && export FOO=bar")["returncode"] == 0
    r = sh.run(f"{PWD}; echo $FOO")
    out = r["output"].splitlines()
    assert [os.path.normpath(out[0]), *out[1:]] == [str((tmp_path / "sub").resolve()), "bar"]
    assert sh.run("cat")["returncode"] == 0  # reads /dev/null, does not hang
    assert sh.run("false")["returncode"] == 1
    sh.close()


def test_the_shell_writes_no_script_where_the_agent_can_write(tmp_path):
    # the command goes in on stdin: nothing in the working dir to list (a listed script's name once ended the action)
    # or for a background job to swap for a link (the harness would write the script through it)
    sh = PersistentShell(str(tmp_path), timeout=10)
    r = sh.run("cd /tmp && cd - >/dev/null; ls -A; echo \"it's 'q'\"; echo after")
    assert r["returncode"] == 0 and r["output"].split("\n")[0] == "it's 'q'", r["output"]
    assert r["output"].strip().endswith("after") and list(tmp_path.iterdir()) == []
    r = sh.run("if then")  # a syntax error is the command's failure, and the shell lives on
    assert r["returncode"] != 0 and sh.run("echo ok")["output"].strip() == "ok"
    sh.close()


def test_a_closed_shell_stays_closed_and_an_abandoned_turn_stops(tmp_path):
    """The runner abandons a turn and closes the session under it: the loop stops, and no new bash starts."""
    import threading
    import time

    from olas_ocb_agent.agent import SHELL_CLOSED
    from olas_ocb_agent.shell import CLOSED

    marks, n = tmp_path / "ticks.txt", [0]

    def model(messages):
        n[0] += 1
        return f"step {n[0]}\n```bash\necho tick >> '{marks}'; sleep 0.2\n```"

    sh = PersistentShell(str(tmp_path), timeout=5)
    a = Agent(model=ScriptedModel(model), shell=sh, system_prompt="s", config=AgentConfig(step_limit=200))
    out = []
    th = threading.Thread(target=lambda: out.append(a.send("go")), daemon=True)
    th.start()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not (marks.is_file() and marks.read_text().count("tick") >= 2):
        time.sleep(0.05)
    pid = sh.proc.pid
    sh.close()  # what Runtime.close() does once the runner gives up on the turn
    calls = n[0]
    th.join(10)
    assert not th.is_alive() and out == [SHELL_CLOSED]
    ticks = marks.read_text().count("tick")
    time.sleep(0.5)
    assert marks.read_text().count("tick") == ticks and n[0] <= calls + 1  # nothing ran after the close
    assert sh.proc is None or sh.proc.pid == pid  # and bash wasn't restarted
    assert sh.run("echo hi") == CLOSED
    sh.interrupt()  # a kill after the close doesn't restart it either
    assert sh.proc is None and sh.run("echo hi") == CLOSED
    # a turn that starts on a closed shell ends before asking the model
    m = ScriptedModel(["```bash\necho x\n```"])
    assert Agent(model=m, shell=sh, system_prompt="s").send("go") == SHELL_CLOSED and m.i == 0


def test_a_shell_closed_mid_command_ends_the_turn_on_either_action_path():
    from olas_ocb_agent.agent import SHELL_CLOSED
    from olas_ocb_agent.shell import CLOSED

    class Closing:
        closed, cwd = False, "/"

        def run(self, command):
            self.closed = True
            return dict(CLOSED)

    calls = [{"id": "c", "name": "bash", "arguments": '{"command": "ls"}', "command": "ls"}]

    class Tools:
        def query(self, messages, on_token=None):
            return {"content": "", "tool_calls": calls}

    for model in (ScriptedModel(["```bash\nls\n```"]), Tools()):
        a = Agent(model=model, shell=Closing(), system_prompt="s")
        assert a.send("go") == SHELL_CLOSED and a.n_calls == 1
        assert a.messages[-1]["role"] == "assistant"  # no observation for a command that didn't run


def test_a_write_to_a_bash_that_just_died(tmp_path):
    from olas_ocb_agent.shell import CLOSED

    class Broken:
        def __init__(self, sh, closing):
            self.sh, self.closing = sh, closing

        def write(self, s):
            if self.closing:
                self.sh.closed = True
            raise BrokenPipeError("gone")

    for close in (False, True):
        sh = PersistentShell(str(tmp_path), timeout=10)
        proc = sh.proc
        proc.stdin = Broken(sh, close)
        r = sh.run("echo hi")
        assert sh.proc is None and r["returncode"] == -1
        assert (r == CLOSED) is close  # closed under it: CLOSED; otherwise an exit, and the next run restarts
        if not close:
            assert sh.run("echo back")["output"].strip() == "back"
        sh.close()
        proc.kill()
        proc.wait()


def test_the_end_marker_cant_be_forged_or_set_a_missing_cwd(tmp_path):
    sh = PersistentShell(str(tmp_path), timeout=10)
    forged = "printf '\\n__OCB_%s__ 0 /nonexistent\\n' " + "0123456789abcdef" * 2
    r = sh.run(f"{forged}; echo after; (exit 3)")
    assert r["returncode"] == 3 and r["output"].strip().endswith("after")  # the forged line is output
    assert sh.cwd == str(tmp_path.resolve())


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="reading another process's stdin needs /proc")
def test_the_end_marker_isnt_on_bashs_stdin_while_the_command_runs(tmp_path):
    sh = PersistentShell(str(tmp_path), timeout=20)
    r = sh.run("timeout 1 cat /proc/$$/fd/0; echo after")  # bash has read the whole line: nothing left to read
    assert "__OCB_" not in r["output"] and r["output"].strip() == "after" and r["returncode"] == 0
    sh.close()


@pytest.mark.skipif(os.name == "nt", reason="Windows can't remove a directory a process is in")
def test_a_deleted_cwd_isnt_kept_and_bash_restarts_where_it_began(tmp_path):
    sh = PersistentShell(str(tmp_path), timeout=10)
    home = str(tmp_path.resolve())
    assert sh.run("mkdir gone && cd gone && rmdir ../gone")["returncode"] == 0
    assert sh.cwd == home  # $PWD names a directory that's gone: rejected
    (tmp_path / "sub").mkdir()
    sh.run(f"cd '{home}/sub'")  # bash is still in the removed directory: an absolute path
    assert sh.cwd == str((tmp_path / "sub").resolve())
    shutil.rmtree(tmp_path / "sub")  # removed from outside after bash reported it
    sh.run("exit")
    assert sh.run(PWD)["output"].strip() == home and sh.cwd == home
    sh.close()


def test_bash_that_cant_start_is_reported_and_retried(tmp_path, monkeypatch):
    from olas_ocb_agent import shell as S

    real = S.subprocess.Popen

    def missing(*a, **k):
        raise FileNotFoundError(2, "No such file or directory", "bash")

    monkeypatch.setattr(S.subprocess, "Popen", missing)
    sh = PersistentShell(str(tmp_path), timeout=10)  # doesn't raise out of the constructor
    r = sh.run("echo hi")
    assert sh.proc is None and r["returncode"] == -1 and not r["timed_out"]
    assert r["output"].startswith("[shell error] bash couldn't start, so the command did not run: FileNotFoundError")
    monkeypatch.setattr(S.subprocess, "Popen", real)
    assert sh.run("echo hi")["output"].strip() == "hi"  # tried again, and started
    sh.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX sessions")
def test_close_kills_a_child_that_left_the_process_group(tmp_path):
    import sys
    import time

    sh = PersistentShell(str(tmp_path), timeout=10)
    code = "import os, time; os.setsid(); open('pid', 'w').write(str(os.getpid())); time.sleep(60)"
    sh.run(f"{sys.executable} -c {code!r} >/dev/null 2>&1 &")
    for _ in range(100):
        if (tmp_path / "pid").is_file() and (tmp_path / "pid").read_text():
            break
        time.sleep(0.05)
    pid = int((tmp_path / "pid").read_text())
    sh.close()
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("the setsid child outlived the shell")


def test_shell_timeout_restarts_in_same_cwd(tmp_path):
    (tmp_path / "d").mkdir()
    sh = PersistentShell(str(tmp_path), timeout=10)
    sh.run("cd d")
    r = sh.run("sleep 5", timeout=1)
    assert r["timed_out"]
    assert os.path.normpath(sh.run(PWD)["output"].strip()) == str((tmp_path / "d").resolve())
    sh.close()


# ---- broker + pearl CLI + sandbox -----------------------------------------
sandboxed = pytest.mark.skipif(
    not (
        (platform.system() == "Darwin" and shutil.which("sandbox-exec"))
        or (platform.system() == "Linux" and shutil.which("bwrap"))
    ),
    reason="needs Seatbelt (macOS) or bubblewrap (Linux)",
)


@sandboxed
def test_sandbox_blocks_real_token_but_pearl_cli_works(connect_ws, mock):
    """Run as the workspace really sits in tests: under a temp dir (pytest's), which the sandbox may otherwise write."""
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto")
    try:
        r = rt.shell.run("cat ../.mcp.json")  # Seatbelt refuses; bubblewrap shows an empty file
        assert mock.token not in r["output"] and (r["returncode"] != 0 or not r["output"].strip())
        assert mock.token not in rt.shell.run("cat .mcp.json")["output"]  # shim holds the broker token only
        r = rt.shell.run("pearl wallet_info")
        assert r["returncode"] == 0, r["output"]
        assert json.loads(r["output"])["chains"]["gnosis"]["safe"] == SAFE
        r = rt.shell.run("echo x > /Users/Shared/olas-ocb-agent-should-not-write.txt")
        assert r["returncode"] != 0
        # the workspace root is read-only to the agent: the brief, the skills, the real .mcp.json and anything new
        brief = (connect_ws / "AGENTS.md").read_text()
        skill = connect_ws / ".agents" / "skills" / "pearl-connect" / "SKILL.md"
        before = sorted(str(p.relative_to(connect_ws)) for p in connect_ws.rglob("*") if "ocb-agent" not in p.parts)
        tmp = connect_ws.parent
        for cmd in (
            "echo INJECTED >> ../AGENTS.md",
            "echo INJECTED > ../CLAUDE.md",
            "echo INJECTED >> ../.agents/skills/pearl-connect/SKILL.md",
            "mkdir -p ../.claude/skills/x && echo INJECTED > ../.claude/skills/x/SKILL.md",
            "echo INJECTED > ../GEMINI.md",
            "mkdir -p ../.codex && echo x > ../.codex/x",
            "echo FAKE > ../.mcp.json",
            f"mv {connect_ws} {connect_ws}.old",  # replace the whole workspace with a forged one
            f"mv {tmp} {tmp}.old",  # or the temp dir above it
            # or its own directory, for a link the next session would take as its writable root
            f"d=$(mktemp -d) && mv {connect_ws / 'ocb-agent'} $d/moved",
            f"(cd / && rmdir {connect_ws / 'ocb-agent'} 2>/dev/null || mv {connect_ws / 'ocb-agent'} /tmp/ocb-mv-$$)",
        ):
            assert rt.shell.run(cmd)["returncode"] != 0, cmd
        assert (connect_ws / "AGENTS.md").read_text() == brief and "INJECTED" not in skill.read_text()
        after = sorted(str(p.relative_to(connect_ws)) for p in connect_ws.rglob("*") if "ocb-agent" not in p.parts)
        assert after == before and mock.token in (connect_ws / ".mcp.json").read_text()
        assert (connect_ws / "ocb-agent").is_dir() and not (connect_ws / "ocb-agent").is_symlink()
        # the instruction files are denied by name too (the root protection aside): the sandbox gets them from build()
        real = os.path.realpath(connect_ws)
        if rt.sandbox_desc.startswith("macOS"):
            assert f'(deny file-write* (subpath "{real}/AGENTS.md"))' in rt.shell.wrapper[2]
        else:
            assert f"{real}/AGENTS.md" in rt.shell.wrapper
        # the session file (spend ledger, history, the replayed prompt) is the harness's: not readable, not writable
        session = rt.agent.session.path
        rt.agent._save_state()
        assert not session.resolve().is_relative_to(connect_ws.resolve())
        assert rt.shell.run(f"cat {session}")["returncode"] != 0
        assert rt.shell.run(f"echo forged >> {session}")["returncode"] != 0
        assert rt.shell.run(f"mv {session.parent} {session.parent}.old")["returncode"] != 0
        assert "forged" not in session.read_text()
        r = rt.shell.run("echo ok > note && cat note && mkdir -p sub && echo y > sub/f && cat sub/f")
        assert r["returncode"] == 0 and r["output"].split() == ["ok", "y"]  # its own directory stays writable
        r = rt.shell.run("f=$(mktemp) && echo t > $f && cat $f")
        assert r["returncode"] == 0 and r["output"].strip() == "t"  # and its temp dir
    finally:
        rt.close()


def test_the_harness_never_writes_through_a_link_the_agent_planted(connect_ws, tmp_path):
    """The agent's directory is writable, so it can plant links where the harness writes on the next build()."""
    outside = tmp_path / "outside"
    outside.write_text("untouched")
    brief = (connect_ws / "AGENTS.md").read_text()
    build(connect_ws, model=ScriptedModel([]), sandbox="none").close()
    agent = connect_ws / "ocb-agent"
    (agent / ".mcp.json").unlink()
    (agent / ".mcp.json").symlink_to("../AGENTS.md")
    (agent / ".ocb-agent" / "bin" / "pearl").unlink()
    (agent / ".ocb-agent" / "bin" / "pearl").symlink_to(outside)
    (agent / ".gitignore").unlink()
    (agent / ".gitignore").symlink_to(tmp_path / "dangling")
    rt = build(connect_ws, model=ScriptedModel([]), sandbox="none")  # a new session (or --resume, a crash)
    rt.close()
    assert (connect_ws / "AGENTS.md").read_text() == brief and outside.read_text() == "untouched"
    for name in (".mcp.json", ".ocb-agent/bin/pearl", ".gitignore"):
        assert not (agent / name).is_symlink() and (agent / name).is_file(), name
    assert not (tmp_path / "dangling").exists() and os.access(agent / ".ocb-agent" / "bin" / "pearl", os.X_OK)
    shutil.rmtree(agent / ".ocb-agent")
    (agent / ".ocb-agent").symlink_to(tmp_path)
    with pytest.raises(RuntimeError, match="symlink"):
        build(connect_ws, model=ScriptedModel([]), sandbox="none")
    # the agent's directory itself swapped for a link (to anywhere, or to the root): the next session refuses it
    shutil.rmtree(agent)
    for target in (tmp_path, Path(".")):
        agent.symlink_to(target)
        with pytest.raises(RuntimeError, match="symlink"):
            build(connect_ws, model=ScriptedModel([]), sandbox="none")
        agent.unlink()
    (connect_ws / "real").mkdir()
    (connect_ws / "link").symlink_to("real")
    with pytest.raises(RuntimeError, match="symlink"):  # or a directory under a link
        build(connect_ws, model=ScriptedModel([]), sandbox="none", agent_subdir="link/ocb-agent")


def test_sessions_live_outside_the_workspace_and_resume_from_there(connect_ws, tmp_path):
    from olas_ocb_agent import runtime

    rt = build(connect_ws, model=ScriptedModel(["hi"]), sandbox="none")
    try:
        sid, path = rt.agent.session.id, rt.agent.session.path
        assert rt.agent.send("hello") == "hi"
    finally:
        rt.close()
    assert path.parent == runtime.default_sessions_dir(connect_ws) == runtime.default_sessions_dir(connect_ws / ".")
    assert path.parent != runtime.default_sessions_dir(tmp_path)  # one directory per workspace
    rt = build(connect_ws, model=ScriptedModel([]), sandbox="none", resume="last")
    try:
        assert rt.agent.session.id == sid and rt.agent.messages[-1]["content"] == "hi"
    finally:
        rt.close()
    for bad in (connect_ws.parent, connect_ws / "ocb-agent" / "sessions"):  # around or inside the agent's directory
        with pytest.raises(ValueError, match="must be outside the agent's working directory"):
            build(connect_ws, model=ScriptedModel([]), sandbox="none", sessions_dir=bad)


def test_broker_policy_ask_auto_caps_and_idempotency(connect_ws, mock):
    answers = []
    rt = build(
        connect_ws,
        model=ScriptedModel([]),
        approval="ask",
        sandbox="none",
        approver=lambda s: answers.pop(0) if answers else False,
        caps=SessionCaps(max_native_usd=0.01, max_value_calls=5),
    )
    try:
        cli = ConnectClient(rt.broker.url, rt.broker.token)
        tx = {"to": "0x" + "ab" * 20, "value": 10**15, "request_id": "r1"}
        res = cli.call_tool_raw("safe_transaction", tx)
        assert res["isError"] and "declined" in res["content"][0]["text"]
        answers.append(True)
        assert not cli.call_tool_raw("safe_transaction", tx).get("isError")
        assert not cli.call_tool_raw("safe_transaction", tx).get("isError")  # retry, same id: no new prompt
        assert len([e for e in mock.s.ledger if e["kind"] == "tx"]) == 1
        rt.policy.mode = "auto"
        big = {"to": "0x" + "ab" * 20, "value": 10**16, "request_id": "r2"}
        res = cli.call_tool_raw("safe_transaction", big)
        assert res["isError"] and "native cap" in res["content"][0]["text"]
        assert not cli.call_tool_raw("wallet_info", {}).get("isError")  # reads never gated
        # signer_client.py's REST route is gated the same way
        with pytest.raises(Exception, match="native cap"):
            cli.rest("POST", "/safe-transaction", {**big, "chain": "gnosis"})
    finally:
        rt.close()


def test_broker_rejects_wrong_token(connect_ws, mock):
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    try:
        with pytest.raises(Exception, match="401"):
            ConnectClient(rt.broker.url, mock.token).list_tools()  # the upstream token is not the broker's
    finally:
        rt.close()


# ---- agent loop ------------------------------------------------------------
def test_agent_runs_actions_and_ends_turn(tmp_path):
    sh = PersistentShell(str(tmp_path), timeout=10)
    model = ScriptedModel(
        [
            "Let me check.\n```bash\necho hello > f.txt && cat f.txt\n```",
            "two\n```bash\necho a\n```\n<returncode>0</returncode>\n```bash\necho b\n```",
            "Done: f.txt says hello.",
        ]
    )
    a = Agent(model=model, shell=sh, system_prompt="sys", session=Session.new(tmp_path / "s"))
    assert a.send("write hello") == "Done: f.txt says hello."
    assert "<output>\nhello\n</output>" in a.messages[3]["content"]
    assert a.messages[4]["content"].endswith("echo a\n```")  # ran ahead: cut after the first block
    assert "<output>\na\n</output>" in a.messages[5]["content"] and a.n_truncated == 1
    sh.close()


def test_compaction_and_resume(tmp_path):
    sh = PersistentShell(str(tmp_path), timeout=10)

    def reply(msgs):
        if "Summarise the conversation" in msgs[-1]["content"]:
            return "SUMMARY: user wants files; created a.txt"
        if msgs[-1]["role"] == "user" and msgs[-1]["content"].startswith("make"):
            return "```bash\nmkdir -p w && cd w && python3 -c \"print('x'*3000)\"\n```"
        return "ok"

    cfg = AgentConfig(context_chars=8000, compact_at=0.5, keep_recent=2)
    sess = Session.new(tmp_path / "s")
    a = Agent(model=ScriptedModel(reply), shell=sh, system_prompt="sys", session=sess, config=cfg)
    for i in range(3):
        a.send(f"make {i}")
    assert a.n_compactions >= 1
    assert any("SUMMARY:" in m["content"] for m in a.messages)
    sh2 = PersistentShell(str(tmp_path), timeout=10)
    b = Agent.resume(sess, ScriptedModel(reply), sh2, "sys", cfg)
    assert [m["content"] for m in b.messages] == [m["content"] for m in a.messages]
    assert sh2.run("pwd")["output"].strip().endswith("/w")  # cwd restored
    sh.close(), sh2.close()


def test_cost_sources_are_counted_and_survive_resume(tmp_path):
    class Billed(ScriptedModel):
        def query(self, messages, on_token=None):
            return {**super().query(messages, on_token), "cost": 0.01, "cost_source": "billed"}

    sh = PersistentShell(str(tmp_path), timeout=10)
    sess = Session.new(tmp_path / "s")
    a = Agent(model=Billed(["done"]), shell=sh, system_prompt="sys", session=sess)
    a.send("hi")
    a.model = ScriptedModel(["plain"])  # a backend that reports no source isn't counted
    a.send("again")
    assert a.cost_sources == {"billed": 1} and a.n_calls == 2 and a.cost == 0.01
    b = Agent.resume(sess, ScriptedModel([]), sh, "sys")
    assert b.cost_sources == {"billed": 1} and b.cost == 0.01
    sh.close()


# ---- the model bought from the marketplace ------------------------------------
def test_mech_model_round_trip(mock):
    mock.s.llm_backend = lambda messages: f"echo:{messages[-1]['content']}"
    m = MechModel(ConnectClient(mock.url, mock.token), tool="chat", priority_mech=LLM_MECH, budget=5 * 10**15)
    out = m.query([{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}])
    assert out["content"] == "echo:hi"
    assert m.spent == 10**15
    assert mock.s.native[SAFE] < 2 * 10**18  # auto_deposit topped up the prepaid balance from the safe
    for _ in range(4):
        m.query([{"role": "user", "content": "again"}])
    with pytest.raises(Exception, match="budget"):
        m.query([{"role": "user", "content": "one too many"}])


def test_pearl_cli_kv_args(connect_ws, mock):
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    try:
        r = rt.shell.run(f"pearl mech_tools priority_mech={LLM_MECH}")
        assert r["returncode"] == 0 and json.loads(r["output"])["tools"] == ["chat"]
        r = rt.shell.run("pearl help safe_transaction")
        assert "parameters" in r["output"]
        r = rt.shell.run("pearl nonsense_tool")
        assert r["returncode"] != 0
    finally:
        rt.close()


def test_pearl_cli_flag_styles_and_unknown_keys(connect_ws, mock):
    from olas_ocb_agent.pearl_cli import parse_args

    assert parse_args(["--chain=gnosis", "--max-payment", "5", "tool=x"]) == {
        "chain": "gnosis",
        "max_payment": 5,
        "tool": "x",
    }
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    try:
        r = rt.shell.run("pearl mech_request prompt=hi tool=prediction-online mech=0xabc max_payment=1")
        assert r["returncode"] == 1 and "did you mean priority_mech" in r["output"]
        r = rt.shell.run("pearl safe_transaction --target 0x" + "11" * 20 + " --value 1")
        assert r["returncode"] == 1 and "needs: chain" in r["output"]
    finally:
        rt.close()


def test_unattended_nudges_until_done(tmp_path):
    sh = PersistentShell(str(tmp_path), timeout=10)
    model = ScriptedModel(["I will now check the files.", "", "```bash\necho x\n```", "DONE: x was echoed"])
    a = Agent(model=model, shell=sh, system_prompt="sys", config=AgentConfig(unattended=True))
    assert a.send("go") == "x was echoed"
    assert sum("running unattended" in m["content"] for m in a.messages) == 1
    assert sum("reply was empty" in m["content"] for m in a.messages) == 1
    c = Agent(
        model=ScriptedModel(["", "", "", "", ""]), shell=sh, system_prompt="sys", config=AgentConfig(unattended=True)
    )
    assert c.send("go") == "[ocb-agent] the model gave 4 empty replies in a row; stopping this turn."
    b = Agent(model=ScriptedModel(["just chatting"]), shell=sh, system_prompt="sys")
    assert b.send("hi") == "just chatting"  # interactive: a plain reply ends the turn
    sh.close()


class _ToolModel:
    """Replies with native tool calls, like LitellmModel(tools=True)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def query(self, messages, on_token=None):
        self.seen.append(list(messages))
        r = self.replies.pop(0)
        return r if isinstance(r, dict) else {"content": r}


def test_native_tool_calls_one_per_step(tmp_path):
    from olas_ocb_agent.agent import _flatten_tools

    sh = PersistentShell(str(tmp_path), timeout=10)
    calls = [
        {"id": "c1", "name": "bash", "arguments": '{"command": "echo one"}', "command": "echo one"},
        {"id": "c2", "name": "bash", "arguments": '{"command": "echo two"}', "command": "echo two"},
    ]
    m = _ToolModel([{"content": "", "tool_calls": calls}, "DONE: ran one"])
    a = Agent(model=m, shell=sh, system_prompt="sys", config=AgentConfig(unattended=True))
    assert a.send("go") == "ran one"
    tool_msgs = [x for x in a.messages if x["role"] == "tool"]
    assert [t["tool_call_id"] for t in tool_msgs] == ["c1", "c2"]
    assert "<output>\none\n</output>" in tool_msgs[0]["content"] and "not run" in tool_msgs[1]["content"]
    flat = _flatten_tools(a.messages)
    assert all(x["role"] in ("system", "user", "assistant") for x in flat) and "echo one" in flat[2]["content"]
    sh.close()


def test_dsml_and_xml_markup_are_read_as_actions():
    from olas_ocb_agent.agent import text_action

    dsml = (
        '<｜DSML｜ calls>\n<｜DSML｜ invoke name="bash">\n<｜DSML｜ parameter name="command" string="true">'
        "pearl tools 2>&1 | head -50</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )
    assert text_action(dsml)[0] == "pearl tools 2>&1 | head -50"
    xml = '<invoke name="bash"><parameter name="command">ls -la</parameter></invoke> and then I made up output'
    cmd, end = text_action(xml)
    assert cmd == "ls -la" and xml[end:].strip().startswith("and then")
    assert text_action('```bash\necho a\n```\n<invoke name="bash">')[0] == "echo a\n"
    # a bash fence closed by DSML markup instead of ``` (seen from DeepSeek V4.1 in OnchainBench)
    mixed = "Now the trade.\n\n```bash\ncast call $M x</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    cmd, end = text_action(mixed)
    assert cmd == "cast call $M x" and mixed[end:].strip() == "</｜DSML｜ calls>"
    assert text_action("```bash\nno close</｜DSML｜ parameter>")[0] == "no close"


def test_a_bash_fence_closes_only_on_a_line_of_its_own():
    from olas_ocb_agent.agent import text_action

    readme = "```bash\ncat > README.md <<'EOF'\n# Title\n\n```python\nprint(1)\n```\nEOF\nrm -rf build\n```\n"
    cmd, end = text_action(readme)  # the heredoc's fenced example is part of the command
    assert cmd == "cat > README.md <<'EOF'\n# Title\n\n```python\nprint(1)\n```\nEOF\nrm -rf build\n"
    assert readme[end:] == "\n"
    invented = "```bash\nls\n```\nOutput:\n```\na.txt\n```\n"
    cmd, end = text_action(invented)  # an invented output block after it isn't
    assert cmd == "ls\n" and invented[end:].startswith("\nOutput:")
    assert text_action("```bash\necho 'a ``` b'\n```  \nmore")[0] == "echo 'a ``` b'\n"  # ``` inside a line
    # a nested fence that never balances: the last ``` line closes the block
    assert text_action("```bash\ncat <<EOF\n```python\nx\nEOF\n```")[0] == "cat <<EOF\n```python\nx\nEOF\n"
    assert text_action("```bash\nls -la```")[0] == "ls -la"  # no ``` line at all: the first ``` anywhere
    assert text_action("```bash\nls") == (None, -1)


def test_a_dropped_tail_is_noted_in_the_observation(tmp_path):
    from olas_ocb_agent.agent import DROPPED

    sh = PersistentShell(str(tmp_path), timeout=10)
    model = ScriptedModel(["```bash\necho a\n```\n<output>made up</output>", "```bash\necho b\n```", "done"])
    a = Agent(model=model, shell=sh, system_prompt="s")
    assert a.send("go") == "done" and a.n_truncated == 1
    first, second = (m["content"] for m in a.messages if m["content"].startswith("<returncode>"))
    assert first.endswith(DROPPED) and "<output>\na\n</output>" in first and DROPPED not in second
    sh.close()


def test_resume_restores_the_call_and_token_counts(tmp_path):
    class Usage(ScriptedModel):
        def query(self, messages, on_token=None):
            out = super().query(messages, on_token)
            log.append(out["content"])
            u = {"prompt_tokens": 10, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 4}}
            return {**out, "cost": 0.01, "usage": u}

    log = []  # every model call's reply, summaries included

    def reply(msgs):
        if "Summarise the conversation" in msgs[-1]["content"]:
            return "SUMMARY"
        return "```bash\necho " + "x" * 3000 + "\n```" if msgs[-1]["content"].startswith("make") else "ok"

    sh = PersistentShell(str(tmp_path), timeout=10)
    sess = Session.new(tmp_path / "s")
    cfg = AgentConfig(context_chars=8000, compact_at=0.5, keep_recent=2)
    a = Agent(model=Usage(reply), shell=sh, system_prompt="sys", session=sess, config=cfg)
    for i in range(3):
        a.send(f"make {i}")
    assert a.n_compactions >= 1 and a.n_calls > 6  # the summary calls are counted too
    a.compact()  # /compact last: its call is saved by itself, with no model reply after it
    n = len(log)  # every call, summaries too, counts once with its cost and its tokens
    assert log.count("SUMMARY") == a.n_compactions >= 2
    assert (a.n_calls, a.prompt_tokens, a.completion_tokens) == (n, 10 * n, 3 * n) and abs(a.cost - 0.01 * n) < 1e-9
    b = Agent.resume(sess, ScriptedModel([]), sh, "sys", cfg)
    assert (b.n_calls, b.prompt_tokens, b.completion_tokens) == (n, 10 * n, 3 * n) and abs(b.cost - a.cost) < 1e-9
    assert (a.cached_tokens, b.cached_tokens, b.n_compactions) == (4 * n, 4 * n, a.n_compactions)
    sh.close()


def test_shell_deadline_and_clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-should-not-leak")
    sh = PersistentShell(str(tmp_path), timeout=10)
    r = sh.run("while true; do echo tick; sleep 0.2; done", timeout=2)  # prints constantly: still ends at the deadline
    assert r["timed_out"]
    assert "sk-should-not-leak" not in sh.run("env")["output"]
    sh.close()


def test_retry_with_same_id_but_different_args_needs_approval():
    answers = [True, False]
    p = Policy(
        mode="ask", caps=SessionCaps(max_native_usd=1.0), approver=lambda s: answers.pop(0), default_chain="gnosis"
    )
    tx = {"target": "0x" + "ab" * 20, "value": 1, "request_id": "r", "chain": "gnosis"}
    assert p.check("safe_transaction", tx).allowed
    assert p.check("safe_transaction", dict(tx)).allowed  # true retry: no prompt
    assert not p.check(
        "safe_transaction", {**tx, "value": 10**17}
    ).allowed  # $0.10 on gnosis: asked, declined  # same id, bigger value: asked, declined


def test_mech_cap_binds_in_usd_on_gnosis():
    p = Policy(
        mode="auto",
        caps=SessionCaps(max_mech_usd=1.0),
        default_chain="gnosis",
        mech_asset_lookup=lambda chain, mech: (18, 1.0),
    )
    req = {"prompt": "q", "tool": "prediction-online", "max_payment": 6 * 10**17, "priority_mech": "0x1"}  # 0.6 xDAI
    assert p.check("mech_request", {**req, "request_id": "a"}).allowed
    d = p.check("mech_request", {**req, "request_id": "b"})  # 1.2 xDAI in total > $1
    assert not d.allowed and "mech payment cap" in d.rule


def test_an_unanswered_approval_is_refused_by_its_own_rule_not_declined():
    """Nobody there, the timeout, a Ctrl-C at the prompt: each its own rule; only a no is 'the user declined'."""
    from olas_ocb_agent.policy import APPROVAL_CANCELLED, APPROVAL_TIMEOUT, DECLINED, NO_APPROVER, UNANSWERED

    tx = {"chain": "gnosis", "target": "0x" + "ab" * 20, "value": 1}
    for answer in (NO_APPROVER, APPROVAL_TIMEOUT, APPROVAL_CANCELLED):
        p = Policy(mode="ask", approver=lambda s, a=answer: a)
        d = p.check("safe_transaction", tx)
        assert not d.allowed and d.rule == UNANSWERED[answer] and f"rule '{answer}'" in d.rule
        assert p.value_calls == 0
    for answer in (False, "y", None):  # a no, or anything that isn't a yes
        assert Policy(mode="ask", approver=lambda s, a=answer: a).check("safe_transaction", tx).rule == DECLINED
    assert Policy(mode="ask").check("safe_transaction", tx).rule == UNANSWERED[NO_APPROVER]  # no approver at all

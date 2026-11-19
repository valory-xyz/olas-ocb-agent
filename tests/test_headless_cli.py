"""The CLI as a headless harness: `ocb-agent -p ... --output json`, the flags a benchmark sets, and multi-turn
sessions through `-p --attended --resume <id>`. The model is a stand-in litellm module (no paid calls)."""

import hashlib
import json
import math
import os
import signal
import sys
import threading
import time
import types

import pytest

from olas_ocb_agent import cli
from olas_ocb_agent.agent import AgentConfig
from olas_ocb_agent.model import ModelError
from olas_ocb_agent.policy import SessionCaps


def _litellm(reply):
    """A litellm stand-in: `reply(messages)` gives each call's text; every call reports 10 prompt tokens (4 of
    them cached), 3 completion tokens and an OpenRouter charge of $0.01."""
    calls, prices = [], {}

    def chunk(text):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=text))])

    def completion(**kw):
        calls.append({**kw, "messages": list(kw["messages"])})  # as sent: the agent's list grows afterwards
        usage = types.SimpleNamespace(cost=0.01)
        return [chunk(reply(kw["messages"])), types.SimpleNamespace(choices=[], usage=usage)]

    def builder(chunks, messages):
        u = {"prompt_tokens": 10, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 4}}
        msg = types.SimpleNamespace(tool_calls=None)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg)], usage=types.SimpleNamespace(model_dump=lambda: u)
        )

    mod = types.SimpleNamespace(
        completion=completion,
        stream_chunk_builder=builder,
        completion_cost=lambda completion_response: 0.0,
        register_model=prices.update,
        exceptions=types.SimpleNamespace(BadRequestError=ValueError),
    )
    return mod, calls, prices


MODEL = "litellm:openrouter/acme/m1?provider=fireworks"
PRICE = {"input_cost_per_token": 3e-7, "output_cost_per_token": 1.2e-6}


def _args(ws, tmp_path, *extra):
    return [
        "--dir",
        str(ws),
        "--model",
        MODEL,
        "--sandbox",
        "none",
        "--approval",
        "auto",
        "--sessions-dir",
        str(tmp_path / "sessions"),
        *extra,
    ]


def _json(capsys) -> tuple[dict, str]:
    cap = capsys.readouterr()
    return json.loads(cap.out), cap.err  # stdout holds the JSON object and nothing else


def test_two_turn_attended_session(connect_ws, mock, monkeypatch, tmp_path, capsys):
    def reply(msgs):
        last = msgs[-1]["content"]
        if last == "check my wallet":
            return "```bash\npearl tools >/dev/null && echo listed\n```"
        if last.startswith("<returncode>"):
            return "Listed. Which chain should I use?"
        return "DONE: used gnosis"

    mod, calls, prices = _litellm(reply)
    monkeypatch.setitem(sys.modules, "litellm", mod)
    price = PRICE
    common = ["--attended", "--output", "json", "--model-price", json.dumps(price)]
    assert cli.main([*_args(connect_ws, tmp_path, *common), "-p", "check my wallet"]) == 0
    one, err = _json(capsys)
    assert "\033" not in json.dumps(one) and "olas-ocb-agent · workspace" in err and "\033" not in err
    assert prices == {"openrouter/acme/m1": price, "acme/m1": price}  # as the bench registers it, both names
    assert calls[0]["extra_body"]["provider"] == {"only": ["fireworks"], "allow_fallbacks": False}
    assert one["reply"] == "Listed. Which chain should I use?" and one["done"] is False and one["error"] is None
    assert (one["model_calls"], one["steps"], one["tools_listed"]) == (2, 2, 1)
    assert (one["prompt_tokens"], one["completion_tokens"], one["cached_tokens"]) == (20, 6, 8)
    assert math.isclose(one["cost_usd"], 0.02) and one["cost_sources"] == {"billed": 2}
    system = one["events"][0]["message"]
    assert system["role"] == "system" and "Running unattended" not in system["content"]  # attended: as the REPL
    assert [e["message"]["role"] for e in one["events"][1:]] == ["user", "assistant", "user", "assistant"]
    sid = one["session_id"]
    assert one["session_file"].endswith(f"{sid}.jsonl")

    assert cli.main([*_args(connect_ws, tmp_path, *common, "--resume", sid), "-p", "gnosis"]) == 0
    two, _ = _json(capsys)
    assert two["session_id"] == sid and two["reply"] == "DONE: used gnosis" and two["done"] is True
    assert (two["model_calls"], two["steps"], two["tools_listed"]) == (3, 1, 0)  # restored, plus this turn's
    assert (two["prompt_tokens"], two["completion_tokens"], two["cached_tokens"]) == (30, 9, 12)
    assert math.isclose(two["cost_usd"], 0.03) and two["cost_sources"] == {"billed": 3}
    assert [e["message"]["content"] for e in two["events"]] == ["gnosis", "DONE: used gnosis"]
    sent = calls[-1]["messages"]  # the resumed model saw the whole first turn
    assert [m["content"] for m in sent[1:3]] == [
        "check my wallet",
        "```bash\npearl tools >/dev/null && echo listed\n```",
    ]


def test_unattended_json_turn_with_the_bench_settings(connect_ws, mock, monkeypatch, tmp_path, capsys):
    run = '```bash\necho "$TRIAL_PROXY" && basename "$PWD"\n```'
    mod, calls, _ = _litellm(lambda msgs: "DONE: all good" if "<returncode>" in msgs[-1]["content"] else run)
    monkeypatch.setitem(sys.modules, "litellm", mod)
    skill = hashlib.sha256((connect_ws / ".agents/skills/pearl-connect/SKILL.md").read_bytes()).hexdigest()
    extra = ["--output", "json", "--agent-subdir", "agent", "--trusted-skill-hash", skill]
    extra += ["--shell-env", "TRIAL_PROXY=http://127.0.0.1:9"]
    assert cli.main([*_args(connect_ws, tmp_path, *extra), "-p", "go"]) == 0
    out, _ = _json(capsys)
    assert out["reply"] == "all good" and out["done"] is True and out["steps"] == 2
    system = calls[0]["messages"][0]["content"]
    assert "Running unattended" in system and "## Skills" in system and "Unverified skill files" not in system
    assert str(connect_ws / "agent") in system
    obs = calls[1]["messages"][-1]["content"]
    assert "<output>\nhttp://127.0.0.1:9\nagent\n</output>" in obs  # the shell's variable, in the agent's own dir


RUN_ECHO = "```bash\necho hi\n```"


def test_json_events_and_the_session_file_carry_timing(connect_ws, mock, monkeypatch, tmp_path, capsys):
    mod, calls, _ = _litellm(lambda msgs: "DONE: ok" if "<returncode>" in msgs[-1]["content"] else RUN_ECHO)
    monkeypatch.setitem(sys.modules, "litellm", mod)
    assert cli.main([*_args(connect_ws, tmp_path, "--output", "json"), "-p", "go"]) == 0
    out, _ = _json(capsys)
    events = out["events"]
    assert [e["message"]["role"] for e in events] == ["system", "user", "assistant", "user", "assistant"]
    assert all(isinstance(e["t"], float) and e["t"] >= 0 for e in events)
    system, request, action, obs, done = events
    assert "dur_s" not in system and "dur_s" not in request
    for reply in (action, done):  # model calls: duration and time to the first streamed token
        assert 0 <= reply["ttft_s"] <= reply["dur_s"] <= reply["t"]
    assert 0 <= obs["dur_s"] <= obs["t"] and "ttft_s" not in obs  # the command
    assert action["t"] <= obs["t"] <= done["t"]
    with open(out["session_file"]) as f:
        logged = [json.loads(x) for x in f.read().splitlines()]
    assert [e for e in logged if e["type"] == "message"] == events  # the same events, timing and all
    assert all("t" in e for e in logged)
    for sent in calls:  # never in what the model is sent
        assert all(set(m) <= {"role", "content"} for m in sent["messages"])


def test_limits_stop_the_turn(connect_ws, mock, monkeypatch, tmp_path, capsys):
    mod, _, _ = _litellm(lambda msgs: "```bash\necho again\n```")
    monkeypatch.setitem(sys.modules, "litellm", mod)
    assert cli.main([*_args(connect_ws, tmp_path, "--output", "json", "--step-limit", "2"), "-p", "loop"]) == 0
    out, _ = _json(capsys)
    assert out["reply"] == "[ocb-agent] step limit 2 reached for this turn." and out["done"] is False
    assert out["steps"] == 2
    assert cli.main([*_args(connect_ws, tmp_path, "--output", "json", "--cost-limit", "0"), "-p", "loop"]) == 0
    out, _ = _json(capsys)
    assert out["reply"].startswith("[ocb-agent] session cost limit $0.00") and out["model_calls"] == 0


def test_refusals_are_reported(connect_ws, mock, monkeypatch, tmp_path, capsys):
    pay = "```bash\npearl safe_transaction target=0x0000000000000000000000000000000000000001 value=1 chain=gnosis\n```"
    mod, _, _ = _litellm(lambda msgs: "DONE: refused" if "<returncode>" in msgs[-1]["content"] else pay)
    monkeypatch.setitem(sys.modules, "litellm", mod)
    args = _args(connect_ws, tmp_path, "--output", "json")
    args[args.index("auto")] = "deny"
    assert cli.main([*args, "-p", "pay"]) == 0
    out, _ = _json(capsys)
    assert [r["rule"] for r in out["broker_refusals"]] == [r["rule"] for r in out["policy_refusals"]]
    assert out["broker_refusals"][0]["tool"] == "safe_transaction" and "deny" in out["broker_refusals"][0]["rule"]


def test_model_error_still_prints_json(connect_ws, mock, monkeypatch, tmp_path, capsys):
    class Broken:
        def query(self, messages, on_token=None):
            raise ModelError("budget exhausted")

    real = cli.build
    monkeypatch.setattr(cli, "build", lambda *a, **k: real(*a, **{**k, "model": Broken()}))
    assert cli.main([*_args(connect_ws, tmp_path, "--output", "json"), "-p", "hi"]) == 1
    out, err = _json(capsys)
    assert out["reply"] == "[model] budget exhausted" and out["error"] == "model" and out["done"] is False
    assert "[model] budget exhausted" in err


def _spy(monkeypatch):
    seen = {}
    real = cli.build

    def spy(*a, **k):
        seen.update(k)
        from olas_ocb_agent.model import ScriptedModel

        return real(*a, **{**k, "model": ScriptedModel(["DONE: ok"])})

    monkeypatch.setattr(cli, "build", spy)
    return seen


def test_flags_reach_the_runtime(connect_ws, mock, monkeypatch, tmp_path):
    seen = _spy(monkeypatch)
    flags = ["--step-limit", "500", "--cost-limit", "inf", "--max-value-calls", "none", "--max-native-usd", "inf"]
    assert cli.main([*_args(connect_ws, tmp_path, *flags), "-p", "hi"]) == 0
    assert seen["config"].step_limit == 500 and seen["config"].cost_limit == math.inf
    assert seen["caps"].max_value_calls is None and seen["caps"].max_native_usd == math.inf
    assert seen["sessions_dir"] == tmp_path / "sessions"
    assert seen["caps"].allow_sign_message is False
    for v, want in (("0", 0), ("60", 60), ("INF", None), ("none", None)):  # 0 blocks; none/inf: no limit
        assert cli.main([*_args(connect_ws, tmp_path, "--max-value-calls", v), "-p", "hi"]) == 0
        assert seen["caps"].max_value_calls == want
    sha = "AB" * 32
    assert (
        cli.main([*_args(connect_ws, tmp_path, "--allow-sign-message", "--trusted-skill-hash", sha), "-p", "hi"]) == 0
    )
    assert seen["caps"].allow_sign_message is True and seen["trusted_skills"] == {"ab" * 32}


def test_max_value_calls_zero_blocks_value_calls(connect_ws, mock, monkeypatch, tmp_path, capsys):
    pay = "```bash\npearl safe_transaction target=0x0000000000000000000000000000000000000001 value=1 chain=gnosis\n```"
    mod, _, _ = _litellm(lambda msgs: "DONE: x" if "<returncode>" in msgs[-1]["content"] else pay)
    monkeypatch.setitem(sys.modules, "litellm", mod)
    assert cli.main([*_args(connect_ws, tmp_path, "--output", "json", "--max-value-calls", "0"), "-p", "pay"]) == 0
    out, _ = _json(capsys)
    assert "limit of 0 value calls" in out["broker_refusals"][0]["rule"]


def test_sign_message_follows_the_flag(connect_ws, mock, monkeypatch, tmp_path, capsys):
    sign = "```bash\npearl sign_message digest=0x" + "ab" * 32 + "\n```"
    mod, calls, _ = _litellm(lambda msgs: "DONE: x" if "<returncode>" in msgs[-1]["content"] else sign)
    monkeypatch.setitem(sys.modules, "litellm", mod)
    base = _args(connect_ws, tmp_path, "--output", "json", "--max-value-calls", "none")
    assert cli.main([*base, "-p", "sign"]) == 0
    off, _ = _json(capsys)
    assert [r["tool"] for r in off["broker_refusals"]] == ["sign_message"]  # off by default
    assert cli.main([*base, "--allow-sign-message", "-p", "sign"]) == 0
    on, _ = _json(capsys)
    assert on["broker_refusals"] == [] and "raw digest signing is off" not in calls[-1]["messages"][-1]["content"]


def test_defaults_are_unchanged(connect_ws, mock, monkeypatch):
    seen = _spy(monkeypatch)
    assert cli.main(["--dir", str(connect_ws), "--sandbox", "none", "-p", "hi"]) == 0
    assert seen["config"] == AgentConfig(unattended=True)  # -p alone: unattended, the default limits
    assert seen["caps"] == SessionCaps() and seen["approval"] == "ask"
    assert seen["sessions_dir"] is None and seen["agent_subdir"] == "ocb-agent"
    assert seen["trusted_skills"] == set() and seen["shell_env"] == {}
    assert seen["model"].startswith("facilitator:")


@pytest.mark.parametrize(
    "bad",
    [
        ["--attended"],  # without -p
        ["--output", "json"],  # without -p
        ["--turn-seconds", "5"],  # without -p
        ["-p", "x", "--model-price", json.dumps(PRICE)],  # not a litellm model
        ["-p", "x", "--model", "litellm:m", "--model-price", "[1]"],  # not an object
        ["-p", "x", "--model", "litellm:m", "--model-price", "nope"],
        ["-p", "x", "--model", "litellm:m", "--model-price", "{}"],  # no prices
        ["-p", "x", "--model", "litellm:m", "--model-price", '{"input_cost_per_token": 1}'],
        ["-p", "x", "--model", "litellm:m", "--model-price", json.dumps({**PRICE, "input_cost_per_token": "1"})],
        ["-p", "x", "--model", "litellm:m", "--model-price", json.dumps({**PRICE, "output_cost_per_token": True})],
        ["-p", "x", "--model", "litellm:m", "--model-price", json.dumps({**PRICE, "output_cost_per_token": -1})],
        ["-p", "x", "--max-value-calls", "-1"],
        ["-p", "x", "--max-value-calls", "lots"],
        *(
            ["-p", "x", f, v]
            for f in ("--max-native-usd", "--max-mech-usd", "--cost-limit", "--model-budget")
            for v in ("nan", "-0.5")
        ),
        *(["-p", "x", f, "-1"] for f in ("--step-limit", "--facilitator-deposit")),
        *(["-p", "x", "--turn-seconds", v] for v in ("0", "nan", "-3")),
        ["-p", "x", "--trusted-skill-hash", "ab" * 31],  # too short
        ["-p", "x", "--trusted-skill-hash", "zz" * 32],  # not hex
        ["-p", "x", "--shell-env", "NOVALUE"],
        ["-p", "x", "--shell-env", "=v"],
        *(
            ["-p", "x", "--shell-env", f"{k}=v"]
            for k in ("PATH", "HOME", "BASH_ENV", "ENV", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "path", "UV_CACHE_DIR")
        ),
    ],
)
def test_bad_flags_are_refused(bad):
    with pytest.raises(SystemExit) as e:
        cli.main(bad)
    assert e.value.code == 2


def test_register_price_uses_litellm(monkeypatch):
    got = {}
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(register_model=got.update))
    cli.register_price("litellm:anthropic/claude-x?tools=1", {"input_cost_per_token": 1.0})
    assert got == {"anthropic/claude-x": {"input_cost_per_token": 1.0}}


# ---- --shell-env -------------------------------------------------------------------------------


def test_shell_env_reserves_what_the_harness_sets(connect_ws):
    from olas_ocb_agent.workspace import Workspace

    assert set(Workspace.open(connect_ws).install_bin()) <= cli.SHELL_ENV_RESERVED
    proxies = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy"]
    for k in [*proxies, "no_proxy", "NODE_USE_ENV_PROXY"]:  # what the bench sends
        assert cli._shell_env(f"{k}=x") == (k, "x")


# ---- failures still report -------------------------------------------------------------------


def _broken(monkeypatch, exc):
    class Broken:
        def query(self, messages, on_token=None):
            raise exc

    from olas_ocb_agent.runtime import build as real  # not cli.build, which an earlier call may have replaced

    monkeypatch.setattr(cli, "build", lambda *a, **k: real(*a, **{**k, "model": Broken()}))


class _LitellmError(Exception):
    pass


_LitellmError.__module__ = "litellm.exceptions"


@pytest.mark.parametrize(
    "exc, kind",
    [
        (RuntimeError("APIConnectionError: openrouter 502"), "provider"),  # the review's case
        (ConnectionResetError("reset by peer"), "provider"),
        (_LitellmError("anything the client raised after its retries"), "provider"),
        (KeyError("choices"), "harness"),
    ],
)
def test_an_exception_in_the_turn_still_prints_json(connect_ws, mock, monkeypatch, tmp_path, capsys, exc, kind):
    _broken(monkeypatch, exc)
    assert cli.main([*_args(connect_ws, tmp_path, "--output", "json"), "-p", "hi"]) == 1
    out, _ = _json(capsys)
    assert out["error"] == kind and out["done"] is False and out["reply"].startswith("[error] ")
    assert out["error_detail"].startswith(type(exc).__name__) and out["model_calls"] == 0 and out["session_id"]


def test_error_detail_is_truncated_and_redacted(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret-value-123")
    e = RuntimeError("key or-secret-value-123, sk-abcdefghijklmnop, Bearer tok123, https://u:pw@h/x " + "y" * 3000)
    d = cli.error_detail(e)
    assert len(d) == cli.DETAIL_MAX and "or-secret-value-123" not in d and "sk-abcdefgh" not in d
    assert "tok123" not in d and "u:pw" not in d and d.startswith("RuntimeError: key ***")


def test_text_mode_raises_as_before_even_with_a_turn_limit(connect_ws, mock, monkeypatch, tmp_path):
    _broken(monkeypatch, KeyError("choices"))
    with pytest.raises(KeyError):
        cli.main([*_args(connect_ws, tmp_path, "--turn-seconds", "30"), "-p", "hi"])
    _broken(monkeypatch, ModelError("budget exhausted"))
    assert cli.main([*_args(connect_ws, tmp_path, "--turn-seconds", "30"), "-p", "hi"]) == 1


def test_a_failed_start_still_prints_json(connect_ws, mock, tmp_path, capsys):
    assert cli.main([*_args(connect_ws, tmp_path, "--output", "json", "--resume", "nope"), "-p", "hi"]) == 1
    out, _ = _json(capsys)
    assert out["error"] == "harness" and out["session_id"] == "nope" and "nope" in out["error_detail"]
    # text mode: one line (no traceback), and the sessions there are to resume
    assert cli.main([*_args(connect_ws, tmp_path, "--resume", "nope"), "-p", "hi"]) == 2
    err = capsys.readouterr().err
    assert "ocb-agent: error: no session 'nope'" in err and "no saved sessions for" in err


# ---- --turn-seconds and SIGTERM ------------------------------------------------------------------


def _gone(pid: int, within: float = 10.0) -> bool:
    deadline = time.time() + within
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_turn_seconds_stops_the_turn_and_reports(connect_ws, mock, monkeypatch, tmp_path, capsys):
    pidfile = tmp_path / "pid"
    hang = f"```bash\nsh -c 'echo $$ > \"{pidfile.as_posix()}\"; exec sleep 60'\n```"
    mod, _, _ = _litellm(lambda msgs: "DONE: x" if "<returncode>" in msgs[-1]["content"] else hang)
    monkeypatch.setitem(sys.modules, "litellm", mod)
    t0 = time.time()
    assert cli.main([*_args(connect_ws, tmp_path, "--output", "json", "--turn-seconds", "1.5"), "-p", "go"]) == 1
    out, _ = _json(capsys)
    assert time.time() - t0 < 20
    assert out["error"] == "timeout" and out["reply"] == "[timeout] the turn passed 1.5 s" and out["done"] is False
    assert out["model_calls"] == 1 and out["steps"] == 1 and out["prompt_tokens"] == 10  # the partial totals
    if os.name == "posix":  # the command still running was killed with the shell's process group
        assert _gone(int(pidfile.read_text()))


@pytest.mark.skipif(os.name != "posix", reason="SIGTERM to the own process: POSIX")
def test_sigterm_closes_the_runtime_and_reports(connect_ws, mock, monkeypatch, tmp_path, capsys):
    pidfile = tmp_path / "pid"
    cmd = f"sh -c 'echo $$ > \"{pidfile}\"; kill -TERM {os.getpid()}; exec sleep 60'"
    mod, _, _ = _litellm(lambda msgs: "DONE: x" if "<returncode>" in msgs[-1]["content"] else f"```bash\n{cmd}\n```")
    monkeypatch.setitem(sys.modules, "litellm", mod)
    before = signal.getsignal(signal.SIGTERM)
    assert cli.main([*_args(connect_ws, tmp_path, "--output", "json"), "-p", "go"]) == 1
    out, _ = _json(capsys)
    assert out["error"] == "terminated" and out["model_calls"] == 1 and _gone(int(pidfile.read_text()))
    assert signal.getsignal(signal.SIGTERM) is before  # restored


def test_a_turn_off_the_main_thread_installs_no_handler(connect_ws, mock, monkeypatch, tmp_path, capsys):
    mod, _, _ = _litellm(lambda msgs: "DONE: ok")
    monkeypatch.setitem(sys.modules, "litellm", mod)
    codes = []
    th = threading.Thread(
        target=lambda: codes.append(cli.main([*_args(connect_ws, tmp_path, "--output", "json"), "-p", "go"]))
    )
    th.start()
    th.join(60)
    assert codes == [0] and _json(capsys)[0]["reply"] == "ok"

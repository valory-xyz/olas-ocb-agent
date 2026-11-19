"""Timing on session events: `t` on every event, `dur_s` (and `ttft_s`) on model replies and command outputs. It is
metadata beside the message, so what the model is sent stays byte-identical, and old sessions still resume."""

import json
import types

import pytest

from olas_ocb_agent import agent as agent_mod
from olas_ocb_agent.agent import Agent, AgentConfig, Session


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(
        agent_mod, "time", types.SimpleNamespace(monotonic=c.monotonic, strftime=agent_mod.time.strftime)
    )
    return c


class Shell:
    cwd = None
    closed = False

    def __init__(self, clock, secs=3.0):
        self.clock, self.secs = clock, secs

    def run(self, cmd):
        self.clock.now += self.secs
        return {"output": f"ran {cmd}", "returncode": 0}


REPLIES = [
    {"content": "```bash\necho one\n```"},
    {"content": "```bash\necho two\n```\n<output>made up</output>"},  # the dropped tail
    {"content": "", "tool_calls": [{"id": "c1", "name": "bash", "arguments": '{"command": "ls"}', "command": "ls"}]},
    {"content": "", "tool_calls": [{"id": "c2", "name": "bash", "arguments": "{}", "command": ""}]},
    {"content": "DONE: finished"},
]


class Model:
    """Scripted replies; records each request as JSON bytes. stream: the first token comes 0.5 s in."""

    def __init__(self, clock, stream=True, secs=2.0):
        self.clock, self.stream, self.secs = clock, stream, secs
        self.sent: list[str] = []
        self.i = 0

    def query(self, messages, on_token=None):
        self.sent.append(json.dumps(messages, sort_keys=True))
        if messages[-1]["content"].startswith("Summarise"):
            reply = {"content": "the summary"}
        else:
            reply = REPLIES[self.i]
            self.i += 1
        if self.stream:
            self.clock.now += 0.5
            on_token(reply["content"])
            self.clock.now += self.secs - 0.5
        else:
            self.clock.now += self.secs
        return {"cost": 0.0, "usage": {}, **reply}


def _run(clock, session, stream=True, model_secs=2.0, shell_secs=3.0):
    model = Model(clock, stream, model_secs)
    a = Agent(
        model=model,
        shell=Shell(clock, shell_secs),
        system_prompt="sys",
        session=session,
        config=AgentConfig(unattended=True),
    )
    assert a.send("go") == "finished"
    a.compact()
    return model.sent, a


def test_the_model_is_sent_the_same_bytes_with_and_without_timing(clock, tmp_path):
    timed, _ = _run(clock, Session(tmp_path / "timed.jsonl"))
    untimed, _ = _run(clock, None, stream=False, model_secs=7.0, shell_secs=11.0)  # no events, other durations
    assert timed == untimed
    for sent in timed:
        for m in json.loads(sent):
            assert not {"t", "dur_s", "ttft_s", "dropped"} & set(m)
    obs = "<returncode>0</returncode>\n<output>\nran {}\n</output>"
    assert [m["content"] for m in json.loads(timed[4])[1:]] == [
        "go",
        "```bash\necho one\n```",
        obs.format("echo one"),
        "```bash\necho two\n```",
        obs.format("echo two") + agent_mod.DROPPED,
        "",
        obs.format("ls"),
        "",
        "<returncode>-1</returncode>\n<output>\nerror: the bash tool needs a 'command'\n</output>",
    ]


def test_events_carry_t_and_durations(clock, tmp_path):
    session = Session(tmp_path / "s.jsonl")
    _run(clock, session)
    events = session.events()
    assert all(isinstance(e["t"], float) for e in events)
    msgs = [e for e in events if e["type"] == "message"]
    assert (msgs[0]["message"]["role"], msgs[0]["t"]) == ("system", 0.0) and "dur_s" not in msgs[0]
    timed = [(e["message"]["role"], e["t"], e.get("dur_s"), e.get("ttft_s")) for e in msgs[1:]]
    assert timed == [
        ("user", 0.0, None, None),  # the request
        ("assistant", 2.0, 2.0, 0.5),  # model: 2 s, first token 0.5 s in
        ("user", 5.0, 3.0, None),  # command: 3 s
        ("assistant", 7.0, 2.0, 0.5),
        ("user", 10.0, 3.0, None),
        ("assistant", 12.0, 2.0, 0.5),  # a native tool call
        ("tool", 15.0, 3.0, None),
        ("assistant", 17.0, 2.0, 0.5),
        ("tool", 17.0, None, None),  # no command: nothing ran
        ("assistant", 19.0, 2.0, 0.5),  # DONE
    ]
    (comp,) = [e for e in events if e["type"] == "compaction"]
    assert (comp["t"], comp["dur_s"], comp["ttft_s"]) == (21.0, 2.0, 0.5)
    state = [e for e in events if e["type"] == "state"]
    assert state[0]["t"] == 2.0 and state[-1]["t"] == 21.0


def test_no_ttft_without_a_streamed_token(clock, tmp_path):
    session = Session(tmp_path / "s.jsonl")
    _run(clock, session, stream=False)
    replies = [e for e in session.events() if e["type"] == "message" and e["message"]["role"] == "assistant"]
    assert replies and all(e["dur_s"] == 2.0 and "ttft_s" not in e for e in replies)


def test_a_session_saved_without_timing_resumes(clock, tmp_path):
    path = tmp_path / "old.jsonl"
    old = [
        {"type": "message", "message": {"role": "system", "content": "old sys"}},
        {"type": "message", "message": {"role": "user", "content": "first"}},
        {"type": "message", "message": {"role": "assistant", "content": "answered"}},
        {"type": "state", "state": {"cost": 0.5, "n_calls": 1, "n_turns": 1}},
    ]
    path.write_text("".join(json.dumps(e) + "\n" for e in old))
    model = Model(clock)
    model.i = len(REPLIES) - 1  # replies DONE at once
    a = Agent.resume(Session(path), model, Shell(clock), "sys", AgentConfig(unattended=True))
    assert (a.cost, a.n_calls, a.n_turns) == (0.5, 1, 1)
    assert a.send("second") == "finished"
    assert [m["content"] for m in json.loads(model.sent[0])] == ["sys", "first", "answered", "second"]
    new = Session(path).events()[len(old) :]
    assert [e["message"]["content"] for e in new if e["type"] == "message"] == ["second", "DONE: finished"]
    assert all("t" in e for e in new)

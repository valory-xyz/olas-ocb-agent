"""The olas-ocb-agent agent loop: mini-swe-agent's design plus what a chat product needs.

Kept from mini-swe-agent: bash is the only tool; actions are parsed from a
fenced block in plain text (so any model works, no tool-calling API needed);
the history is linear, so the trajectory is exactly what the model saw.

Added: multi-turn chat (a reply without a bash block ends the turn and goes
to the user), context compaction, an append-only session log that can be
resumed, streaming callbacks and per-turn limits.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ACTION_RE = re.compile(r"```(?:bash|sh)[ \t]*\n(.*?)```", re.S)  # any closing ```: the fallback, and for summaries
OPEN_RE = re.compile(r"```(?:bash|sh)[ \t]*\n")
FENCE_RE = re.compile(r"^```([^\s`]*)[ \t]*$", re.M)  # a fence line: ``` alone closes, ```lang opens a nested block
# Some models emit their native tool-call markup as text when the provider doesn't parse it
# (DeepSeek's DSML, Anthropic-style XML). Read it as the action rather than losing the step.
INVOKE_RE = re.compile(
    r"<[^>]*invoke name=\"bash\"[^>]*>\s*<[^>]*parameter name=\"command\"[^>]*>(.*?)</[^>]*parameter>", re.S
)

# A bash fence the model closed with its tool-call markup instead of ``` (DeepSeek V4.1 does this, in OnchainBench).
MIXED_RE = re.compile(r"```(?:bash|sh)[ \t]*\n(.*?)</[^>]*parameter>", re.S)


def fenced_action(text: str) -> tuple[int, str, int] | None:
    """(start, command, end) of the first fenced bash block. It closes on a ``` line of its own; a ```lang line
    inside it (a heredoc holding a fenced example) opens a nested block, whose own ``` line doesn't close the
    bash block. If the nesting never balances, the last ``` line closes it; with no ``` line at all, the first ```
    anywhere does (a block closed on its command's line)."""
    m = OPEN_RE.search(text)
    if not m:
        return None
    depth, last = 0, None
    for f in FENCE_RE.finditer(text, m.end()):
        if f.group(1):
            depth += 1
        elif depth:
            depth, last = depth - 1, f
        else:
            return m.start(), text[m.end() : f.start()], f.end()
    if last is not None:
        return m.start(), text[m.end() : last.start()], last.end()
    old = ACTION_RE.match(text, m.start())
    return (m.start(), old.group(1), old.end()) if old else None


def text_action(text: str) -> tuple[str | None, int]:
    """(command, end offset) of the first action in plain text: a fenced bash block, else tool-call markup,
    else a bash fence closed by tool-call markup."""
    m = fenced_action(text)
    n = INVOKE_RE.search(text)
    if m and (not n or m[0] < n.start()):
        return m[1], m[2]
    if n:
        end = text.find(">", text.find("invoke>", n.end()))
        return n.group(1), (end + 1 if end > 0 else n.end())
    x = MIXED_RE.search(text)
    if x:
        end = text.find(">", text.find("invoke>", x.end()))
        return x.group(1), (end + 1 if end > 0 else x.end())
    return None, -1


NUDGE_EMPTY = "Your reply was empty. Run the next command in one ```bash block, or answer."
NUDGE_DONE = (
    "You're running unattended, so nobody will reply to that. If the task is finished, reply with no bash "
    "block, starting with 'DONE:' and your report. Otherwise run the next command in one ```bash block."
)

OBSERVATION = "<returncode>{rc}</returncode>{timeout}\n<output>\n{output}\n</output>"
DROPPED = (
    "\n<note>your reply went on after its bash block; that part was dropped unread. "
    "Send one bash block per reply and end the reply there.</note>"
)
SHELL_CLOSED = "[ocb-agent] the shell was closed (the session ended); stopping this turn."
# The limit is checked before each model call: the call that crosses it finishes (and its command runs), then the
# turn stops before the next call. Its spend is restored on resume, so only a higher limit lets it go on.
COST_LIMIT = (
    "[ocb-agent] session cost limit ${limit:.2f} reached (${cost:.4f} spent): this turn stopped before its next "
    "model call, and the session can't go on at this limit. Start a new session (run ocb-agent without --resume), "
    "or resume this one with a higher --cost-limit."
)


def count_turns(messages: list[dict]) -> int:
    """The user's requests in a message list (not observations, nudges or a compaction summary): for sessions saved
    before the turn count was."""
    skip = ("<returncode>", "[Summary of the earlier conversation]", NUDGE_EMPTY, NUDGE_DONE)
    return sum(1 for m in messages if m["role"] == "user" and not str(m.get("content") or "").startswith(skip))


SUMMARY_PROMPT = (
    "Summarise the conversation so far for your own later use; it will replace the older messages. Keep: "
    "the user's goals and instructions, decisions taken, facts learned (addresses, amounts, tx hashes, "
    "request ids, file paths), what was done and what is still pending, and any errors. Be complete but terse. "
    "Do not include a bash block."
)


@dataclass
class AgentConfig:
    step_limit: int = 30  # model calls per user turn (--step-limit; a benchmark may set its own)
    cost_limit: float = 2.0  # USD per session for API (litellm) backends (marketplace models use the model budget)
    context_chars: int = 400_000  # ~100k tokens at 4 chars/token
    compact_at: float = 0.75  # compact when the history passes this share of context_chars
    keep_recent: int = 6  # messages kept verbatim after a compaction
    unattended: bool = False  # headless: the turn ends only on a 'DONE:' reply
    max_nudges: int = 3  # consecutive no-action replies nudged before giving up


def _secs(s: float) -> float:
    return round(s, 3)


def _flatten_tools(messages: list[dict]) -> list[dict]:
    """Tool calls and results as plain text, for the summary request (and for backends without tools)."""
    out = []
    for m in messages:
        if m.get("tool_calls"):
            cmds = "\n".join(json.loads(c["function"]["arguments"] or "{}").get("command", "") for c in m["tool_calls"])
            out.append({"role": "assistant", "content": (m.get("content") or "") + f"\n```bash\n{cmds}\n```"})
        elif m["role"] == "tool":
            out.append({"role": "user", "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


class Session:
    """Append-only JSONL: one event per message, plus compaction events. Replay = resume."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def new(cls, root: Path) -> Session:
        sid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        return cls(root / f"{sid}.jsonl")

    @property
    def id(self) -> str:
        return self.path.stem

    def append(self, event: dict) -> None:
        with self._lock, self.path.open("a") as f:  # the broker thread saves spend too
            f.write(json.dumps(event) + "\n")

    def events(self) -> list[dict]:
        """Every event logged so far, in order."""
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def replay(self) -> tuple[list[dict], dict]:
        """The message list and state to resume from. Timing (`t`, `dur_s`, `ttft_s`) sits on the event, outside the
        message, so it never reaches the model; sessions saved before it was logged replay the same."""
        messages, state = [], {}
        for ev in self.events():
            if ev["type"] == "message":
                messages.append(ev["message"])
            elif ev["type"] == "compaction":
                messages = ev["messages"]
            elif ev["type"] == "state":
                state.update(ev["state"])
        return messages, state


@dataclass
class UI:
    """Callbacks; the CLI prints, the eval runner records, tests ignore."""

    on_token: Callable[[str], None] = lambda s: None
    on_action: Callable[[str], None] = lambda s: None
    on_observation: Callable[[dict], None] = lambda o: None
    on_info: Callable[[str], None] = lambda s: None


@dataclass
class Agent:
    model: object
    shell: object
    system_prompt: str
    session: Session | None = None
    config: AgentConfig = field(default_factory=AgentConfig)
    messages: list = field(default_factory=list)
    ledger: object = None  # snapshot() -> dict / restore(dict): the session's spend outside the agent (runtime)
    cost: float = 0.0
    n_calls: int = 0
    n_compactions: int = 0
    n_truncated: int = 0  # replies cut after their first bash block (the model ran ahead, e.g. invented output)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # prompt tokens the provider served from its cache (part of prompt_tokens), where reported
    cost_sources: dict = field(default_factory=dict)  # model calls per cost source: billed (by the provider), estimate
    n_turns: int = 0  # the user's requests this session (send calls)
    last_request: str = ""  # the latest of them (first 500 characters), for the recap on --resume
    # monotonic start of the current turn (of the agent, before its first turn): events log `t` from it
    t0: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        self.t0 = time.monotonic()
        if not self.messages:
            self._add({"role": "system", "content": self.system_prompt})

    # ---- public --------------------------------------------------------
    @classmethod
    def resume(cls, session: Session, model, shell, system_prompt: str, config: AgentConfig | None = None, ledger=None):
        messages, state = session.replay()
        if messages and messages[0]["role"] == "system":  # today's prompt: the sandbox in place now, the edited brief
            messages[0] = {"role": "system", "content": system_prompt}
        agent = cls(
            model=model,
            shell=shell,
            system_prompt=system_prompt,
            session=session,
            config=config or AgentConfig(),
            messages=messages,
            ledger=ledger,
            cost=float(state.get("cost") or 0.0),  # caps are per session, not per process
            cost_sources=dict(state.get("cost_sources") or {}),
            n_calls=int(state.get("n_calls") or 0),  # and so are the counts the session reports
            prompt_tokens=int(state.get("prompt_tokens") or 0),
            completion_tokens=int(state.get("completion_tokens") or 0),
            cached_tokens=int(state.get("cached_tokens") or 0),
            n_compactions=int(state.get("n_compactions") or 0),
            n_turns=int(state.get("n_turns") or count_turns(messages)),
            last_request=str(state.get("last_request") or ""),
        )
        if ledger is not None:
            ledger.restore(state)
        if state.get("cwd") and Path(state["cwd"]).is_dir():
            shell.run(f"cd {json.dumps(state['cwd'])}")
        return agent

    def _query(self, messages: list[dict], on_token=None) -> tuple[dict, dict]:
        """A model call: (reply, timing). timing has `dur_s` (request sent to reply complete) and, once a token
        arrived, `ttft_s`. If the call fails after paying (a mech follow-up, a facilitator deposit), its spend is
        saved before the error goes up, so `--resume` keeps it. Record-only fields (`dropped`) never reach the
        model."""
        messages = [{k: v for k, v in m.items() if k != "dropped"} if "dropped" in m else m for m in messages]
        start, first = time.monotonic(), []

        def token(s: str) -> None:
            if not first:
                first.append(time.monotonic())
            if on_token:
                on_token(s)

        try:
            reply = self.model.query(messages, on_token=token)
        except Exception:
            self._save_state()
            raise
        timing = {"dur_s": _secs(time.monotonic() - start)}
        if first:
            timing["ttft_s"] = _secs(first[0] - start)
        return reply, timing

    def _run(self, cmd: str) -> tuple[dict, dict]:
        """A shell command: (output, timing), `dur_s` from sending it to its returncode."""
        start = time.monotonic()
        out = self.shell.run(cmd)
        return out, {"dur_s": _secs(time.monotonic() - start)}

    def _t(self) -> float:
        return _secs(time.monotonic() - self.t0)

    def _count(self, reply: dict) -> None:
        """Every model call, the turn's and compaction's summary alike: one call, its cost and its tokens."""
        self.n_calls += 1
        self.cost += reply.get("cost", 0.0) or 0.0
        u = reply.get("usage") or {}
        self.prompt_tokens += int(u.get("prompt_tokens") or 0)
        self.completion_tokens += int(u.get("completion_tokens") or 0)
        self.cached_tokens += int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        if src := reply.get("cost_source"):
            self.cost_sources[src] = self.cost_sources.get(src, 0) + 1

    def _save_state(self) -> None:
        """cwd plus everything spent and counted so far, so --resume continues the same caps and totals."""
        if self.session:
            state = {
                "cost": self.cost,
                "cost_sources": self.cost_sources,
                "n_calls": self.n_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cached_tokens": self.cached_tokens,
                "n_compactions": self.n_compactions,
                "n_turns": self.n_turns,
                "last_request": self.last_request,
                **(self.ledger.snapshot() if self.ledger else {}),
            }
            if self.shell is not None:
                state["cwd"] = self.shell.cwd
            self.session.append({"type": "state", "state": state, "t": self._t()})

    def send(self, user_text: str, ui: UI | None = None) -> str:
        """One user turn: loop until the model replies without an action. Returns that reply."""
        ui = ui or UI()
        self.t0 = time.monotonic()  # the turn's events log `t` from here
        self._add({"role": "user", "content": user_text})
        self.n_turns += 1
        self.last_request = user_text[:500]
        nudges = 0
        for _ in range(self.config.step_limit):
            if getattr(self.shell, "closed", False):  # the session was closed under a running turn
                return self._stop(SHELL_CLOSED, ui)
            self._maybe_compact(ui)
            if self.cost >= self.config.cost_limit:  # checked before each model call: no call starts past the limit
                return self._stop(COST_LIMIT.format(limit=self.config.cost_limit, cost=self.cost), ui)
            reply, timing = self._query(self.messages, on_token=ui.on_token)
            self._count(reply)
            self._save_state()  # the call's spend, before anything else can fail
            text = reply["content"] or ""
            calls = reply.get("tool_calls") or []
            if calls:  # native tool calling: run the first call; answer the rest without running them
                self._add(
                    {
                        "role": "assistant",
                        "content": text,
                        "tool_calls": [
                            {
                                "id": c["id"],
                                "type": "function",
                                "function": {"name": c["name"], "arguments": c["arguments"]},
                            }
                            for c in calls
                        ],
                    },
                    timing,
                )
                nudges = 0
                cmd = calls[0]["command"]
                ui.on_action(cmd)
                out, ran = (
                    self._run(cmd)
                    if cmd
                    else ({"output": "error: the bash tool needs a 'command'", "returncode": -1}, {})
                )
                if out.get("closed"):
                    return self._stop(SHELL_CLOSED, ui)
                ui.on_observation(out)
                self._add({"role": "tool", "tool_call_id": calls[0]["id"], "content": self._observation(out)}, ran)
                for c in calls[1:]:
                    self._add(
                        {
                            "role": "tool",
                            "tool_call_id": c["id"],
                            "content": "not run: one command per step. Call it again after reading the output above.",
                        }
                    )
                self._save_state()
                continue
            cmd, end = text_action(text)
            dropped = cmd is not None and bool(text[end:].strip())
            message = {"role": "assistant", "content": text}
            if dropped:
                # Anything after the first action was written without seeing its output (often an invented
                # <output> and more commands). Keep the reply up to the action, run it, drop the rest, and say so.
                # The model's context gets the cut reply; the session and transcript also keep what was cut.
                text = text[:end]
                message = {"role": "assistant", "content": text, "dropped": message["content"][end:]}
                self.n_truncated += 1
            self._add(message, timing)
            if cmd is None:
                done = text.lstrip().upper().startswith("DONE")
                if text.strip() and (done or not self.config.unattended):
                    return text.lstrip()[5:].lstrip() if done and self.config.unattended else text
                if nudges >= self.config.max_nudges:  # say why the turn ended rather than return nothing
                    return (
                        text or f"[ocb-agent] the model gave {nudges + 1} empty replies in a row; stopping this turn."
                    )
                nudges += 1
                if text and not text.endswith("\n"):  # end the streamed reply, so the next one starts on a new line
                    ui.on_token("\n")
                self._add({"role": "user", "content": NUDGE_DONE if text.strip() else NUDGE_EMPTY})
                continue
            nudges = 0
            ui.on_action(cmd)
            out, ran = self._run(cmd)
            if out.get("closed"):
                return self._stop(SHELL_CLOSED, ui)
            ui.on_observation(out)
            self._add({"role": "user", "content": self._observation(out) + (DROPPED if dropped else "")}, ran)
            self._save_state()
        return self._stop(f"[ocb-agent] step limit {self.config.step_limit} reached for this turn.", ui)

    @staticmethod
    def _stop(msg: str, ui: UI) -> str:
        ui.on_info(msg)
        return msg

    def compact(self, ui: UI | None = None) -> None:
        ui = ui or UI()
        k = self.config.keep_recent
        if len(self.messages) <= k + 2:
            return
        head, middle, tail = self.messages[:1], self.messages[1:-k], self.messages[-k:]
        while tail and tail[0]["role"] != "user":  # start the kept tail on a user message (never on a tool result)
            middle.append(tail.pop(0))
        req = [*head, *_flatten_tools(middle), {"role": "user", "content": SUMMARY_PROMPT}]
        reply, timing = self._query(req)
        self._count(reply)
        summary = ACTION_RE.sub("", reply["content"]).strip()
        self.messages = [
            *head,
            {"role": "user", "content": f"[Summary of the earlier conversation]\n{summary}"},
            {"role": "assistant", "content": "Understood; continuing from that summary."},
            *tail,
        ]
        self.n_compactions += 1
        if self.session:
            self.session.append({"type": "compaction", "messages": self.messages, "t": self._t(), **timing})
        self._save_state()  # the summary call is counted like any other, so a resume keeps it
        ui.on_info(f"[ocb-agent] compacted {len(middle)} messages into a summary")

    # ---- internals -----------------------------------------------------
    @staticmethod
    def _observation(out: dict) -> str:
        timeout = (
            "\n<note>the command timed out and was killed; the shell was restarted</note>"
            if out.get("timed_out")
            else ""
        )
        return OBSERVATION.format(rc=out["returncode"], timeout=timeout, output=out["output"].rstrip("\n"))

    def _chars(self) -> int:
        return sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls", ""))) for m in self.messages)

    def _maybe_compact(self, ui: UI) -> None:
        if self._chars() > self.config.compact_at * self.config.context_chars:
            self.compact(ui)

    def _add(self, message: dict, timing: dict | None = None) -> None:
        """Log a message. Its event carries `t` (seconds since the turn started) and the timing of the model call
        or command that produced it, beside the message: the message itself, what the model is sent, is unchanged."""
        self.messages.append(message)
        if self.session:
            self.session.append({"type": "message", "message": message, "t": self._t(), **(timing or {})})

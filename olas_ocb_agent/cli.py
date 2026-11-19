"""olas-ocb-agent REPL.

    ocb-agent                                          # the user's Pearl Connect workspace, Gemini via the facilitator
    python -m olas_ocb_agent.cli --dir . --model litellm:anthropic/claude-sonnet-5-5
    python -m olas_ocb_agent.cli -p "what's in my safe?"   # one turn, headless
    python -m olas_ocb_agent.cli --resume last
    ocb-agent -p "task" --approval auto --output json          # headless, e.g. a benchmark: one JSON object on stdout
    ocb-agent -p "next" --attended --resume <id> --output json # a multi-turn session, one turn per process
    ocb-agent --list-sessions

In the REPL: /compact, /cost, /exit. Ctrl-C stops the running command (and cancels an approval prompt).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import select
import signal
import sys
import threading
import time
import urllib.error
from pathlib import Path

from . import __version__
from .agent import UI, AgentConfig
from .chains import fmt_amount, fmt_usd, parse_prices
from .interim.facilitator import DEFAULT_MODEL
from .model import ModelError, import_litellm
from .policy import (
    APPROVAL_CANCELLED,
    APPROVAL_TIMEOUT,
    CALL_DEADLINE_S,
    INPUT_CLOSED,
    NO_APPROVER,
    SessionCaps,
    capped,
)
from .runtime import SessionNotFound, build, list_sessions, session_started
from .workspace import CACHE_VARS, discover_connect_workspace

DIM, BOLD, YEL, RED, RST = "\033[2m", "\033[1m", "\033[33m", "\033[31m", "\033[0m"
_prompt_lock = threading.Lock()  # one approval prompt at a time; held while it reads stdin
_cancel = threading.Event()  # set on Ctrl-C: the prompt up gives stdin back to the REPL
CANCELLED = object()  # _read_line's answer when the prompt was cancelled


APPROVAL_TIMEOUT_S = CALL_DEADLINE_S  # an unanswered approval is refused after 10 minutes


def palette(on: bool) -> tuple[str, str, str, str, str]:
    """(DIM, BOLD, YEL, RED, RST) when `on`, else five empty strings: no ANSI codes in a pipe or a file."""
    return (DIM, BOLD, YEL, RED, RST) if on else ("",) * 5


def approver(summary: str, timeout: float = APPROVAL_TIMEOUT_S) -> bool | str:
    """True (yes), False (the user said no), or a policy rule when nobody answered: no terminal (or end of input),
    the timeout, or a Ctrl-C while the prompt was up."""
    _, bold, yel, _, rst = palette(sys.stdout.isatty())
    with _prompt_lock:
        # the REPL clears _cancel before each turn; a Ctrl-C since then (even while the summary was being built,
        # before this prompt took the lock) cancels this prompt too
        if _cancel.is_set():
            print(f"\n{yel}[approval cancelled]{rst}")
            return APPROVAL_CANCELLED
        print(f"\n{yel}{bold}approve?{rst}{yel} {summary}{rst}")
        if not sys.stdin.isatty():
            print(f"{yel}[nobody can approve here (stdin is not a terminal): refused, not declined]{rst}")
            return NO_APPROVER
        answer = _read_line(f"{yel}  [y/N, refused if unanswered after {int(timeout // 60)} min] {rst}", timeout)
        if answer is CANCELLED:
            print(f"\n{yel}[approval cancelled]{rst}")
            return APPROVAL_CANCELLED
        if answer is None:
            print(f"\n{yel}[no answer in time: refused, not declined]{rst}")
            return APPROVAL_TIMEOUT
        if answer == "":  # end of input (Ctrl-D) at a terminal: no answer to this call (a later one can be asked)
            print(f"\n{yel}[end of input: refused, not declined]{rst}")
            return INPUT_CLOSED
        return answer.strip().lower() in ("y", "yes")


def cancel_prompt(wait: float = 2.0) -> None:
    """After Ctrl-C: the approval prompt up (if any) stops reading stdin, and this returns once it has, so the
    REPL's next input() is the only reader. The cancel stays set, so a prompt the stopped turn opens later (its
    summary still being built) is cancelled too, until the REPL starts the next turn (new_turn)."""
    _cancel.set()
    if _prompt_lock.acquire(timeout=wait):
        _prompt_lock.release()


def new_turn() -> None:
    """A new REPL turn: its approval prompts may read stdin again."""
    _cancel.clear()


def _read_line(prompt: str, timeout: float, tick: float = 0.1):
    """One line from stdin within `timeout`; None when the time is up, CANCELLED after cancel_prompt(). POSIX polls
    with select, so nothing is left reading stdin afterwards (a reader thread would swallow the user's next line).
    Windows can't select on stdin, so it falls back to a reader thread, which a cancel or timeout leaves behind."""
    sys.stdout.write(prompt)
    sys.stdout.flush()
    end = time.monotonic() + timeout
    if os.name != "posix":
        return _read_line_thread(end, tick)
    while (left := end - time.monotonic()) > 0:
        if _cancel.is_set():
            return CANCELLED
        ready, _, _ = select.select([sys.stdin], [], [], min(tick, left))
        if ready and not _cancel.is_set():
            return sys.stdin.readline()
    return None


def _read_line_thread(end: float, tick: float):  # pragma: no cover - Windows only (no select on stdin there)
    answer: list = []

    def read():
        try:
            answer.append(input() + "\n")
        except EOFError:
            answer.append("")

    t = threading.Thread(target=read, daemon=True)
    t.start()
    while not answer and (left := end - time.monotonic()) > 0:
        if _cancel.is_set():
            return CANCELLED
        t.join(min(tick, left))
    return answer[0] if answer else None


def console_ui(color: bool = True) -> UI:
    dim, bold, yel, red, rst = palette(color)

    def obs(o):
        lines = o["output"].splitlines()
        preview = "\n".join(lines[:12]) + (f"\n… {len(lines) - 12} more lines" if len(lines) > 12 else "")
        c = red if o["returncode"] else dim
        print(f"{c}{preview}{rst}" if preview else f"{c}(rc={o['returncode']}){rst}")

    # whether streamed text left the cursor mid-line: the next line then starts with a newline, and only then (no
    # stray blank line before a turn's first output)
    mid_line = [False]

    def token(s):
        if s:
            sys.stdout.write(f"{dim}{s}{rst}")
            sys.stdout.flush()
            mid_line[0] = not s.endswith("\n")

    def line(text):
        print(("\n" if mid_line[0] else "") + text)
        mid_line[0] = False

    return UI(
        on_token=token,
        on_action=lambda c: line(f"{bold}$ {c.strip()}{rst}"),
        on_observation=obs,
        on_info=lambda s: line(f"{yel}{s}{rst}"),
    )


def _note(text: str, color: str = YEL) -> None:
    """A status line the user should see (a deposit, a blocked call, a resume recap): on stderr, so in JSON mode
    nothing new reaches stdout, even from a turn's thread after its report."""
    print(f"{color}{text}{RST}" if sys.stderr.isatty() else text, file=sys.stderr)


def count(n: int, noun: str) -> str:
    """'1 turn', '2 turns', '0 turns'."""
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _of(used: float, cap, fmt=fmt_usd) -> str:
    """'$0.05 of $0.10 ($0.05 left)', or '$0.05 (no cap)'."""
    return f"{fmt(used)} of {fmt(cap)} ({fmt(cap - used)} left)" if capped(cap) else f"{fmt(used)} (no cap)"


def model_spend(rt) -> str:
    """What the model cost this session: in the marketplace's payment asset and USD (facilitator, mech), else USD
    against the session cost limit."""
    m, a = rt.agent.model, rt.agent
    if rt.model_asset is not None and hasattr(m, "spent"):
        d, price, sym = rt.model_asset[0], rt.model_asset[1], rt.model_asset[2]
        out = f"{fmt_amount(m.spent, d, sym, price)} paid to the marketplace"
        if rt.model_budget is not None:
            left = max(0, rt.model_budget - m.spent)
            out += f", of a {fmt_amount(rt.model_budget, d, sym)} model budget ({fmt_amount(left, d, sym)} left)"
        return out
    return f"{_of(a.cost, a.config.cost_limit)} (session cost limit)"


def cost_report(rt) -> str:
    """/cost: model spend, each limit and what's left of it, the facilitator pre-deposit, and blocked calls."""
    a, p, m = rt.agent, rt.policy, rt.agent.model
    c = p.caps
    lines = [f"model: {count(a.n_calls, 'call')}, {model_spend(rt)}"]
    if getattr(m, "max_deposit", 0) or getattr(m, "deposited", 0):
        lines.append(
            f"facilitator pre-deposit: {m.amount(m.deposited)} moved from the safe, of a "
            f"{m.amount(m.max_deposit)} allowance"
        )
    calls = (
        "no cap" if c.max_value_calls is None else f"of {c.max_value_calls} ({c.max_value_calls - p.value_calls} left)"
    )
    lines.append(f"value calls: {p.value_calls} {calls}")
    lines.append(f"native value sent: {_of(p.spent_native, c.max_native_usd)}; ERC-20 amounts are not counted")
    lines.append(f"mech payments: {_of(p.spent_mech, c.max_mech_usd)}")
    blocked = [e for e in p.log if not e["allowed"]]
    lines.append(f"blocked calls: {len(blocked)}" + (f" (last: {blocked[-1]['rule']})" if blocked else ""))
    return "\n".join(lines)


def recap(rt) -> str:
    """Printed on --resume: where the session stands."""
    a, p = rt.agent, rt.policy
    sid = a.session.id
    last = a.last_request.replace("\n", " ")
    return (
        f"resumed session {sid} (started {session_started(sid)}): {count(a.n_turns, 'turn')}, "
        f"{count(a.n_calls, 'model call')}, "
        f"{model_spend(rt)}\n"
        f"value calls {p.value_calls}, native {fmt_usd(p.spent_native)}, mech {fmt_usd(p.spent_mech)}\n"
        f"last request: {(last[:200] + '…') if len(last) > 200 else last or '(none)'}"
    )


def print_sessions(root: Path, file=None, sessions_dir: Path | None = None) -> None:
    """The workspace's saved sessions, newest first, as --resume takes them."""
    file = file or sys.stdout
    rows = list_sessions(root, sessions_dir)
    if not rows:
        print(f"no saved sessions for {root}", file=file)
        return
    print(f"saved sessions for {root} (newest first; resume one with --resume <id>):", file=file)
    for r in rows:
        last = r["last_request"].replace("\n", " ")[:60]
        print(
            f"  {r['id']}  started {r['started']}  {count(r['turns'], 'turn')}  "
            f"{count(r['model_calls'], 'model call')}  {fmt_usd(r['usd'])}" + (f"  {last!r}" if last else ""),
            file=file,
        )


def _amount(v: str) -> float:
    """A USD (or seconds) limit: a number >= 0, or inf; never NaN."""
    x = float(v)
    if math.isnan(x) or x < 0:
        raise argparse.ArgumentTypeError(f"expected a number >= 0 (or inf), got {v!r}")
    return x


def _count(v: str) -> int:
    n = int(v)
    if n < 0:
        raise argparse.ArgumentTypeError(f"expected a count >= 0, got {v!r}")
    return n


def _count_or_none(v: str) -> int | None:
    """--max-value-calls: a count (0 blocks every value call); 'none' or 'inf' mean no limit."""
    return None if v.strip().lower() in ("none", "inf") else _count(v)


def _seconds(v: str) -> float:
    x = _amount(v)
    if x == 0:
        raise argparse.ArgumentTypeError("expected a number of seconds > 0")
    return x


def _sha256(v: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", v):
        raise argparse.ArgumentTypeError(f"expected a SHA-256 as 64 hex characters, got {v!r}")
    return v.lower()


# What --shell-env may not set: the harness's own shell setup (Workspace.install_bin's variables, PATH first, so
# `pearl` stays the harness's), HOME, and what makes bash or the loader run other code.
SHELL_ENV_RESERVED = {"PATH", "HOME", "BASH_ENV", "ENV", "PAGER", "GIT_PAGER", "PIP_NO_INPUT"}
SHELL_ENV_RESERVED |= set(CACHE_VARS)


def _shell_env(v: str) -> tuple[str, str]:
    k, sep, val = v.partition("=")
    if not sep or not k:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {v!r}")
    if k.upper() in SHELL_ENV_RESERVED or k.upper().startswith(("LD_", "DYLD_")):
        raise argparse.ArgumentTypeError(f"--shell-env can't set {k}: the harness sets up the shell's {k}")
    return k, val


def _price(v: str) -> dict:
    """--model-price: a LiteLLM model_cost entry with numeric input and output prices per token."""
    entry = json.loads(v)
    if not isinstance(entry, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    for k in ("input_cost_per_token", "output_cost_per_token"):
        x = entry.get(k)
        if isinstance(x, bool) or not isinstance(x, (int, float)) or math.isnan(x) or x < 0:
            raise argparse.ArgumentTypeError(f"--model-price needs a number >= 0 for {k}")
    return entry


# A model API's failure (after LiteLLM's own retries): its client's connection, timeout, 429 or 5xx errors.
PROVIDER_MODULES = ("litellm", "openai", "anthropic", "httpx", "httpcore")
PROVIDER_TEXT = re.compile(
    r"APIConnectionError|APITimeoutError|RateLimitError|InternalServerError|ServiceUnavailable|BadGateway|Overloaded"
    r"|\b(?:429|50[0-4]|529)\b|timed out|Connection (?:reset|refused|aborted)",
    re.I,
)
DETAIL_MAX = 1500


def error_kind(e: BaseException) -> str:
    """'provider' for a model API or transport failure, else 'harness'."""
    if isinstance(e, (ConnectionError, TimeoutError, urllib.error.URLError)):
        return "provider"
    if any(c.__module__.split(".")[0] in PROVIDER_MODULES for c in type(e).__mro__[:-1]):
        return "provider"
    return "provider" if PROVIDER_TEXT.search(f"{type(e).__name__}: {e}") else "harness"


SECRET_RE = re.compile(r"(?:sk|pk|rk)-[A-Za-z0-9_\-]{8,}|Bearer\s+\S+|(?<=://)[^/\s:@]+:[^/\s@]+(?=@)")


def error_detail(e: BaseException) -> str:
    """The exception as one truncated line, with key-like strings and the process's secret env values redacted."""
    text = f"{type(e).__name__}: {e}"
    for k, v in os.environ.items():
        if len(v) >= 8 and re.search(r"KEY|TOKEN|SECRET|PASSWORD|AUTH", k, re.I):
            text = text.replace(v, "***")
    return SECRET_RE.sub("***", text)[:DETAIL_MAX]


class Terminated(Exception):
    """SIGTERM arrived during a -p turn."""


def register_price(model: str, entry: dict) -> None:
    """Give LiteLLM the model's price (a LiteLLM model_cost entry, USD per token) in this process, under its name
    with and without the openrouter/ prefix (responses may name the model either way), so the cost estimate and
    --cost-limit use it when the provider reports no charge."""
    litellm = import_litellm()
    name = model.partition(":")[2].partition("?")[0]
    litellm.register_model({name: entry, name.removeprefix("openrouter/"): entry})


class TurnRecorder:
    """What a -p turn logged in this process: its session events, and what the broker refused or listed."""

    def __init__(self, rt, resumed: bool):
        self.broker_refusals: list[dict] = []
        self.tools_listed = 0
        # a new session logged its system prompt before the recorder was in place
        session = rt.agent.session
        logged = [] if resumed else session.events()
        self.events = [e for e in logged if e.get("type") in ("message", "compaction")]
        append = session.append

        def recording(event: dict) -> None:
            append(event)
            if event.get("type") in ("message", "compaction"):
                self.events.append(event)

        session.append = recording
        if rt.broker is not None:
            shown = rt.broker.on_refusal  # the one-line [blocked] note, kept

            def refused(tool, args, rule):
                self.broker_refusals.append({"tool": tool, "args": args, "rule": rule})
                if shown is not None:
                    shown(tool, args, rule)

            rt.broker.on_refusal = refused
            rt.broker.on_list = self._listed

    def _listed(self, tools) -> None:
        self.tools_listed += 1


def turn_report(
    rt, rec: TurnRecorder, reply: str, calls_before: int, error: str | None = None, detail: str | None = None
) -> dict:
    """What --output json prints after a -p turn. Counts, tokens and cost are session totals (restored across
    --resume); `steps` is this turn's model calls. error: None, 'model' (a model budget or cap), 'provider',
    'harness', 'timeout' (--turn-seconds) or 'terminated' (SIGTERM)."""
    a = rt.agent
    last = a.messages[-1] if a.messages else {}
    done = not error and last.get("role") == "assistant"
    done = done and (last.get("content") or "").lstrip().upper().startswith("DONE")
    return {
        "reply": reply,
        "session_id": a.session.id,
        "done": done,
        "error": error,
        "error_detail": detail,
        "model_calls": a.n_calls,
        "steps": a.n_calls - calls_before,
        "prompt_tokens": a.prompt_tokens,
        "completion_tokens": a.completion_tokens,
        "cached_tokens": a.cached_tokens,
        "cost_usd": a.cost,
        "cost_sources": dict(a.cost_sources),  # billed: what the provider charged; estimate: tokens x price
        "compactions": a.n_compactions,
        "policy_refusals": [e for e in list(rt.policy.log) if not e["allowed"]],  # this turn's
        "broker_refusals": list(rec.broker_refusals),  # this turn's: every call the broker refused, with its rule
        "tools_listed": rec.tools_listed,  # this turn's tool listings through the broker (`pearl tools`)
        "session_file": str(a.session.path),
        "events": list(rec.events),  # the session events this turn logged (messages, compactions); a new session's
    }  # first starts with the system prompt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ocb-agent")
    ap.add_argument("--dir", type=Path, help="workspace (default: the Pearl Connect workspace)")
    ap.add_argument(
        "--model",
        default=f"facilitator:{DEFAULT_MODEL}",
        help="facilitator:<gemini model>[@chain] | mech:<tool>[@mech] | litellm:<model> (default: %(default)s; "
        "litellm: needs the [litellm] extra)",
    )
    ap.add_argument(
        "--approval",
        default="ask",
        choices=["ask", "auto", "deny"],
        help="calls that move funds or sign: ask = prompt for each one (default); auto = allow them within the "
        "session caps; deny = refuse them all",
    )
    ap.add_argument(
        "--sandbox",
        default="auto",
        choices=["auto", "none", "seatbelt", "bwrap"],
        help="the agent shell's OS sandbox: auto = Seatbelt on macOS, bubblewrap on Linux, none where neither is "
        "available (default); seatbelt or bwrap = that one, or fail; none = no sandbox",
    )
    ap.add_argument(
        "--max-native-usd",
        type=_amount,
        default=0.10,
        help="session cap on native-token value sent (xDAI, ETH, ...), in USD (inf: none); ERC-20 transfers and "
        "approvals are not capped",
    )
    ap.add_argument(
        "--max-mech-usd", type=_amount, default=1.00, help="session cap on mech payments, in USD (inf: none)"
    )
    ap.add_argument(
        "--max-value-calls",
        type=_count_or_none,
        default=SessionCaps().max_value_calls,
        metavar="N",
        help="session cap on calls that move funds or sign (0 blocks them; 'none': no limit)",
    )
    ap.add_argument(
        "--step-limit", type=_count, default=AgentConfig().step_limit, help="model calls per user turn, at most"
    )
    ap.add_argument(
        "--cost-limit",
        type=_amount,
        default=AgentConfig().cost_limit,
        help="USD per session for litellm models; the turn stops once it is reached",
    )
    ap.add_argument(
        "--native-usd",
        action="append",
        metavar="CHAIN=PRICE",
        help="USD price of a chain's native token (any EVM chain; repeatable); overrides olas_ocb_agent/chains.py",
    )
    ap.add_argument(
        "--model-budget",
        type=_amount,
        default=0.05,
        help="USD that marketplace model calls may consume this session (converted to the mech's payment asset; "
        "default: %(default)s)",
    )
    ap.add_argument(
        "--facilitator-deposit",
        type=_count,
        default=0,
        help="base units (wei on gnosis) the session may move from the safe into the mech facilitator's pre-deposit, "
        "which pays for the default model; 50000000000000000 is 0.05 xDAI, about 50 calls (default 0: never). "
        "Each deposit is shown on screen and in /cost",
    )
    ap.add_argument("--resume", help="session id, or 'last' (--list-sessions shows them)")
    ap.add_argument("--list-sessions", action="store_true", help="list the workspace's saved sessions and exit")
    ap.add_argument(
        "--allow-brief-edits",
        action="store_true",
        help="in a plain project folder, let the agent write AGENTS.md and CLAUDE.md (read-only by default: they "
        "brief the next session)",
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument(
        "--allow-sign-message",
        action="store_true",
        help="let the agent ask for raw digest signatures (sign_message; off by default: a digest can't be read)",
    )
    ap.add_argument("-p", "--print", dest="prompt", help="run one turn headless and print the reply")
    headless = ap.add_argument_group("headless turns (-p), e.g. for benchmark runs")
    headless.add_argument(
        "--attended",
        action="store_true",
        help="a -p turn may end with a question or text to the user, as in the REPL; continue with -p --resume <id>"
        " (default: unattended, the turn ends only on a 'DONE:' reply)",
    )
    headless.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="json: print one JSON object with the reply, session id and counts on stdout (also when the turn fails "
        "or times out); progress goes to stderr. No interactive approvals: use --approval auto or deny",
    )
    headless.add_argument(
        "--turn-seconds",
        type=_seconds,
        metavar="N",
        help="stop the turn after N seconds (killing its running command) and still report",
    )
    headless.add_argument("--sessions-dir", type=Path, help="where session files live (default: under the state dir)")
    headless.add_argument(
        "--agent-subdir", default="ocb-agent", help="the agent's working directory inside a Connect workspace"
    )
    headless.add_argument(
        "--trusted-skill-hash",
        type=_sha256,
        action="append",
        default=[],
        metavar="SHA256",
        help="treat a workspace skill file with this SHA-256 as instructions, this session only (repeatable); "
        "a launcher such as Pearl passes one for each skill it ships",
    )
    headless.add_argument(
        "--shell-env",
        type=_shell_env,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="set a variable in the agent's shell only, not in the harness (repeatable), e.g. a logging web proxy",
    )
    headless.add_argument(
        "--model-price",
        type=_price,
        metavar="JSON",
        help="register a litellm model's price in this process: a LiteLLM model_cost entry "
        '(e.g. {"input_cost_per_token": 3e-7, "output_cost_per_token": 1.2e-6})',
    )
    ap.add_argument(
        "--trust-skill",
        type=Path,
        action="append",
        metavar="SKILL.md",
        help="approve a workspace skill file as instructions (its hash is recorded outside the sandbox) and exit",
    )
    a = ap.parse_args(argv)
    if not a.prompt and (a.attended or a.output == "json" or a.turn_seconds):
        ap.error("--attended, --output json and --turn-seconds apply to a -p turn")
    if a.model_price is not None:
        if not a.model.startswith("litellm:"):
            ap.error("--model-price is for a litellm: model")
        try:
            register_price(a.model, a.model_price)
        except ModelError as e:  # litellm isn't installed
            print(f"ocb-agent: error: {e}", file=sys.stderr)
            return 2
    if a.trust_skill:
        from .skill_trust import trust

        for p in a.trust_skill:
            print(f"trusted {p} (sha256 {trust(p)})")
        return 0

    root = a.dir or discover_connect_workspace()
    if root is None:
        print("No Pearl Connect workspace found; pass --dir.", file=sys.stderr)
        return 2
    if a.list_sessions:
        print_sessions(root, sessions_dir=a.sessions_dir)
        return 0
    if a.output == "json":
        stdout = sys.stdout
        with contextlib.redirect_stdout(sys.stderr):  # the banner, the progress and anything a library prints
            code, report = _run(a, root)
        stdout.write(json.dumps(report) + "\n")
        stdout.flush()
        return code
    return _run(a, root)[0]


def _raise_terminated(signum, frame):
    raise Terminated


def _turn(rt, a: argparse.Namespace, ui: UI) -> tuple[str, str | None, str | None]:
    """One -p turn: (reply, error kind or None, error detail or None). In JSON mode, or with --turn-seconds, the
    turn runs in a worker thread so a timeout or SIGTERM still reports; the caller then closes the runtime, which
    kills the shell's process group (and so the command still running)."""
    if a.output == "text" and not a.turn_seconds:
        try:
            return rt.agent.send(a.prompt, ui), None, None
        except ModelError as e:  # a budget or cap inside the model backend
            return f"[model] {e}", "model", None
        except Exception as e:  # noqa: BLE001 - a provider's error: one line, not a traceback (a bug re-raises)
            return _text_error(e)
    stopped = threading.Event()  # once the turn is abandoned, the worker prints nothing more

    def guard(f):
        return lambda *x: None if stopped.is_set() else f(*x)

    ui = UI(*(guard(f) for f in (ui.on_token, ui.on_action, ui.on_observation, ui.on_info)))
    box: dict = {}

    def work():
        try:
            box["reply"] = rt.agent.send(a.prompt, ui)
        except BaseException as e:  # noqa: BLE001 - reported below, by kind
            box["exc"] = e

    main = threading.current_thread() is threading.main_thread()
    old = signal.signal(signal.SIGTERM, _raise_terminated) if main else None
    worker = threading.Thread(target=work, name="ocb-agent-turn", daemon=True)
    try:
        worker.start()
        worker.join(a.turn_seconds)
    except Terminated:
        stopped.set()
        return "[terminated] the turn was stopped by SIGTERM", "terminated", "SIGTERM"
    finally:
        if main:
            signal.signal(signal.SIGTERM, old)
    if worker.is_alive():
        stopped.set()
        return f"[timeout] the turn passed {a.turn_seconds:g} s", "timeout", f"--turn-seconds {a.turn_seconds:g}"
    e = box.get("exc")
    if e is None:
        return box["reply"], None, None
    if isinstance(e, ModelError):
        return f"[model] {e}", "model", error_detail(e)
    if a.output == "text":
        return _text_error(e)
    detail = error_detail(e)
    return f"[error] {detail}", error_kind(e), detail


def _text_error(e: BaseException) -> tuple[str, str, str]:
    """A -p text turn's failure: a model provider's (or the network's) error becomes one redacted
    `ocb-agent: error:` line; anything else is a bug in the harness and keeps its traceback."""
    if error_kind(e) != "provider":
        raise e
    detail = error_detail(e)
    return f"ocb-agent: error: {detail}", "provider", detail


def _run(a: argparse.Namespace, root: Path) -> tuple[int, dict | None]:
    text_mode = a.output == "text"
    color = text_mode and sys.stdout.isatty()  # no ANSI codes when stdout is a pipe or a file
    dim, bold, yel, red, rst = palette(color)
    err_red, err_rst = palette(text_mode and sys.stderr.isatty())[3:]  # a -p turn's error line goes to stderr
    try:
        rt = build(
            root,
            model=a.model,
            approval=a.approval,
            approver=approver,
            sandbox=a.sandbox,
            caps=SessionCaps(
                max_native_usd=a.max_native_usd,
                max_mech_usd=a.max_mech_usd,
                max_value_calls=a.max_value_calls,
                allow_sign_message=a.allow_sign_message,
            ),
            config=AgentConfig(
                step_limit=a.step_limit, cost_limit=a.cost_limit, unattended=bool(a.prompt) and not a.attended
            ),
            resume=a.resume,
            model_budget=a.model_budget,
            sessions_dir=a.sessions_dir,
            max_deposit=a.facilitator_deposit,
            native_prices=parse_prices(a.native_usd),
            trusted_skills=set(a.trusted_skill_hash),
            agent_subdir=a.agent_subdir,
            shell_env=dict(a.shell_env),
            allow_brief_edits=a.allow_brief_edits,
        )
    except Exception as e:  # a bad --resume, a sandbox that can't start: JSON mode still reports
        if isinstance(e, SessionNotFound):  # what there is to resume (stderr, in either mode)
            print_sessions(root, sys.stderr, a.sessions_dir)
        if text_mode:
            if not isinstance(e, (ValueError, RuntimeError, OSError)):
                raise
            # a mistyped model, price, session or sandbox: one line, not a traceback
            print(f"ocb-agent: error: {e}", file=sys.stderr)
            return 2, None
        detail = error_detail(e)
        print(f"[error] {detail}", file=sys.stderr)
        return 1, {"reply": f"[error] {detail}", "session_id": a.resume, "done": False, "error": "harness"} | {
            "error_detail": detail
        }
    ui = console_ui(color)
    print(
        f"{dim}olas-ocb-agent · workspace {rt.workspace.root} · sandbox: {rt.sandbox_desc} · approval: {a.approval}"
        f" · session {rt.agent.session.id} · version {__version__}{rst}"
    )
    _announce(rt, a)
    try:
        if a.prompt:
            rec = None if text_mode else TurnRecorder(rt, resumed=bool(a.resume))
            before = rt.agent.n_calls
            reply, error, detail = _turn(rt, a, ui)
            print(f"\n{err_red}{reply}{err_rst}" if error else "", file=sys.stderr if error else sys.stdout)
            report = turn_report(rt, rec, reply, before, error, detail) if rec else None
            return (0 if reply and not error else 1), report
        while True:
            try:
                text = input(f"\n{bold}> {rst}").strip()
            except EOFError:
                break
            if not text:
                continue
            if text in ("/exit", "/quit"):
                break
            if text == "/compact":
                rt.agent.compact(ui)
                continue
            if text == "/cost":
                print(cost_report(rt))
                continue
            try:
                new_turn()
                rt.agent.send(text, ui)
                print()
            except ModelError as e:  # budget exhausted, gate timeout, facilitator refusal: report, keep the session
                print(f"\n{red}[model] {e}{rst}")
            except KeyboardInterrupt:
                rt.shell.interrupt()
                cancel_prompt()  # an approval prompt up gives stdin back before the next input()
                print(f"\n{yel}[interrupted]{rst}")
            except Exception as e:  # noqa: BLE001 - e.g. a provider's error: one line, and the session goes on
                print(f"\n{red}[error] {error_detail(e)}{rst}")
    finally:
        rt.close()
    return 0, None


def _announce(rt, a: argparse.Namespace) -> None:
    """What the user should know at the start, and the screen hooks: what auto mode's cap covers, that nobody can
    approve without a terminal, every blocked call, the facilitator's deposits, the recap on --resume. All on stderr."""
    if rt.broker is not None:
        if a.approval == "auto":
            cap = f"capped at {fmt_usd(a.max_native_usd)}" if capped(a.max_native_usd) else "not capped (no cap)"
            _note(f"auto: native value {cap} this session; ERC-20 transfers and approvals are not capped")
        elif a.approval == "ask" and not sys.stdin.isatty():
            _note(
                "ask: stdin is not a terminal, so nobody can approve: every value call will be refused "
                "(run interactively, or use --approval auto with caps)"
            )
        rt.broker.on_refusal = lambda tool, args, rule: _note(f"[blocked] {tool}: {rule}", RED)
    if hasattr(rt.agent.model, "log"):  # the facilitator's deposits and retries
        rt.agent.model.log = _note
    if a.resume:
        _note(recap(rt), DIM)


if __name__ == "__main__":
    sys.exit(main())

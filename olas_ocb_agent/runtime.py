"""Wire a session together: workspace, broker, sandboxed shell, model, agent."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .agent import Agent, AgentConfig, Session, count_turns
from .broker import Broker
from .chains import STABLE_SYMBOLS, native_symbol, native_usd
from .connect_client import ConnectClient, ConnectError
from .interim.facilitator_assets import facilitator_asset, facilitator_symbol
from .model import LitellmModel, MechModel
from .policy import CALL_DEADLINE_S, Policy, SessionCaps, read_only_from
from .sandbox import BWRAP, wrapper_for
from .shell import PersistentShell
from .workspace import Workspace, build_system_prompt


def state_dir() -> Path:
    """$XDG_STATE_HOME/olas-ocb-agent, by default ~/.local/state/olas-ocb-agent: not a path the sandbox may write."""
    xdg = os.environ.get("XDG_STATE_HOME", "")
    return (Path(xdg) if os.path.isabs(xdg) else Path.home() / ".local" / "state") / "olas-ocb-agent"


STATE_DIR = state_dir()


def default_sessions_dir(root: Path) -> Path:
    """Where a workspace's sessions live by default: one directory per workspace, named by its resolved path."""
    return STATE_DIR / "sessions" / hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()


class SessionNotFound(FileNotFoundError):
    """--resume named a session that isn't there (or there is none to resume)."""

    def __init__(self, msg: str, sessions_dir: Path):
        super().__init__(msg)
        self.sessions_dir = sessions_dir


@dataclass
class Runtime:
    agent: Agent
    workspace: Workspace
    shell: PersistentShell
    broker: Broker | None
    policy: Policy
    sandbox_desc: str
    model_asset: tuple | None = None  # (decimals, USD price, symbol) of a marketplace model's payment asset
    model_budget: int | None = None  # in that asset's base units

    def close(self):
        self.shell.close()
        if self.broker:
            self.broker.stop()


_NO_CONNECT = (
    "a Connect workspace (no .mcp.json found): run in Pearl Connect's workspace, or pass --model litellm:<model>"
)
NEEDS_CONNECT = {  # the model kinds Connect pays for, and what to say without it
    "mech": f"a mech model needs {_NO_CONNECT}",
    "facilitator": f"the facilitator model needs {_NO_CONNECT}",
}


def make_model(
    spec: str,
    upstream: ConnectClient | None,
    model_budget: int | None = None,
    max_deposit: int | None = None,
    default_chain: str | None = None,
):
    """spec: 'facilitator:<gemini model>[@chain]' | 'mech:<tool>[@<mech address>]' | 'litellm:<model>' | a model
    instance. model_budget is in base units of the payment asset (build() converts from USD)."""
    if not isinstance(spec, str):
        return spec
    kind, _, rest = spec.partition(":")
    if kind in NEEDS_CONNECT and upstream is None:
        raise ValueError(NEEDS_CONNECT[kind])
    if kind == "mech":
        tool, _, mech = rest.partition("@")
        return MechModel(upstream, tool=tool, priority_mech=mech or None, budget=model_budget)
    if kind == "facilitator":  # facilitator:<gemini model>[@chain]
        from .interim.facilitator import make as make_facilitator

        return make_facilitator(
            upstream,
            rest,
            default_chain=default_chain,
            budget=model_budget,
            max_deposit=max_deposit or 0,
        )
    if kind == "litellm":  # litellm:<model>[?tools=1&provider=<openrouter provider>]
        return LitellmModel.from_spec(rest)
    raise ValueError(f"unknown model spec {spec!r}")


def build(
    root: Path,
    model,
    approval: str = "ask",
    approver=None,
    sandbox: str = "auto",
    caps: SessionCaps | None = None,
    timeout: int = 120,
    config: AgentConfig | None = None,
    resume: str | None = None,
    model_budget: float | None = 0.05,
    sessions_dir: Path | None = None,
    max_deposit: int | None = None,
    native_prices: dict | None = None,
    trusted_skills: set[str] | None = None,
    agent_subdir: str = "ocb-agent",
    shell_env: dict | None = None,
    allow_brief_edits: bool = False,
) -> Runtime:
    """model_budget: USD that marketplace model calls may consume this session (default $0.05),
    converted to the payment asset's base units here. It replaces cost_limit for
    marketplace backends, which report no USD
    cost. allow_brief_edits: in a plain directory, let the agent write AGENTS.md and CLAUDE.md (read-only by default;
    a Connect workspace's brief is always read-only)."""
    ws = Workspace.open(
        Path(root), agent_subdir=agent_subdir, trusted_skills=trusted_skills, allow_brief_edits=allow_brief_edits
    )  # skills beyond these are unverified
    policy = Policy(mode=approval, caps=caps or SessionCaps(), approver=approver, native_prices=native_prices or {})
    kind = model.split(":")[0] if isinstance(model, str) else ""
    if kind in NEEDS_CONNECT and not ws.connect:  # before the budget conversion, which needs Connect's chain
        raise ValueError(NEEDS_CONNECT[kind])
    broker = upstream = None
    env = ws.install_bin() | (shell_env or {})  # shell_env: e.g. a benchmark's per-trial web proxy
    if ws.connect:
        upstream = ConnectClient(*ws.upstream_mcp())
        # the broker gets its own upstream session; the model client is separate so a long
        # mech model call never blocks the agent's own `pearl` calls
        broker = Broker(ConnectClient(upstream.mcp_url, upstream.token), policy).start()
        chains = {}
        try:  # the chain that calls omitting `chain` act on, for USD caps
            chains = upstream.rest("GET", "/wallet").get("chains") or {}
            policy.default_chain = next((c for c, e in chains.items() if e.get("actionable")), None)
        except Exception:  # noqa: BLE001 - best effort; without a default chain, value calls need `chain`
            policy.default_chain = None
        policy.mech_asset_lookup = mech_asset_lookup(upstream, chains, policy.native_prices)
        policy.token_lookup = token_lookup(chains)  # the approval prompt's ERC-20 amounts
        # read-only tools as Connect declares them; best effort: the fallback list until it does
        with contextlib.suppress(Exception):
            policy.read_only = read_only_from(upstream.list_tools()) or policy.read_only
        ws.install_shim(broker.shim_mcp_json())
    # the session file holds the spend ledger and the history --resume replays: the harness's alone, kept outside
    # the agent's directory and denied to the sandbox (read and write; every workspace's sessions, not just this one's)
    sdir = Path(sessions_dir) if sessions_dir else default_sessions_dir(ws.root)
    sdir.mkdir(parents=True, exist_ok=True)
    all_sessions = STATE_DIR / "sessions"
    all_sessions.mkdir(parents=True, exist_ok=True)
    agent_dir, real_sdir = ws.agent_dir.resolve(), sdir.resolve()
    if agent_dir.is_relative_to(real_sdir) or real_sdir.is_relative_to(agent_dir):
        raise ValueError(f"the sessions dir {sdir} must be outside the agent's working directory {ws.agent_dir}")
    # the shell may write its own directory only, not the workspace root that holds the brief and the skills (nor
    # rename it or add files to it, even when the root sits under a temp dir)
    # ~/.operate is denied as a whole (other services' keys and configs); the workspace inside it is read again
    wrapper, sandbox_desc = wrapper_for(
        ws.agent_dir,
        deny_read=[*ws.deny_read, sdir, all_sessions],
        deny_write=ws.deny_write,
        protect=[ws.root] if ws.connect else [],
        mode=sandbox,
        allow_read=[ws.root],
    )
    if sandbox_desc == BWRAP and not ws.connect and (links := [p.name for p in ws.deny_write if p.is_symlink()]):
        # a mount can cover a file, not a link: the agent could remove the link and write a brief of its own
        raise RuntimeError(
            f"{', '.join(links)} in {ws.root} is a symlink, which bubblewrap can't keep read-only: replace it with "
            "a copy, or pass --allow-brief-edits to let the agent edit the brief"
        )
    shell = PersistentShell(str(ws.agent_dir), env=env, wrapper=wrapper, timeout=timeout)
    if broker is not None:
        shell.hold, shell.max_hold = broker.busy, CALL_DEADLINE_S
        shell.on_kill = broker.command_killed  # its calls not yet sent are refused
    system = build_system_prompt(ws, sandbox_desc, timeout, unattended=bool(config and config.unattended))
    budget = asset = None
    if model_budget is not None and kind in ("mech", "facilitator"):
        chain = (model.partition("@")[2] if kind == "facilitator" else "") or policy.default_chain
        mech = model.partition("@")[2] if kind == "mech" else ""
        if kind == "mech":
            asset = policy.mech_asset_lookup(chain, mech) if (mech and policy.mech_asset_lookup) else None
        else:
            asset = facilitator_asset(chain, policy.native_prices)
            asset = asset and (*asset, facilitator_symbol(chain))
        if asset is None or asset[1] is None:
            raise ValueError(
                f"the model's payment asset on {chain!r} isn't known, so the ${model_budget} model budget can't "
                "be converted: name the mech (mech:<tool>@<address>, paid in native or a USD stablecoin)"
            )
        budget = int(model_budget / asset[1] * 10 ** asset[0])  # USD -> base units
    m = make_model(model, upstream, budget, max_deposit, policy.default_chain)
    if broker is not None and m.__class__.__name__ == "FacilitatorModel":
        from .interim.marketplace_gate import MarketplaceGate

        m.gate = broker.gate = MarketplaceGate(m)
    if asset is not None and hasattr(m, "asset"):
        m.asset = asset  # so the model states its deposits in xDAI and USD
    ledger = SpendLedger(policy, m, asset)
    if resume:
        path = (
            sdir / f"{resume}.jsonl"
            if resume != "last"
            else max(sdir.glob("*.jsonl"), default=None, key=lambda p: p.stat().st_mtime)
        )
        if path is None or not path.exists():
            shell.close()
            if broker is not None:
                broker.stop()
            what = "no saved session to resume" if resume == "last" else f"no session {resume!r}"
            raise SessionNotFound(f"{what} for {ws.root}", sdir)
        agent = Agent.resume(Session(path), m, shell, system, config, ledger=ledger)
    else:
        agent = Agent(
            model=m,
            shell=shell,
            system_prompt=system,
            session=Session.new(sdir),
            config=config or AgentConfig(),
            ledger=ledger,
        )
    policy.on_record = agent._save_state  # spend counted by the broker is saved at once
    if broker is not None and broker.gate is not None:
        broker.gate.on_floor = agent._save_state  # and so is the gate's pending floor
    if hasattr(m, "on_deposit"):
        m.on_deposit = agent._save_state  # a pre-deposit is saved the moment it is made, not with the next state
    return Runtime(
        agent=agent,
        workspace=ws,
        shell=shell,
        broker=broker,
        policy=policy,
        sandbox_desc=sandbox_desc,
        model_asset=asset,
        model_budget=budget,
    )


def session_started(sid: str) -> str:
    """'20261116-101500-ab12cd' (Session.new's ids) -> '2026-11-16 10:15'; '?' for an id of another shape."""
    if len(sid) < 15 or not (sid[:8] + sid[9:15]).isdigit():
        return "?"
    return f"{sid[0:4]}-{sid[4:6]}-{sid[6:8]} {sid[9:11]}:{sid[11:13]}"


def list_sessions(root: Path, sessions_dir: Path | None = None) -> list[dict]:
    """A workspace's saved sessions, newest first: id, start time, turns, model calls, spend in USD."""
    sdir = Path(sessions_dir) if sessions_dir else default_sessions_dir(Path(root))
    out = []
    for p in sorted(sdir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            messages, state = Session(p).replay()
        except (OSError, ValueError, KeyError):
            continue  # unreadable: not resumable either
        sid = p.stem
        started = session_started(sid)
        usd = sum(float(state.get(k) or 0) for k in ("cost", "spent_native", "spent_mech", "model_usd"))
        out.append(
            {
                "id": sid,
                "started": started,
                "turns": int(state.get("n_turns") or count_turns(messages)),
                "model_calls": int(state.get("n_calls") or 0),
                "usd": usd,
                "last_request": state.get("last_request") or "",
            }
        )
    return out


class SpendLedger:
    """What the session spent outside the agent (broker caps, model budget, pre-deposits), saved in the
    session's JSONL so `--resume` continues the same caps."""

    POLICY = ("spent_native", "spent_mech", "value_calls")
    MODEL = ("spent", "deposited")

    def __init__(self, policy: Policy, model, asset: tuple | None = None):
        self.policy, self.model, self.asset = policy, model, asset

    def snapshot(self) -> dict:
        out = {k: getattr(self.policy, k) for k in self.POLICY}
        # the approved (request_id, arguments hash) pairs, so a retry after --resume isn't counted as new spend
        out["approved"] = sorted(map(list, self.policy.approved.copy()))  # copy: the broker thread may add one
        out.update({f"model_{k}": getattr(self.model, k) for k in self.MODEL if hasattr(self.model, k)})
        if self.asset is not None and hasattr(self.model, "spent"):  # the marketplace model's spend in USD (listings)
            out["model_usd"] = self.model.spent / 10 ** self.asset[0] * self.asset[1]
        if (gate := getattr(self.model, "gate", None)) is not None:  # the agent's mech request still to settle
            out["gate_pending_floor"] = gate.pending_floor
        return out

    def restore(self, state: dict) -> None:
        for k in self.POLICY:
            if k in state:
                setattr(self.policy, k, state[k])
        self.policy.approved = {tuple(x) for x in state.get("approved") or []}
        for k in self.MODEL:
            if f"model_{k}" in state and hasattr(self.model, k):
                setattr(self.model, k, state[f"model_{k}"])
        if (gate := getattr(self.model, "gate", None)) is not None and "gate_pending_floor" in state:
            gate.pending_floor = state["gate_pending_floor"]


def mech_asset_lookup(upstream: ConnectClient, chains: dict, prices: dict):
    """(chain, mech) -> (decimals, USD price) of the mech's payment asset, read through Connect and the chain.

    `payment_token` comes from Connect's `mech_tools`; a token's decimals and symbol are read on-chain via
    the chain's RPC from Connect's /wallet. Stablecoins (by symbol) count at $1; anything else is unknown.
    Results are cached per (chain, mech); failures return None, and the caller falls back to the default.
    """
    cache: dict = {}

    tokens = token_lookup(chains)

    def resolve(chain, mech):
        try:
            info = upstream.call_tool("mech_tools", {"chain": chain, "priority_mech": mech})
        except ConnectError:
            return None
        token = str(info.get("payment_token") or "")
        if token == "native":
            return 18, native_usd(chain, prices), native_symbol(chain)
        t = tokens(chain, token) if token.startswith("0x") else None
        return None if t is None else (t[0], t[2], t[1])

    def lookup(chain: str | None, mech: str):
        key = (chain, mech.lower())
        if key not in cache:
            cache[key] = resolve(chain, mech)
        return cache[key]

    return lookup


def token_lookup(chains: dict):
    """(chain, token) -> (decimals, symbol, USD price or None) of an ERC-20, read on-chain through the chain's RPC
    from Connect's /wallet; stablecoins (by symbol) count at $1. Cached once read; None when it can't be."""
    cache: dict = {}

    def resolve(chain, token):
        rpc = (chains.get(chain) or {}).get("rpc")
        if not rpc:
            return None
        try:
            decimals = int(_eth_call(rpc, token, "0x313ce567"), 16)  # decimals()
            symbol = _decode_symbol(_eth_call(rpc, token, "0x95d89b41"))  # symbol()
        except (OSError, ValueError, KeyError):
            return None
        return decimals, symbol, (1.0 if symbol.upper() in STABLE_SYMBOLS else None)

    def lookup(chain: str | None, token: str):
        key = (chain, str(token).lower())
        if key not in cache and (found := resolve(chain, token)) is not None:  # a failure is tried again next time
            cache[key] = found
        return cache.get(key)

    return lookup


def _eth_call(rpc: str, to: str, data: str) -> str:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_call", "params": [{"to": to, "data": data}, "latest"]})
    headers = {"Content-Type": "application/json", "User-Agent": f"olas-ocb-agent/{__version__}"}
    req = urllib.request.Request(rpc, data=body.encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=20) as r:  # nosec B310 - the chain RPC Connect reports
        return json.loads(r.read())["result"]


def _decode_symbol(hexdata: str) -> str:
    """An ABI-encoded string, or a bytes32 symbol (older tokens)."""
    raw = bytes.fromhex(hexdata[2:])
    if len(raw) >= 96:
        n = int.from_bytes(raw[32:64], "big")
        return raw[64 : 64 + n].decode(errors="replace")
    return raw.rstrip(b"\x00").decode(errors="replace")

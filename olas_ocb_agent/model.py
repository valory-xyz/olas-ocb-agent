"""Model backends. All take the chat message list and return {"content", "cost", "usage"}.

- MechModel: buys each completion from an Olas marketplace mech through
  Connect's off-chain flow (signed digest, prepaid balance; no transaction,
  no API key). The intended default once chat models are served as marketplace mechs that Connect can pay;
  until then the CLI defaults to FacilitatorModel (olas_ocb_agent/interim/facilitator.py).
- LitellmModel: any provider litellm supports (bring-your-own-key, evals); needs the [litellm] extra.
- ScriptedModel: deterministic replies for tests and oracle runs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import uuid
from collections.abc import Callable

from .connect_client import ConnectClient, ConnectError


class ModelError(RuntimeError):
    pass


def _litellm_missing(reason: str) -> ModelError:
    return ModelError(
        f"the litellm backend needs olas-ocb-agent[litellm]: {reason} "
        "(install the package with its litellm extra, e.g. uv tool install 'olas-ocb-agent[facilitator,litellm]')"
    )


def require_litellm() -> None:
    """Fail early, with the extra to install, when litellm isn't installed (without importing it: that is slow)."""
    if sys.modules.get("litellm") is None and importlib.util.find_spec("litellm") is None:
        raise _litellm_missing("litellm is not installed")


def import_litellm():
    """litellm, or a ModelError that names the extra to install."""
    try:
        import litellm  # optional dependency
    except ImportError as e:
        raise _litellm_missing(str(e)) from e
    return litellm


BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run one bash command in the persistent shell and get its return code and output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command"}},
            "required": ["command"],
        },
    },
}


def parse_spec(spec: str) -> tuple[str, dict]:
    """'openrouter/x?tools=1&provider=fireworks' -> ('openrouter/x', {'tools': True, 'provider': 'fireworks'})."""
    name, _, q = spec.partition("?")
    opts = {}
    for part in filter(None, q.split("&")):
        k, _, v = part.partition("=")
        opts[k] = {"1": True, "true": True, "0": False, "false": False}.get(v.lower(), v) if v else True
    return name, opts


def billed_cost(chunks: list) -> float | None:
    """What the provider charged for a streamed call: OpenRouter's `usage.cost` on the stream's usage chunk."""
    for ch in reversed(chunks):
        u = getattr(ch, "usage", None)
        if u is None:
            continue
        c = getattr(u, "cost", None)
        if c is None:
            c = (getattr(u, "model_extra", None) or {}).get("cost")
        if c is not None:
            return float(c)
    return None


class LitellmModel:
    """tools=True: native tool calling with one `bash` tool. provider=<slug>: pin an OpenRouter provider
    (no fallbacks), as the harness leaderboard pins one provider per run."""

    def __init__(self, name: str, tools: bool = False, provider: str | None = None, **kwargs):
        self.name = name
        self.tools = tools
        self.kwargs = kwargs
        if provider:
            self.kwargs["extra_body"] = {"provider": {"only": [provider], "allow_fallbacks": False}}
        if name.startswith("openrouter/"):  # OpenRouter puts what it charged in the usage (usage.cost)
            self.kwargs.setdefault("extra_body", {})["usage"] = {"include": True}

    @classmethod
    def from_spec(cls, spec: str) -> LitellmModel:
        require_litellm()
        name, opts = parse_spec(spec)
        return cls(name, tools=bool(opts.get("tools")), provider=opts.get("provider"))

    def query(self, messages: list[dict], on_token: Callable[[str], None] | None = None) -> dict:
        litellm = import_litellm()

        # the model must not write the observation itself; rate limits and timeouts are retried with backoff
        kwargs = {"stop": ["<returncode>"], "num_retries": 3, **self.kwargs}
        if self.tools:
            kwargs.update(tools=[BASH_TOOL], tool_choice="auto")
        try:
            resp = litellm.completion(
                model=self.name, messages=messages, stream=True, stream_options={"include_usage": True}, **kwargs
            )
        except litellm.exceptions.BadRequestError:
            kwargs.pop("stop")  # some providers reject stop sequences
            resp = litellm.completion(
                model=self.name, messages=messages, stream=True, stream_options={"include_usage": True}, **kwargs
            )
        chunks, parts = [], []
        for ch in resp:
            chunks.append(ch)
            delta = ch.choices[0].delta.content if ch.choices else None
            if delta:
                parts.append(delta)
                if on_token:
                    on_token(delta)
        full = litellm.stream_chunk_builder(chunks, messages=messages)
        cost, source = billed_cost(chunks), "billed"
        if cost is None:  # no provider charge in the stream: estimate from tokens and the registered price
            source = "estimate"
            try:
                cost = float(litellm.completion_cost(completion_response=full))
            except Exception:  # noqa: BLE001 - litellm raises varied errors for models without a price entry
                cost = 0.0
        usage = getattr(full, "usage", None)
        out = {
            "content": "".join(parts),
            "cost": cost,
            "cost_source": source,
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else {},
        }
        calls = getattr(full.choices[0].message, "tool_calls", None) if full and full.choices else None
        if calls:
            out["tool_calls"] = []
            for c in calls:
                try:
                    cmd = json.loads(c.function.arguments or "{}").get("command", "")
                except ValueError:
                    cmd = ""
                out["tool_calls"].append(
                    {"id": c.id, "name": c.function.name, "arguments": c.function.arguments, "command": cmd}
                )
        return out


class MechModel:
    """A completion bought from a marketplace mech, off-chain.

    Calls go straight from the harness to the connect server (not through the
    agent's broker): the user sets a model budget for the session instead of
    approving each call. The prompt is the messages list as JSON, for mech tools that take a chat.
    """

    def __init__(
        self,
        client: ConnectClient,
        tool: str,
        priority_mech: str | None = None,
        budget: int | None = None,
        timeout: int = 300,
    ):
        self.client = client
        self.tool = tool
        self.priority_mech = priority_mech
        self.budget = budget  # total base units of the payment token this session may spend on the model
        self.spent = 0
        self.timeout = timeout

    def query(self, messages: list[dict], on_token: Callable[[str], None] | None = None) -> dict:
        cap = None
        if self.budget is not None:
            cap = self.budget - self.spent  # the mech refuses before payment if it costs more
            if cap <= 0:
                raise ModelError(f"model budget exhausted ({self.spent} of {self.budget} base units spent)")
        prompt = json.dumps(messages)
        args = {
            "prompt": prompt,
            "tool": self.tool,
            "legacy_on_chain": False,
            "timeout": self.timeout,
            "request_id": f"olas-ocb-agent-{uuid.uuid4().hex}",
        }
        for k, v in (("priority_mech", self.priority_mech), ("max_payment", cap)):
            if v is not None:
                args[k] = v
        t0, paid = time.time(), None
        try:
            res = self.client.call_tool("mech_request", args)
            paid = mech_charge(res, cap)  # the charge is in mech_request's answer; Connect's mech_result has none
            pending = list(res.get("pending_request_ids") or [])
            while pending and time.time() - t0 < 2 * self.timeout:  # paid already: resume, never resend
                res = self.client.call_tool("mech_result", {"request_id": pending[0]})
                if "payment" in res:
                    paid = mech_charge(res, cap)
                if res.get("delivered") is False:  # Connect's mech_result: not delivered yet, poll again
                    continue
                pending = list(res.get("pending_request_ids") or [])
        except ConnectError as e:
            if paid is not None:  # the request went out (paid) and only the follow-up failed: count it
                self.spent += paid
            if "exceeds max_payment" in str(e) and self.budget is not None:
                raise ModelError(
                    f"model budget exhausted: the next call costs more than the "
                    f"{self.budget - self.spent} base units left ({e})"
                ) from e
            raise ModelError(f"mech model call failed: {e}") from e
        self.spent += paid  # counted before parsing: a paid call is spent even if unreadable
        text = extract_mech_answer(res)
        if on_token:
            on_token(text)  # no token streaming over the marketplace: one chunk
        return {"content": text, "cost": 0.0, "usage": {"mech_payment": paid, "latency_s": round(time.time() - t0, 2)}}


def mech_charge(res: dict, cap: int | None) -> int:
    """Base units a mech_request answer says it charged. `payment` is the integer charge; a Connect that merges the
    mech report's payment label into the answer puts a dict there, and an answer may carry none: then the call
    counts as the whole cap, never less than it could have cost, and never crashes after it was paid."""
    p = res.get("payment")
    if p is None or isinstance(p, bool):
        return cap or 0
    try:
        return max(int(p), 0)
    except (TypeError, ValueError, OverflowError):
        return cap or 0


TEXT_KEYS = ("content", "text", "completion")
WRAPPER_KEYS = ("response", "result")


def _message(item):
    """Descend a delivery's wrappers to the message holding the completion. A mech's inline off-chain delivery is
    two deep ({request_id, status, content_cid, response: {schema_version, result: <text>, tool, cost_dict}}); a
    result file, or a delivery already unwrapped once, is one deep ({result: <text>})."""
    for _ in WRAPPER_KEYS:
        if not isinstance(item, dict) or any(k in item for k in TEXT_KEYS):
            break
        key = next((k for k in WRAPPER_KEYS if k in item), None)
        if key is None:
            break
        item = item[key]
    return item


def extract_mech_answer(res: dict) -> str:
    """The completion in a mech answer: mech_request's `delivery_results` (first delivery), mech_result's `result`,
    then Connect's own reading of it (top-level `completion`), which also renders native tool calls."""
    results = res.get("delivery_results") or ({"r": res["result"]} if "result" in res else {})
    completion = res.get("completion") if isinstance(res.get("completion"), str) else None
    if not results:
        if completion is not None:
            return completion
        raise ModelError(f"mech delivered no result: {json.dumps(res)[:300]}")
    message = _message(next(iter(results.values())))
    text = message
    if isinstance(message, dict):
        text = next((message[k] for k in TEXT_KEYS if isinstance(message.get(k), str)), None)
    if isinstance(text, str) and (text or not completion):
        return text
    if completion is not None:
        return completion
    return json.dumps(message) if isinstance(message, (dict, list)) else str(message)


class ScriptedModel:
    """replies: list of strings (consumed in order) or a function(messages) -> str."""

    def __init__(self, replies):
        self.replies = replies
        self.i = 0

    def query(self, messages: list[dict], on_token=None) -> dict:
        if callable(self.replies):
            text = self.replies(messages)
        else:
            text = "I have nothing further to do." if self.i >= len(self.replies) else self.replies[self.i]
            self.i += 1
        if on_token:
            on_token(text)
        return {"content": text, "cost": 0.0, "usage": {}}

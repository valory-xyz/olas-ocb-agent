"""What the broker lets the agent do without asking, what it asks about, and what it refuses.

This sits *in front of* Connect's own guardrail, never instead of it: a call
the broker allows still has to pass the connect server's rules. The broker's
job is the part Connect can't know about: this session, this user, right now.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from .chains import fmt_amount, fmt_usd, native_symbol, native_usd
from .interim.readonly_fallback import FALLBACK_READ_ONLY  # until Connect flags them; the runtime prefers its flags

# One deadline for a broker call: an unanswered approval is refused, and the marketplace gate
# gives up, when it passes; the agent's shell command waits for the broker meanwhile (shell.py `hold`).
CALL_DEADLINE_S = 600

# What an approver returns besides True (approved) and False (the user said no): nobody answered. Each is its own
# rule, so neither the agent nor the user is told that someone declined a call nobody saw.
NO_APPROVER = "no-approver"  # nobody can answer here (stdin is not a terminal, or no approver at all)
APPROVAL_TIMEOUT = "approval-timeout"  # the prompt was up, and nobody answered in time
APPROVAL_CANCELLED = "approval-cancelled"  # the user stopped the command (Ctrl-C) while its prompt was up
INPUT_CLOSED = "input-closed"  # at a terminal, the user ended input (Ctrl-D) instead of answering
DECLINED = "broker: the user declined this call"
UNANSWERED = {
    INPUT_CLOSED: (
        "broker: rule 'input-closed': the user ended the approval prompt's input (Ctrl-D) without answering, so it "
        "was not sent; nobody declined it"
    ),
    NO_APPROVER: (
        "broker: rule 'no-approver': nobody can approve calls in this session (no terminal to ask), so it was not "
        "sent; nobody declined it. The user can run interactively, or choose --approval auto with caps"
    ),
    APPROVAL_TIMEOUT: (
        f"broker: rule 'approval-timeout': nobody answered the approval prompt within {CALL_DEADLINE_S // 60} "
        "minutes, so it was not sent; nobody declined it"
    ),
    APPROVAL_CANCELLED: (
        "broker: rule 'approval-cancelled': the user stopped the command (Ctrl-C) while its approval prompt was up; "
        "not sent"
    ),
}

# Calls that move funds or produce a signature someone could use to move funds.
VALUE_TOOLS = {"safe_transaction", "send_transaction", "mech_request", "sign_message"}
REST_VALUE_ROUTES = {
    "/safe-transaction": "safe_transaction",
    "/sign-message": "sign_message",
}  # signer_client.py's routes
REST_READ_ROUTES = {"/wallet"}


@dataclass
class Decision:
    allowed: bool
    rule: str = ""  # named rule when refused, Connect-style


@dataclass
class SessionCaps:
    """Per-session ceilings in USD, converted per chain (olas_ocb_agent/chains.py, or user prices). None = no cap."""

    max_native_usd: float | None = 0.10  # value sent from the safe (or EOA)
    max_mech_usd: float | None = 1.00  # sum of declared max_payment across mech requests
    max_value_calls: int | None = 20
    allow_sign_message: bool = False  # raw digests are opaque: off unless the user turns it on


@dataclass
class Policy:
    """mode: 'ask' (default: prompt for every value call), 'auto' (allow within caps), 'deny'."""

    mode: str = "ask"
    caps: SessionCaps = field(default_factory=SessionCaps)
    approver: object = None  # callable(summary: str) -> bool, used in 'ask' mode
    spent_native: float = 0.0  # USD
    spent_mech: float = 0.0  # USD
    default_chain: str | None = None  # for calls that omit `chain` (set by the runtime from wallet_info)
    native_prices: dict = field(default_factory=dict)  # user overrides: chain -> native USD price (any EVM chain)
    # (chain, mech address) -> (decimals, USD price[, symbol]) of that mech's payment asset, or None (by the runtime)
    mech_asset_lookup: Callable | None = None
    # (chain, token address) -> (decimals, symbol, USD price or None) of an ERC-20, or None: only for the approval
    # prompt's amounts (set by the runtime); ERC-20 amounts aren't counted in the caps
    token_lookup: Callable | None = None
    read_only: set = field(default_factory=lambda: set(FALLBACK_READ_ONLY))  # never includes VALUE_TOOLS
    value_calls: int = 0
    approved: set = field(default_factory=set)  # (request_id, hash of the arguments) of every approved call
    log: list = field(default_factory=list)
    on_record: Callable | None = None  # called after spend is counted, so it is saved at once (set by the runtime)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check(self, tool: str, args: dict) -> Decision:
        with self._lock:
            d = self._check(tool, args)
            self.log.append({"tool": tool, "args": args, "allowed": d.allowed, "rule": d.rule})
            return d

    def _check(self, tool: str, args: dict) -> Decision:
        if tool in self.read_only and tool not in VALUE_TOOLS:
            return Decision(True)
        rid = args.get("request_id")
        if rid and (rid, _args_hash(tool, args)) in self.approved:
            return Decision(True)  # idempotent retry of something already approved: no new spend
        if self.mode == "deny":
            return Decision(False, "broker: value calls disabled for this session (approval mode 'deny')")
        if tool == "sign_message" and not self.caps.allow_sign_message:
            return Decision(False, "broker: raw digest signing is off for this session")
        chain = args.get("chain") or self.default_chain
        try:  # a negative amount would lower the counters and lift the caps, so it is refused, never counted
            wei = _int(args.get("value")) if tool in ("safe_transaction", "send_transaction") else 0
            pay = _int(args.get("max_payment")) if tool == "mech_request" else 0
        except NegativeAmount:
            return Decision(False, "broker: rule 'negative-amount': value and max_payment can't be negative")
        except ValueError:
            return Decision(
                False, "broker: rule 'bad-amount': value and max_payment must be integers (or 0x hex) below 2**256"
            )
        native = 0.0
        if wei:
            price = native_usd(chain, self.native_prices)
            if price is None:
                return Decision(
                    False,
                    f"broker: no USD price for {chain!r}'s native token, so value can't be capped "
                    f"(the user can set one: --native-usd {chain}=<price>)",
                )
            native = wei / 1e18 * price
        if tool == "mech_request" and args.get("max_payment") is None:
            # Connect's default cap is 0.1 of the payment asset, whose units differ per chain;
            # without an explicit cap the session total can't be counted, so ask for one
            return Decision(
                False,
                "broker: set max_payment explicitly (base units of the mech's payment "
                "asset; see mech_tools max_delivery_rate)",
            )
        mech = 0.0
        if tool == "mech_request":
            if not args.get("priority_mech"):  # its payment asset (and so the USD cap) depends on the mech
                return Decision(
                    False, "broker: name the mech (priority_mech; see mech_tools) so its price can be capped"
                )
            asset = self.mech_asset_lookup(chain, args["priority_mech"]) if self.mech_asset_lookup else None
            if asset is None or asset[1] is None:
                return Decision(
                    False,
                    f"broker: the payment asset of mech {args['priority_mech']} on {chain!r} isn't known "
                    "(not native or a USD stablecoin), so its price can't be capped",
                )
            mech = pay / 10 ** asset[0] * asset[1]
        if native < 0 or mech < 0:  # a negative price (a user's --native-usd, a mech asset): never lower the counters
            return Decision(False, "broker: rule 'negative-amount': the call would count a negative USD amount")
        c = self.caps
        if c.max_value_calls is not None and self.value_calls + 1 > c.max_value_calls:
            return Decision(False, f"broker: session limit of {c.max_value_calls} value calls reached")
        if c.max_native_usd is not None and self.spent_native + native > c.max_native_usd + 1e-12:
            return Decision(False, f"broker: session native cap ${c.max_native_usd:.2f} would be exceeded")
        if c.max_mech_usd is not None and self.spent_mech + mech > c.max_mech_usd + 1e-12:
            return Decision(False, f"broker: session mech payment cap ${c.max_mech_usd:.2f} would be exceeded")
        if tool not in VALUE_TOOLS and self.mode != "ask":
            return Decision(False, f"broker: unknown tool '{tool}' is not allowed without asking")
        if self.mode == "ask":
            answer = self.approver(self._summary(tool, args, chain, native, mech)) if self.approver else NO_APPROVER
            if isinstance(answer, str) and answer in UNANSWERED:  # nobody answered: not the same as a no
                return Decision(False, UNANSWERED[answer])
            if isinstance(answer, str) or not answer:
                return Decision(False, DECLINED)
        self.value_calls += 1
        self.spent_native += native
        self.spent_mech += mech
        if rid:
            self.approved.add((rid, _args_hash(tool, args)))
        if self.on_record:
            self.on_record()
        return Decision(True)

    def _summary(self, tool: str, args: dict, chain: str | None, native: float, mech: float) -> str:
        """What the approval prompt shows: amounts in the token's own units and USD, and what the session has left
        once this call is counted."""
        price = native_usd(chain, self.native_prices)
        token = None
        data = str(args.get("data") or "")
        if tool in ("safe_transaction", "send_transaction") and data[:10].lower() in ERC20 and self.token_lookup:
            target = args.get("target") or args.get("to")
            with contextlib.suppress(Exception):  # best effort: without it, the prompt shows base units
                token = self.token_lookup(chain, target) if target else None
        asset = None
        if tool == "mech_request" and self.mech_asset_lookup and args.get("priority_mech"):
            asset = self.mech_asset_lookup(chain, args["priority_mech"])
        out = summarize(
            tool,
            {**args, "chain": chain} if chain else args,
            native=(native_symbol(chain), price) if price is not None else None,
            token=token,
            mech=asset,
        )
        c, left = self.caps, []
        if capped(c.max_native_usd):
            left.append(
                f"native {fmt_usd(c.max_native_usd - self.spent_native - native)} of {fmt_usd(c.max_native_usd)}"
            )
        elif c.max_native_usd is not None:  # inf: say so rather than "$inf"
            left.append("native: no cap")
        if tool == "mech_request" and capped(c.max_mech_usd):
            left.append(f"mech {fmt_usd(c.max_mech_usd - self.spent_mech - mech)} of {fmt_usd(c.max_mech_usd)}")
        if c.max_value_calls is not None:
            left.append(f"{c.max_value_calls - self.value_calls - 1} of {c.max_value_calls} value calls")
        return out + (f"; left after this call: {', '.join(left)}" if left else "")


def capped(cap) -> bool:
    """A cap that limits something: not None (no cap) and not inf (the CLI's 'inf: none')."""
    return cap is not None and not math.isinf(cap)


def _args_hash(tool: str, args: dict) -> str:
    """A retry is the same call only if its arguments match what was approved."""
    return hashlib.sha256(json.dumps([tool, args], sort_keys=True, default=str).encode()).hexdigest()


class NegativeAmount(ValueError):
    """A negative value or max_payment: refused, since counting it would lower the session's spend."""


_AMOUNT = re.compile(r"(?P<neg>-)?(?:0[xX](?P<hex>[0-9a-fA-F]+)|(?P<dec>[0-9]+))", re.ASCII)


def _int(v) -> int:
    """A non-negative amount below 2**256 (a uint256) from an int, a decimal string or a 0x hex string; None and ""
    are 0.

    Raises NegativeAmount for a negative amount, ValueError for anything else: not an integer, or too large for
    a uint256 (which would also overflow the float conversion to USD).
    """
    if v is None:
        return 0
    if isinstance(v, int):
        n = int(v)
    else:
        s = str(v).strip() or "0"
        m = _AMOUNT.fullmatch(s)
        if m is None:  # ASCII only: no "1_000", "+5" or other scripts' digits, which int() would take
            raise ValueError(f"not an amount: {s[:40]!r}")
        n = (int(m["hex"], 16) if m["hex"] else int(m["dec"])) * (-1 if m["neg"] else 1)
    if n < 0:
        raise NegativeAmount(n)
    if n >= 2**256:
        raise ValueError("amount too large for a uint256")
    return n


ERC20 = {"0xa9059cbb": "ERC-20 transfer", "0x095ea7b3": "ERC-20 approve"}


def summarize(
    tool: str,
    args: dict,
    *,
    native: tuple[str, float | None] | None = None,
    token: tuple[int, str, float | None] | None = None,
    mech: tuple | None = None,
) -> str:
    """One line a person can approve. native: (symbol, USD price) of the chain's native token; token: (decimals,
    symbol, USD price or None) of the ERC-20 at the target; mech: (decimals, USD price[, symbol]) of the mech's payment
    asset. Without them, amounts are shown in base units."""
    if tool in ("safe_transaction", "send_transaction"):
        who = "safe" if tool == "safe_transaction" else "agent EOA"
        data = str(args.get("data", "0x"))
        sel = data[:10] if len(data) >= 10 else "(plain transfer)"
        try:
            wei = _int(args.get("value"))
        except ValueError:
            wei = None
        value = fmt_amount(wei, 18, *native) if native and wei is not None else f"{args.get('value', 0)} wei"
        out = (
            f"{who} call on {args.get('chain', 'default chain')}: to {args.get('target') or args.get('to')} "
            f"value {value}, selector {sel}, {max(0, (len(data) - 2) // 2)} bytes of calldata"
        )
        erc20 = ERC20.get(sel.lower())
        if erc20 and len(data) >= 138:  # say who gets what: a user can't approve what they can't read
            party = "0x" + data[34:74].lower()
            role = "recipient" if erc20.endswith("transfer") else "spender"
            n = int(data[74:138], 16)
            if erc20.endswith("approve") and n == 2**256 - 1:  # unlimited, whether or not the token could be read
                amount = f"unlimited {token[1]}" if token else "unlimited (2**256-1) of the token at the target"
            elif token:
                amount = fmt_amount(n, token[0], token[1], token[2])
            else:
                amount = f"{n} base units of the token at the target"
            out += f" ({erc20}: {role} {party}, amount {amount}; ERC-20 amounts are not counted in the session caps)"
        return out
    if tool == "mech_request":
        pay = args.get("max_payment", "default")
        if mech and mech[1] is not None:
            with contextlib.suppress(ValueError):
                pay = fmt_amount(_int(pay), mech[0], mech[2] if len(mech) > 2 else "of the payment asset", mech[1])
        return (
            f"pay mech {args.get('priority_mech', '(marketplace picks)')} for tool {args.get('tool')}, "
            f"max_payment {pay}: {str(args.get('prompt', ''))[:120]!r}"
        )
    if tool == "sign_message":
        return f"sign raw digest {args.get('digest')}"
    return f"{tool}({args})"


def read_only_from(tools: list[dict]) -> set | None:
    """The tools Connect flags read-only (MCP `readOnlyHint`); None if it flags none (fall back)."""
    flagged = {t["name"] for t in tools if (t.get("annotations") or {}).get("readOnlyHint")}
    return (flagged - VALUE_TOOLS) or None

"""Property test: cap exhaustion and idempotency against olas-ocb-agent's real Policy (the formal-checking plan's experiment 2.2).

Random sequences of value calls (native transfers and mech requests, on chains with native and stablecoin mechs),
with retries that reuse a request id with the same or changed arguments, in auto and ask mode. After every call:
- the counters stay within the caps (invariant 1);
- a retry of any approved (id, args) changes no counter and asks nobody (invariant 2);
- a retry with changed args goes through every check again (it is logged as a new check and counted if allowed);
- the counters never decrease: amounts are drawn as ints, decimal and hex strings, zero, negatives and junk, and a
  negative or non-numeric amount is refused by a named rule, never counted.
"""

from __future__ import annotations

import random
import re

from olas_ocb_agent.policy import Policy, SessionCaps

MECHS = {"0xnative": (18, 1.0), "0xusdc": (6, 1.0), "0xodd": (18, None)}  # (decimals, USD) per mech


def _amount(rng: random.Random, n: int):
    """n in one of the encodings a caller can send, or (sometimes) zero, a negative or junk."""
    r = rng.random()
    if r < 0.08:
        return rng.choice([0, "0", "0x0"])
    if r < 0.18:
        return rng.choice([-n, str(-n), "-" + hex(n), -1])
    if r < 0.24:
        return rng.choice(["abc", "1.5", 1.5, "0x", "--1", [1]])
    return rng.choice([n, str(n), hex(n)])


def _call(rng: random.Random, rid: str | None):
    if rng.random() < 0.5:
        value = _amount(rng, rng.choice([10**14, 10**15, 5 * 10**16, 10**17]))  # xDAI at $1
        return "safe_transaction", {"chain": "gnosis", "target": "0x" + "ab" * 20, "value": value, "request_id": rid}
    mech = rng.choice(list(MECHS))
    pay = _amount(
        rng, rng.choice([10**4, 10**5, 5 * 10**5]) if mech == "0xusdc" else rng.choice([10**15, 10**16, 10**17])
    )
    return "mech_request", {
        "chain": "gnosis",
        "prompt": "q",
        "tool": "t",
        "max_payment": pay,
        "priority_mech": mech,
        "request_id": rid,
    }


def run(seed: int, n_calls: int = 40) -> dict:
    rng = random.Random(seed)
    mode = rng.choice(["auto", "ask"])
    asked = []

    def approver(summary):
        asked.append(summary)
        return rng.random() < 0.8

    p = Policy(mode=mode, caps=SessionCaps(), approver=approver, mech_asset_lookup=lambda c, m: MECHS.get(m))
    history: list[tuple[str, dict, bool]] = []
    out = {"violations": [], "replays": 0, "changed_retries": 0, "allowed": 0, "refused_amounts": 0}
    for _ in range(n_calls):
        low = (p.spent_native, p.spent_mech, p.value_calls)
        r = rng.random()
        if history and r < 0.25:  # replay an earlier call exactly
            tool, args, was_allowed = rng.choice(history)
            before = (p.spent_native, p.spent_mech, p.value_calls, len(asked))
            d = p.check(tool, dict(args))
            if was_allowed and args.get("request_id"):
                out["replays"] += 1
                if not d.allowed or (p.spent_native, p.spent_mech, p.value_calls, len(asked)) != before:
                    out["violations"].append("an approved retry changed counters or asked again")
        elif history and r < 0.4:  # same id, changed args
            tool, args, _ = rng.choice(history)
            if not args.get("request_id"):
                continue
            new_tool, new_args = _call(rng, args["request_id"])
            if new_args == args:
                continue
            out["changed_retries"] += 1
            n_log = len(p.log)
            d = p.check(new_tool, new_args)
            if len(p.log) != n_log + 1:
                out["violations"].append("a changed-args retry was not checked")
            history.append((new_tool, new_args, d.allowed))
        else:
            rid = f"r{rng.randrange(1000)}" if rng.random() < 0.7 else None
            tool, args = _call(rng, rid)
            d = p.check(tool, args)
            out["allowed"] += d.allowed
            history.append((tool, args, d.allowed))
        c = p.caps
        if p.spent_native < low[0] or p.spent_mech < low[1] or p.value_calls < low[2]:
            out["violations"].append(f"a counter decreased: {low} -> {p.spent_native, p.spent_mech, p.value_calls}")
        last = p.log[-1] if p.log else None
        if last and not last["allowed"] and "amount" in last["rule"]:
            out["refused_amounts"] += 1
            amount = last["args"].get("value", last["args"].get("max_payment"))
            if not _bad(amount):
                out["violations"].append(f"a valid amount {amount!r} was refused as {last['rule']}")
        if p.spent_native > c.max_native_usd + 1e-9 or p.spent_mech > c.max_mech_usd + 1e-9:
            out["violations"].append(f"cap exceeded: native {p.spent_native}, mech {p.spent_mech}")
        if p.value_calls > c.max_value_calls:
            out["violations"].append("value-call cap exceeded")
    out["hit_cap"] = any("cap" in e["rule"] for e in p.log if not e["allowed"])
    return out


def _bad(v) -> bool:
    """The oracle, written apart from the policy's parser: a valid amount is a non-negative int or a string of
    decimal digits or 0x and hex digits (every draw here is far below 2**256)."""
    if isinstance(v, int):
        return v < 0
    return not (isinstance(v, str) and re.fullmatch(r"[0-9]+|0[xX][0-9a-fA-F]+", v.strip()))


def test_caps_and_retries_hold_over_random_sequences():
    rs = [run(seed) for seed in range(2000)]
    assert [v for r in rs for v in r["violations"]] == []
    assert sum(r["replays"] for r in rs) > 1000 and sum(r["changed_retries"] for r in rs) > 1000  # both exercised
    assert sum(r["hit_cap"] for r in rs) > 1000  # the caps are reached, not just approached
    assert sum(r["refused_amounts"] for r in rs) > 1000  # negative and junk amounts were drawn and refused


def test_negative_zero_hex_and_junk_amounts():
    mechs = {"0xm": (18, 1.0)}
    p = Policy(mode="auto", mech_asset_lookup=lambda c, m: mechs.get(m))
    tx = {"chain": "gnosis", "target": "0x" + "ab" * 20}
    mech = {"chain": "gnosis", "prompt": "q", "tool": "t", "priority_mech": "0xm"}
    for v in (-(10**20), "-100000000000000000000", "-0x56bc75e2d63100000", -1):
        d = p.check("safe_transaction", {**tx, "value": v})
        assert not d.allowed and "'negative-amount'" in d.rule
        d = p.check("mech_request", {**mech, "max_payment": v})
        assert not d.allowed and "'negative-amount'" in d.rule
    for v in ("abc", 1.5, "0x"):
        d = p.check("safe_transaction", {**tx, "value": v})
        assert not d.allowed and "'bad-amount'" in d.rule
    assert (p.spent_native, p.spent_mech, p.value_calls) == (0, 0, 0)
    assert p.check("safe_transaction", {**tx, "value": 0}).allowed and p.spent_native == 0  # zero is a plain call
    assert p.check("safe_transaction", {**tx, "value": None}).allowed and p.spent_native == 0
    assert p.check("safe_transaction", {**tx, "value": "0x2386f26fc10000"}).allowed  # 0.01 xDAI in hex
    assert abs(p.spent_native - 0.01) < 1e-12
    assert not p.check("safe_transaction", {**tx, "value": 10**17}).allowed  # still capped at $0.10


def test_replaying_any_approved_args_set_is_free():
    p = Policy(mode="auto")
    a = {"chain": "gnosis", "target": "0x" + "ab" * 20, "value": 10**15, "request_id": "r1"}
    b = {**a, "value": 2 * 10**15}
    p.check("safe_transaction", a)
    p.check("safe_transaction", b)
    calls = p.value_calls
    assert p.check("safe_transaction", a).allowed and p.value_calls == calls  # both sets were approved: free

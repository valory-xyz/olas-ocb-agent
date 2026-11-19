"""FacilitatorModel: signing matches the agents' golden vectors; the call flow works against the mock."""

import pytest
from mock_connect.facilitator import RATE, MockFacilitator
from mock_connect.server import PRED_MECH, MockConnect, MockState

from olas_ocb_agent.connect_client import ConnectClient
from olas_ocb_agent.interim.facilitator import (
    FacilitatorModel,
    canonical_request_data,
    compute_domain_separator,
    compute_safe_message_hash,
    derive_request_id,
    to_gemini,
)
from olas_ocb_agent.model import ModelError

# Golden vectors from valory/connections/x402/tests/test_mech.py (pinned against mech_interact and the facilitator)
_MP, _MECH, _REQ = "0x" + "11" * 20, "0x" + "22" * 20, "0x" + "33" * 20
_PT = bytes.fromhex("ba699a34be8fe0e7725e93dcbce1701b0211a8ca61330aaeb8a05bf2ec7abed1")


def test_golden_vectors():
    assert (
        compute_domain_separator(100, _MP).hex() == "df50cdbe42bf9d9976fcb9374107c6cf0450b566360eaa04dacb3bb0c1cc8845"
    )
    rid = derive_request_id(
        marketplace_address=_MP,
        mech_address=_MECH,
        requester=_REQ,
        data=b'{"prompt":"hi","tool":"t","nonce":"n"}',
        delivery_rate=10**16,
        payment_type=_PT,
        nonce=42,
        chain_id=100,
    )
    assert rid.hex() == "76a226b6cf8c7da71bd6340d00e54002d9b01d1e8db5670d4bd3282863633fb0"
    assert (
        compute_safe_message_hash(rid, "0x" + "44" * 20, 100).hex()
        == "1f045c3aa1ba9e74c2ff21d51cc0be889abe3e7f8d1e1ffaaafa16ab635d9855"
    )
    assert canonical_request_data(
        method="post",
        path="/models/gemini-2.5-flash:generateContent",
        query={"b": "2", "a": "1"},
        body_b64="eyJwcm9tcHQiOiJoaSJ9",
        expires_at=1_700_000_060,
    ) == (
        b'{"body_b64":"eyJwcm9tcHQiOiJoaSJ9","expires_at":1700000060,"method":"POST",'
        b'"path":"/models/gemini-2.5-flash:generateContent","query":{"a":"1","b":"2"}}'
    )


def test_gemini_body_shape():
    body = to_gemini(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"},
        ]
    )
    assert body["systemInstruction"] == {"parts": [{"text": "S"}]}
    assert [c["role"] for c in body["contents"]] == ["user", "model"]
    assert len(body["contents"][0]["parts"]) == 2


@pytest.fixture
def chain():
    m = MockConnect(MockState())
    f = MockFacilitator(m.s, backend=lambda req: "echo:" + req["contents"][-1]["parts"][-1]["text"])
    yield m, f
    f.stop()
    m.stop()


def model(m, f, **kw):
    return FacilitatorModel(
        ConnectClient(m.url, m.token), chain="gnosis", model="gemini-2.5-flash", base_url=f.url, **kw
    )


def test_deposit_then_call_then_replay_after_lost_response(chain):
    m, f = chain
    fm = model(m, f, max_deposit=3 * RATE)
    out = fm.query([{"role": "user", "content": "hi"}])
    assert out["content"] == "echo:hi"
    assert fm.deposited == 3 * RATE and f.charged == RATE  # one deposit of the whole allowance
    assert next(e for e in m.s.ledger if e["kind"] == "tx")["data"].startswith("0xaa67c919")  # depositFor(safe)
    f.drop_first_response = True
    out = fm.query([{"role": "user", "content": "again"}])  # charged, 504, replayed: one charge
    assert out["content"] == "echo:again" and f.charged == 2 * RATE and len(f.calls) == 2


def test_no_deposit_allowance_fails_clearly(chain):
    m, f = chain
    with pytest.raises(ModelError, match="pre-deposit too low"):
        model(m, f).query([{"role": "user", "content": "hi"}])
    assert not [e for e in m.s.ledger if e["kind"] == "tx"]


def test_budget_caps_calls(chain):
    m, f = chain
    fm = model(m, f, max_deposit=10 * RATE, budget=2 * RATE)
    fm.query([{"role": "user", "content": "1"}])
    fm.query([{"role": "user", "content": "2"}])
    with pytest.raises(ModelError, match="budget"):
        fm.query([{"role": "user", "content": "3"}])


@pytest.mark.parametrize("gated", [False, True])
def test_shared_nonce_collision_and_gate(gated, connect_ws, mock):
    """Live finding 2026-11-08: a Connect mech_request signed while the facilitator still held a model
    call's slot made the facilitator's settlement fail and pause the safe. The gate prevents it."""
    from olas_ocb_agent.runtime import build

    mock.s.connect_settle_delay = 0.3
    f = MockFacilitator(mock.s, backend=lambda req: "ok", settle_delay=3.0)  # longer than a cold `pearl` start
    from mock_connect.facilitator import TRACKER
    from mock_connect.server import SAFE

    mock.s.ledger.append(
        {
            "kind": "tx",
            "to": TRACKER,
            "value": 10 * RATE,  # a prior depositFor(safe)
            "data": "0xaa67c919" + SAFE[2:].rjust(64, "0"),
        }
    )
    fm = FacilitatorModel(ConnectClient(mock.url, mock.token), chain="gnosis", model="gemini-2.5-flash", base_url=f.url)
    rt = build(connect_ws, model=fm, approval="auto", sandbox="none")
    try:
        if gated:
            assert fm.gate is rt.broker.gate is not None
            fm.gate.poll = 0.1
        else:
            fm.gate = rt.broker.gate = None
        fm.query([{"role": "user", "content": "model call 1"}])  # row held at nonce 0
        r = rt.shell.run(
            "pearl mech_request prompt=q tool=prediction-online max_payment=20000000000000000 "
            f"priority_mech={PRED_MECH}"
        )
        assert r["returncode"] == 0, r["output"]

        def settled(cond, limit=10.0):  # poll, not a fixed sleep: settlement runs on a thread (3 s delay)
            import time

            t0 = time.time()
            while not cond() and time.time() - t0 < limit:
                time.sleep(0.05)
            return cond()

        if gated:
            fm.query([{"role": "user", "content": "model call 2"}])
            assert settled(lambda: mock.s.market_nonce == 3)
            assert not f.paused and mock.s.connect_settle_failures == 0
        else:
            assert settled(lambda: f.paused)  # the collision seen live
            with pytest.raises(ModelError, match="settlement_paused"):
                fm.query([{"role": "user", "content": "model call 2"}])
    finally:
        rt.close()
        f.stop()


def test_gate_pending_state_has_a_deadline():
    """An agent mech request that never settles must not block model calls forever."""
    from olas_ocb_agent.interim.marketplace_gate import MarketplaceGate

    class F:  # the facilitator's view: the on-chain nonce never moves
        def info(self):
            class Info:
                on_chain_nonce, next_nonce = 5, 5

            return Info()

    g = MarketplaceGate(F(), settle_timeout=0.2, poll=0.05)
    g.pending_floor = 6
    with pytest.raises(ModelError, match="timed out"):
        g.before_model_call()  # first failed wait: the model call fails
    g.before_model_call()  # second failed wait: the floor is dropped, the call proceeds
    g.after_model_call()
    assert g.pending_floor is None

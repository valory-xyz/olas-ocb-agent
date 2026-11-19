"""FacilitatorModel error and retry paths, with a faked transport; plus `pearl` and broker leftovers."""

import io
import json
import urllib.error
import urllib.request

import pytest

from olas_ocb_agent import pearl_cli
from olas_ocb_agent.interim.facilitator import (
    FacilitatorModel,
    RequesterInfo,
    compute_safe_message_hash,
    derive_request_id,
    from_gemini,
)
from olas_ocb_agent.model import ModelError

OK = {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}], "usageMetadata": {}}


def _info(**kw):
    base = dict(
        chain_id=100,
        on_chain_nonce=0,
        marketplace="0x" + "11" * 20,
        mech="0x" + "22" * 20,
        payment_type=b"\x00" * 32,
        rate=10,
        next_nonce=0,
        available=100,
        max_ttl=300,
        server_time=None,
    )
    return RequesterInfo(**{**base, **kw})


class _Client:
    def __init__(self, wallet=None, sig=None):
        self.wallet, self.calls = wallet, []
        self.sig = {"signature": "0x" + "ab" * 65} if sig is None else sig

    def rest(self, method, path, payload=None):
        return self.wallet

    def call_tool(self, name, args):
        self.calls.append((name, args))
        return self.sig if name == "sign_message" else {"tx_hash": "0xtx"}


def _model(responses, **kw):
    fm = FacilitatorModel(
        _Client({"chains": {"gnosis": {"safe": "0x" + "33" * 20, "rpc": "http://rpc"}}}), chain="gnosis", **kw
    )
    fm.info = _info
    fm._wallet()  # info() normally loads the safe first
    seq = list(responses)

    def http(method, url, body=None, timeout=200):
        r = seq.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    fm._http = http
    return fm


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("olas_ocb_agent.interim.facilitator.time.sleep", lambda s: None)


def test_primitive_input_checks_and_empty_gemini():
    with pytest.raises(ValueError):
        derive_request_id(
            marketplace_address="0x" + "1" * 40,
            mech_address="0x" + "2" * 40,
            requester="0x" + "3" * 40,
            data=b"",
            delivery_rate=1,
            payment_type=b"\x00",
            nonce=0,
            chain_id=1,
        )
    with pytest.raises(ValueError):
        compute_safe_message_hash(b"\x01", "0x" + "4" * 40, 1)
    with pytest.raises(ModelError, match="no candidates"):
        from_gemini({})


def test_query_retries_transport_conflict_and_replay():
    seen = []
    fm = _model(
        [
            OSError("reset"),
            (409, {"Retry-After": "0"}, {"detail": {"error": "nonce_mismatch"}}),
            (504, {"Retry-After": "0"}, {"detail": "gateway"}),
            (200, {}, OK),
        ]
    )
    assert fm.query([{"role": "user", "content": "x"}], on_token=seen.append)["content"] == "ok"
    assert seen == ["ok"] and fm.spent == 10


def test_query_failures():
    with pytest.raises(ModelError, match="Gemini error"):
        _model([(400, {}, {"error": {"code": 400, "message": "bad"}})]).query([])
    with pytest.raises(ModelError, match="facilitator refused"):
        _model([(500, {}, {"detail": {"error": "boom"}})]).query([])
    with pytest.raises(ModelError, match="did not settle"):
        _model([(502, {}, {"detail": "x"})] * 12).query([])
    with pytest.raises(ModelError, match="pre-deposit too low"):
        _model([(402, {}, {"detail": {"error": "insufficient", "context": {"required": 10}}})]).query([])
    with pytest.raises(ModelError, match="budget exhausted"):
        _model([], budget=5).query([])


def test_wallet_info_sign_and_deposit_errors(monkeypatch):
    with pytest.raises(ModelError, match="no safe"):
        FacilitatorModel(_Client({"chains": {}}), chain="gnosis")._wallet()
    fm = _model([])
    fm._safe = "0x" + "33" * 20
    fm.client = _Client(sig={"nope": 1})
    with pytest.raises(ModelError, match="unexpected sign_message"):
        fm._sign(b"\x00" * 32)
    fm._rpc = None
    with pytest.raises(ModelError, match="no RPC"):
        fm._eth_call("0x1", "0x")
    fm._rpc = "http://rpc"
    fm._http = lambda *a, **k: (200, {}, {"error": "x"})
    with pytest.raises(ModelError, match="eth_call failed"):
        fm._eth_call("0x1", "0x")
    fm._http = lambda *a, **k: (200, {}, {"result": "0x" + "0" * 64})
    with pytest.raises(ModelError, match="no balance tracker"):
        fm.balance_tracker(_info())
    with pytest.raises(ModelError, match="deposit allowance"):
        fm.deposit(1, _info())


def test_requester_info_errors_and_http_error_body(monkeypatch):
    fm = FacilitatorModel(_Client({"chains": {"gnosis": {"safe": "0x" + "33" * 20}}}), chain="gnosis")
    fm._http = lambda *a, **k: (404, {}, {"detail": "unknown safe"})
    with pytest.raises(ModelError, match="requester info failed"):
        fm.info()
    d = {
        "chain_id": 100,
        "marketplace_address": "0x1",
        "mech_address": "0x2",
        "payment_type": "0x" + "00" * 32,
        "delivery_rates": {},
        "next_nonce": 0,
        "balance": 0,
        "available": 0,
        "max_ttl_secs": 300,
    }
    fm._http = lambda *a, **k: (200, {"Date": "not a date"}, d)
    with pytest.raises(ModelError, match="no rate for 'chat'"):
        fm.info()

    def boom(*a, **k):
        raise urllib.error.HTTPError("u", 502, "x", {}, io.BytesIO(b"<html>bad gateway"))

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    code, _, body = FacilitatorModel(_Client(), chain="gnosis")._http("GET", "http://x")
    assert code == 502 and "bad gateway" in body["detail"]


# ---- `pearl` and broker leftovers ---------------------------------------------------------


def test_pearl_cli_json_args_hints_and_unknown_tool(connect_ws, mock, monkeypatch, capsys):
    monkeypatch.chdir(connect_ws)
    assert pearl_cli.main(["mech_tools", json.dumps({"chain": "gnosis"})]) == 0
    assert pearl_cli.main(["mech_request", "prompt=q", "tool=t", "mech=0x1"]) == 1
    assert "did you mean priority_mech" in capsys.readouterr().err
    assert pearl_cli.main(["mech_request", "prompt=q", "zzz=1", "tool=t"]) == 1  # unknown key with no hint
    assert pearl_cli.main(["safe_transaction", "value=1"]) == 1
    assert "needs: chain, target" in capsys.readouterr().err
    assert pearl_cli.main(["no_such_tool"]) == 2


def test_broker_get_requires_token(connect_ws, mock):
    from olas_ocb_agent.model import ScriptedModel
    from olas_ocb_agent.runtime import build

    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(rt.broker.url.removesuffix("/mcp/") + "/wallet", timeout=5)
        assert e.value.code == 401
    finally:
        rt.close()

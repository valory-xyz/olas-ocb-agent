"""Multi-chain: chain data, asset resolution and refusals on unknown chains (the suite on every chain is in
OnchainBench)."""

import pytest
from mock_connect.server import DEAR_MECH, PRED_MECH, USDC, MockConnect, MockState, chain_profile

from olas_ocb_agent import cli
from olas_ocb_agent.chains import OLAS_EVM_CHAINS, native_usd, parse_prices
from olas_ocb_agent.connect_client import ConnectClient
from olas_ocb_agent.interim.facilitator import make as make_facilitator
from olas_ocb_agent.interim.facilitator_assets import facilitator_asset
from olas_ocb_agent.model import ScriptedModel
from olas_ocb_agent.policy import Policy
from olas_ocb_agent.runtime import _decode_symbol, build, mech_asset_lookup

# ---- chain data ------------------------------------------------------------------


def test_chain_data_and_overrides():
    assert set(OLAS_EVM_CHAINS) == {
        "ethereum",
        "gnosis",
        "polygon",
        "arbitrum",
        "optimism",
        "base",
        "celo",
        "mode",
        "robinhood",
    }
    assert native_usd("gnosis") == 1.0 and native_usd("nope") is None
    assert native_usd("nope", {"nope": 2.5}) == 2.5 and native_usd("base", {"base": 1.0}) == 1.0
    assert facilitator_asset("gnosis") == (18, 1.0) and facilitator_asset("base") == (6, 1.0)
    assert facilitator_asset("ethereum") is None and facilitator_asset("nope") is None
    assert parse_prices(["a=1.5", " b = 2"]) == {"a": 1.5, "b": 2.0} and parse_prices(None) == {}
    with pytest.raises(ValueError):
        parse_prices(["no-equals"])


def test_any_evm_chain_with_a_user_price():
    tx = {"target": "0x" + "ab" * 20, "value": 10**16, "chain": "mychain"}
    assert "--native-usd mychain=" in Policy(mode="auto").check("safe_transaction", tx).rule
    assert Policy(mode="auto", native_prices={"mychain": 1.0}).check("safe_transaction", tx).allowed


def test_mech_payment_asset_resolution_in_the_broker():
    req = {"prompt": "q", "tool": "t", "max_payment": 10**6, "chain": "ethereum"}
    assert "name the mech" in Policy(mode="auto").check("mech_request", req).rule  # always named
    assert "isn't known" in Policy(mode="auto").check("mech_request", {**req, "priority_mech": "0x1"}).rule
    p = Policy(mode="auto", mech_asset_lookup=lambda chain, mech: (6, 1.0))
    assert p.check("mech_request", {**req, "priority_mech": "0x1"}).allowed and p.spent_mech == pytest.approx(1.0)


# ---- runtime's token lookup ----------------------------------------------------------


def test_mech_asset_lookup_native_token_and_failures(monkeypatch):
    m = MockConnect(MockState(chain="base"))  # base: mechs paid in the mock USDC
    try:
        c = ConnectClient(m.url, m.token)
        chains = c.rest("GET", "/wallet")["chains"]
        look = mech_asset_lookup(c, chains, {})
        assert look("base", PRED_MECH) == (6, 1.0, "USDC")  # USDC: decimals and symbol read via the mock RPC
        assert look("base", PRED_MECH.upper().replace("X", "x")) == (6, 1.0, "USDC")  # cached, case-insensitive
        assert look("base", "0x" + "99" * 20) is None  # unknown mech: Connect refuses
        assert mech_asset_lookup(c, {}, {})("base", PRED_MECH) is None  # no RPC for the chain
        bad = {"base": {"rpc": m.url.removesuffix("/mcp/") + "/rpc"}}
        m.s.mechs[PRED_MECH]["price"] = 1
        monkeypatch.setattr(m, "rpc", lambda req: {"jsonrpc": "2.0", "id": 1, "error": {}})
        assert mech_asset_lookup(c, bad, {})("base", PRED_MECH) is None  # RPC can't answer
    finally:
        m.stop()
    g = MockConnect(MockState())  # gnosis: native xDAI
    try:
        c = ConnectClient(g.url, g.token)
        assert mech_asset_lookup(c, {}, {})("gnosis", PRED_MECH) == (18, 1.0, "xDAI")
    finally:
        g.stop()
    assert _decode_symbol("0x" + b"WXDAI".hex().ljust(64, "0")) == "WXDAI"  # bytes32-style symbol


def test_model_budget_needs_a_known_asset_and_facilitator_chain(connect_ws, mock):
    with pytest.raises(ValueError, match="isn't known"):
        build(connect_ws, model="facilitator:gemini-2.5-flash@ethereum", approval="auto", sandbox="none")
    with pytest.raises(ValueError, match="no chain for the facilitator"):
        make_facilitator(ConnectClient(mock.url, mock.token), "gemini-2.5-flash", default_chain=None)
    assert make_facilitator(ConnectClient(mock.url, mock.token), "m", default_chain="base").chain == "base"
    rt = build(connect_ws, model="mech:chat@" + PRED_MECH, approval="auto", sandbox="none", model_budget=0.05)
    try:
        assert rt.agent.model.budget == 5 * 10**16  # the mech reports native on gnosis
    finally:
        rt.close()
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none", native_prices={"x": 3.0})
    assert rt.policy.native_prices == {"x": 3.0}
    rt.close()


def test_cli_accepts_native_prices(connect_ws, mock, monkeypatch):
    seen = {}
    real = cli.build

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **{**k, "model": ScriptedModel(["DONE: ok"])})

    monkeypatch.setattr(cli, "build", spy)
    assert cli.main(["--dir", str(connect_ws), "--native-usd", "mychain=2", "--sandbox", "none", "-p", "hi"]) == 0
    assert seen["native_prices"] == {"mychain": 2.0}


# ---- the suite on every chain in chains.py -----------------------------------------------


def test_mock_chain_profiles_price_a_unit_near_a_cent():
    for chain in OLAS_EVM_CHAINS:
        p = chain_profile(chain)
        assert 0 < p.unit / 1e18 * OLAS_EVM_CHAINS[chain].native_usd < 0.02


def test_mock_mech_pricing_follows_the_chain():
    s = MockState(chain="polygon")
    assert s.mechs[DEAR_MECH]["price"] == 500_000 and s.mech_balance() is s.usdc and s.profile.mech_sym == "USDC"
    assert MockState().mechs[DEAR_MECH]["price"] == 5 * 10**17
    m = MockConnect(MockState(chain="polygon"))
    try:
        assert m.call("mech_tools", {"priority_mech": DEAR_MECH}, via="mcp")["payment_token"] == USDC
        assert "error" in m.rpc({"id": 1, "method": "eth_call", "params": [{"to": "0x1", "data": "0x"}]})
    finally:
        m.stop()

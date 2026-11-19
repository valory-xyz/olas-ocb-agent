"""Model calls through the Olas mech facilitator: Gemini, paid per call from the service safe.

Interim: goes away once chat models are served as full marketplace mechs that Connect pays for;
`MechModel` replaces it.

This is how Pearl's own agents (trader, Optimus, Basius) get an LLM without
an API key (USE_MECH_FACILITATOR). Each call is a mech-marketplace request
signed for the safe:

  1. GET  {base}/mech/{chain}/requester/{safe}   -> nonce, rate, mech, payment type, pre-deposit
  2. request_id = MechMarketplace.getRequestId(canonical bytes of the upstream call, rate, nonce, ...)
  3. the safe's owner (the agent EOA) signs the SafeMessage digest of request_id:
     Connect's `sign_message` (raw digest, unprefixed) does exactly that, so no key is ever here
  4. POST {base}/mech/chat/{chain} {request_id, safe_address, signature, nonce, expires_at, upstream}
     -> the Gemini response, passed through; the facilitator settles on-chain from the pre-deposit

The signing primitives mirror valory/connections/x402/mech_signing.py byte for byte (the tests pin
the same golden vectors). Retry semantics follow its mech.py client: 409 requester_busy and
nonce_mismatch re-read the nonce; 502/504 replay the same signed body (same request_id, so
never charged twice); 402 means the pre-deposit is too low.
"""

from __future__ import annotations

import base64
import contextlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

from eth_abi import encode as abi_encode
from eth_utils import keccak

from .. import __version__
from ..chains import fmt_amount
from ..connect_client import ConnectClient
from ..model import ModelError
from .facilitator_assets import facilitator_asset, facilitator_symbol

FACILITATOR = "https://x402.olas.autonolas.tech"
DEFAULT_MODEL = "gemini-2.5-pro"  # the CLI's default model
GEMINI_PREFIX = "/v1beta"

# ---- signing primitives (mirror of mech_signing.py) ---------------------------------
_MARKETPLACE_NAME, _MARKETPLACE_VERSION = "MechMarketplace", "1.1.0"
_DOMAIN_TYPEHASH = keccak(text="EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)")
_SAFE_DOMAIN_TYPEHASH = keccak(text="EIP712Domain(uint256 chainId,address verifyingContract)")
_SAFE_MESSAGE_TYPEHASH = keccak(text="SafeMessage(bytes message)")
SEL_DEPOSIT_FOR = "aa67c919"  # depositFor(address)
SEL_MAP_TRACKERS = "4eb07dd3"  # mapPaymentTypeBalanceTrackers(bytes32)


def canonical_request_data(*, method: str, path: str, query: dict, body_b64: str, expires_at: int) -> bytes:
    payload = {
        "method": method.upper(),
        "path": path,
        "query": dict(sorted(query.items())),
        "body_b64": body_b64,
        "expires_at": int(expires_at),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_domain_separator(chain_id: int, marketplace: str) -> bytes:
    return keccak(
        abi_encode(
            ["bytes32", "bytes32", "bytes32", "uint256", "address"],
            [
                _DOMAIN_TYPEHASH,
                keccak(text=_MARKETPLACE_NAME),
                keccak(abi_encode(["string"], [_MARKETPLACE_VERSION])),
                chain_id,
                marketplace,
            ],
        )
    )


def derive_request_id(
    *,
    marketplace_address,
    mech_address,
    requester,
    data: bytes,
    delivery_rate: int,
    payment_type: bytes,
    nonce: int,
    chain_id: int,
) -> bytes:
    if len(payment_type) != 32:
        raise ValueError("payment_type must be 32 bytes")
    inner = keccak(
        abi_encode(
            ["address", "address", "address", "bytes32", "uint256", "bytes32", "uint256"],
            [marketplace_address, mech_address, requester, keccak(data), delivery_rate, payment_type, nonce],
        )
    )
    return keccak(b"\x19\x01" + compute_domain_separator(chain_id, marketplace_address) + inner)


def compute_safe_message_hash(request_id: bytes, safe: str, chain_id: int) -> bytes:
    if len(request_id) != 32:
        raise ValueError("request_id must be 32 bytes")
    domain = keccak(abi_encode(["bytes32", "uint256", "address"], [_SAFE_DOMAIN_TYPEHASH, chain_id, safe]))
    struct = keccak(
        abi_encode(["bytes32", "bytes32"], [_SAFE_MESSAGE_TYPEHASH, keccak(abi_encode(["bytes32"], [request_id]))])
    )
    return keccak(b"\x19\x01" + domain + struct)


# ---- Gemini request/response shapes -------------------------------------------------
def to_gemini(messages: list[dict]) -> dict:
    from ..agent import _flatten_tools

    messages = _flatten_tools(messages)
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = "model" if m["role"] == "assistant" else "user"
        if contents and contents[-1]["role"] == role:  # Gemini wants alternating turns
            contents[-1]["parts"].append({"text": m["content"]})
        else:
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
    body = {"contents": contents}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return body


def from_gemini(resp: dict) -> tuple[str, dict]:
    cands = resp.get("candidates") or []
    if not cands:
        raise ModelError(f"Gemini returned no candidates: {json.dumps(resp)[:300]}")
    parts = (cands[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    usage = resp.get("usageMetadata") or {}
    return text, {
        "prompt_tokens": usage.get("promptTokenCount"),
        "completion_tokens": usage.get("candidatesTokenCount"),
        "finish_reason": cands[0].get("finishReason"),
    }


@dataclass
class RequesterInfo:
    chain_id: int
    on_chain_nonce: int
    marketplace: str
    mech: str
    payment_type: bytes
    rate: int
    next_nonce: int
    available: int
    max_ttl: int
    server_time: int | None


class FacilitatorModel:
    """olas-ocb-agent model backend: Gemini via the mech facilitator, signed through Connect.

    budget: most the session's model calls may consume, in base units of the chain's payment
      asset (wei of xDAI on gnosis). max_deposit: most the session may move from the safe into
      the mech pre-deposit, one `safe_transaction` to the balance tracker each time the
      pre-deposit can't cover a call. 0 = never deposit; the call fails with the shortfall.
    """

    def __init__(
        self,
        client: ConnectClient,
        chain: str,
        model: str = DEFAULT_MODEL,
        base_url: str = FACILITATOR,
        budget: int | None = None,
        max_deposit: int = 0,
        ttl: int = 240,
        log: Callable[[str], None] | None = None,
    ):
        self.client = client
        self.chain = chain
        self.model = model
        self.base = base_url.rstrip("/")
        self.budget = budget
        self.max_deposit = max_deposit
        self.ttl = ttl
        self.log = log or (lambda s: None)
        self.spent = 0  # paid in delivery rates
        self.deposited = 0
        self.gate = None  # MarketplaceGate, when the agent can also make mech requests from this safe
        self.on_deposit: Callable[[], None] = lambda: None  # saves the session's state (set by the runtime)
        asset = facilitator_asset(chain)
        # (decimals, USD price, symbol) of the payment asset; the runtime replaces it with the user's prices
        self.asset = asset and (*asset, facilitator_symbol(chain))
        self._safe: str | None = None
        self._rpc: str | None = None

    def amount(self, n: int) -> str:
        """Base units of the payment asset as a person reads them: '0.5 xDAI (about $0.50)'."""
        if not self.asset:
            return f"{n} base units of the payment asset"
        return fmt_amount(n, self.asset[0], self.asset[2], self.asset[1])

    # ---- plumbing ------------------------------------------------------------------
    def _http(self, method: str, url: str, body: dict | None = None, timeout: float = 200) -> tuple[int, dict, dict]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "User-Agent": f"olas-ocb-agent/{__version__}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310 - fixed https origin
                return r.status, dict(r.headers), json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                parsed = json.loads(raw or b"{}")
            except ValueError:
                parsed = {"detail": raw.decode(errors="replace")[:500]}
            return e.code, dict(e.headers), parsed

    def _wallet(self) -> None:
        if self._safe:
            return
        w = self.client.rest("GET", "/wallet")
        entry = (w.get("chains") or {}).get(self.chain) or {}
        if not entry.get("safe"):
            raise ModelError(f"Connect has no safe on {self.chain}; configured: {list(w.get('chains') or {})}")
        self._safe, self._rpc = entry["safe"], entry.get("rpc")

    def info(self) -> RequesterInfo:
        self._wallet()
        code, headers, d = self._http("GET", f"{self.base}/mech/{self.chain}/requester/{self._safe}", timeout=30)
        if code != 200:
            raise ModelError(f"facilitator requester info failed ({code}): {d.get('detail')}")
        st = None
        with contextlib.suppress(TypeError, ValueError):
            st = int(parsedate_to_datetime(headers.get("Date") or headers.get("date")).timestamp())
        rate = (d.get("delivery_rates") or {}).get("chat")
        if rate is None:
            raise ModelError("facilitator quotes no rate for 'chat' on this chain")
        return RequesterInfo(
            chain_id=int(d["chain_id"]),
            on_chain_nonce=int(d.get("on_chain_nonce", d["next_nonce"])),
            marketplace=d["marketplace_address"],
            mech=d["mech_address"],
            payment_type=bytes.fromhex(d["payment_type"][2:]),
            rate=int(rate),
            next_nonce=int(d["next_nonce"]),
            available=int(d["available"]),
            max_ttl=int(d["max_ttl_secs"]),
            server_time=st,
        )

    def _sign(self, digest: bytes) -> str:
        res = self.client.call_tool("sign_message", {"digest": "0x" + digest.hex()})
        sig = res.get("signature") if isinstance(res, dict) else res
        if not isinstance(sig, str) or not sig.startswith("0x"):
            raise ModelError(f"unexpected sign_message result: {str(res)[:200]}")
        return sig

    def signed_body(self, upstream: dict, info: RequesterInfo, nonce: int) -> dict:
        now = info.server_time or int(time.time())
        expires_at = now + min(self.ttl, info.max_ttl)
        data = canonical_request_data(
            method=upstream["method"],
            path=upstream["path"],
            query=upstream["query"],
            body_b64=upstream["body_b64"],
            expires_at=expires_at,
        )
        rid = derive_request_id(
            marketplace_address=info.marketplace,
            mech_address=info.mech,
            requester=self._safe,
            data=data,
            delivery_rate=info.rate,
            payment_type=info.payment_type,
            nonce=nonce,
            chain_id=info.chain_id,
        )
        digest = compute_safe_message_hash(rid, self._safe, info.chain_id)
        return {
            "request_id": "0x" + rid.hex(),
            "safe_address": self._safe,
            "signature": self._sign(digest),
            "nonce": str(nonce),
            "expires_at": expires_at,
            "upstream": upstream,
        }

    # ---- pre-deposit -----------------------------------------------------------------
    def _eth_call(self, to: str, data: str) -> str:
        if not self._rpc:
            raise ModelError("no RPC for this chain in Connect's /wallet")
        _, _, d = self._http(
            "POST",
            self._rpc,
            {"jsonrpc": "2.0", "id": 1, "method": "eth_call", "params": [{"to": to, "data": data}, "latest"]},
            timeout=30,
        )
        if "result" not in d:
            raise ModelError(f"eth_call failed: {d}")
        return d["result"]

    def balance_tracker(self, info: RequesterInfo) -> str:
        out = self._eth_call(info.marketplace, "0x" + SEL_MAP_TRACKERS + info.payment_type.hex())
        addr = "0x" + out[-40:]
        if int(addr, 16) == 0:
            raise ModelError("the marketplace maps no balance tracker to the facilitator's payment type")
        return addr

    def deposit(self, amount: int, info: RequesterInfo | None = None) -> str:
        """depositFor(safe) on the tracker, sent by the safe through Connect. Returns the tx hash."""
        info = info or self.info()
        if self.deposited + amount > self.max_deposit:
            raise ModelError(
                f"depositing {amount} would exceed this session's deposit allowance "
                f"({self.deposited} of {self.max_deposit} used)"
            )
        tracker = self.balance_tracker(info)
        data = "0x" + SEL_DEPOSIT_FOR + self._safe[2:].lower().rjust(64, "0")
        rid = f"olas-ocb-agent-deposit-{info.next_nonce}-{amount}-{int(time.time()) // 600}"
        res = self.client.call_tool(
            "safe_transaction",
            {"target": tracker, "value": amount, "data": data, "chain": self.chain, "request_id": rid},
        )
        self.deposited += amount
        self.on_deposit()  # saved now: a kill before the next state write must not deposit again on --resume
        tx = res.get("tx_hash") if isinstance(res, dict) else str(res)
        self.log(
            f"[ocb-agent] moved {self.amount(amount)} from the safe into the mech facilitator's pre-deposit, which "
            f"pays for model calls ({self.amount(self.deposited)} of the session's {self.amount(self.max_deposit)} "
            f"allowance used; tracker {tracker}, tx {tx})"
        )
        return tx

    # ---- the model interface ------------------------------------------------------------
    def query(self, messages: list[dict], on_token: Callable[[str], None] | None = None) -> dict:
        if self.gate is None:
            return self._query(messages, on_token)
        self.gate.before_model_call()
        try:
            return self._query(messages, on_token)
        finally:
            self.gate.after_model_call()

    def _query(self, messages: list[dict], on_token: Callable[[str], None] | None = None) -> dict:
        t0 = time.time()
        body = json.dumps(to_gemini(messages)).encode()
        upstream = {
            "method": "POST",
            "path": f"{GEMINI_PREFIX}/models/{self.model}:generateContent",
            "query": {},
            "body_b64": base64.b64encode(body).decode("ascii"),
        }
        info = self.info()
        if self.budget is not None and self.spent + info.rate > self.budget:
            raise ModelError(f"model budget exhausted ({self.spent} of {self.budget} spent)")
        deposited_once = False
        nonce = info.next_nonce
        signed = None
        for _attempt in range(12):
            if signed is None:
                signed = self.signed_body(upstream, info, nonce)
            try:
                code, headers, resp = self._http("POST", f"{self.base}/mech/chat/{self.chain}", signed)
            except (OSError, TimeoutError) as e:  # lost response: replay the same signed body
                self.log(f"[ocb-agent] facilitator transport error ({e}); replaying the same request")
                time.sleep(2)
                continue
            if 200 <= code < 300 and not (isinstance(resp, dict) and "detail" in resp):
                text, usage = from_gemini(resp)
                self.spent += info.rate
                if on_token:
                    on_token(text)
                usage.update(
                    mech_payment=info.rate, latency_s=round(time.time() - t0, 2), request_id=signed["request_id"]
                )
                return {"content": text, "cost": 0.0, "usage": usage}
            detail = resp.get("detail") if isinstance(resp, dict) else resp
            err = detail.get("error") if isinstance(detail, dict) else None
            ctx = (detail.get("context") or {}) if isinstance(detail, dict) else {}
            if code in (502, 504) or err == "request_in_progress":
                time.sleep(max(1.0, min(30.0, float(headers.get("Retry-After", 2) or 2))))
                continue  # same body, same request_id: the facilitator never charges it twice
            if code == 402:
                room = self.max_deposit - self.deposited
                if room > 0 and not deposited_once:
                    self.deposit(room, info)  # the rest of the session's allowance, in one transaction
                    deposited_once = True
                    for _ in range(20):  # wait until the facilitator sees the deposit
                        time.sleep(3)
                        info = self.info()
                        if info.available >= info.rate:
                            break
                    nonce, signed = info.next_nonce, None
                    continue
                raise ModelError(self._low_deposit(ctx, info))
            if code == 409 and err in ("requester_busy", "nonce_mismatch"):
                time.sleep(max(1.0, min(30.0, float(headers.get("Retry-After", 2) or 2))))
                info = self.info()
                nonce, signed = info.next_nonce, None
                continue
            if code >= 400 and not err and isinstance(resp, dict) and "error" in resp:  # Gemini error passed through
                raise ModelError(f"Gemini error ({code}): {json.dumps(resp['error'])[:400]}")
            raise ModelError(f"facilitator refused ({code}): {json.dumps(detail)[:400]}")
        raise ModelError("facilitator call did not settle after retries")

    def _low_deposit(self, ctx: dict, info: RequesterInfo) -> str:
        """The 402 in words, with the command to run: what the balance is, and how to top it up or avoid it."""
        need = int(ctx.get("required") or info.rate)
        suggest = self.deposited + 50 * need  # the allowance is per session: what's used, plus about 50 calls
        have = int(ctx.get("available") or 0)
        if not self.max_deposit:
            used = "this session may not top it up (--facilitator-deposit is 0, the default)"
        else:
            used = f"this session's deposit allowance ({self.amount(self.max_deposit)}) is used up"
        return (
            f"mech pre-deposit too low: the model ({self.model} through the mech facilitator) is paid from a prepaid "
            f"balance on {self.chain}, and it is too low: {self.amount(have)} available, {self.amount(need)} needed "
            "per call; "
            f"{used}. Run again with --facilitator-deposit {suggest} (in base units: {self.amount(suggest)} in all, "
            "about 50 more calls, moved from the safe when needed), or use --model litellm:<model> with your own "
            "API key"
        )


def make(client: ConnectClient, spec: str, default_chain: str | None = None, **kw) -> FacilitatorModel:
    """spec: 'gemini-2.5-pro' or 'gemini-2.5-pro@base' (model@chain); without @chain, the safe's actionable chain."""
    model, _, chain = spec.partition("@")
    chain = chain or default_chain
    if not chain:
        raise ValueError(
            "no chain for the facilitator (the safe has no actionable chain): use facilitator:<model>@<chain>"
        )
    return FacilitatorModel(client, chain=chain, model=model or DEFAULT_MODEL, **kw)

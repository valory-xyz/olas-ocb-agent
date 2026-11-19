"""A mock of the Olas mech facilitator (x402.olas.autonolas.tech): requester info, /mech/chat, eth_call.

It checks what the real one checks that a client can get wrong: the request_id recomputed from
the canonical bytes, the Safe-message signature (made by the mock connect server's sign_message),
the nonce, the expiry and the pre-deposit. Deposits are the mock chain's depositFor transactions
to the tracker. `drop_first_response` charges a call and then answers 504, like a lost response.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mock_connect.server import SAFE, MockState
from olas_ocb_agent.interim.facilitator import (
    SEL_DEPOSIT_FOR,
    SEL_MAP_TRACKERS,
    canonical_request_data,
    compute_safe_message_hash,
    derive_request_id,
)

MARKETPLACE = "0x" + "4d" * 20
MECH = "0x" + "0e" * 20
TRACKER = "0x" + "7c" * 20
PAYMENT_TYPE = "ba699a34be8fe0e7725e93dcbce1701b0211a8ca61330aaeb8a05bf2ec7abed1"
RATE = 10**15
CHAIN_ID = 100


def mock_signature(digest_hex: str) -> str:  # what the mock connect server's sign_message returns
    return "0x" + hashlib.sha256(str(digest_hex).encode()).hexdigest() * 2 + "1b"


class MockFacilitator:
    def __init__(self, chain_state: MockState, backend=None, settle_delay: float = 0.0):
        self.cs = chain_state
        self.settle_delay = settle_delay  # >0: rows are held, then settled in nonce order in the background
        self.held: list = []  # nonces signed and served, not yet settled
        self.paused = False
        self.backend = backend or (lambda body: "ok")  # callable(gemini request body) -> text
        self.nonce = 0
        self.charged = 0
        self.served: dict = {}  # request_id -> response (replays return it, uncharged)
        self.calls: list = []
        self.drop_first_response = False
        self._dropped = False
        self.lock = threading.Lock()
        fac = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                parts = self.path.strip("/").split("/")
                if len(parts) == 4 and parts[0] == "mech" and parts[2] == "requester":
                    with fac.lock:
                        return self._send(200, fac.info(parts[3]))
                self._send(404, {"detail": "Not Found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if self.path.rstrip("/") == "/rpc":
                    return self._send(200, fac.rpc(body))
                parts = self.path.strip("/").split("/")
                if len(parts) == 3 and parts[:2] == ["mech", "chat"]:
                    with fac.lock:
                        code, obj = fac.chat(body)
                    return self._send(code, obj)
                self._send(404, {"detail": "Not Found"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        chain_state.rpc_url = self.url + "/rpc"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def _settle(self, nonce: int) -> None:
        time.sleep(self.settle_delay)
        with self.lock, self.cs.market_lock:
            if self.held and self.held[0] == nonce and self.cs.market_nonce == nonce:
                self.cs.market_nonce += 1
                self.charged += RATE
                self.held.pop(0)
            else:  # the slot was taken by another payer: settlement fails, the safe is paused
                self.paused = True

    def deposits(self) -> int:
        return sum(
            e["value"]
            for e in self.cs.ledger
            if e["kind"] == "tx"
            and e["to"] == TRACKER
            and e["data"].startswith("0x" + SEL_DEPOSIT_FOR)
            and e["data"].endswith(SAFE[2:].lower())
        )

    def info(self, safe):
        bal = self.deposits() - self.charged
        on_chain = self.cs.market_nonce
        return {
            "chain": "gnosis",
            "chain_id": CHAIN_ID,
            "marketplace_address": MARKETPLACE,
            "mech_address": MECH,
            "payment_type": "0x" + PAYMENT_TYPE,
            "delivery_rates": {"chat": RATE, "coingecko": RATE},
            "next_nonce": on_chain + len(self.held),
            "on_chain_nonce": on_chain,
            "held": len(self.held),
            "held_nonces": list(self.held),
            "balance": bal,
            "available": bal - RATE * len(self.held),
            "max_ttl_secs": 300,
        }

    def rpc(self, req):
        p = req["params"][0]
        if p["to"].lower() == MARKETPLACE and p["data"].startswith("0x" + SEL_MAP_TRACKERS):
            return {"jsonrpc": "2.0", "id": req["id"], "result": "0x" + TRACKER[2:].rjust(64, "0")}
        return {"jsonrpc": "2.0", "id": req["id"], "error": {"message": "unsupported"}}

    def chat(self, b):
        def err(code, error, detail, context=None):
            return code, {"detail": {"error": error, "detail": detail, "context": context or {}}}

        rid = b.get("request_id")
        if rid in self.served:
            return 200, self.served[rid]
        up = b["upstream"]
        if int(b["expires_at"]) < time.time():
            return err(400, "expired", "request expired")
        if self.paused:
            return err(
                503,
                "settlement_paused",
                "Recent settlements for this requester on this chain have failed",
                {"retry_after_secs": "30"},
            )
        nonce = int(b["nonce"])
        expected = self.cs.market_nonce + len(self.held)
        if nonce != expected:
            return err(409, "nonce_mismatch", "wrong nonce", {"expected": expected})
        data = canonical_request_data(
            method=up["method"],
            path=up["path"],
            query=up.get("query") or {},
            body_b64=up.get("body_b64", ""),
            expires_at=int(b["expires_at"]),
        )
        want = derive_request_id(
            marketplace_address=MARKETPLACE,
            mech_address=MECH,
            requester=b["safe_address"],
            data=data,
            delivery_rate=RATE,
            payment_type=bytes.fromhex(PAYMENT_TYPE),
            nonce=nonce,
            chain_id=CHAIN_ID,
        )
        if "0x" + want.hex() != rid:
            return err(400, "bad_request_id", "request_id does not match the canonical request")
        digest = "0x" + compute_safe_message_hash(want, b["safe_address"], CHAIN_ID).hex()
        if b["signature"] != mock_signature(digest):
            return err(401, "bad_signature", "signature is not the safe owner's over the SafeMessage digest")
        avail = self.deposits() - self.charged
        if avail < RATE:
            return err(
                402,
                "insufficient_deposit",
                "pre-deposit too low",
                {"balance": avail, "reserved": 0, "available": avail, "required": RATE},
            )
        if not up["path"].startswith("/v1beta/models/") or not up["path"].endswith(":generateContent"):
            return 404, {"error": {"code": 404, "message": "unknown upstream path"}}
        req = json.loads(base64.b64decode(up["body_b64"]))
        text = self.backend(req)
        resp = {
            "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": len(up["body_b64"]) // 4, "candidatesTokenCount": len(text) // 4},
        }
        if self.settle_delay:
            self.held.append(nonce)
            threading.Thread(target=self._settle, args=(nonce,), daemon=True).start()
        else:
            self.charged += RATE
            self.cs.market_nonce += 1
        self.served[rid] = resp
        self.calls.append({"request_id": rid, "model": up["path"].split("/")[-1].split(":")[0], "request": req})
        if self.drop_first_response and not self._dropped:
            self._dropped = True
            return 504, {"detail": "gateway timeout"}
        return 200, resp

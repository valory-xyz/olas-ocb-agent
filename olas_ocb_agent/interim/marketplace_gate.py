"""Serialise the two payers that take mech-marketplace nonces from the same safe.

Interim: goes away once every off-chain request, model calls included, goes through Connect, so Connect is
the only payer per Safe.

The marketplace nonce (mapNonces[requester]) belongs to the safe and is shared by every mech.
olas-ocb-agent has two payers on it: the harness's model calls through the facilitator, and the agent's
own `mech_request`s through Connect. Neither can see the other's unsettled requests, so a
Connect request signed at a nonce the facilitator is still holding makes the facilitator's
settlement fail. The facilitator then pauses the safe ("settlement_paused"); this happened in live use, with
several settlements left stuck.

The gate makes them take turns:
  - before an agent mech_request: wait until the facilitator holds no rows for the safe
    (next_nonce == on_chain_nonce), and note the on-chain nonce n0;
  - after it: the next model call waits until the on-chain nonce passes n0, i.e. Connect's
    request has settled, before the facilitator signs a new slot.
"""

from __future__ import annotations

import threading
import time

from ..model import ModelError


class MarketplaceGate:
    def __init__(self, facilitator, settle_timeout: float = 300.0, poll: float = 3.0, log=None):
        self.f = facilitator
        self.settle_timeout = settle_timeout
        self.poll = poll
        self.log = log or (lambda s: None)
        self.lock = threading.RLock()
        self.pending_floor: int | None = None
        self.on_floor = lambda: None  # called when a new floor is set, so it is saved at once (set by the runtime)
        self._settle_failures = 0  # consecutive failed waits for the current pending floor
        self._n0: int | None = None

    def _wait(self, cond, what: str, deadline: float | None = None):
        end = time.time() + self.settle_timeout
        end = min(end, deadline) if deadline is not None else end  # the broker call's deadline
        while True:
            info = self.f.info()
            if cond(info):
                return info
            if time.time() > end:
                raise ModelError(
                    f"marketplace gate: timed out waiting for {what} "
                    f"(on-chain nonce {info.on_chain_nonce}, facilitator next {info.next_nonce})"
                )
            time.sleep(self.poll)

    # ---- the agent's own mech requests (called by the broker) ------------------------
    def before_agent_mech_request(self, deadline: float | None = None) -> None:
        self.lock.acquire()
        try:
            if self.pending_floor is not None:
                self._settle_pending(deadline)
            info = self._wait(lambda i: i.next_nonce <= i.on_chain_nonce, "the model's mech calls to settle", deadline)
            self._n0 = info.on_chain_nonce
        except Exception:
            self.lock.release()
            raise

    def after_agent_mech_request(self, sent: bool) -> None:
        try:
            if sent and self._n0 is not None:
                self.pending_floor = self._n0 + 1
                self.on_floor()  # saved in the session state: a restart must not forget the pending request
        finally:
            self._n0 = None
            self.lock.release()

    # ---- the harness's model calls (called by FacilitatorModel) -------------------------
    def _settle_pending(self, deadline: float | None = None) -> None:
        floor = self.pending_floor
        self.log(f"[ocb-agent] waiting for the agent's mech request to settle on-chain (nonce >= {floor})")
        try:
            self._wait(lambda i: i.on_chain_nonce >= floor, "the agent's mech request to settle", deadline)
        except ModelError:
            self._settle_failures += 1
            if self._settle_failures < 2:
                raise
            # deadline: after 2 failed waits the request is treated as never settling (its signed slot has
            # expired; the facilitator rejects stale nonces), so the floor is dropped
            self.log(f"[ocb-agent] the agent's mech request at nonce {floor - 1} never settled; resuming model calls")
        self.pending_floor = None
        self._settle_failures = 0

    def before_model_call(self) -> None:
        self.lock.acquire()
        try:
            if self.pending_floor is not None:
                self._settle_pending()
        except Exception:
            self.lock.release()
            raise

    def after_model_call(self) -> None:
        self.lock.release()

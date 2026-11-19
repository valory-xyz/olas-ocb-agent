"""Local broker between the sandboxed agent and the connect server.

The harness process reads the real .mcp.json (as Claude Code does) and keeps
the connect token in memory. The agent only ever sees a shim .mcp.json that
points here with a broker token. The broker speaks the same MCP and REST API
as the connect server, so Connect's skills, signer_client.py and the `pearl`
CLI work unchanged; every value call passes the session Policy first.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from . import __version__
from .connect_client import ConnectClient, ConnectError
from .policy import CALL_DEADLINE_S, REST_READ_ROUTES, REST_VALUE_ROUTES, Policy

KILLED = "broker: the command that made this call was stopped (timed out) before it was sent; not sent"


def rpc_error(e: Exception) -> dict:
    """A JSON-RPC error for a request the broker couldn't handle: malformed JSON, or a crash."""
    code = -32700 if isinstance(e, json.JSONDecodeError) else -32603
    return {"code": code, "message": broker_error(e)}


def broker_error(e: Exception) -> str:
    return str(e) if isinstance(e, ConnectError) else f"broker: could not handle the request: {type(e).__name__}: {e}"


class Broker:
    def __init__(self, upstream: ConnectClient, policy: Policy, host: str = "127.0.0.1", port: int = 0):
        self.upstream = upstream
        self.policy = policy
        self.token = secrets.token_urlsafe(24)
        self.gate = None  # MarketplaceGate: serialises agent mech requests with facilitator model calls
        self._lock = threading.Lock()  # one upstream MCP session; serialize calls on it
        self.epoch = 0  # bumped when the shell kills a command: its calls not yet sent are refused
        self._sending = None  # (tool, request_id) of the MCP call being sent upstream, if any
        self.in_flight = 0  # POSTs being handled (approvals, gate waits, upstream calls); see busy()
        self._count_lock = threading.Lock()
        self.on_refusal = None  # called (tool, args, rule) when policy refuses a call; the eval logs these
        self.on_list = None  # called (tools) when the agent lists tools: the eval records what it saw
        broker = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, code: int, obj: dict | None, extra: dict | None = None):
                body = b"" if obj is None else json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _authed(self) -> bool:
                if self.headers.get("Authorization", "") != f"Bearer {broker.token}":
                    self._send(401, {"detail": "unauthorized"})
                    return False
                return True

            def _body(self) -> dict:
                n = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(n) or b"{}")

            def do_GET(self):
                if not self._authed():
                    return
                path = urlsplit(self.path).path.rstrip("/")
                if path not in REST_READ_ROUTES:
                    return self._send(404, {"detail": f"broker: route {path} not exposed"})
                try:
                    with broker._lock:
                        self._send(200, broker.upstream.rest("GET", self.path))
                except ConnectError as e:
                    self._send(400, {"detail": str(e)})

            def do_POST(self):
                with broker.counting():
                    try:
                        self._post()
                    except Exception as e:  # noqa: BLE001 - a malformed body or a crash: a JSON-RPC error back
                        with contextlib.suppress(OSError):  # unless the client is gone
                            self._send(200, {"jsonrpc": "2.0", "id": None, "error": rpc_error(e)})

            def _post(self):
                if not self._authed():
                    return
                path = urlsplit(self.path).path.rstrip("/")
                body = self._body()
                if path == "/mcp":
                    return self._mcp(body)
                if path in REST_VALUE_ROUTES:
                    tool, epoch = REST_VALUE_ROUTES[path], broker.epoch
                    d = broker.policy.check(tool, body)
                    if not d.allowed:
                        broker.refused(tool, body, d.rule)
                        return self._send(400, {"detail": d.rule})
                else:  # reads are GET-only (/wallet); nothing else is forwarded
                    return self._send(404, {"detail": f"broker: route {path} not exposed"})
                try:
                    forwarded, res = broker._forward(
                        epoch, tool, body, lambda: broker.upstream.rest("POST", path, body)
                    )
                    if not forwarded:
                        return self._send(400, {"detail": KILLED})
                    self._send(200, res)
                except ConnectError as e:
                    self._send(400, {"detail": str(e)})

            def _mcp(self, msg: dict):
                mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
                sid = {"Mcp-Session-Id": "ocb-agent-broker"}
                if mid is None:  # notification
                    return self._send(202, None, sid)

                def ok(result):
                    self._send(200, {"jsonrpc": "2.0", "id": mid, "result": result}, sid)

                if method == "initialize":
                    return ok(
                        {
                            "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "ocb-agent-broker", "version": __version__},
                        }
                    )
                if method == "ping":
                    return ok({})
                try:
                    if method == "tools/list":
                        with broker._lock:
                            tools = broker.upstream.list_tools()
                        if broker.on_list is not None:
                            with contextlib.suppress(Exception):  # reporting must never change what the agent gets
                                broker.on_list(tools)
                        return ok({"tools": tools})
                    if method == "tools/call":
                        name, args = params.get("name", ""), params.get("arguments") or {}
                        deadline, epoch = time.time() + CALL_DEADLINE_S, broker.epoch
                        d = broker.policy.check(name, args)
                        if not d.allowed:
                            broker.refused(name, args, d.rule)
                            return ok({"content": [{"type": "text", "text": d.rule}], "isError": True})
                        gated = name == "mech_request" and broker.gate is not None
                        if gated:
                            broker.gate.before_agent_mech_request(deadline)
                        sent = True  # if the call raises, it may still have been sent and paid: wait for it
                        try:
                            forwarded, res = broker._forward(
                                epoch, name, args, lambda: broker.upstream.call_tool_raw(name, args), gated
                            )
                            if not forwarded:  # killed: _forward released the gate, as nothing was sent
                                gated = False
                                return ok({"content": [{"type": "text", "text": KILLED}], "isError": True})
                            sent = not res.get("isError") or bool(
                                (res.get("structuredContent") or {}).get("request_ids")
                            )
                            return ok(res)
                        finally:
                            if gated:
                                broker.gate.after_agent_mech_request(sent)
                except ConnectError as e:
                    return ok({"content": [{"type": "text", "text": str(e)}], "isError": True})
                except Exception as e:  # noqa: BLE001 - e.g. the gate timed out: refuse with a reason, don't drop the call
                    return ok(
                        {"content": [{"type": "text", "text": f"broker: {type(e).__name__}: {e}"}], "isError": True}
                    )
                self._send(
                    200, {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"{method} not supported"}}
                )

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.url = f"http://{host}:{self.server.server_address[1]}/mcp/"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def refused(self, tool: str, args: dict, rule: str) -> None:
        if self.on_refusal is not None:
            with contextlib.suppress(Exception):  # reporting must never change what the agent gets back
                self.on_refusal(tool, args, rule)

    @contextlib.contextmanager
    def counting(self):
        with self._count_lock:
            self.in_flight += 1
        try:
            yield
        finally:
            with self._count_lock:
                self.in_flight -= 1

    def _forward(self, epoch: int, name: str, args: dict, call, gated: bool = False) -> tuple[bool, object]:
        """Send a value call upstream (MCP or REST): (True, call()) under the upstream lock, or (False, None) if the
        command that made it was killed since it arrived. The kill is checked here, right before the send: the call
        may have waited on the gate, an approval or a slow call ahead of it. A killed gated call releases the gate as
        not sent. While the call is being sent, a kill reports it (command_killed)."""
        with self._lock:
            with self._count_lock:
                killed = self.epoch != epoch
                if not killed:
                    self._sending = (name, args.get("request_id") if isinstance(args, dict) else None)
            if killed:
                if gated:
                    self.gate.after_agent_mech_request(False)
                return False, None
            try:
                return True, call()
            finally:
                self._sending = None

    def command_killed(self) -> str:
        """Refuse the killed command's calls not yet sent. Returns a note for the agent naming the call already
        being sent at the kill (its result is lost with the command), or "" if none was."""
        with self._count_lock:
            self.epoch += 1
            sending = self._sending
        if sending is None:
            return ""
        name, rid = sending
        if rid is None:
            return f"broker: a {name} call was already sent to Connect when the command was killed; check its outcome"
        return (
            f"broker: a {name} call with request_id {rid} was already sent to Connect when the command was killed; "
            "check its outcome, and if you retry, reuse that request_id"
        )

    def busy(self) -> bool:
        """A call is waiting in the broker: the agent's shell command shouldn't time out under it."""
        return self.in_flight > 0

    def start(self) -> Broker:
        self._thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def shim_mcp_json(self) -> dict:
        return {
            "mcpServers": {
                "pearl-connect": {
                    "type": "http",
                    "url": self.url,
                    "headers": {"Authorization": f"Bearer {self.token}"},
                }
            }
        }

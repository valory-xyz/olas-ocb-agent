"""Check that `olas-ocb-agent[litellm,facilitator]`, installed alone (no dev extra), can run LiteLLM's retry and
streaming paths: the agent's own LitellmModel against a local fake OpenAI endpoint whose first request fails with
a 500, so LiteLLM retries it through tenacity. No network beyond 127.0.0.1, no API keys.

Run it with the clean venv's python: python scripts/check_litellm_extra.py
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"  # the bundled price map: no fetch at import

import tenacity  # noqa: F401 - must be installed by the extra

from olas_ocb_agent.model import LitellmModel

hits = []


class FakeOpenAI(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        hits.append(self.path)
        if len(hits) == 1:  # a transient provider error
            body = json.dumps({"error": {"message": "overloaded", "type": "server_error"}}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        base = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "fake"}
        chunks = [
            {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "DONE: "}}]},
            {**base, "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]},
            {**base, "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
        ]
        for c in chunks:
            self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    # max_retries=0: the OpenAI client doesn't retry itself, so the second request is LiteLLM's (tenacity) retry
    model = LitellmModel("openai/fake", api_base=base, api_key="not-a-key", max_retries=0)
    tokens = []
    reply = model.query([{"role": "user", "content": "hi"}], on_token=tokens.append)
    server.shutdown()
    print(f"requests: {len(hits)}, reply: {reply['content']!r}, usage: {reply['usage'].get('total_tokens')}")
    ok = len(hits) == 2 and reply["content"] == "DONE: ok" and "".join(tokens) == "DONE: ok"
    ok = ok and "tenacity" in sys.modules
    print("ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""OpenAI-compatible VLM client (the places importer's name suggester)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bebop_vision.vlm import ChatVLMClient, VLMError


def test_chat_vlm_client_roundtrip():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(n))
            payload = json.dumps(
                {"choices": [{"message": {"content": '{"name": "Kitchen"}'}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = ChatVLMClient(
            f"http://127.0.0.1:{httpd.server_address[1]}", model="m", timeout=5.0
        )
        out = client.complete([{"role": "user", "content": "hi"}], max_tokens=64)
        assert out == '{"name": "Kitchen"}'
        assert seen["path"] == "/v1/chat/completions"
        assert seen["body"]["model"] == "m"
        assert seen["body"]["max_tokens"] == 64
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_chat_vlm_client_wraps_transport_errors():
    client = ChatVLMClient("http://127.0.0.1:1", timeout=0.5)
    with pytest.raises(VLMError):
        client.complete([{"role": "user", "content": "hi"}])

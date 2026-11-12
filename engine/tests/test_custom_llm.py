"""Tests for OpenAI-compatible custom LLM rewrite and /llm/models endpoint."""

from __future__ import annotations

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import pytest
from fastapi.testclient import TestClient

from sottovox_engine.cleanup import (
    _normalize_openai_base_url,
    clean_transcript_custom_openai,
    list_custom_models,
)
from sottovox_engine.server import app


class _MockOpenAIState:
    last_auth_header: str | None = None
    last_path: str | None = None
    fail_next: bool = False


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        _MockOpenAIState.last_path = self.path
        _MockOpenAIState.last_auth_header = self.headers.get("Authorization")
        if _MockOpenAIState.fail_next:
            _MockOpenAIState.fail_next = False
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error":"unauthorized"}')
            return
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "gpt-4o-mini"}, {"id": "custom-qwen-7b"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        _MockOpenAIState.last_path = self.path
        _MockOpenAIState.last_auth_header = self.headers.get("Authorization")
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        if _MockOpenAIState.fail_next:
            _MockOpenAIState.fail_next = False
            self.send_response(500)
            self.end_headers()
            return
        if self.path == "/v1/chat/completions":
            body = json.dumps(
                {"choices": [{"message": {"content": "Open the gate at 9 AM."}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture()
def mock_openai_server():
    _MockOpenAIState.last_auth_header = None
    _MockOpenAIState.last_path = None
    _MockOpenAIState.fail_next = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_normalize_openai_base_url() -> None:
    assert _normalize_openai_base_url("https://api.openai.com") == "https://api.openai.com/v1"
    assert _normalize_openai_base_url("https://api.openai.com/v1/") == "https://api.openai.com/v1"
    with pytest.raises(ValueError):
        _normalize_openai_base_url("")
    with pytest.raises(ValueError):
        _normalize_openai_base_url("localhost:11434")


def test_list_custom_models_and_auth_header(mock_openai_server: str) -> None:
    models, err = asyncio.run(list_custom_models(mock_openai_server, api_key="sk-test-123"))
    assert err is None
    assert models == ["custom-qwen-7b", "gpt-4o-mini"]
    assert _MockOpenAIState.last_auth_header == "Bearer sk-test-123"
    assert _MockOpenAIState.last_path == "/v1/models"


def test_clean_transcript_custom_openai_success_and_fallback(mock_openai_server: str) -> None:
    res = asyncio.run(
        clean_transcript_custom_openai(
            "um open the gate at nine am",
            base_url=mock_openai_server,
            api_key="sk-test-123",
            model="gpt-4o-mini",
        )
    )
    assert res.applied is True
    assert res.model == "gpt-4o-mini"
    assert res.text == "Open the gate at 9 AM."
    assert res.error is None

    _MockOpenAIState.fail_next = True
    fallback = asyncio.run(
        clean_transcript_custom_openai(
            "um open the gate please",
            base_url=mock_openai_server,
            api_key="sk-test-123",
            model="gpt-4o-mini",
        )
    )
    assert fallback.applied is True
    assert fallback.model == "rules fallback"
    assert fallback.error is not None
    assert fallback.text == "Open the gate please"


def test_fastapi_llm_models_route(mock_openai_server: str) -> None:
    client = TestClient(app)
    resp = client.post(
        "/llm/models",
        data={"base_url": f"{mock_openai_server}/v1", "api_key": "sk-route"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["models"] == ["custom-qwen-7b", "gpt-4o-mini"]
    assert payload["error"] is None

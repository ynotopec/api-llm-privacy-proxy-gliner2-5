import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import HTTPException

sys.modules.setdefault(
    "torch",
    SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    ),
)

import app
from privacy_proxy_core.redaction import RedactionContext, RedactionStats
from privacy_proxy_core.sanitizers.gliner2 import GLiNER2Sanitizer


def test_payload_limits_reject_oversized_strings(monkeypatch):
    monkeypatch.setattr(app.settings, "max_string_chars", 5)

    with pytest.raises(HTTPException) as error:
        app.validate_payload_limits({"messages": [{"content": "secret"}]})

    assert error.value.status_code == 413
    assert error.value.detail == "string_too_large_to_filter"


def test_payload_limits_reject_excessive_depth(monkeypatch):
    monkeypatch.setattr(app.settings, "max_json_depth", 2)

    with pytest.raises(HTTPException) as error:
        app.validate_payload_limits({"a": {"b": {"c": "value"}}})

    assert error.value.status_code == 413
    assert error.value.detail == "json_too_deep"


def test_inference_runs_outside_event_loop(monkeypatch):
    sanitizer = GLiNER2Sanitizer("test/model", entity_types=["email"])
    sanitizer.model = Mock()
    sanitizer.model.extract_entities.return_value = []

    async def already_loaded():
        return None

    sanitizer.ensure_loaded = already_loaded
    to_thread = Mock(return_value=asyncio.sleep(0, result=[]))
    monkeypatch.setattr(asyncio, "to_thread", to_thread)

    result = asyncio.run(
        sanitizer.sanitize_text("hello", RedactionContext(), RedactionStats())
    )

    assert result == "hello"
    to_thread.assert_called_once()


def test_auth_uses_constant_time_comparison(monkeypatch):
    monkeypatch.setattr(app.settings, "inbound_api_keys", ["expected"])
    compare = Mock(return_value=False)
    monkeypatch.setattr(app.secrets, "compare_digest", compare)
    request = SimpleNamespace(headers={"authorization": "Bearer supplied"})

    with pytest.raises(HTTPException):
        app.require_auth(request)

    compare.assert_called_once_with("supplied", "expected")


def test_inbound_token_is_not_forwarded_upstream(monkeypatch):
    monkeypatch.setattr(app.settings, "upstream_api_key", "")
    request = SimpleNamespace(
        headers={
            "authorization": "Bearer inbound-secret",
            "openai-organization": "org-test",
        }
    )

    headers = app.build_upstream_headers(request)

    assert "authorization" not in headers
    assert headers["openai-organization"] == "org-test"


def test_connection_nominated_request_header_is_not_forwarded(monkeypatch):
    monkeypatch.setattr(app.settings, "upstream_api_key", "")
    request = SimpleNamespace(
        headers={"connection": "x-internal, keep-alive", "x-internal": "secret"}
    )

    headers = app.build_upstream_headers(request)

    assert "x-internal" not in headers
    assert "connection" not in headers


def test_connection_nominated_response_header_is_not_forwarded():
    headers = httpx.Headers(
        {"connection": "x-upstream-internal", "x-upstream-internal": "secret"}
    )

    assert app.upstream_response_headers(headers) == {}


def test_spanless_repeated_entities_are_all_redacted():
    sanitizer = GLiNER2Sanitizer("test/model", min_score=0.5)
    text = "jane@example.com and jane@example.com"

    spans = sanitizer._parse_result(
        text,
        {"email": [{"text": "jane@example.com", "confidence": 0.9}]},
    )
    output = sanitizer._build_output(
        text, spans, RedactionContext(), RedactionStats()
    )

    assert output == "[EMAIL_1] and [EMAIL_1]"


def test_entity_labels_cannot_break_placeholder_syntax():
    ctx = RedactionContext()

    placeholder = ctx.placeholder('B-email"]\\nmalicious', "secret")

    assert placeholder == "[EMAIL_MALICIOUS_1]"


def test_stream_request_is_sanitized_before_native_upstream_stream(monkeypatch):
    request = SimpleNamespace(
        method="POST",
        headers={},
        stream=lambda: _request_chunks(
            b'{"model":"test-anonym","stream":true,'
            b'"messages":[{"role":"user","content":"Jane Doe"}]}'
        ),
    )
    sanitized = {
        "model": "test-anonym",
        "stream": True,
        "messages": [{"role": "user", "content": "[PERSON_1]"}],
    }
    sanitize_payload = AsyncMock(
        return_value=(sanitized, RedactionStats(tokens=2, spans=1))
    )
    streamed_response = app.StreamingResponse(iter([b"data: first\n\n"]))
    forward = AsyncMock(return_value=streamed_response)
    monkeypatch.setattr(app.sanitizer, "sanitize_payload", sanitize_payload)
    monkeypatch.setattr(app, "forward_request", forward)
    monkeypatch.setattr(app.settings, "upstream_base_url", "http://upstream/v1")

    response = asyncio.run(app.proxy_openai(request, "chat/completions"))

    sanitize_payload.assert_awaited_once()
    assert forward.await_args.args[2]["messages"][0]["content"] == "[PERSON_1]"
    assert forward.await_args.kwargs["stream"] is True
    assert response is streamed_response


async def _request_chunks(*chunks):
    for chunk in chunks:
        yield chunk


def test_revision_identifies_native_streaming_build():
    assert app.APP_REVISION == "gliner2.5-input-filter-native-streaming"
    assert "settings.filter_output" not in Path(app.__file__).read_text(encoding="utf-8")


def test_prometheus_escapes_untrusted_entity_labels():
    metrics = app.GlobalMetrics()

    async def render():
        await metrics.add(1, 1, {'email"\\\ninjected': 1})
        return await metrics.prometheus()

    output = asyncio.run(render())

    assert 'label="email\\"\\\\\\ninjected"' in output
    assert "\ninjected" not in output

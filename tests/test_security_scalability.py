import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

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


def test_streaming_is_rejected_when_output_filtering_is_enabled(monkeypatch):
    """SSE must not silently bypass the configured response sanitizer."""
    monkeypatch.setattr(app.settings, "inbound_api_keys", [])
    monkeypatch.setattr(app.settings, "filter_output", True)

    body = b'{"model":"test","stream":true,"messages":[]}'
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
        },
        receive,
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(app.proxy_openai(request, "chat/completions"))

    assert error.value.status_code == 400
    assert error.value.detail == "streaming_requires_filter_output_disabled"


def test_prometheus_escapes_untrusted_entity_labels():
    metrics = app.GlobalMetrics()

    async def render():
        await metrics.add(1, 1, {'email"\\\ninjected': 1})
        return await metrics.prometheus()

    output = asyncio.run(render())

    assert 'label="email\\"\\\\\\ninjected"' in output
    assert "\ninjected" not in output

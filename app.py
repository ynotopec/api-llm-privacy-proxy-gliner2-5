from __future__ import annotations

import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.background import BackgroundTask

from privacy_proxy_core.redaction import PrivacySanitizerBase, RedactionContext, RedactionStats
from privacy_proxy_core.metrics import GlobalMetrics, metrics
from privacy_proxy_core.settings import Settings, settings, suffix_model_id, unsuffix_model_id, unsuffix_model_path

APP_REVISION = "gliner2.5-buffered-streaming"


class GLiNER2ProxySanitizer(PrivacySanitizerBase):
    def __init__(self) -> None:
        super().__init__()
        self._delegate: Any = None

    async def ensure_loaded(self) -> None:
        from privacy_proxy_core.sanitizers.gliner2 import GLiNER2Sanitizer
        if self._delegate is None:
            entity_types = [
                x.strip()
                for x in os.getenv(
                    "PRIVACY_ENTITY_TYPES",
                    "person,full_name,first_name,last_name,date_of_birth,email,phone_number,address,"
                    "street_address,city,state_or_region,postal_code,country,government_id,national_id_number,"
                    "passport_number,drivers_license_number,tax_id,bank_account,account_number,iban,"
                    "payment_card,card_number,username,ip_address,password,api_key,access_token,secret",
                ).split(",")
                if x.strip()
            ]
            self._delegate = GLiNER2Sanitizer(
                settings.privacy_model_id,
                device=settings.device,
                entity_types=entity_types,
                min_score=settings.min_entity_score,
            )
            if settings.model_idle_unload_seconds > 0:
                check_interval = int(os.getenv("MODEL_IDLE_CHECK_SECONDS", "30"))
                await self._delegate.start_idle_watcher(check_interval=check_interval)
        self._delegate.unload_if_idle()
        await self._delegate.ensure_loaded()

    async def start_idle_watcher(self) -> None:
        """Initialize the delegate and its idle watcher for app startup."""
        await self.ensure_loaded()

    async def stop_idle_watcher(self) -> None:
        """Stop the delegate watcher and release model memory."""
        if self._delegate is not None:
            await self._delegate.stop_idle_watcher()

    async def sanitize_text(
        self, text: str, ctx: RedactionContext, stats: RedactionStats,
    ) -> str:
        if self._delegate is None:
            await self.ensure_loaded()
        return await self._delegate.sanitize_text(text, ctx, stats)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split())) if text else 0

    async def sanitize_payload(
        self, payload: Any, settings: Settings,
    ) -> tuple[Any, RedactionStats]:
        from privacy_proxy_core.redaction import RedactionContext, RedactionStats
        ctx = RedactionContext()
        stats = RedactionStats()
        sanitized = await self._sanitize_any(payload, ctx, stats, None, settings)
        return sanitized, stats

    async def _sanitize_any(
        self, value: Any, ctx: RedactionContext, stats: RedactionStats,
        parent_key: Optional[str], settings: Settings,
    ) -> Any:
        if parent_key in settings.skip_json_keys:
            return value
        if isinstance(value, str):
            return await self.sanitize_text(value, ctx, stats)
        if isinstance(value, list):
            return [await self._sanitize_any(v, ctx, stats, None, settings) for v in value]
        if isinstance(value, dict):
            return {k: await self._sanitize_any(v, ctx, stats, k, settings) for k, v in value.items()}
        return value


sanitizer = GLiNER2ProxySanitizer()

app = FastAPI(title="OpenAI Privacy Filter Proxy GLiNER2.5", version="1.0.2")

# ── routes (identiques aux autres repos via le core) ────────────


@app.on_event("startup")
async def log_revision() -> None:
    log = logging.getLogger("llm-privacy-proxy")
    log.info("Starting GLiNER2 proxy revision=%s", APP_REVISION)
    app.state.upstream_client = httpx.AsyncClient(
        timeout=httpx.Timeout(600.0, connect=30.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    await sanitizer.start_idle_watcher()


@app.on_event("shutdown")
async def shutdown_sanitizer() -> None:
    await sanitizer.stop_idle_watcher()
    client = getattr(app.state, "upstream_client", None)
    if client is not None:
        await client.aclose()


def rewrite_request_model_ids(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    out = dict(value)
    model_id = out.get("model")
    if isinstance(model_id, str):
        model_id = model_id.strip()
        if not model_id or model_id == settings.model_suffix:
            raise HTTPException(status_code=400, detail="invalid_model_id")
        out["model"] = unsuffix_model_id(model_id)
    return out


def rewrite_response_model_ids(value: Any, *, models_endpoint: bool = False) -> Any:
    if isinstance(value, list):
        return [rewrite_response_model_ids(item, models_endpoint=models_endpoint) for item in value]
    if not isinstance(value, dict):
        return value
    out = dict(value)
    model_id = out.get("model")
    if isinstance(model_id, str):
        out["model"] = suffix_model_id(model_id)
    if models_endpoint:
        object_id = out.get("id")
        if isinstance(object_id, str) and out.get("object") == "model":
            out["id"] = suffix_model_id(object_id)
        data = out.get("data")
        if isinstance(data, list):
            out["data"] = [rewrite_response_model_ids(item, models_endpoint=True) for item in data]
    return out


def extract_bearer(req: Request) -> str:
    auth = req.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return ""
    return auth.split(" ", 1)[1].strip()


def require_auth(req: Request, *, metrics_auth: bool = False) -> None:
    if metrics_auth and not settings.metrics_require_auth:
        return
    if not settings.inbound_api_keys:
        return
    token = extract_bearer(req)
    if not any(secrets.compare_digest(token, key) for key in settings.inbound_api_keys):
        raise HTTPException(status_code=401, detail="invalid_or_missing_api_token")


def build_upstream_headers(req: Request) -> Dict[str, str]:
    excluded = {
        "host", "content-length", "authorization", "connection", "keep-alive",
        "proxy-authenticate", "proxy-authorization",
        "te", "trailers", "transfer-encoding", "upgrade",
    }
    headers = {k: v for k, v in req.headers.items() if k.lower() not in excluded}
    if settings.upstream_api_key:
        headers["authorization"] = f"Bearer {settings.upstream_api_key}"
    headers["content-type"] = "application/json"
    return headers


def headers_for_modified_body(resp: Response) -> Dict[str, str]:
    excluded = {"content-length", "content-encoding", "transfer-encoding", "connection"}
    return {k: v for k, v in resp.headers.items() if k.lower() not in excluded}


def response_models_endpoint(full_path: str) -> bool:
    return full_path == "models" or full_path.startswith("models/")


def add_response_model_suffixes(upstream_resp: Response, full_path: str) -> Response:
    content_type = upstream_resp.headers.get("content-type", "")
    if "application/json" not in content_type:
        return upstream_resp
    try:
        response_payload = json.loads(upstream_resp.body)
    except Exception:
        return upstream_resp
    response_payload = rewrite_response_model_ids(
        response_payload, models_endpoint=response_models_endpoint(full_path)
    )
    return JSONResponse(
        content=response_payload,
        status_code=upstream_resp.status_code,
        headers=headers_for_modified_body(upstream_resp),
    )


async def forward_request(
    req: Request, full_path: str, sanitized_payload: Any, stream: bool = False,
) -> Response:
    if not settings.llm_enabled:
        raise HTTPException(status_code=503, detail="llm_disabled")
    full_path = unsuffix_model_path(full_path)
    url = f"{settings.upstream_base_url}/{full_path}"
    client: httpx.AsyncClient = app.state.upstream_client

    if stream:
        upstream_req = client.build_request(
            method=req.method,
            url=url,
            headers=build_upstream_headers(req),
            params=dict(req.query_params),
            json=sanitized_payload,
        )
        upstream_stream = await client.send(upstream_req, stream=True)

        async def close_upstream() -> None:
            await upstream_stream.aclose()

        content_type = upstream_stream.headers.get("content-type", "application/json")
        headers = {
            k: v for k, v in upstream_stream.headers.items()
            if k.lower() not in {
                "content-length", "connection", "keep-alive",
                "proxy-authenticate", "proxy-authorization", "te", "trailers",
                "transfer-encoding", "upgrade",
            }
        }
        return StreamingResponse(
            upstream_stream.aiter_bytes(),
            status_code=upstream_stream.status_code,
            headers=headers,
            media_type=content_type,
            background=BackgroundTask(close_upstream),
        )

    upstream = await client.request(
        method=req.method,
        url=url,
        headers=build_upstream_headers(req),
        params=dict(req.query_params),
        json=sanitized_payload,
    )

    content_type = upstream.headers.get("content-type", "application/json")
    headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in {"content-length", "content-encoding", "transfer-encoding", "connection"}
    }
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=headers,
        media_type=content_type,
    )


def text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


def sanitized_chat_content(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    fallback = ""
    for message in messages:
        if isinstance(message, dict):
            content = text_from_message_content(message.get("content"))
            if content:
                fallback = content
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = text_from_message_content(message.get("content"))
        if content:
            return content
    return fallback


def validate_payload_limits(payload: Any) -> None:
    """Reject payloads that could bypass filtering or exhaust recursion/memory."""
    stack: list[tuple[Any, int]] = [(payload, 0)]
    nodes = 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > settings.max_json_nodes:
            raise HTTPException(status_code=413, detail="too_many_json_values")
        if depth > settings.max_json_depth:
            raise HTTPException(status_code=413, detail="json_too_deep")
        if isinstance(value, str) and len(value) > settings.max_string_chars:
            # Passing an oversized value through unchanged would disclose PII.
            raise HTTPException(status_code=413, detail="string_too_large_to_filter")
        if isinstance(value, dict):
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)


async def read_limited_body(req: Request) -> bytes:
    """Read a possibly chunked request without buffering past the limit."""
    body = bytearray()
    async for chunk in req.stream():
        if len(body) + len(chunk) > settings.max_request_bytes:
            raise HTTPException(status_code=413, detail="request_body_too_large")
        body.extend(chunk)
    return bytes(body)


def llm_disabled_response_payload(full_path: str, sanitized_payload: Any, in_stats: RedactionStats) -> dict[str, Any]:
    if full_path == "chat/completions" and isinstance(sanitized_payload, dict):
        return {
            "id": "privacy-proxy-sanitized",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": suffix_model_id(str(sanitized_payload.get("model", ""))),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": sanitized_chat_content(sanitized_payload),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "privacy": {
                "filtered_tokens": in_stats.tokens,
                "filtered_spans": in_stats.spans,
                "filtered_labels": in_stats.labels,
            },
        }
    return {
        "object": "privacy_proxy.sanitized_payload",
        "llm_enabled": False,
        "data": sanitized_payload,
        "privacy": {
            "filtered_tokens": in_stats.tokens,
            "filtered_spans": in_stats.spans,
            "filtered_labels": in_stats.labels,
        },
    }


def completion_as_stream_event(full_path: str, payload: Any) -> Any:
    """Convert a buffered completion response into one OpenAI-compatible event."""
    if not isinstance(payload, dict):
        return payload

    event = dict(payload)
    choices = event.get("choices")
    if full_path == "chat/completions" and isinstance(choices, list):
        event["object"] = "chat.completion.chunk"
        streamed_choices: list[Any] = []
        for choice in choices:
            if not isinstance(choice, dict):
                streamed_choices.append(choice)
                continue
            streamed_choice = dict(choice)
            message = streamed_choice.pop("message", None)
            if isinstance(message, dict):
                streamed_choice["delta"] = message
            streamed_choices.append(streamed_choice)
        event["choices"] = streamed_choices
    return event


def buffered_streaming_response(
    full_path: str, payload: Any, source: Response,
) -> StreamingResponse:
    """Emit an output-filtered response as a single safe SSE event."""
    event = completion_as_stream_event(full_path, payload)
    body = (
        f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}"
        "\n\ndata: [DONE]\n\n"
    ).encode("utf-8")
    headers = headers_for_modified_body(source)
    headers.pop("content-type", None)
    headers.pop("Content-Type", None)

    async def events():
        yield body

    return StreamingResponse(
        events(),
        status_code=source.status_code,
        headers=headers,
        media_type="text/event-stream",
    )


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": settings.privacy_model_id,
        "upstream_configured": bool(settings.upstream_base_url),
        "llm_enabled": settings.llm_enabled,
        "filter_output": settings.filter_output,
        "model_suffix": settings.model_suffix,
        "device": settings.device,
        "model_idle_unload_seconds": settings.model_idle_unload_seconds,
        "max_concurrent_inferences": settings.max_concurrent_inferences,
        "revision": APP_REVISION,
    }


@app.get("/metrics")
async def get_metrics(req: Request) -> PlainTextResponse:
    require_auth(req, metrics_auth=True)
    return PlainTextResponse(await metrics.prometheus(), media_type="text/plain")


@app.api_route("/v1/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy_openai(req: Request, full_path: str) -> Response:
    if req.method == "OPTIONS":
        return Response(status_code=204)

    require_auth(req)

    if req.method in ("GET", "DELETE"):
        if not settings.llm_enabled:
            raise HTTPException(status_code=503, detail="llm_disabled")
        upstream_resp = await forward_request(req, full_path, sanitized_payload=None)
        return add_response_model_suffixes(upstream_resp, full_path)

    content_length = req.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > settings.max_request_bytes:
                raise HTTPException(status_code=413, detail="request_body_too_large")
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid_content_length")

    body = await read_limited_body(req)
    try:
        payload = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="expected_json_body")

    validate_payload_limits(payload)

    wants_stream = bool(isinstance(payload, dict) and payload.get("stream") is True)
    start = time.perf_counter()

    sanitized_payload, in_stats = await sanitizer.sanitize_payload(payload, settings)
    sanitized_payload = rewrite_request_model_ids(sanitized_payload)
    if wants_stream and settings.filter_output and isinstance(sanitized_payload, dict):
        # Ask upstream for one complete response so entity spans cannot cross
        # SSE event boundaries.  The filtered result is converted back to SSE.
        sanitized_payload["stream"] = False
    await metrics.add(in_stats.tokens, in_stats.spans, in_stats.labels)

    if not settings.llm_enabled:
        response_payload = llm_disabled_response_payload(full_path, sanitized_payload, in_stats)
        response = JSONResponse(content=response_payload)
        response.headers["x-privacy-filtered-tokens"] = str(in_stats.tokens)
        response.headers["x-privacy-filtered-spans"] = str(in_stats.spans)
        response.headers["x-privacy-filter-latency-ms"] = str(round((time.perf_counter() - start) * 1000, 2))
        return response

    if wants_stream and not settings.filter_output:
        return await forward_request(req, full_path, sanitized_payload, stream=True)

    upstream_resp = await forward_request(req, full_path, sanitized_payload, stream=False)

    upstream_resp.headers["x-privacy-filtered-tokens"] = str(in_stats.tokens)
    upstream_resp.headers["x-privacy-filtered-spans"] = str(in_stats.spans)
    upstream_resp.headers["x-privacy-filter-latency-ms"] = str(round((time.perf_counter() - start) * 1000, 2))

    rewritten_resp = add_response_model_suffixes(upstream_resp, full_path)
    content_type = rewritten_resp.headers.get("content-type", "")
    if "application/json" not in content_type:
        return rewritten_resp

    try:
        response_payload = json.loads(rewritten_resp.body)
    except Exception:
        return rewritten_resp

    out_stats = RedactionStats()
    if settings.filter_output:
        response_payload, out_stats = await sanitizer.sanitize_payload(response_payload, settings)
        await metrics.add(out_stats.tokens, out_stats.spans, out_stats.labels, count_request=False)

    if wants_stream and settings.filter_output:
        return buffered_streaming_response(full_path, response_payload, rewritten_resp)

    final = JSONResponse(
        content=response_payload,
        status_code=rewritten_resp.status_code,
        headers=headers_for_modified_body(rewritten_resp),
    )
    if settings.filter_output:
        final.headers["x-privacy-filtered-output-tokens"] = str(out_stats.tokens)
        final.headers["x-privacy-filtered-output-spans"] = str(out_stats.spans)
    return final


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        log_level=os.getenv("LOG_LEVEL", "info"),
    )

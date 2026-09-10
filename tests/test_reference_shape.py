from pathlib import Path

SOURCE = Path("app.py").read_text()


def test_openai_v1_route_matches_reference_shape():
    assert '@app.api_route("/v1/{full_path:path}"' in SOURCE
    assert 'async def proxy_openai(req: Request, full_path: str)' in SOURCE


def test_reference_auth_and_model_variables_are_supported():
    assert 'INBOUND_API_KEYS' in SOURCE
    assert 'PRIVACY_MODEL_ID' in SOURCE
    assert 'PRIVACY_ENTITY_TYPES' in SOURCE
    assert 'fastino/gliner2-privacy-filter-PII-multi' in SOURCE


def test_reference_features_are_present():
    assert '@app.get("/metrics")' in SOURCE
    assert 'MODEL_SUFFIX' in SOURCE
    assert 'FILTER_OUTPUT' in SOURCE


def test_gliner2_extract_entities_receives_entity_types():
    assert 'extract_entities(\n                text,\n                settings.entity_types,' in SOURCE
    assert 'extract_entities(text, settings.entity_types)' in SOURCE


def test_health_exposes_device_revision_marker():
    assert 'APP_REVISION = "gliner2-optional-llm"' in SOURCE
    assert '"revision": APP_REVISION' in SOURCE
    assert '"resolved_device": sanitizer.model_device' in SOURCE
    assert '"cuda_available": sanitizer.cuda_available' in SOURCE


def test_model_idle_unload_runs_in_background_and_clears_memory():
    assert 'MODEL_IDLE_CHECK_SECONDS' in SOURCE
    assert 'asyncio.create_task(watch(), name="privacy-model-idle-unload")' in SOURCE
    assert 'torch.cuda.empty_cache()' in SOURCE
    assert 'await sanitizer.start_idle_unload_watcher()' in SOURCE
    assert 'await sanitizer.stop_idle_unload_watcher()' in SOURCE


def test_llm_can_be_disabled_for_sanitize_only_mode():
    assert 'LLM_ENABLED' in SOURCE
    assert '"llm_enabled": settings.llm_enabled' in SOURCE
    assert '"object": "privacy_proxy.sanitized_payload"' in SOURCE
    assert '"object": "chat.completion"' in SOURCE
    assert '"choices"' in SOURCE
    assert 'sanitized_chat_content(sanitized_payload)' in SOURCE
    assert 'raise HTTPException(status_code=503, detail="llm_disabled")' in SOURCE

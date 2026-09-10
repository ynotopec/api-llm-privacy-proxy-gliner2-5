import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

sys.modules.setdefault(
    "torch",
    SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    ),
)

from privacy_proxy_core.redaction import RedactionContext, RedactionStats
from privacy_proxy_core.sanitizers.gliner2 import GLiNER2Sanitizer
from privacy_proxy_core.settings import DEFAULT_PRIVACY_MODEL_ID, Settings


def test_gliner25_is_the_default_model(monkeypatch):
    monkeypatch.delenv("PRIVACY_MODEL_ID", raising=False)

    configured = Settings()

    assert DEFAULT_PRIVACY_MODEL_ID == "fastino/gliner2.5-multi-v1"
    assert configured.privacy_model_id == DEFAULT_PRIVACY_MODEL_ID


def test_privacy_model_can_still_be_configured(monkeypatch):
    monkeypatch.setenv("PRIVACY_MODEL_ID", "organization/custom-gliner2.5")

    assert Settings().privacy_model_id == "organization/custom-gliner2.5"


def test_gliner25_extract_entities_uses_scored_spans():
    sanitizer = GLiNER2Sanitizer(
        DEFAULT_PRIVACY_MODEL_ID,
        entity_types=["person", "email"],
        min_score=0.6,
    )
    sanitizer.model = Mock()
    sanitizer.model.extract_entities.return_value = [
        {"text": "Jane Doe", "label": "person", "score": 0.9, "start": 8, "end": 16}
    ]
    sanitizer.ensure_loaded = AsyncMock()

    async def run() -> tuple[str, RedactionStats]:
        stats = RedactionStats()
        output = await sanitizer.sanitize_text(
            "Contact Jane Doe", RedactionContext(), stats
        )
        return output, stats

    output, stats = asyncio.run(run())

    assert output == "Contact [PERSON_1]"
    assert stats.spans == 1
    sanitizer.model.extract_entities.assert_called_once_with(
        "Contact Jane Doe",
        ["person", "email"],
        threshold=0.6,
        include_confidence=True,
        include_spans=True,
    )

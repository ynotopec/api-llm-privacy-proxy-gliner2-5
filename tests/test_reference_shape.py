import asyncio
import sys
from pathlib import Path
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

APP_SOURCE = Path("app.py").read_text()
REQUIREMENTS = Path("requirements.txt").read_text().splitlines()


def test_gliner25_is_the_default_model(monkeypatch):
    monkeypatch.delenv("PRIVACY_MODEL_ID", raising=False)

    configured = Settings()

    assert DEFAULT_PRIVACY_MODEL_ID == "fastino/gliner2.5-multi-v1"
    assert configured.privacy_model_id == DEFAULT_PRIVACY_MODEL_ID


def test_gliner_runtime_dependencies_are_installed_explicitly():
    assert "numpy>=1.26.0,<3.0.0" in REQUIREMENTS
    assert "torch>=2.2.0,<3.0.0" in REQUIREMENTS
    assert "transformers>=4.46.0,<6.0.0" in REQUIREMENTS
    assert "accelerate>=1.0.0,<2.0.0" in REQUIREMENTS
    assert "peft>=0.13.0,<1.0.0" in REQUIREMENTS


def test_installer_resolves_the_lazy_gliner25_auto_extractor():
    installer = Path("install.sh").read_text()

    assert "from gliner2 import AutoExtractor" in installer


def test_sanitizer_loads_the_extractor_architecture():
    sanitizer_source = Path(
        "privacy_proxy_core/sanitizers/gliner2.py"
    ).read_text()

    assert "from gliner2 import AutoExtractor" in sanitizer_source
    assert "AutoExtractor.from_pretrained(self.model_id)" in sanitizer_source
    assert "GLiNER2.from_pretrained" not in sanitizer_source


def test_privacy_model_can_still_be_configured(monkeypatch):
    monkeypatch.setenv("PRIVACY_MODEL_ID", "organization/custom-gliner2.5")

    assert Settings().privacy_model_id == "organization/custom-gliner2.5"


def test_llm_enabled_setting_is_respected(monkeypatch):
    monkeypatch.setenv("UPSTREAM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LLM_ENABLED", "false")

    assert Settings().llm_enabled is False


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


def test_gliner25_label_keyed_result_is_parsed():
    sanitizer = GLiNER2Sanitizer(DEFAULT_PRIVACY_MODEL_ID, min_score=0.5)

    spans = sanitizer._parse_result(
        "Email jane@example.com",
        {
            "email": [
                {
                    "text": "jane@example.com",
                    "confidence": 0.95,
                    "span": [6, 22],
                }
            ]
        },
    )

    assert spans == [(6, 22, "email")]


def test_gliner25_canonical_nested_result_is_parsed():
    sanitizer = GLiNER2Sanitizer(DEFAULT_PRIVACY_MODEL_ID, min_score=0.5)

    spans = sanitizer._parse_result(
        "Email jane@example.com",
        {
            "entities": {
                "email": [
                    {
                        "text": "jane@example.com",
                        "confidence": 0.95,
                        "start": 6,
                        "end": 22,
                    }
                ]
            }
        },
    )

    assert spans == [(6, 22, "email")]


def test_proxy_exposes_startup_and_shutdown_lifecycle():
    assert "async def start_idle_watcher(self) -> None:" in APP_SOURCE
    assert "async def stop_idle_watcher(self) -> None:" in APP_SOURCE
    assert "await sanitizer.start_idle_watcher()" in APP_SOURCE
    assert "await sanitizer.stop_idle_watcher()" in APP_SOURCE


def test_idle_unload_can_run_without_application_settings(monkeypatch):
    sanitizer = GLiNER2Sanitizer(DEFAULT_PRIVACY_MODEL_ID)
    sanitizer.model = Mock()
    sanitizer._model_device = "cpu"
    sanitizer._last_used_at = 1.0
    monkeypatch.setenv("MODEL_IDLE_UNLOAD_SECONDS", "1")

    sanitizer.unload_if_idle()

    assert sanitizer.model is None
    assert sanitizer._model_device == "unloaded"

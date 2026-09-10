"""GLiNER2-based privacy sanitizer — used by gliner2 proxy variant."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
from typing import Any, List, Tuple

import torch
from privacy_proxy_core.redaction import PrivacySanitizerBase, RedactionContext, RedactionStats

log = logging.getLogger("privacy-proxy.gliner2")


class GLiNER2Sanitizer(PrivacySanitizerBase):
    """Entity extraction via GLiNER2.5.

    Uses ``extract_entities`` with threshold + include_spans.
    """

    def __init__(self, model_id: str, device: str = "auto",
                 torch_dtype_str: str = "auto",
                 entity_types: List[str] | None = None,
                 min_score: float = 0.50) -> None:
        super().__init__()
        self.model_id = model_id
        self.device = device
        self.torch_dtype_str = torch_dtype_str
        self.entity_types = entity_types or []
        self.min_score = min_score
        self.model: Any = None
        self._model_device: str = "unloaded"
        self._cuda_available: bool | None = None
        self._load_lock = asyncio.Lock()
        self._unload_task: asyncio.Task[None] | None = None

    def _on_idle_unload(self) -> None:
        log.info("Unloading GLiNER2 model after idle timeout")
        self.model = None
        self._model_device = "unloaded"
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def start_idle_watcher(self, check_interval: int = 30) -> None:
        """Background task to periodically check idle timeout."""
        if self._unload_task is not None:
            return

        interval = max(1, min(check_interval,
                              int(os.getenv("MODEL_IDLE_UNLOAD_SECONDS", "300"))))

        async def _watch() -> None:
            try:
                while True:
                    await asyncio.sleep(interval)
                    async with self._load_lock:
                        self.unload_if_idle()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("GLiNER2 idle watcher stopped")

        self._unload_task = asyncio.create_task(_watch(), name="gliner2-idle")

    async def stop_idle_watcher(self) -> None:
        task = self._unload_task
        self._unload_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._on_idle_unload()

    async def ensure_loaded(self) -> None:
        self.unload_if_idle()
        if self.model is not None:
            return
        async with self._load_lock:
            if self.model is not None:
                return
            device = self._resolve_device(self.device)
            log.info("Loading GLiNER2 model: %s on %s", self.model_id, device)

            from gliner2 import AutoExtractor
            from huggingface_hub import hf_hub_download

            try:
                config_path = hf_hub_download(self.model_id, "tokenizer_config.json", repo_type="model")
                with open(config_path, encoding="utf-8") as f:
                    tkn_cfg = json.load(f)
                if "extra_special_tokens" in tkn_cfg and isinstance(tkn_cfg["extra_special_tokens"], list):
                    existing = tkn_cfg.get("additional_special_tokens", [])
                    tkn_cfg["additional_special_tokens"] = list(
                        dict.fromkeys(existing + tkn_cfg["extra_special_tokens"])
                    )
                    tkn_cfg["extra_special_tokens"] = {}
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(tkn_cfg, f)
                    log.info("Patched tokenizer_config.json extra_special_tokens to dict format")
            except Exception:
                log.warning("Failed to patch tokenizer config, proceeding with original", exc_info=True)

            # Boundary extractors must be constructed on the target device.
            # Moving the fully initialized wrapper afterwards is unsupported
            # by some GLiNER2.5 releases and leaves the model on CPU.
            self.model = AutoExtractor.from_pretrained(
                self.model_id,
                map_location=device,
            )
            self._model_device = device
            self._touch()
            log.info("GLiNER2 loaded on %s", device)

    async def sanitize_text(
        self, text: str, ctx: RedactionContext, stats: RedactionStats,
    ) -> str:
        max_chars = int(os.getenv("MAX_STRING_CHARS", "200000"))

        if not text or len(text) > max_chars:
            return text

        await self.ensure_loaded()
        self._touch()

        try:
            result = self.model.extract_entities(
                text,
                self.entity_types,
                threshold=self.min_score,
                include_confidence=True,
                include_spans=True,
            )
        except Exception as exc:
            log.exception("GLiNER2 inference failed")
            raise RuntimeError(f"privacy_filter_failed: {exc}") from exc

        spans = self._parse_result(text, result)
        if not spans:
            return text

        return self._build_output(text, spans, ctx, stats)

    def _parse_result(self, text: str, result: Any) -> List[Tuple[int, int, str]]:
        spans: List[Tuple[int, int, str]] = []
        entities: list[Any]
        if isinstance(result, list):
            entities = result
        elif isinstance(result, dict):
            if isinstance(result.get("entities"), list):
                entities = result["entities"]
            else:
                entities = []
                # GLiNER2.5 returns its normal result as
                # {"entities": {"label": [{...}]}}.  Legacy checkpoints may
                # return the label mapping directly, so accept both shapes.
                grouped = result.get("entities", result)
                if not isinstance(grouped, dict):
                    return spans
                for label, values in grouped.items():
                    if not isinstance(values, list):
                        continue
                    for value in values:
                        if isinstance(value, dict):
                            entities.append({"label": label, **value})
                        elif isinstance(value, str):
                            entities.append({"label": label, "text": value})
        else:
            return spans

        for ent in entities:
            if not isinstance(ent, dict):
                continue
            score = float(ent.get("score", ent.get("confidence", 1.0)) or 0.0)
            if score < self.min_score:
                continue
            start = ent.get("start")
            end = ent.get("end")
            raw_span = ent.get("span")
            if (
                (start is None or end is None)
                and isinstance(raw_span, (list, tuple))
                and len(raw_span) == 2
            ):
                start, end = raw_span
            value = ent.get("text") or ent.get("value") or ent.get("word")
            label = ent.get("label") or ent.get("entity_group") or "private"

            if not isinstance(start, int) or not isinstance(end, int):
                if value:
                    idx = text.find(str(value))
                    if idx >= 0:
                        start, end = idx, idx + len(str(value))
                    else:
                        continue
                else:
                    continue

            start, end = int(start), int(end)
            if 0 <= start < end <= len(text):
                spans.append((start, end, str(label)))

        return spans

    def _build_output(
        self, text: str, spans: List[Tuple[int, int, str]],
        ctx: RedactionContext, stats: RedactionStats,
    ) -> str:
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        merged: List[Tuple[int, int, str]] = []
        for span in spans:
            if not merged or span[0] >= merged[-1][1]:
                merged.append(span)
            elif span[1] > merged[-1][1]:
                merged[-1] = (merged[-1][0], span[1], merged[-1][2])

        parts: list[str] = []
        last = 0
        for start, end, label in merged:
            original = text[start:end]
            parts.append(text[last:start])
            parts.append(ctx.placeholder(label, original))
            last = end
            stats.add(label, self.count_tokens(original))
        parts.append(text[last:])
        return "".join(parts)

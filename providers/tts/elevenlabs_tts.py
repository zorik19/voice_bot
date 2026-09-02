from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import numpy as np

from config import settings


class ElevenLabsTTS:
    """Стриминговый TTS через REST (без SDK — меньше зависимостей).
    Формат pcm_16000 — сразу наш внутренний, ресемплинг не нужен."""

    _URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30)

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        text = text.strip()
        if not text:
            return
        url = self._URL.format(voice_id=settings.elevenlabs_voice_id)
        async with self._client.stream(
            "POST",
            url,
            params={"output_format": "pcm_16000"},
            headers={"xi-api-key": settings.elevenlabs_api_key},
            json={
                "text": text,
                "model_id": settings.elevenlabs_model,
                "language_code": "ru",
            },
        ) as resp:
            resp.raise_for_status()
            tail = b""
            async for raw in resp.aiter_bytes(chunk_size=8192):
                raw = tail + raw
                if len(raw) % 2:  # int16 = 2 байта, бережём выравнивание
                    raw, tail = raw[:-1], raw[-1:]
                else:
                    tail = b""
                if raw:
                    yield np.frombuffer(raw, dtype=np.int16)

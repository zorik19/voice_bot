from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator

import httpx
import numpy as np

from audio.pronunciation import normalize_for_tts
from config import settings

_URL = "https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis"

# Запас ниже реального лимита Yandex TTS v3 на длину текста в одном запросе.
# Резать здесь, а не до normalize_for_tts: нормализация разворачивает числа
# в слова ("2100000" -> "два миллиона сто тысяч") и текст может вырасти в 2-3
# раза — резка "по сырому" тексту до нормализации лимит не гарантирует.
# Калибровано вживую: 230 символов ещё проходит, 260 уже падает с "Too long
# text" — берём 200 с запасом (другой микс пунктуации/тире может отличаться).
_MAX_CHARS_PER_REQUEST = 200


def _chunk_text(text: str, max_len: int = _MAX_CHARS_PER_REQUEST) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while len(text) > max_len:
        cut = text.rfind(" ", 0, max_len)
        cut = cut if cut > 0 else max_len
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


class YandexTTS:
    """SpeechKit TTS v3: голоса нового поколения с амплуа.
    Ответ — поток JSON-строк с base64-чанками LPCM 16kHz."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30)

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        text = normalize_for_tts(text.strip())
        if not text:
            return
        for piece in _chunk_text(text):
            async for chunk in self._synthesize_one(piece):
                yield chunk

    async def _synthesize_one(self, text: str) -> AsyncIterator[np.ndarray]:
        hints = [{"voice": settings.yandex_voice}, {"speed": settings.yandex_speed}]
        if settings.yandex_role:
            hints.append({"role": settings.yandex_role})
        body = {
            "text": text,
            "hints": hints,
            "outputAudioSpec": {
                "rawAudio": {
                    "audioEncoding": "LINEAR16_PCM",
                    "sampleRateHertz": str(settings.sample_rate),
                }
            },
            "loudnessNormalizationType": "LUFS",
        }
        async with self._client.stream(
            "POST",
            _URL,
            headers={"Authorization": f"Api-Key {settings.yandex_api_key}"},
            json=body,
        ) as resp:
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Yandex TTS v3 {resp.status_code}: "
                    f"{(await resp.aread()).decode('utf-8', 'replace')}"
                )
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                data = payload.get("result", {}).get("audioChunk", {}).get("data")
                if data:
                    yield np.frombuffer(base64.b64decode(data), dtype=np.int16)
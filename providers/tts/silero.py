from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import torch

from audio.io import resample
from config import settings


class SileroTTS:
    """Полностью локальный русский TTS, ноль ключей, ~50-150мс на фразу (CPU).
    Модель качается с серверов Silero при первом запуске (доступны из РФ)."""

    def __init__(self) -> None:
        self._model, _ = torch.hub.load(
            "snakers4/silero-models",
            "silero_tts",
            language="ru",
            speaker="v4_ru",
            trust_repo=True,
        )
        self._model.to(torch.device("cpu"))

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        text = text.strip()
        if not text:
            return
        audio_f32: torch.Tensor = await asyncio.to_thread(
            self._model.apply_tts,
            text=text,
            speaker=settings.silero_speaker,
            sample_rate=settings.silero_sample_rate,
        )
        pcm16 = (audio_f32.numpy() * 32767).astype(np.int16)
        pcm16 = resample(pcm16, settings.silero_sample_rate, settings.sample_rate)
        # Silero отдаёт фразу целиком; режем на чанки ~0.5с, чтобы barge-in
        # мог остановить воспроизведение между ними
        chunk = settings.sample_rate // 2
        for i in range(0, len(pcm16), chunk):
            yield pcm16[i : i + chunk]

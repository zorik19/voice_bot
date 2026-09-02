from __future__ import annotations

import asyncio

import numpy as np
from faster_whisper import WhisperModel

from config import settings
from core.types import Transcript


class WhisperSTT:
    """Локальный STT. Модель скачается с HuggingFace при первом запуске.

    Whisper — батчевый, не стриминговый: распознаём фразу целиком после того,
    как VAD определил её конец. Для локального демо этого достаточно; настоящий
    стриминг появится с Yandex SpeechKit (providers/stt/yandex.py)."""

    def __init__(self) -> None:
        device = settings.whisper_device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self._model = WhisperModel(
            settings.whisper_model,
            device=device,
            compute_type=settings.whisper_compute,
        )

    async def transcribe(self, audio_pcm16: np.ndarray) -> Transcript:
        audio_f32 = audio_pcm16.astype(np.float32) / 32768.0
        # CPU-bound — уводим в поток, event loop не блокируем
        segments, _info = await asyncio.to_thread(
            self._model.transcribe,
            audio_f32,
            language="ru",
            beam_size=1,           # скорость важнее на демо
            vad_filter=False,      # VAD уже отработал снаружи
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return Transcript(text=text, is_final=True)

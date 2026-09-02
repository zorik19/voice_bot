from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

import numpy as np

from core.types import LLMToken, Message, ToolCall, Transcript


class STTProvider(Protocol):
    async def transcribe(self, audio_pcm16: np.ndarray) -> Transcript:
        """Распознать законченную фразу (int16 mono, sample_rate из конфига)."""
        ...


class LLMProvider(Protocol):
    def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMToken | ToolCall]:
        """Стримит текст токенами; tool call отдаёт целиком, когда собран."""
        ...


class TTSProvider(Protocol):
    def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        """Текст -> чанки PCM int16 mono, sample_rate из конфига (16kHz)."""
        ...


class AudioTransport(Protocol):
    """Источник/приёмник аудио: микрофон+колонки, файл, AudioSocket (Asterisk).
    Ресемплинг во внутренние 16kHz — зона ответственности транспорта."""

    def input_stream(self) -> AsyncIterator[np.ndarray]:
        """Входящие чанки PCM int16 16kHz (кадры по frame_ms)."""
        ...

    async def play(self, pcm16: np.ndarray) -> None:
        """Отправить чанк на воспроизведение (неблокирующе)."""
        ...

    def stop_playback(self) -> None:
        """Мгновенно заглушить воспроизведение и сбросить очередь (barge-in)."""
        ...

    @property
    def is_playing(self) -> bool: ...

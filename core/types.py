from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any


class DialogState(StrEnum):
    IDLE = auto()
    LISTENING = auto()   # ждём/слушаем речь клиента
    THINKING = auto()    # LLM генерирует ответ
    SPEAKING = auto()    # бот говорит, параллельно следим за barge-in


@dataclass(slots=True)
class AudioChunk:
    """Внутренний формат пайплайна: PCM mono int16.
    Частота — settings.sample_rate (16kHz). Ресемплинг — забота транспорта."""
    data: bytes
    timestamp_ms: int = 0


@dataclass(slots=True)
class Transcript:
    text: str
    is_final: bool = True
    confidence: float | None = None


@dataclass(slots=True)
class LLMToken:
    text: str


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass(slots=True)
class Message:
    role: str  # system | user | assistant | tool
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

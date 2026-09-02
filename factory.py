from __future__ import annotations

from config import settings
from core.interfaces import LLMProvider, STTProvider, TTSProvider


def make_stt() -> STTProvider:
    match settings.stt_provider:
        case "whisper":
            from providers.stt.whisper_local import WhisperSTT
            return WhisperSTT()
        case "yandex":
            from providers.stt.yandex import YandexSTT
            return YandexSTT()
    raise ValueError(f"Неизвестный STT: {settings.stt_provider}")


def make_llm() -> LLMProvider:
    match settings.llm_provider:
        case "claude":
            from providers.llm.claude import ClaudeLLM
            return ClaudeLLM()
        case "gigachat":
            from providers.llm.gigachat_llm import GigaChatLLM
            return GigaChatLLM()
    raise ValueError(f"Неизвестный LLM: {settings.llm_provider}")


def make_tts() -> TTSProvider:
    match settings.tts_provider:
        case "silero":
            from providers.tts.silero import SileroTTS
            return SileroTTS()
        case "elevenlabs":
            from providers.tts.elevenlabs_tts import ElevenLabsTTS
            return ElevenLabsTTS()
        case "yandex":
            from providers.tts.yandex import YandexTTS
            return YandexTTS()
    raise ValueError(f"Неизвестный TTS: {settings.tts_provider}")

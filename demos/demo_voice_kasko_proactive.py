"""Полное голосовое демо: проактивный обзвон "посчитал КАСКО и не оплатил".

Тот же пайплайн, что и demo_voice.py (микрофон -> VAD -> STT -> LLM -> TTS ->
колонки, с фоновым офисным шумом и barge-in), но со сценарием
prompts.kasko_proactive вместо консультации по действующему полису.

    python -m demos.demo_voice_kasko_proactive

Важно: используй НАУШНИКИ, иначе бот будет слышать сам себя через микрофон
и barge-in-защита сработает не идеально. Ctrl+C — завершить звонок.
"""
from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, ".")

from audio.io import LocalAudioTransport
from domain.kasko_offer import sample_offer
from factory import make_llm, make_stt, make_tts
from orchestrator import Orchestrator
from prompts.kasko_proactive import build_greeting, build_system_prompt
from rag.rag_store import KnowledgeBase


async def main() -> None:
    print("Загружаю модели (первый запуск скачает whisper и silero)...")
    offer = sample_offer()
    orchestrator = Orchestrator(
        stt=make_stt(),
        llm=make_llm(),
        tts=make_tts(),
        transport=LocalAudioTransport(),
        system_prompt=build_system_prompt(offer),
        greeting=build_greeting(offer),
        kb=KnowledgeBase("knowledge/kasko_proactive.yaml"),
    )
    print("Готово. Звонок начался — говори в микрофон.\n")
    await orchestrator.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nЗвонок завершён.")
"""Полное голосовое демо: микрофон -> VAD -> STT -> LLM -> TTS -> колонки.

    python -m demos.demo_voice

Важно: используй НАУШНИКИ, иначе бот будет слышать сам себя через микрофон
и barge-in-защита сработает не идеально (для колонок пороги в конфиге придётся
поднять). Ctrl+C — завершить звонок.
"""
from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, ".")

from audio.io import LocalAudioTransport

from factory import make_llm, make_stt, make_tts
from orchestrator import Orchestrator
from prompts.kasko_faq import ClientCard, build_system_prompt, GREETING_TEMPLATE
from rag.rag_store import KnowledgeBase


async def main() -> None:
    print("Загружаю модели (первый запуск скачает whisper и silero)...")
    card = ClientCard(name_io="Анатолий Михайлович", car="Kia Sportage 2021")
    orchestrator = Orchestrator(
        stt=make_stt(),
        llm=make_llm(),
        tts=make_tts(),
        transport=LocalAudioTransport(),
        system_prompt=build_system_prompt(card),
        greeting=GREETING_TEMPLATE.format(name_io=card.name_io),
        kb=KnowledgeBase("knowledge/kasko_faq.yaml"),
    )
    print("Готово. Звонок начался — говори в микрофон.\n")
    await orchestrator.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nЗвонок завершён.")

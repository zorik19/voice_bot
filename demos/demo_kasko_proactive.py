"""Текстовое демо: проактивный обзвон "клиент посчитал КАСКО и не оплатил".

Реплики клиента вводятся с клавиатуры, бот отвечает голосом. Данные клиента —
фикстура domain.kasko_offer.sample_offer (текущий расчёт + оптимальные условия
+ таблица готовых альтернатив), в проде приходят одним запросом к внутреннему
API на старте звонка.




                # с озвучкой
    python -m demos.demo_kasko_proactive --no-tts   # чисто текстом (отладка логики)
"""
from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, ".")

import numpy as np
import sounddevice as sd

from config import settings
from core.call_log import CallLogger
from core.types import LLMToken, Message, ToolCall
from domain.kasko_offer import sample_offer
from factory import make_llm, make_tts
from prompts.kasko_proactive import build_greeting, build_system_prompt
from rag.rag_store import KnowledgeBase

_SENTENCE_END = (".", "!", "?", "…")
_MAX_CHUNK_LEN = 220


def _split_sentences(text: str) -> list[str]:
    """Дробим на предложения перед TTS: длинный ответ одним куском может
    упереться в лимит длины запроса конкретного провайдера (видели на
    Yandex TTS v3 — 400 'Too long text'). Предложение без точки длиннее
    _MAX_CHUNK_LEN дополнительно режем по границе слова — точку ждать
    бесконечно нельзя."""
    parts: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in _SENTENCE_END and len(buf.strip()) > 3:
            parts.append(buf.strip())
            buf = ""
    if buf.strip():
        parts.append(buf.strip())

    chunks: list[str] = []
    for part in parts:
        while len(part) > _MAX_CHUNK_LEN:
            cut = part.rfind(" ", 0, _MAX_CHUNK_LEN)
            cut = cut if cut > 40 else _MAX_CHUNK_LEN
            chunks.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            chunks.append(part)
    return chunks


async def main(use_tts: bool) -> None:
    llm = make_llm()
    tts = make_tts() if use_tts else None
    log = CallLogger()
    kb = KnowledgeBase("knowledge/kasko_proactive.yaml")
    offer = sample_offer()
    history = [Message(role="system", content=build_system_prompt(offer))]

    GREETING = build_greeting(offer)

    async def speak(text: str) -> None:
        print(f"\n🤖 Анна: {text}")
        log.event("assistant", text=text)
        if tts is None:
            return
        for sentence in _split_sentences(text):
            try:
                chunks = [c async for c in tts.synthesize_stream(sentence)]
            except Exception as exc:  # не роняем демо из-за одного фрагмента TTS
                log.event("tts_error", error=repr(exc))
                print(f"   ⚠️ TTS ошибка на фрагменте: {exc}")
                continue
            if chunks:
                sd.play(np.concatenate(chunks), samplerate=settings.sample_rate, blocking=True)

    async def llm_turn() -> None:
        text = ""
        async for item in llm.stream_chat(history):
            if isinstance(item, LLMToken):
                text += item.text
            elif isinstance(item, ToolCall):
                pass  # в этом сценарии тулов нет, на всякий случай игнорируем
        if text.strip():
            await speak(text.strip())
        history.append(Message(role="assistant", content=text))

    history.append(Message(role="assistant", content=GREETING))
    await speak(GREETING)

    while True:
        try:
            user = input("\n👤 Клиент: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.lower() in ("выход", "exit", "quit"):
            break
        log.event("user", text=user)

        items = await kb.search(user)
        user_content = user
        if items:
            log.event("rag", hits=[(it.id, round(it.score, 3)) for it in items])
            print(f"   🔎 RAG: {[(it.id, round(it.score, 3)) for it in items]}")
            user_content = f"{user}\n\n{KnowledgeBase.format_context(items)}"

        history.append(Message(role="user", content=user_content))
        await llm_turn()

    print(f"\nЛог звонка: {log.path}")


if __name__ == "__main__":
    asyncio.run(main(use_tts="--no-tts" not in sys.argv))
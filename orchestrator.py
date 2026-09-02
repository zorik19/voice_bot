from __future__ import annotations

import asyncio
import contextlib
import time

from audio.vad import BargeInDetector, StreamingVAD
from config import settings
from core.call_log import CallLogger
from core.interfaces import AudioTransport, LLMProvider, STTProvider, TTSProvider
from core.types import LLMToken, Message, ToolCall
from rag.rag_store import KnowledgeBase

SENTENCE_END = (".", "!", "?", "…")

# Переспросы и прощание при тишине — общие для любого голосового сценария
REPROMPTS = [
    "Простите, я вас не расслышала. Повторите, пожалуйста.",
    "Кажется, связь прерывается. Вы меня слышите?",
]
GOODBYE_NO_ANSWER = "Похоже, связь пропала. Я перезвоню позже, всего доброго!"


class Orchestrator:
    """Главный цикл звонка.

    Состояния неявные, через факты:
      - идёт response-task или играет звук  -> SPEAKING (следим за barge-in)
      - иначе                               -> LISTENING (крутим VAD)
    THINKING живёт внутри response-task между STT и первым звуком.
    """

    def __init__(
        self,
        stt: STTProvider,
        llm: LLMProvider,
        tts: TTSProvider,
        transport: AudioTransport,
        system_prompt: str,
        greeting: str,
        kb: KnowledgeBase | None = None,
    ) -> None:
        self._stt, self._llm, self._tts, self._transport = stt, llm, tts, transport
        self._log = CallLogger()
        self._response_task: asyncio.Task | None = None
        self._reprompts = 0
        self._kb = kb

        self._history: list[Message] = [Message(role="system", content=system_prompt)]
        self._greeting = greeting

    # ------------------------------------------------------------- main loop

    async def run(self) -> None:
        await self._transport.start()
        self._log.event("call_start")

        # приветствие — фиксированное, без LLM: мгновенный старт
        self._history.append(Message(role="assistant", content=self._greeting))
        self._response_task = asyncio.create_task(self._speak(self._greeting))

        vad = StreamingVAD()
        barge = BargeInDetector()
        last_activity = time.monotonic()

        was_speaking = False
        async for frame in self._transport.input_stream():
            if self._is_speaking:
                was_speaking = True
                if barge.process(frame):
                    self._log.event("barge_in")
                    await self._cancel_response()
                    barge.reset()
                    vad._reset()
                    last_activity = time.monotonic()
                continue

            if was_speaking:
                # бот только что договорил — клиенту нужно время на ответ
                was_speaking = False
                last_activity = time.monotonic()

            barge.reset()
            ev = vad.process(frame)
            if ev.is_speech:
                last_activity = time.monotonic()

            if vad.utterance_ended:
                audio = vad.pop_utterance()
                self._reprompts = 0
                self._response_task = asyncio.create_task(self._respond(audio))
                last_activity = time.monotonic()
            elif time.monotonic() - last_activity > settings.silence_timeout_s:
                if self._reprompts >= settings.max_reprompts:
                    self._log.event("call_end", reason="silence")
                    await self._speak(GOODBYE_NO_ANSWER)
                    return
                text = REPROMPTS[min(self._reprompts, len(REPROMPTS) - 1)]
                self._reprompts += 1
                self._log.event("reprompt", n=self._reprompts)
                self._response_task = asyncio.create_task(self._speak(text))
                last_activity = time.monotonic()

    @property
    def _is_speaking(self) -> bool:
        task_alive = self._response_task is not None and not self._response_task.done()
        return task_alive or self._transport.is_playing

    async def _cancel_response(self) -> None:
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._response_task
        self._transport.stop_playback()

    # -------------------------------------------------------------- response

    async def _respond(self, audio_pcm16) -> None:
        t0 = time.monotonic()
        transcript = await self._stt.transcribe(audio_pcm16)
        self._log.event(
            "stt", text=transcript.text,
            latency_ms=round((time.monotonic() - t0) * 1000),
        )
        if not transcript.text:
            return

        rag_context = ""
        if self._kb:
            items = await self._kb.search(transcript.text)
            if items:
                self._log.event("rag", hits=[(i.id, round(i.score, 3)) for i in items])
                rag_context = KnowledgeBase.format_context(items)

        user_content = transcript.text
        if rag_context:
            user_content = f"{transcript.text}\n\n{rag_context}"
        self._history.append(Message(role="user", content=user_content))

        spoken_parts: list[str] = []
        try:
            await self._llm_turn(spoken_parts, t0)
        except asyncio.CancelledError:
            # barge-in: фиксируем в истории то, что успели сказать,
            # иначе LLM будет считать, что договорил
            if spoken_parts:
                self._history.append(Message(
                    role="assistant",
                    content=" ".join(spoken_parts) + " [клиент перебил]",
                ))
            raise
        except Exception as exc:  # не роняем звонок из-за одной ошибки
            self._log.event("error", error=repr(exc))
            await self._speak(
                "Простите, у меня небольшие технические неполадки. "
                "Повторите, пожалуйста, ещё раз."
            )

    async def _llm_turn(self, spoken_parts: list[str], t0: float) -> None:
        """Один проход LLM. В FAQ-сценарии тулов нет — просто стримим ответ.

        Приём токенов от LLM и синтез+воспроизведение речи идут двумя
        параллельными корутинами через очередь, а не одна за другой в одном
        потоке управления: иначе пока идёт сетевой запрос к TTS за одно
        предложение, оркестратор не читает следующие токены от LLM вообще —
        генерация следующего предложения буквально ставится на паузу, и между
        фразами в динамиках повисает тишина. Тут TTS-запрос за предложение N
        и генерация предложения N+1 текут одновременно."""
        sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
        full_text_parts: list[str] = []
        first_audio_logged = False

        async def _produce() -> None:
            buffer = ""
            async for item in self._llm.stream_chat(self._history):
                if isinstance(item, LLMToken):
                    buffer += item.text
                    full_text_parts.append(item.text)
                    while sentence := self._pop_sentence(buffer):
                        buffer = buffer[len(sentence):]
                        await sentence_queue.put(sentence)
                elif isinstance(item, ToolCall):
                    pass  # тулов в этом сценарии нет, на всякий случай игнорируем
            if buffer.strip():
                await sentence_queue.put(buffer)
            await sentence_queue.put(None)  # сигнал конца потока

        async def _consume() -> None:
            nonlocal first_audio_logged
            while (sentence := await sentence_queue.get()) is not None:
                if not first_audio_logged:
                    self._log.event(
                        "first_sentence",
                        latency_ms=round((time.monotonic() - t0) * 1000),
                    )
                    first_audio_logged = True
                spoken_parts.append(sentence.strip())
                await self._speak(sentence, log=False)

        await asyncio.gather(_produce(), _consume())

        full_text = "".join(full_text_parts)
        if full_text.strip():
            self._log.event("assistant", text=full_text.strip())

        self._history.append(Message(role="assistant", content=full_text))

    @staticmethod
    def _pop_sentence(buffer: str) -> str | None:
        """Вернуть первый достаточно длинный законченный кусок из буфера.
        min_len не даёт резать на коротких обрывках — так TTS получает более
        цельные фразы и звучит связнее, а не «слово — пауза — слово».
        max_len — твёрдый потолок: если модель выдаёт длинное предложение без
        точки, ждать её бесконечно нельзя — упрёмся в лимит длины запроса TTS
        (видели 'Too long text' от Yandex TTS v3)."""
        min_len = 40
        max_len = 220
        for i, ch in enumerate(buffer):
            if ch in SENTENCE_END and i >= max(3, min_len):
                return buffer[: i + 1]
        if len(buffer) >= max_len:
            cut = buffer.rfind(" ", 0, max_len)
            return buffer[: cut if cut > min_len else max_len]
        return None

    # ------------------------------------------------------------------- tts

    async def _speak(self, text: str, log: bool = True) -> None:
        if log:
            self._log.event("assistant", text=text.strip())
        async for chunk in self._tts.synthesize_stream(text):
            await self._transport.play(chunk)
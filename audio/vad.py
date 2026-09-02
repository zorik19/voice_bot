from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from silero_vad import load_silero_vad

from config import settings

_FRAME_SAMPLES = 512  # требование silero-vad для 16kHz


@dataclass(slots=True)
class VADEvent:
    is_speech: bool
    prob: float


class StreamingVAD:
    """Покадровый VAD + детекция конца фразы.

    Использование:
        vad = StreamingVAD()
        for frame in frames:                # frame: int16, ровно 512 сэмплов
            ev = vad.process(frame)
            if vad.utterance_ended:
                audio = vad.pop_utterance() # вся фраза целиком, int16
    """

    def __init__(
        self,
        threshold: float | None = None,
        silence_ms: int | None = None,
        min_speech_ms: int | None = None,
    ) -> None:
        self._model = load_silero_vad()
        self._threshold = threshold or settings.vad_threshold
        self._silence_frames = (silence_ms or settings.vad_silence_ms) // settings.frame_ms
        self._min_speech_frames = (
            (min_speech_ms or settings.vad_min_speech_ms) // settings.frame_ms
        )

        self._speech_frames = 0
        self._silent_frames = 0
        self._in_utterance = False
        self._buffer: list[np.ndarray] = []
        self.utterance_ended = False

    def process(self, frame_int16: np.ndarray) -> VADEvent:
        if len(frame_int16) != _FRAME_SAMPLES:
            raise ValueError(f"VAD ждёт кадры по {_FRAME_SAMPLES} сэмплов")

        frame_f32 = frame_int16.astype(np.float32) / 32768.0
        prob = self._model(torch.from_numpy(frame_f32), settings.sample_rate).item()
        is_speech = prob >= self._threshold

        if is_speech:
            self._speech_frames += 1
            self._silent_frames = 0
            if not self._in_utterance and self._speech_frames >= 2:
                self._in_utterance = True
        else:
            self._silent_frames += 1
            self._speech_frames = 0

        if self._in_utterance:
            self._buffer.append(frame_int16)
            if self._silent_frames >= self._silence_frames:
                total_speech = len(self._buffer) - self._silent_frames
                if total_speech >= self._min_speech_frames:
                    self.utterance_ended = True
                else:  # слишком короткий всплеск — шум, сбрасываем
                    self._reset()

        return VADEvent(is_speech=is_speech, prob=prob)

    def pop_utterance(self) -> np.ndarray:
        audio = np.concatenate(self._buffer) if self._buffer else np.array([], np.int16)
        self._reset()
        return audio

    def _reset(self) -> None:
        self._buffer.clear()
        self._in_utterance = False
        self._speech_frames = 0
        self._silent_frames = 0
        self.utterance_ended = False
        self._model.reset_states()


class BargeInDetector:
    """Отдельный VAD с жёсткими порогами: пока бот говорит, клиент должен
    говорить громче и дольше, чтобы считаться перебившим (защита от эха)."""

    def __init__(self) -> None:
        self._model = load_silero_vad()
        self._needed = settings.barge_in_min_speech_ms // settings.frame_ms
        self._streak = 0

    def process(self, frame_int16: np.ndarray) -> bool:
        frame_f32 = frame_int16.astype(np.float32) / 32768.0
        prob = self._model(torch.from_numpy(frame_f32), settings.sample_rate).item()
        self._streak = self._streak + 1 if prob >= settings.barge_in_threshold else 0
        return self._streak >= self._needed

    def reset(self) -> None:
        self._streak = 0
        self._model.reset_states()

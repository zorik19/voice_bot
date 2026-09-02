from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator

import numpy as np
import sounddevice as sd

from config import settings
from pathlib import Path

FRAME_SAMPLES = settings.sample_rate * settings.frame_ms // 1000  # 512


def resample(pcm16: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return pcm16
    n_out = int(len(pcm16) * dst_rate / src_rate)
    x_out = np.linspace(0, len(pcm16) - 1, n_out)
    return np.interp(x_out, np.arange(len(pcm16)), pcm16).astype(np.int16)


class LocalAudioTransport:
    """Микрофон -> кадры int16 16kHz; постоянный OutputStream для бесшовного
    воспроизведения чанков + мгновенный стоп (barge-in)."""

    def __init__(self) -> None:
        self._in_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=200)
        self._out_buf = bytearray()
        self._out_lock = threading.Lock()
        self._playing = asyncio.Event()
        self._stop_flag = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._in_stream: sd.InputStream | None = None
        self._out_stream: sd.OutputStream | None = None
        self._player_task: asyncio.Task | None = None

        # --- фоновый шум офиса ---
        self._amb = np.zeros(0, dtype=np.int16)
        self._amb_pos = 0
        self._amb_gain = 0.0  # текущий уровень, плавно ползёт к целевому
        amb_path = Path("assets/office_ambience.raw")
        if amb_path.exists():
            self._amb = np.frombuffer(amb_path.read_bytes(), dtype=np.int16)

    # ---------- вход (микрофон) ----------

    def _on_input(self, indata: np.ndarray, frames: int, time_, status) -> None:
        frame = indata[:, 0].copy()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._put_nowait_safe, frame)

    def _on_output(self, outdata: np.ndarray, frames: int, time_, status) -> None:
        need = frames * 2  # int16 = 2 байта
        with self._out_lock:
            take = bytes(self._out_buf[:need])
            del self._out_buf[:need]
            speaking = len(take) > 0
        chunk = np.frombuffer(take, dtype=np.int16)

        out = np.zeros(frames, dtype=np.float32)
        out[: len(chunk)] = chunk

        if len(self._amb):
            target = settings.ambience_gain if speaking else settings.ambience_gain_idle
            step = 1.0 / (0.8 * settings.sample_rate)  # плавный переход ~0.8с
            amb = np.empty(frames, dtype=np.float32)
            for i in range(frames):
                self._amb_gain += np.clip(target - self._amb_gain, -step, step)
                amb[i] = self._amb[self._amb_pos] * self._amb_gain
                self._amb_pos = (self._amb_pos + 1) % len(self._amb)
            out += amb

        outdata[:, 0] = np.clip(out, -32768, 32767).astype(np.int16)

    def _put_nowait_safe(self, frame: np.ndarray) -> None:
        try:
            self._in_queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass

    async def input_stream(self) -> AsyncIterator[np.ndarray]:
        self._loop = asyncio.get_running_loop()
        self._in_stream = sd.InputStream(
            samplerate=settings.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            callback=self._on_input,
        )
        self._in_stream.start()
        try:
            while True:
                yield await self._in_queue.get()
        finally:
            self._in_stream.stop()
            self._in_stream.close()

    # ---------- выход (колонки) ----------

    async def start(self) -> None:
        self._out_stream = sd.OutputStream(
            samplerate=settings.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            callback=self._on_output,
        )
        self._out_stream.start()


    async def play(self, pcm16: np.ndarray) -> None:
        with self._out_lock:
            self._out_buf += pcm16.tobytes()

    def stop_playback(self) -> None:
        with self._out_lock:
            self._out_buf.clear()

    @property
    def is_playing(self) -> bool:
        with self._out_lock:
            return len(self._out_buf) > 0
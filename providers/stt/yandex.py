from __future__ import annotations

import numpy as np

from core.types import Transcript


class YandexSTT:
    """TODO (прод-этап): Yandex SpeechKit STT v3, gRPC streaming.

    Почему он для прода: настоящий стриминг (partial-транскрипты по ходу речи)
    и модели, натренированные на телефонных звонках 8kHz — то, чего не даёт Whisper.

    План:
      - pip install yandex-speechkit  (или grpcio + сгенерённые stubs)
      - RecognizeStreaming: шлём чанки PCM, получаем partial/final
      - авторизация: Api-Key + folder_id из config
    Док: https://yandex.cloud/ru/docs/speechkit/stt/api/streaming-api-v3
    """

    def __init__(self) -> None:
        raise NotImplementedError("Yandex STT — прод-этап, см. docstring")

    async def transcribe(self, audio_pcm16: np.ndarray) -> Transcript:
        raise NotImplementedError

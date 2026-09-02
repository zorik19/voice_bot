from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="VOICEBOT_", extra="ignore"
    )

    # --- выбор провайдеров ---
    stt_provider: str = "whisper"      # whisper | yandex
    llm_provider: str = "gigachat"     # gigachat | claude
    tts_provider: str = "silero"       # silero | elevenlabs | yandex

    # --- GigaChat ---
    gigachat_auth_key: str = ""        # Authorization key из ЛК developers.sber.ru
    gigachat_scope: str = "GIGACHAT_API_PERS"   # PERS — физлицо, CORP/B2B — юрлицо
    gigachat_model: str = "GigaChat-2-Pro"      # если недоступна — "GigaChat"

    # --- Claude ---
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"
    claude_max_tokens: int = 400       # реплики короткие, длиннее не нужно

    # --- ElevenLabs ---
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "EXAVITQu4vr4xnSDxMaL"  # заменишь на свой
    elevenlabs_model: str = "eleven_flash_v2_5"

    # --- Yandex SpeechKit (прод, TODO) ---
    yandex_api_key: str = ""
    yandex_folder_id: str = ""
    yandex_voice: str = "masha"  # marina | lera | dasha | masha | alexander | anton
    yandex_role: str = ""  # friendly | good | neutral (не у всех голосов)
    yandex_speed: float = 1.1

    # --- Whisper ---
    whisper_model: str = "small"       # small | medium; medium точнее, но медленнее
    whisper_device: str = "auto"       # auto | cpu | cuda
    whisper_compute: str = "int8"      # int8 на CPU/M-серии, float16 на CUDA

    # --- Silero TTS ---
    silero_speaker: str = "xenia"      # xenia | baya | kseniya | aidar | eugene
    silero_sample_rate: int = 24_000

    # --- аудио-пайплайн (внутренний формат: PCM 16kHz mono int16) ---
    sample_rate: int = 16_000
    frame_ms: int = 32                 # 512 сэмплов @16kHz — размер окна silero-vad

    # --- VAD, профиль mic / phone ---
    audio_profile: str = "mic"
    vad_threshold: float = 0.5         # порог "это речь" в LISTENING
    vad_silence_ms: int = 700          # тишина после речи = конец фразы
    vad_min_speech_ms: int = 250       # короче — считаем шумом

    # --- barge-in (защита от самоперебивания) ---
    barge_in_threshold: float = 0.75   # порог выше, чем обычный VAD
    barge_in_min_speech_ms: int = 400  # клиент должен говорить дольше, чтобы перебить

    # --- диалог ---
    silence_timeout_s: float = 6.0     # клиент молчит — переспрашиваем
    max_reprompts: int = 2             # исчерпали — вежливо прощаемся

    log_dir: str = "logs"

    ambience_gain: float = 0.07  # громкость офисного фона, 0 = выключить
    ambience_gain_idle: float = 0.03  # фон, когда бот молчит (слушает/думает)

    def apply_profile(self) -> None:
        """Профиль phone: телефонный канал шумнее — пороги жёстче."""
        if self.audio_profile == "phone":
            self.vad_threshold = 0.6
            self.vad_silence_ms = 800
            self.barge_in_threshold = 0.85
            self.barge_in_min_speech_ms = 500


settings = Settings()
settings.apply_profile()

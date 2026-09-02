from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from config import settings


class CallLogger:
    """Каждый звонок — отдельный JSONL: все реплики, тайминги, tool calls.
    Основа для будущей аналитики и разбора неудачных диалогов."""

    def __init__(self) -> None:
        Path(settings.log_dir).mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self._path = Path(settings.log_dir) / f"call_{stamp}.jsonl"
        self._t0 = time.monotonic()

    def event(self, kind: str, **payload: Any) -> None:
        record = {
            "t_ms": round((time.monotonic() - self._t0) * 1000),
            "kind": kind,
            **payload,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @property
    def path(self) -> Path:
        return self._path

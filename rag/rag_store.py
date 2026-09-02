from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

# multilingual-e5: требует префиксы "query:" / "passage:" — это не магия,
# модель так обучена различать запрос и документ
_MODEL_NAME = "intfloat/multilingual-e5-small"

# Короткие фатические реплики ("да", "это я", "не уверен", "алло") — на них
# эмбеддинг короткой фразы часто ловит случайное совпадение по косинусу
# с посторонним триггером (см. лог: "да, это я" -> already_insured 0.858).
# Материал скрипта им не нужен: ход диалога по шагам ведёт сам системный
# промпт, а не RAG. Список — закрытый набор частиц/связок, не открытый
# словарь возражений: "дорого", "нет машины" и т.п. сюда не попадают
# и продолжают уходить в обычный поиск.
_PUNCT_RE = re.compile(r"[^\w\s-]", re.UNICODE)
_FILLER_WORDS = {
    "да", "нет", "не", "неа", "ага", "угу", "ну", "ладно", "хорошо", "окей", "ок",
    "конечно", "понятно", "принято", "ясно", "верно", "именно", "так",
    "наверное", "знаю", "уверен", "уверена", "спасибо", "пожалуйста",
    "это", "я", "вы", "слушаю", "алло", "мм",
}
_MAX_FILLER_WORDS = 4


def _is_filler(text: str) -> bool:
    words = _PUNCT_RE.sub("", text.lower()).split()
    if not words or len(words) > _MAX_FILLER_WORDS:
        return False
    return all(w in _FILLER_WORDS for w in words)


@dataclass(slots=True)
class KBItem:
    id: str
    type: str
    response: str
    score: float = 0.0


class KnowledgeBase:
    """RAG по скрипту: реплика клиента -> релевантные блоки возражений/FAQ.

    Эмбеддятся ТРИГГЕРЫ (варианты реплик клиента), не ответы: матчинг
    реплика-с-репликой точнее, чем реплика-с-ответом. Хранилище — numpy
    в памяти: на 4 сценария и ~150 триггеров векторная БД не нужна."""

    def __init__(self, kb_path: str | Path) -> None:
        data = yaml.safe_load(Path(kb_path).read_text(encoding="utf-8"))
        self.scenario: str = data["scenario"]
        self._items: list[dict] = data["items"]

        self._model = SentenceTransformer(_MODEL_NAME)
        trigger_texts: list[str] = []
        self._trigger_to_item: list[int] = []  # индекс триггера -> индекс item
        for i, item in enumerate(self._items):
            for trig in item["triggers"]:
                trigger_texts.append(f"query: {trig}")
                self._trigger_to_item.append(i)
        self._vectors = self._model.encode(
            trigger_texts, normalize_embeddings=True, show_progress_bar=False
        )  # (n_triggers, dim), нормированы -> dot = cosine

    def search_sync(self, text: str, top_k: int = 2, min_score: float = 0.86) -> list[KBItem]:
        if _is_filler(text):
            return []
        query = self._model.encode(
            [f"query: {text}"], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = self._vectors @ query
        best: dict[int, float] = {}  # item_idx -> max score по его триггерам
        for trig_idx in np.argsort(scores)[::-1][:top_k * 4]:
            item_idx = self._trigger_to_item[trig_idx]
            s = float(scores[trig_idx])
            if s >= min_score and (item_idx not in best or s > best[item_idx]):
                best[item_idx] = s
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            KBItem(
                id=self._items[i]["id"],
                type=self._items[i]["type"],
                response=" ".join(self._items[i]["response"].split()),
                score=s,
            )
            for i, s in ranked
        ]

    async def search(self, text: str, top_k: int = 2) -> list[KBItem]:
        """Async-обёртка: encode — CPU-bound, уводим из event loop."""
        return await asyncio.to_thread(self.search_sync, text, top_k)

    @staticmethod
    def format_context(items: list[KBItem]) -> str:
        if not items:
            return ""
        blocks = "\n\n".join(f"[{it.id}] {it.response}" for it in items)
        return (
            "МАТЕРИАЛ СКРИПТА по реплике клиента (перескажи коротко своими словами, "
            "сохраняя смысл и обещания дословно; если материал не подходит к ситуации — "
            "игнорируй его):\n" + blocks
        )

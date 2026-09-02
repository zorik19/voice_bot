from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from config import settings
from core.types import LLMToken, Message, ToolCall


class ClaudeLLM:
    """Стриминг текста + function calling.

    Текст отдаём токенами сразу (для низкой задержки TTS),
    tool call — целиком после завершения стрима."""

    def __init__(self) -> None:
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMToken | ToolCall]:
        system, api_messages = self._convert(messages)
        kwargs: dict[str, Any] = {
            "model": settings.claude_model,
            "max_tokens": settings.claude_max_tokens,
            "system": system,
            "messages": api_messages,
        }
        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield LLMToken(text=text)
            final = await stream.get_final_message()

        for block in final.content:
            if block.type == "tool_use":
                yield ToolCall(name=block.name, arguments=block.input, id=block.id)

    # ---------- конвертация нашего формата в формат Anthropic API ----------

    @staticmethod
    def _convert(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        system = ""
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system = m.content
            elif m.role == "tool":
                out.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content,
                    }],
                })
            elif m.role == "assistant" and m.tool_calls:
                content: list[dict[str, Any]] = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                content += [
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                    for tc in m.tool_calls
                ]
                out.append({"role": "assistant", "content": content})
            else:
                out.append({"role": m.role, "content": m.content})
        return system, out

    @staticmethod
    def _convert_tool(tool: dict[str, Any]) -> dict[str, Any]:
        """Наш формат тулов = формат Anthropic, отдаём как есть.
        (Для GigaChat в gigachat.py будет конвертация в их functions.)"""
        return tool

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from gigachat import GigaChat
from gigachat.models import Chat, Function, FunctionCall, Messages, MessagesRole

from config import settings
from core.types import LLMToken, Message, ToolCall


class GigaChatLLM:
    """GigaChat со стримингом и function calling (официальный SDK Сбера).

    Авторизация: Authorization key (base64 client_id:secret) из личного
    кабинета developers.sber.ru — SDK сам обменивает его на access token.
    verify_ssl_certs=False, чтобы не ставить сертификат НУЦ Минцифры;
    для прода поставить сертификат и включить проверку."""

    def __init__(self) -> None:
        self._client = GigaChat(
            credentials=settings.gigachat_auth_key,
            scope=settings.gigachat_scope,
            model=settings.gigachat_model,
            verify_ssl_certs=False,
        )

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMToken | ToolCall]:
        chat = Chat(
            messages=self._convert(messages),
            max_tokens=settings.claude_max_tokens,  # общий лимит длины реплики
            functions=[self._convert_tool(t) for t in tools] if tools else None,
        )

        fn_call: FunctionCall | None = None
        async for chunk in self._client.astream(chat):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield LLMToken(text=delta.content)
            if delta.function_call is not None:
                fn_call = delta.function_call

        if fn_call is not None:
            args = fn_call.arguments
            if isinstance(args, str):  # на случай, если придёт строкой
                args = json.loads(args or "{}")
            # у GigaChat нет id вызова — кладём имя функции, оно же
            # вернётся в tool_call_id и попадёт в name function-сообщения
            yield ToolCall(name=fn_call.name, arguments=args or {}, id=fn_call.name)

    # ---------- конвертация нашего формата в формат GigaChat ----------

    @staticmethod
    def _convert(messages: list[Message]) -> list[Messages]:
        out: list[Messages] = []
        for m in messages:
            match m.role:
                case "system":
                    out.append(Messages(role=MessagesRole.SYSTEM, content=m.content))
                case "user":
                    out.append(Messages(role=MessagesRole.USER, content=m.content))
                case "tool":
                    out.append(Messages(
                        role=MessagesRole.FUNCTION,
                        content=m.content,
                        name=m.tool_call_id,  # сюда мы положили имя функции
                    ))
                case "assistant" if m.tool_calls:
                    tc = m.tool_calls[0]  # GigaChat: один вызов за ход
                    out.append(Messages(
                        role=MessagesRole.ASSISTANT,
                        content=m.content,
                        function_call=FunctionCall(name=tc.name, arguments=tc.arguments),
                    ))
                case "assistant":
                    out.append(Messages(role=MessagesRole.ASSISTANT, content=m.content))
        return out

    @staticmethod
    def _convert_tool(tool: dict[str, Any]) -> Function:
        """Наш anthropic-совместимый формат -> Function GigaChat."""
        return Function(
            name=tool["name"],
            description=tool["description"],
            parameters=tool["input_schema"],
        )

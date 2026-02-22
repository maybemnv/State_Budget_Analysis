import json
from typing import Any
from langchain_core.callbacks import AsyncCallbackHandler
from fastapi import WebSocket


class WebSocketStreamingCallback(AsyncCallbackHandler):
    """Streams LangChain agent events to a WebSocket connection."""

    def __init__(self, websocket: WebSocket) -> None:
        self.ws = websocket

    async def _send(self, payload: dict[str, Any]) -> None:
        await self.ws.send_text(json.dumps(payload))

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await self._send({"type": "thought", "content": token})

    async def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        tool_name = serialized.get("name", "unknown")
        try:
            args = json.loads(input_str)
        except Exception:
            args = input_str
        await self._send({"type": "tool_call", "tool": tool_name, "args": args})

    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        try:
            result = json.loads(output)
        except Exception:
            result = output
        await self._send({"type": "tool_result", "result": result})

    async def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        await self._send({"type": "answer", "content": finish.return_values.get("output", "")})

    async def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        await self._send({"type": "error", "message": str(error)})

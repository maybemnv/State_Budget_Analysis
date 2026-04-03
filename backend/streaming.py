import json
from typing import Any
from langchain_core.callbacks import AsyncCallbackHandler
from fastapi import WebSocket, WebSocketDisconnect
from .utils.json_utils import sanitize_for_json


class WebSocketStreamingCallback(AsyncCallbackHandler):
    """Streams LangChain agent events to a WebSocket connection.
    
    Message types match frontend TypeScript interfaces:
    - thought: { type: 'thought', content: string }
    - tool_call: { type: 'tool_call', tool: string, args: object }
    - tool_result: { type: 'tool_result', tool: string, result: object }
    - chart: { type: 'chart', spec: object }
    - answer: { type: 'answer', content: string }
    - error: { type: 'error', message: string }
    - done: { type: 'done' }
    """

    def __init__(self, websocket: WebSocket) -> None:
        self.ws = websocket
        self._current_tool: str | None = None

    async def _send(self, payload: dict[str, Any]) -> None:
        try:
            sanitized = sanitize_for_json(payload)
            await self.ws.send_text(json.dumps(sanitized))
        except (WebSocketDisconnect, RuntimeError):
            # Connection closed or error occurred - ignore quietly
            pass
        except Exception:
            # Connection likely closed, ignore to prevent callback spam
            pass

    async def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        """Called when LLM starts generating - we accumulate thought content."""
        pass

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Stream individual tokens as thought content."""
        # Accumulate tokens into complete thoughts
        await self._send({
            "type": "thought",
            "content": token
        })

    async def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        """Called when a tool starts executing."""
        tool_name = serialized.get("name", "unknown")
        self._current_tool = tool_name
        
        try:
            args = json.loads(input_str)
        except Exception:
            args = {"raw": input_str}
        
        await self._send({
            "type": "tool_call",
            "tool": tool_name,
            "args": args
        })

    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when a tool finishes executing."""
        try:
            result = json.loads(output)
        except Exception:
            result = {"output": output}
        
        # Include the tool name so frontend can match with the call
        await self._send({
            "type": "tool_result",
            "tool": self._current_tool or "unknown",
            "result": result
        })
        self._current_tool = None

    async def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        """Called when the agent finishes with a final answer."""
        await self._send({
            "type": "answer",
            "content": finish.return_values.get("output", "")
        })

    async def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        """Called when agent decides to take an action."""
        pass

    async def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when there's an error in the chain."""
        await self._send({
            "type": "error",
            "message": str(error)
        })

    async def on_text(self, text: str, **kwargs: Any) -> None:
        """Called when agent outputs text."""
        pass

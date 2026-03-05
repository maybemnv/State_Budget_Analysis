import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException

from ..session import get_session
from ..agent.analyst_agent import run_agent
from ..agent.output_parser import parse_output
from ..streaming import WebSocketStreamingCallback
from ..schemas import ChatRequest, ChatResponse


logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()

    session = get_session(session_id)
    if session is None:
        logger.warning(f"WebSocket rejected: session not found: {session_id}")
        await websocket.send_text(json.dumps({"type": "error", "message": "Session not found"}))
        await websocket.close(code=4004, reason="Session not found")
        return

    logger.info(f"WebSocket accepted: session_id={session_id}")
    callback = WebSocketStreamingCallback(websocket)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                message = data.get("message", "")
            except json.JSONDecodeError:
                message = raw

            logger.debug(f"WS message: {message[:100]}...")
            result = await run_agent(session_id, message)
            parsed = parse_output(result.get("output", ""))

            if parsed["chart_spec"]:
                await callback._send({"type": "chart", "spec": parsed["chart_spec"]})

            await callback._send({"type": "done"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session_id={session_id}")
    except Exception as e:
        logger.exception(f"WebSocket error: session_id={session_id}, error={e}")
        try:
            await callback._send({"type": "error", "message": f"Server error: {str(e)}"})
        except Exception:
            pass


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(session_id: str, body: ChatRequest) -> ChatResponse:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await run_agent(session_id, body.message)
    parsed = parse_output(result.get("output", ""))

    steps = [
        {
            "tool": action.tool,
            "args": action.tool_input if isinstance(action.tool_input, dict) else {"input": action.tool_input},
            "result": observation if isinstance(observation, dict) else {"output": str(observation)},
        }
        for action, observation in result.get("intermediate_steps", [])
    ]

    return ChatResponse(
        answer=parsed["answer"],
        chart_spec=parsed["chart_spec"],
        has_error=parsed["has_error"],
        steps=steps,
    )

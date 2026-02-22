import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from ..session import get_session
from ..agent.analyst_agent import run_agent
from ..agent.output_parser import parse_output
from ..streaming import WebSocketStreamingCallback
from ..schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    if get_session(session_id) is None:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    callback = WebSocketStreamingCallback(websocket)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw).get("message", "")
            except json.JSONDecodeError:
                message = raw

            result = await run_agent(session_id, message)
            parsed = parse_output(result.get("output", ""))

            if parsed["chart_spec"]:
                await callback._send({"type": "chart", "spec": parsed["chart_spec"]})

            await callback._send({"type": "done"})

    except WebSocketDisconnect:
        pass


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(session_id: str, body: ChatRequest) -> ChatResponse:
    if get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await run_agent(session_id, body.message)
    parsed = parse_output(result.get("output", ""))

    steps = [
        {"tool": action.tool, "args": action.tool_input, "result": observation}
        for action, observation in result.get("intermediate_steps", [])
    ]

    return ChatResponse(
        answer=parsed["answer"],
        chart_spec=parsed["chart_spec"],
        has_error=parsed["has_error"],
        steps=steps,
    )

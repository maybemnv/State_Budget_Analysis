import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from ..session import get_session
from ..agent.analyst_agent import build_agent
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
                data = json.loads(raw)
                message = data.get("message", "")
            except json.JSONDecodeError:
                message = raw

            agent = build_agent(session_id)
            try:
                result = await agent.ainvoke(
                    {"input": message, "session_id": session_id},
                    config={"callbacks": [callback]},
                )
                # Final answer is sent via on_agent_finish callback
            except Exception as e:
                await callback._send({"type": "error", "message": str(e)})

            await callback._send({"type": "done"})

    except WebSocketDisconnect:
        pass


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(session_id: str, body: ChatRequest) -> ChatResponse:
    """Synchronous fallback — returns complete agent response as JSON."""
    if get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")

    agent = build_agent(session_id)
    result = await agent.ainvoke({"input": body.message, "session_id": session_id})

    steps = []
    for action, observation in result.get("intermediate_steps", []):
        steps.append({
            "tool": action.tool,
            "args": action.tool_input,
            "result": observation,
        })

    return ChatResponse(
        answer=result.get("output", ""),
        steps=steps,
    )

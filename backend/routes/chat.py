import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from ..session import get_session
from ..agent.analyst_agent import run_agent
from ..agent.output_parser import parse_output
from ..streaming import WebSocketStreamingCallback
from ..schemas import ChatRequest, ChatResponse, SessionInfo

router = APIRouter(tags=["chat"])


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time agent communication.
    
    Accepts: { message: string }
    Streams: thought, tool_call, tool_result, chart, answer, error, done
    """
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

            result = await run_agent(session_id, message)
            parsed = parse_output(result.get("output", ""))

            # Send chart spec if present
            if parsed["chart_spec"]:
                await callback._send({
                    "type": "chart",
                    "spec": parsed["chart_spec"]
                })

            # Signal completion
            await callback._send({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await callback._send({
                "type": "error",
                "message": f"Server error: {str(e)}"
            })
        except:
            pass


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(session_id: str, body: ChatRequest) -> ChatResponse:
    """HTTP chat endpoint (non-streaming)."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await run_agent(session_id, body.message)
    parsed = parse_output(result.get("output", ""))

    steps = [
        {
            "tool": action.tool,
            "args": action.tool_input if isinstance(action.tool_input, dict) else {"input": action.tool_input},
            "result": observation if isinstance(observation, dict) else {"output": str(observation)}
        }
        for action, observation in result.get("intermediate_steps", [])
    ]

    return ChatResponse(
        answer=parsed["answer"],
        chart_spec=parsed["chart_spec"],
        has_error=parsed["has_error"],
        steps=steps,
    )


@router.get("/sessions/{session_id}", response_model=SessionInfo)
def get_session_info(session_id: str) -> dict:
    """Get session metadata without raw data."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    meta = session["metadata"]
    return {
        "session_id": session_id,
        "filename": meta["filename"],
        "shape": list(meta["shape"]),  # Convert tuple to list for JSON
        "columns": meta["columns"],
        "dtypes": meta["dtypes"],
    }


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    """Delete a session and free resources."""
    from ..session import delete_session as _delete
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    _delete(session_id)
    return {"status": "deleted", "session_id": session_id}

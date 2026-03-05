# WebSocket Debugging Guide

## Issue: "Connection failed" error in UI

### What Was Fixed:

1. **Better Error Logging**
   - Added console logs in WebSocket client (`src/lib/api.ts`)
   - Added backend logging in WebSocket endpoint (`backend/routes/chat.py`)
   - Added session loading logs in workspace page

2. **Improved Error Handling**
   - WebSocket error no longer triggers false positive "Connection failed"
   - Connection status shown properly in UI
   - Disabled suggested queries when not connected

3. **Backend Logging**
   ```python
   print(f"WebSocket connection attempt for session: {session_id}")
   print(f"Session found: {session is not None}")
   print(f"Received message: {message[:50]}...")
   ```

---

## How to Debug:

### 1. Open Browser DevTools

Press F12 → Console tab

### 2. Refresh the Workspace Page

You should see logs like:
```
Workspace loading with session ID: {uuid}
Fetching session info from backend...
Session info loaded: {info}
Connecting to WebSocket: ws://localhost:8000/ws/{uuid}
```

### 3. Check for These Logs:

**Frontend (Browser Console):**
```
✅ Expected:
- "Workspace loading with session ID: ..."
- "Fetching session info from backend..."
- "Session info loaded: ..."
- "Connecting to WebSocket: ws://localhost:8000/ws/..."
- "WebSocket connected"

❌ If you see:
- "Failed to load session: Session not found" → Backend session expired
- "WebSocket error: ..." → Connection issue
- "WebSocket closed: 4004 Session not found" → Session ID mismatch
```

**Backend (Terminal):**
```
✅ Expected:
- "WebSocket connection attempt for session: ..."
- "Session found, accepting connection: ..."
- "Received message: ..."

❌ If you see:
- "Session not found: ..." → Session expired or wrong ID
```

---

## Common Issues & Fixes:

### Issue 1: "Session not found" (Code 4004)

**Cause:** Session expired or was deleted

**Solution:**
1. Upload a new file
2. Sessions expire after 1 hour (configurable in `backend/config.py`)

---

### Issue 2: WebSocket Won't Connect

**Cause:** Backend not running or wrong URL

**Check:**
```bash
# Test backend is running
curl http://localhost:8000/health

# Should return: {"status":"ok","version":"2.0.0"}
```

**Solution:**
```bash
cd D:\Projects\State_Budget_Analysis
.venv\Scripts\Activate
uv run uvicorn backend.main:app --reload --port 8000
```

---

### Issue 3: CORS Error

**Cause:** Frontend origin not allowed in backend CORS

**Check Backend Logs:**
```
INFO:     127.0.0.1:xxxxx - "GET /sessions/{id} HTTP/1.1" 200 OK
```

**Solution:** Already configured in `backend/main.py`:
```python
allow_origins=[
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*",
]
```

---

### Issue 4: Session Info Loads but WebSocket Fails

**Cause:** Session ID format mismatch

**Check:**
1. Frontend logs: What session ID is being used?
2. Backend logs: What session ID is being received?

**Solution:**
- Make sure session ID from upload matches what's in workspace URL
- Session ID should be UUID format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`

---

## Test WebSocket Manually:

Open browser console and run:

```javascript
// Test WebSocket connection
const sessionId = 'your-session-id-from-url'
const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`)

ws.onopen = () => console.log('✅ Connected!')
ws.onmessage = (e) => console.log('Message:', JSON.parse(e.data))
ws.onerror = (e) => console.error('Error:', e)
ws.onclose = (e) => console.log('Closed:', e.code, e.reason)

// Send a test message
setTimeout(() => {
  ws.send(JSON.stringify({ message: 'Describe the dataset' }))
}, 1000)
```

---

## Quick Fix Steps:

1. **Restart Backend:**
   ```bash
   taskkill /F /PID <backend_pid>
   cd D:\Projects\State_Budget_Analysis
   uv run uvicorn backend.main:app --reload --port 8000
   ```

2. **Restart Frontend:**
   ```bash
   taskkill /F /PID <frontend_pid>
   cd D:\Projects\State_Budget_Analysis\frontend
   npm run dev
   ```

3. **Clear Browser Cache:**
   - Ctrl+Shift+Delete
   - Or use Incognito mode

4. **Upload New File:**
   - Go to http://localhost:3000
   - Upload a CSV file
   - Watch console logs

5. **Check Both Terminals:**
   - Backend terminal should show connection logs
   - Frontend terminal should show build logs

---

## Expected Flow:

```
1. Upload file → POST /upload
   Backend: Creates session with UUID
   Frontend: Redirects to /workspace/{UUID}

2. Workspace loads → GET /sessions/{UUID}
   Backend: Returns session info
   Frontend: Displays sidebar with column info

3. WebSocket connects → WS /ws/{UUID}
   Backend: "WebSocket connection attempt for session: {UUID}"
   Backend: "Session found, accepting connection"
   Frontend: "WebSocket connected"

4. User sends message
   Frontend: WS.send({ message: "..." })
   Backend: "Received message: ..."
   Backend: Runs agent
   Frontend: Receives streamed events
```

---

## Current Status:

- ✅ Backend running on http://localhost:8000
- ✅ Frontend running on http://localhost:3000
- ✅ Session info loading (sidebar shows data)
- ⚠️ WebSocket connection failing

**Next Step:** Check browser console logs to see exact error.

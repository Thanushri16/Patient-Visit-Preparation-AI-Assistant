"""FastAPI entry point and in-memory typed session management for the chatbot."""

import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from .chatbot import get_chatbot_response, load_api_key
    from .moderation import moderate_text
    from .models import ChatSession, ConversationState
    from .persistence import JsonVisitRepository
except ImportError:  # pragma: no cover - allows running as a script
    from chatbot import get_chatbot_response, load_api_key
    from moderation import moderate_text
    from models import ChatSession, ConversationState
    from persistence import JsonVisitRepository

app = FastAPI(title="Healthcare Chatbot")
project_root = Path(__file__).resolve().parent.parent
conversation_histories: dict[str, ChatSession] = {}
SESSION_TTL_SECONDS = 15 * 60
visit_repository = JsonVisitRepository(project_root / "db" / "visits")


def create_chat_session(session_id: str, now: float) -> ChatSession:
    return ChatSession(
        state=ConversationState(session_id=session_id),
        expires_at=now + SESSION_TTL_SECONDS,
    )


def get_or_create_chat_session(session_id: str, now: float) -> ChatSession:
    session = conversation_histories.get(session_id)
    if session is None:
        session = create_chat_session(session_id, now)
        conversation_histories[session_id] = session
    return session


def prune_expired_sessions(now: float | None = None) -> None:
    current_time = time.time() if now is None else now
    expired_ids = [
        session_id
        for session_id, session in conversation_histories.items()
        if session.expires_at <= current_time
    ]
    for session_id in expired_ids:
        conversation_histories.pop(session_id, None)


api_key = None
try:
    api_key = load_api_key()
except RuntimeError:
    api_key = None

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    from openai import OpenAI

    client = OpenAI()
else:
    client = None


# Built once, at startup. None when no knowledge store is configured, which
# leaves the chain exactly as it was: intake never depends on a database.
try:
    from rag.integration import build_knowledge_branch
except ImportError:  # pragma: no cover
    from src.rag.integration import build_knowledge_branch
knowledge_branch = build_knowledge_branch(client) if client is not None else None

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Healthcare Chatbot</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #f4f7fb;
      margin: 0;
      padding: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
    }
    .chat-container {
      width: 100%;
      max-width: 700px;
      height: 80vh;
      background: white;
      border-radius: 16px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.1);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .chat-header {
      background: #2563eb;
      color: white;
      padding: 16px 20px;
      font-size: 18px;
      font-weight: bold;
    }
    .messages {
      flex: 1;
      padding: 16px;
      overflow-y: auto;
      background: #fafcff;
    }
    .message {
      margin: 8px 0;
      padding: 10px 12px;
      border-radius: 10px;
      max-width: 80%;
      line-height: 1.4;
    }
    .user {
      background: #dbeafe;
      margin-left: auto;
      text-align: right;
    }
    .assistant {
      background: #e5e7eb;
      margin-right: auto;
    }
    .input-area {
      display: flex;
      border-top: 1px solid #e5e7eb;
      padding: 12px;
      gap: 8px;
    }
    input {
      flex: 1;
      padding: 10px 12px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      font-size: 14px;
    }
    button {
      padding: 10px 14px;
      border: none;
      border-radius: 8px;
      background: #2563eb;
      color: white;
      cursor: pointer;
    }
    button:hover { background: #1d4ed8; }
  </style>
</head>
<body>
  <div class="chat-container">
    <div class="chat-header">Healthcare Appointment Prep Chat</div>
    <div class="messages" id="messages"></div>
    <div class="input-area">
      <input id="userInput" placeholder="Type your message..." />
      <button onclick="sendMessage()">Send</button>
    </div>
  </div>

  <script>
    const messagesDiv = document.getElementById('messages');
    const input = document.getElementById('userInput');
    const initialAssistantMessage = __INITIAL_ASSISTANT_MESSAGE__;
    let sessionId = localStorage.getItem('healthcareChatSessionId');
    let sessionExpiry = localStorage.getItem('healthcareChatSessionExpiry');
    const now = Date.now();

    if (!sessionId || !sessionExpiry || now > Number(sessionExpiry)) {
      sessionId = crypto.randomUUID();
      localStorage.setItem('healthcareChatSessionId', sessionId);
      localStorage.setItem('healthcareChatSessionExpiry', String(now + 15 * 60 * 1000));
    }

    addMessage(initialAssistantMessage, 'assistant');

    function addMessage(text, role) {
      const msg = document.createElement('div');
      msg.className = `message ${role}`;
      msg.textContent = text;
      messagesDiv.appendChild(msg);
      messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    async function sendMessage() {
      const prompt = input.value.trim();
      if (!prompt) return;

      addMessage(prompt, 'user');
      input.value = '';
      addMessage('Thinking...', 'assistant');

      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: prompt, session_id: sessionId })
      });

      const data = await response.json();
      messagesDiv.removeChild(messagesDiv.lastChild);
      addMessage(data.reply, 'assistant');
    }

    input.addEventListener('keydown', function(event) {
      if (event.key === 'Enter') {
        sendMessage();
      }
    });
  </script>
</body>
</html>
"""

INITIAL_ASSISTANT_MESSAGE = (
    "Hello — I’m your healthcare assistant. "
    "What can I help you with today?"
)

HTML_TEMPLATE = HTML_TEMPLATE.replace(
    "__INITIAL_ASSISTANT_MESSAGE__",
    json.dumps(INITIAL_ASSISTANT_MESSAGE),
)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return HTMLResponse(content=HTML_TEMPLATE)


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("message", "")
    session_id = data.get("session_id", "default")

    if not prompt:
        return JSONResponse(
            {
                "reply": (
                    "I didn't receive a message. Type what you'd like help with, "
                    "or say menu to see the options."
                ),
                "intent": "unknown",
                "state": {},
                "is_emergency": False,
                "safety_triggered": False,
            }
        )

    if not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 200:
        return JSONResponse({"reply": "A valid session ID is required."}, status_code=400)

    session_id = session_id.strip()

    if client is None:
        return JSONResponse({"reply": "OpenAI API key is not configured yet."})

    prune_expired_sessions()
    now = time.time()
    session = get_or_create_chat_session(session_id, now)
    input_moderation = moderate_text(prompt, stage="input")
    session.state.rag = None
    reply = get_chatbot_response(
        session.messages,
        prompt,
        client,
        state=session.state,
        visit_repository=visit_repository,
        knowledge_branch=knowledge_branch,
    )
    session.expires_at = now + SESSION_TTL_SECONDS
    return JSONResponse(
        {
            "reply": reply,
            "intent": session.state.workflow.value if session.state.workflow else "unknown",
            "state": session.state.model_dump(mode="json"),
            "is_emergency": session.state.emergency_detected,
            # Every guardrail action that changed what the assistant would
            # otherwise have said counts as triggered, including a request that
            # was declined and an embedded payload that was stripped out.
            "safety_triggered": input_moderation.action
            in {"block", "escalate", "redirect", "neutralize"},
            # Only present when the turn used the knowledge branch, so every
            # existing consumer of this payload sees exactly what it saw before.
            **(
                {"citations": [c.model_dump(mode="json") for c in session.state.rag.citations]}
                if session.state.rag and session.state.rag.citations
                else {}
            ),
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000)

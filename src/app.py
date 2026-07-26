import os
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from .chatbot import get_chatbot_response, load_api_key
except ImportError:  # pragma: no cover - allows running as a script
    from chatbot import get_chatbot_response, load_api_key

app = FastAPI(title="Healthcare Chatbot")
project_root = Path(__file__).resolve().parent.parent
conversation_histories = {}
SESSION_TTL_SECONDS = 15 * 60


def prune_expired_sessions():
    now = time.time()
    expired_ids = [
        session_id
        for session_id, (_, expires_at) in conversation_histories.items()
        if expires_at <= now
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
    let sessionId = localStorage.getItem('healthcareChatSessionId');
    let sessionExpiry = localStorage.getItem('healthcareChatSessionExpiry');
    const now = Date.now();

    if (!sessionId || !sessionExpiry || now > Number(sessionExpiry)) {
      sessionId = crypto.randomUUID();
      localStorage.setItem('healthcareChatSessionId', sessionId);
      localStorage.setItem('healthcareChatSessionExpiry', String(now + 15 * 60 * 1000));
    }

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


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return HTMLResponse(content=HTML_TEMPLATE)


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("message", "")
    session_id = data.get("session_id", "default")

    if not prompt:
        return JSONResponse({"reply": "Please enter a message."})

    if client is None:
        return JSONResponse({"reply": "OpenAI API key is not configured yet."})

    prune_expired_sessions()
    now = time.time()
    if session_id not in conversation_histories:
        conversation_histories[session_id] = ([], now + SESSION_TTL_SECONDS)

    message_history, _ = conversation_histories[session_id]
    reply = get_chatbot_response(message_history, prompt, client)
    conversation_histories[session_id] = (message_history, now + SESSION_TTL_SECONDS)
    return JSONResponse({"reply": reply})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000)

import uvicorn
from fastapi import FastAPI
from starlette.responses import RedirectResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

from configuration.config import WEB_STATS_DIR
from schemas import CreateSessionRequest, Question, SessionQuestion
from service import ChatService

service = ChatService()

app = FastAPI(
    title="EcommerceG电商图谱助手",
    description="这是电商图谱助手后端接口",
)
app.mount("/static", StaticFiles(directory=WEB_STATS_DIR), name="static")


@app.get("/")
async def index():
    return RedirectResponse(url="/static/index.html")


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": service.list_sessions()}


@app.post("/api/sessions")
async def create_session(payload: CreateSessionRequest):
    return service.create_session(payload.title)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    service.delete_session(session_id)
    return {"ok": True}


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    return {"messages": service.get_session_messages(session_id)}


@app.post("/api/chat")
async def chat_non_stream(question: Question):
    session = service.create_session("临时会话")

    def token_generator():
        yield from service.chat_stream(question.message, session["session_id"])

    full = "".join(list(token_generator()))
    return {"message": full}


@app.post("/api/chat/stream")
async def chat_stream(question: SessionQuestion):
    def token_generator():
        yield from service.chat_stream(question.message, question.session_id)

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

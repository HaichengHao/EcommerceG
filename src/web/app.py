# @Time    : 2026/5/14 09:54
# @Author  : hero
# @File    : app.py
import uvicorn
from fastapi import FastAPI
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles
from schemas import Question, Answer
from service import ChatService

from configuration.config import *

#tips:实例化Chatservice对象
service = ChatService()

app = FastAPI(
    title="EcommerceG电商图谱助手",
    description="这是电商图谱助手后端接口"
)
app.mount("/static", StaticFiles(directory=WEB_STATS_DIR), name="static")


@app.get('/')
async def index():
    return RedirectResponse(url='/static/index.html')


@app.post('/api/chat')
async def read_item(question: Question):
    result = service.chat(question)
    return Answer(message=result)

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
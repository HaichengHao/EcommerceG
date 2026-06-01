from pydantic import BaseModel


class Question(BaseModel):
    message: str


class SessionQuestion(Question):
    session_id: str


class Answer(BaseModel):
    message: str


class CreateSessionRequest(BaseModel):
    title: str | None = None

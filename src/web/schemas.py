# @Time    : 2026/5/14 09:54
# @Author  : hero
# @File    : schemas.py

from pydantic import BaseModel
class Question(BaseModel):
    message: str
class Answer(BaseModel):
    message: str
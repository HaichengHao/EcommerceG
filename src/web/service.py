import os
import uuid
from datetime import datetime
from typing import Iterator

import redis
import torch
from dotenv import load_dotenv
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import ChatOpenAI
from neo4j_graphrag.types import SearchType

from configuration.config import EMBEDDING_MODEL, NEO4J_CONFIG

load_dotenv()


class ChatService:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_CONFIG["uri"],
            username=NEO4J_CONFIG["auth"][0],
            password=NEO4J_CONFIG["auth"][1],
        )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cuda:0" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.llm = ChatOpenAI(
            model="glm-4.7",
            api_key=os.getenv("zhipu_key"),
            base_url=os.getenv("zhipu_base_url"),
        )

        self.neo4j_vectors = {
            "Trademark": Neo4jVector.from_existing_index(
                self.embedding_model,
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name="trademark_vector_index",
                keyword_index_name="trademark_fulltext_index",
                search_type=SearchType.HYBRID,
            ),
            "SPU": Neo4jVector.from_existing_index(
                self.embedding_model,
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name="spu_vector_index",
                keyword_index_name="spu_fulltext_index",
                search_type=SearchType.HYBRID,
            ),
            "SKU": Neo4jVector.from_existing_index(
                self.embedding_model,
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name="sku_vector_index",
                keyword_index_name="sku_fulltext_index",
                search_type=SearchType.HYBRID,
            ),
            "Category1": Neo4jVector.from_existing_index(
                self.embedding_model,
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name="category1_vector_index",
                keyword_index_name="category1_fulltext_index",
                search_type=SearchType.HYBRID,
            ),
            "Category2": Neo4jVector.from_existing_index(
                self.embedding_model,
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name="category2_vector_index",
                keyword_index_name="category2_fulltext_index",
                search_type=SearchType.HYBRID,
            ),
            "Category3": Neo4jVector.from_existing_index(
                self.embedding_model,
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name="category3_vector_index",
                keyword_index_name="category3_fulltext_index",
                search_type=SearchType.HYBRID,
            ),
        }

        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser()

        self.redis_url = os.getenv("REDIS_DOCKER", "redis://127.0.0.1:6379/0")
        self.redis_key_prefix = "ecommerceg:chat:"
        self.redis_session_index_key = "ecommerceg:sessions"
        self.redis_session_title_key = "ecommerceg:session_titles"
        self.redis_session_ctime_key = "ecommerceg:session_ctimes"
        self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        self._memory_sessions: dict[str, dict] = {}
        self._memory_histories: dict[str, InMemoryChatMessageHistory] = {}

    def _redis_available(self) -> bool:
        try:
            return bool(self.redis_client.ping())
        except Exception:
            return False

    def _history(self, session_id: str):
        if self._redis_available():
            return RedisChatMessageHistory(
                session_id=session_id,
                url=self.redis_url,
                key_prefix=self.redis_key_prefix,
                ttl=None,
            )
        if session_id not in self._memory_histories:
            self._memory_histories[session_id] = InMemoryChatMessageHistory()
        return self._memory_histories[session_id]

    def create_session(self, title: str | None = None) -> dict:
        session_id = uuid.uuid4().hex
        created_at = datetime.now().isoformat(timespec="seconds")
        title_value = title or "新会话"

        if self._redis_available():
            self.redis_client.zadd(self.redis_session_index_key, {session_id: datetime.now().timestamp()})
            self.redis_client.hset(self.redis_session_title_key, session_id, title_value)
            self.redis_client.hset(self.redis_session_ctime_key, session_id, created_at)
        else:
            self._memory_sessions[session_id] = {
                "session_id": session_id,
                "title": title_value,
                "created_at": created_at,
                "last_score": datetime.now().timestamp(),
            }

        return {"session_id": session_id, "title": title_value, "created_at": created_at}

    def list_sessions(self) -> list[dict]:
        if self._redis_available():
            session_ids = self.redis_client.zrevrange(self.redis_session_index_key, 0, -1)
            titles = self.redis_client.hgetall(self.redis_session_title_key)
            ctimes = self.redis_client.hgetall(self.redis_session_ctime_key)

            result = []
            for sid in session_ids:
                result.append(
                    {
                        "session_id": sid,
                        "title": titles.get(sid, "新会话"),
                        "created_at": ctimes.get(sid, ""),
                    }
                )
            return result
        sessions = sorted(
            self._memory_sessions.values(),
            key=lambda x: x.get("last_score", 0),
            reverse=True,
        )
        return [
            {
                "session_id": item["session_id"],
                "title": item["title"],
                "created_at": item["created_at"],
            }
            for item in sessions
        ]

    def delete_session(self, session_id: str) -> None:
        if self._redis_available():
            self.redis_client.zrem(self.redis_session_index_key, session_id)
            self.redis_client.hdel(self.redis_session_title_key, session_id)
            self.redis_client.hdel(self.redis_session_ctime_key, session_id)
            self.redis_client.delete(f"{self.redis_key_prefix}{session_id}")
        self._memory_sessions.pop(session_id, None)
        self._memory_histories.pop(session_id, None)

    def get_session_messages(self, session_id: str) -> list[dict]:
        messages = self._history(session_id).messages
        result = []
        for m in messages:
            role = "assistant"
            if isinstance(m, HumanMessage):
                role = "user"
            result.append({"role": role, "content": m.content})
        return result

    def _build_history_context(self, session_id: str, max_turns: int = 6) -> str:
        messages = self._history(session_id).messages
        if not messages:
            return ""
        tail = messages[-(max_turns * 2):]
        lines = []
        for m in tail:
            if isinstance(m, HumanMessage):
                lines.append(f"用户: {m.content}")
            else:
                lines.append(f"助手: {m.content}")
        return "\n".join(lines)

    def chat_stream(self, question: str, session_id: str) -> Iterator[str]:
        history = self._history(session_id)
        history_context = self._build_history_context(session_id)

        result = self._generate_cypher(question)
        cypher = result["cypher_query"]
        entities_to_align = result["entities_to_align"]

        aligned_entities = self._entity_align(entities_to_align)
        query_result = self._execute_cypher(cypher, aligned_entities)

        full_answer = ""
        for chunk in self._generate_answer_stream(question, query_result, history_context):
            full_answer += chunk
            yield chunk

        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=full_answer))

        if self._redis_available():
            if self.redis_client.hget(self.redis_session_title_key, session_id) in (None, "新会话"):
                self.redis_client.hset(self.redis_session_title_key, session_id, question[:24])
            self.redis_client.zadd(self.redis_session_index_key, {session_id: datetime.now().timestamp()})
        elif session_id in self._memory_sessions:
            if self._memory_sessions[session_id]["title"] == "新会话":
                self._memory_sessions[session_id]["title"] = question[:24]
            self._memory_sessions[session_id]["last_score"] = datetime.now().timestamp()

    def _generate_cypher(self, question: str) -> dict:
        prompt = """
你是一个专业的Neo4j Cypher查询生成器。你的任务是根据用户问题生成一条Cypher查询语句，用于从知识图谱中获取回答用户问题所需的信息。

用户问题：{question}

知识图谱结构信息：{schema_info}

要求：
1. 生成参数化的Cypher查询语句，使用 $param_0, $param_1, ... 作为占位符
2. 识别需要对齐的实体，并记录原始名称和节点标签
3. 必须严格使用以下JSON格式输出结果：
{{
  "cypher_query": "生成的Cypher语句",
  "entities_to_align": [
    {{
      "param_name": "param_0",
      "entity": "原始实体名称",
      "label": "节点类型"
    }}
  ]
}}
"""
        template = PromptTemplate.from_template(prompt)
        rendered = template.format(question=question, schema_info=self.graph.schema)
        output = self.llm.invoke(rendered)
        return self.json_parser.invoke(output)

    def _entity_align(self, entities_to_align: list[dict]) -> list[dict]:
        for index, entity_to_align in enumerate(entities_to_align):
            label = entity_to_align["label"]
            entity = entity_to_align["entity"]
            aligned_entity = self.neo4j_vectors[label].similarity_search(entity, k=1)[0].page_content
            entities_to_align[index]["entity"] = aligned_entity
        return entities_to_align

    def _execute_cypher(self, cypher: str, aligned_entities: list[dict]):
        params = {aligned_entity["param_name"]: aligned_entity["entity"] for aligned_entity in aligned_entities}
        return self.graph.query(cypher, params=params)

    def _generate_answer_stream(self, question: str, query_result, history_context: str) -> Iterator[str]:
        prompt = """
你是一个电商智能客服，根据用户问题、会话历史和数据库查询结果，生成一段简洁、准确、连贯的自然语言回答。

会话历史：
{history_context}

用户问题: {question}
数据库返回结果: {query_result}
"""
        rendered = prompt.format(
            history_context=history_context or "(无历史)",
            question=question,
            query_result=query_result,
        )
        for chunk in self.llm.stream(rendered):
            content = getattr(chunk, "content", "")
            if isinstance(content, str) and content:
                yield content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        if text:
                            yield text

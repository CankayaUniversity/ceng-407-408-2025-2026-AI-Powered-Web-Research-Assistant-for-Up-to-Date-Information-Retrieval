import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

import cache_store
import chat_store
from agent import build_agent
from config import (
    CACHE_TTL_SECONDS,
    DEFAULT_MODEL_KEY,
    HISTORY_TURN_LIMIT,
    MODEL_REGISTRY,
)
from config import load_environment
from fact_extraction import extract_claims

TOOL_LOG_PREVIEW_CHARS = 2000


load_environment()
app = FastAPI()

print("Building agents…")
AGENTS: dict[str, object] = {
    key: build_agent(info["id"]) for key, info in MODEL_REGISTRY.items()
}
print(f"Agents ready: {list(AGENTS.keys())}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TitleUpdate(BaseModel):
    title: str


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"message": "Deep research agent is active!"}


@app.get("/api/models")
def api_models():
    return {
        "default": DEFAULT_MODEL_KEY,
        "models": [
            {"key": key, **info} for key, info in MODEL_REGISTRY.items()
        ],
    }


@app.get("/api/chats")
def api_list_chats(q: str | None = None):
    if q:
        return {"chats": chat_store.search_chats(q)}
    return {"chats": chat_store.list_chats()}


@app.get("/api/chats/{chat_id}")
def api_get_chat(chat_id: str):
    chat = chat_store.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.patch("/api/chats/{chat_id}")
def api_update_chat(chat_id: str, body: TitleUpdate):
    chat = chat_store.update_title(chat_id, body.title)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.delete("/api/chats/{chat_id}")
def api_delete_chat(chat_id: str):
    if not chat_store.delete_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"ok": True}


@app.post("/api/cache/clear")
def api_clear_cache():
    cache_store.clear()
    return {"ok": True}


def _sse(event_type: str, payload: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _resolve_model_key(model_key: str | None) -> str:
    if model_key and model_key in MODEL_REGISTRY:
        return model_key
    return DEFAULT_MODEL_KEY


def _build_history(chat: dict | None) -> list:
    if not chat:
        return []
    turns = chat.get("turns", []) or []
    history: list = []
    for turn in turns[-HISTORY_TURN_LIMIT:]:
        question = turn.get("question")
        answer = turn.get("answer")
        if question:
            history.append(HumanMessage(content=question))
        if answer:
            history.append(AIMessage(content=answer))
    return history


@app.get("/ask_agent_stream")
def ask_agent_stream(
    question: str,
    chat_id: str | None = None,
    model: str | None = None,
):
    def generate():
        try:
            model_key = _resolve_model_key(model)
            model_info = MODEL_REGISTRY[model_key]

            if chat_id:
                chat = chat_store.get_chat(chat_id) or chat_store.create_chat(question)
            else:
                chat = chat_store.create_chat(question)
            resolved_chat_id = chat["id"]

            prior_turns = chat.get("turns", []) or []
            has_memory = len(prior_turns) > 0

            yield _sse(
                "start",
                {
                    "question": question,
                    "chat_id": resolved_chat_id,
                    "model": model_key,
                    "model_label": model_info["label"],
                    "memory_turns": len(prior_turns),
                },
            )

            # Cache check (skip when memory is active — answer depends on context).
            if not has_memory:
                cached = cache_store.get(model_key, question, CACHE_TTL_SECONDS)
                if cached:
                    saved_chat = chat_store.append_turn(
                        resolved_chat_id,
                        {
                            "question": question,
                            "answer": cached.get("answer", ""),
                            "facts": cached.get("facts", []),
                            "sources": cached.get("sources", []),
                            "trust_signals": cached.get("trust_signals", {}),
                            "model": model_key,
                            "from_cache": True,
                        },
                    )
                    yield _sse(
                        "cached",
                        {
                            "cached_at": cached.get("cached_at", ""),
                            "original_question": cached.get("original_question", question),
                        },
                    )
                    yield _sse(
                        "final",
                        {
                            "question": question,
                            "chat_id": resolved_chat_id,
                            "chat_title": (saved_chat or {}).get("title", ""),
                            "model": model_key,
                            "from_cache": True,
                            "answer": cached.get("answer", ""),
                            "facts": cached.get("facts", []),
                            "sources": cached.get("sources", []),
                            "trust_signals": cached.get("trust_signals", {}),
                        },
                    )
                    yield _sse("done", {"chat_id": resolved_chat_id})
                    return

            agent = AGENTS[model_key]
            history_messages = _build_history(chat)
            input_messages = history_messages + [HumanMessage(content=question)]

            tool_messages_collected: list[dict] = []
            final_answer = ""

            for chunk in agent.stream({"messages": input_messages}, stream_mode="updates"):
                if not isinstance(chunk, dict):
                    continue
                for _node_name, node_data in chunk.items():
                    messages = node_data.get("messages", []) if isinstance(node_data, dict) else []
                    for message in messages:
                        message_type = getattr(message, "type", None)

                        if message_type == "ai" and getattr(message, "tool_calls", None):
                            for tool_call in message.tool_calls:
                                yield _sse(
                                    "tool_call",
                                    {
                                        "id": tool_call.get("id", ""),
                                        "name": tool_call.get("name", "tool"),
                                        "args": tool_call.get("args", {}),
                                    },
                                )
                        elif message_type == "tool":
                            content_str = str(message.content)
                            tool_name = getattr(message, "name", "")
                            tool_messages_collected.append({"name": tool_name, "content": content_str})
                            yield _sse(
                                "tool_result",
                                {
                                    "id": getattr(message, "tool_call_id", ""),
                                    "name": tool_name,
                                    "preview": content_str[:600],
                                    "length": len(content_str),
                                },
                            )
                        elif message_type == "ai" and getattr(message, "content", None):
                            final_answer = message.content
                            yield _sse("answer", {"text": final_answer})

            yield _sse("extracting", {})

            cleaned_answer = final_answer.replace("\n", " ")
            extraction = extract_claims(
                cleaned_answer,
                tool_messages_collected,
                model_id=model_info["id"],
            )

            saved_chat = chat_store.append_turn(
                resolved_chat_id,
                {
                    "question": question,
                    "answer": final_answer,
                    "facts": extraction["facts"],
                    "sources": extraction["sources"],
                    "trust_signals": extraction["trust_signals"],
                    "model": model_key,
                    "from_cache": False,
                },
            )

            # Cache only when memory was NOT used — context-dependent answers must not pollute global cache.
            if not has_memory and final_answer:
                cache_store.put(
                    model_key,
                    question,
                    {
                        "answer": final_answer,
                        "facts": extraction["facts"],
                        "sources": extraction["sources"],
                        "trust_signals": extraction["trust_signals"],
                    },
                )

            yield _sse(
                "final",
                {
                    "question": question,
                    "chat_id": resolved_chat_id,
                    "chat_title": (saved_chat or {}).get("title", ""),
                    "model": model_key,
                    "from_cache": False,
                    "answer": final_answer,
                    "facts": extraction["facts"],
                    "sources": extraction["sources"],
                    "trust_signals": extraction["trust_signals"],
                },
            )
            yield _sse("done", {"chat_id": resolved_chat_id})
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/ask_agent")
def ask_agent(question: str, model: str | None = None):
    try:
        model_key = _resolve_model_key(model)
        model_info = MODEL_REGISTRY[model_key]
        agent = AGENTS[model_key]

        input_message = {"messages": [HumanMessage(content=question)]}
        result = agent.invoke(input_message)

        print("\n" + "=" * 40)
        print(f"AGENT REASONING TRACE ({model_info['label']})")
        for message in result["messages"]:
            if message.type == "ai" and hasattr(message, "tool_calls") and message.tool_calls:
                print("\n[THOUGHT]: The agent decided to use tools.")
                for tool_call in message.tool_calls:
                    print(f" TOOL USED: {tool_call['name']}")
                    print(f" TOOL INPUT: {tool_call['args']}")
            elif message.type == "tool":
                tool_output = str(message.content)
                print(f"\n[TOOL RESULT ({message.name})]:")
                if len(tool_output) <= TOOL_LOG_PREVIEW_CHARS:
                    print(tool_output)
                else:
                    print(tool_output[:TOOL_LOG_PREVIEW_CHARS])
                    print(f"... (truncated, total chars: {len(tool_output)})")
        print("=" * 40 + "\n")

        final_answer = result["messages"][-1].content.replace("\n", " ")
        tool_messages = [
            {"name": message.name, "content": str(message.content)}
            for message in result["messages"]
            if message.type == "tool"
        ]
        extraction = extract_claims(final_answer, tool_messages, model_id=model_info["id"])

        print("FACT EXTRACTION SUMMARY")
        for idx, fact in enumerate(extraction["facts"], start=1):
            print(f" FACT {idx}: {fact['claim_text']}")
            print(f"  URLs: {fact['evidence_urls']}")
            print(f"  FLAGS: {fact['fact_quality_flags']}")
        print(f" TRUST SIGNALS: {extraction['trust_signals']}")

        return {
            "your_question": question,
            "model": model_key,
            "agent_answer": final_answer,
            "facts": extraction["facts"],
            "sources": extraction["sources"],
            "trust_signals": extraction["trust_signals"],
        }
    except Exception as e:
        return {"error": f"Agent error: {str(e)}"}

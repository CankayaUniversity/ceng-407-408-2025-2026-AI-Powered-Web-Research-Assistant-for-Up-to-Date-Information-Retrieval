import asyncio
import json
import logging
import threading
from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

import cache_store
import chat_store
import settings_store
from agent import build_agent
from config import (
    DEFAULT_MODEL_KEY,
    MODEL_REGISTRY,
    load_environment,
)
from fact_extraction import extract_claims
from verification import verify_answer

TOOL_LOG_PREVIEW_CHARS = 2000


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("deep_research")


load_environment()
app = FastAPI()

logger.info("Building agents for models: %s", list(MODEL_REGISTRY.keys()))
AGENTS: dict[str, object] = {
    key: build_agent(info["id"]) for key, info in MODEL_REGISTRY.items()
}
logger.info("Agents ready: %s", list(AGENTS.keys()))


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_static(request, call_next):
    """Prevent browser caching of /static/* and / so JS/CSS edits show up on reload."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/") or path == "/":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


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


class SettingsPatch(BaseModel):
    cache_ttl_seconds: int | None = None
    history_turn_limit: int | None = None
    verification_enabled: bool | None = None
    fact_extraction_enabled: bool | None = None


@app.get("/api/settings")
def api_get_settings():
    return {
        "settings": settings_store.get_all(),
        "defaults": settings_store.DEFAULTS,
    }


@app.patch("/api/settings")
def api_patch_settings(body: SettingsPatch):
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    return {"settings": settings_store.update(patch)}


@app.post("/api/settings/reset")
def api_reset_settings():
    return {"settings": settings_store.reset()}


def _sse(event_type: str, payload: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _resolve_model_key(model_key: str | None) -> str:
    if model_key and model_key in MODEL_REGISTRY:
        return model_key
    return DEFAULT_MODEL_KEY


# Verification should be done by the strongest available model regardless of
# which model wrote the draft — a 3B model verifying its own work catches
# very little. Ordered preference: Llama 3.1 (8B) > Qwen 2.5 (7B) > Llama 3.2 (3B).
_VERIFIER_PREFERENCE = ("llama", "qwen", "llama32")


def _verifier_model_id() -> str:
    for key in _VERIFIER_PREFERENCE:
        if key in MODEL_REGISTRY:
            return MODEL_REGISTRY[key]["id"]
    # Fallback: same as the user-selected default
    return MODEL_REGISTRY[DEFAULT_MODEL_KEY]["id"]


def _build_history(chat: dict | None, exclude_last: bool = False) -> list:
    if not chat:
        return []
    turns = chat.get("turns", []) or []
    if exclude_last and turns:
        turns = turns[:-1]  # the turn we're about to replace
    history_limit = settings_store.get("history_turn_limit") or 0
    if history_limit <= 0:
        return []
    history: list = []
    for turn in turns[-history_limit:]:
        question = turn.get("question")
        answer = turn.get("answer")
        if question:
            history.append(HumanMessage(content=question))
        if answer:
            history.append(AIMessage(content=answer))
    return history


def _today_context_message() -> SystemMessage:
    today = date.today()
    yesterday = today - timedelta(days=1)
    today_pretty = today.strftime("%A, %B %d, %Y")
    yesterday_pretty = yesterday.strftime("%A, %B %d, %Y")
    return SystemMessage(
        content=(
            f"Today's date is {today.isoformat()} ({today_pretty}). "
            f"Yesterday was {yesterday.isoformat()} ({yesterday_pretty}). "
            "Relative time words ('yesterday', 'today', 'this week') in user questions are "
            "automatically substituted with the absolute date when they reach the search tool, "
            "so you do not need to translate them manually — but you MUST still anchor your final "
            "answer to the correct date. If the user asked about 'yesterday', report only results "
            f"from {yesterday_pretty}, even if the search returned articles from other days. "
            "Your training data is from before today and is unreliable for any time-sensitive fact. "
            "For any question about current people, prices, events, statistics, dates, or recent developments, "
            "you MUST use your search tools and base your answer exclusively on the retrieved content. "
            "Do not state remembered facts from training as if they were current truth."
        )
    )


def _question_with_date_prefix(question: str) -> str:
    """Inline a short date marker right before the user's question.

    Small models (Llama 3.2 3B in particular) have weak long-range attention
    and may not anchor to the SystemMessage by the time they process the
    question. Repeating the date adjacent to the question is cheap insurance.
    """
    today = date.today()
    yesterday = today - timedelta(days=1)
    return (
        f"[Context: Today is {today.isoformat()} ({today.strftime('%A, %B %d, %Y')}). "
        f"Yesterday was {yesterday.isoformat()} ({yesterday.strftime('%A, %B %d, %Y')}).]\n\n"
        f"{question}"
    )


@app.get("/ask_agent_stream")
async def ask_agent_stream(
    request: Request,
    question: str,
    chat_id: str | None = None,
    model: str | None = None,
    regenerate: bool = False,
):
    async def generate():
        cancelled = threading.Event()
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

            settings = settings_store.get_all()
            cache_ttl = settings.get("cache_ttl_seconds", 24 * 3600)
            verification_enabled = settings.get("verification_enabled", True)
            fact_extraction_enabled = settings.get("fact_extraction_enabled", True)

            # Cache check (skip when memory is active OR when this is an explicit
            # regenerate — the user is asking for a fresh attempt, not a cached one).
            if not has_memory and not regenerate and cache_ttl > 0:
                cached = cache_store.get(model_key, question, cache_ttl)
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
            history_messages = _build_history(chat, exclude_last=regenerate)
            # Today's-date context sits IMMEDIATELY before the question (not before
            # history) so the model's attention to the date is maximal when it reads
            # the question. The question itself also carries an inline date marker
            # for small models with weak long-range attention.
            input_messages = (
                history_messages
                + [_today_context_message()]
                + [HumanMessage(content=_question_with_date_prefix(question))]
            )

            # ── Run the agent in a background thread ─────────────────────────
            # We need real cancellation when the client disconnects, plus the
            # ability to interleave disconnect checks with chunk processing.
            # Solution: agent.stream() runs in a thread, producing chunks into
            # an asyncio.Queue. The async loop pulls with a short timeout so it
            # can poll request.is_disconnected() in between.
            chunk_queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def _enqueue(item):
                loop.call_soon_threadsafe(chunk_queue.put_nowait, item)

            def run_agent():
                stream_iter = None
                try:
                    stream_iter = agent.stream(
                        {"messages": input_messages},
                        stream_mode=["updates", "messages"],
                    )
                    for chunk in stream_iter:
                        if cancelled.is_set():
                            return
                        _enqueue(("chunk", chunk))
                except Exception as exc:
                    _enqueue(("error", exc))
                finally:
                    # Explicitly close the stream so the underlying httpx
                    # connection to Ollama is released — this is what actually
                    # tells Ollama to stop generating.
                    if stream_iter is not None and hasattr(stream_iter, "close"):
                        try:
                            stream_iter.close()
                        except Exception:
                            pass
                    _enqueue(("done", None))

            agent_thread = threading.Thread(target=run_agent, daemon=True)
            agent_thread.start()

            tool_messages_collected: list[dict] = []
            final_answer = ""

            while True:
                try:
                    item = await asyncio.wait_for(chunk_queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        cancelled.set()
                        logger.info("Client disconnected — stopping agent (chat=%s)", resolved_chat_id)
                        return
                    continue

                kind, payload = item
                if kind == "done":
                    break
                if kind == "error":
                    raise payload

                # kind == "chunk": LangGraph yields (stream_mode, data) tuples
                if not isinstance(payload, tuple) or len(payload) != 2:
                    continue
                stream_mode, data = payload

                if stream_mode == "messages":
                    # data is (message_chunk, metadata) — emit each token
                    msg_chunk = data[0] if isinstance(data, tuple) and data else None
                    if msg_chunk is None:
                        continue
                    content = getattr(msg_chunk, "content", "")
                    if isinstance(content, str) and content:
                        yield _sse("token", {"text": content})
                    continue

                if stream_mode != "updates":
                    continue
                if not isinstance(data, dict):
                    continue
                for _node_name, node_data in data.items():
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

            # Verification pass — re-check the draft answer against tool results
            # and rewrite any claim the search results contradict or don't support.
            if verification_enabled and final_answer and tool_messages_collected:
                if await request.is_disconnected():
                    return
                yield _sse("verifying", {})
                verified_answer, was_changed = await asyncio.to_thread(
                    verify_answer,
                    question=question,
                    answer=final_answer,
                    tool_messages=tool_messages_collected,
                    model_id=_verifier_model_id(),
                    today_iso=date.today().isoformat(),
                )
                if was_changed:
                    final_answer = verified_answer
                    yield _sse("answer", {"text": final_answer, "verified": True})

            if await request.is_disconnected():
                return

            yield _sse("extracting", {})

            cleaned_answer = final_answer.replace("\n", " ")
            extraction = await asyncio.to_thread(
                extract_claims,
                cleaned_answer,
                tool_messages_collected,
                model_id=model_info["id"],
                use_llm=fact_extraction_enabled,
            )

            turn_payload = {
                "question": question,
                "answer": final_answer,
                "facts": extraction["facts"],
                "sources": extraction["sources"],
                "trust_signals": extraction["trust_signals"],
                "model": model_key,
                "from_cache": False,
                "regenerated": regenerate,
            }
            if regenerate:
                saved_chat = chat_store.replace_last_turn(resolved_chat_id, turn_payload)
            else:
                saved_chat = chat_store.append_turn(resolved_chat_id, turn_payload)

            # Cache only on the first-write path. Regenerations bypass cache by design.
            if not has_memory and not regenerate and final_answer and cache_ttl > 0:
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
            logger.exception("Agent stream failed: %s", exc)
            yield _sse("error", {"message": str(exc)})
        finally:
            # Always signal the agent thread to exit, even on disconnect/exception
            cancelled.set()

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
    """Legacy non-streaming endpoint. Useful for command-line / Swagger testing.
    The streaming endpoint /ask_agent_stream is what the UI uses."""
    try:
        model_key = _resolve_model_key(model)
        model_info = MODEL_REGISTRY[model_key]
        agent = AGENTS[model_key]

        input_message = {
            "messages": [
                _today_context_message(),
                HumanMessage(content=_question_with_date_prefix(question)),
            ]
        }
        result = agent.invoke(input_message)

        logger.info("=== agent reasoning trace (%s) ===", model_info["label"])
        for message in result["messages"]:
            if message.type == "ai" and getattr(message, "tool_calls", None):
                for tool_call in message.tool_calls:
                    logger.info("  tool=%s args=%s", tool_call["name"], tool_call["args"])
            elif message.type == "tool":
                tool_output = str(message.content)
                preview = tool_output[:TOOL_LOG_PREVIEW_CHARS]
                truncated = " ...(truncated)" if len(tool_output) > TOOL_LOG_PREVIEW_CHARS else ""
                logger.info("  result[%s]: %s%s", message.name, preview, truncated)

        # Keep newlines so markdown formatting in the answer survives.
        final_answer = result["messages"][-1].content
        tool_messages = [
            {"name": message.name, "content": str(message.content)}
            for message in result["messages"]
            if message.type == "tool"
        ]
        extraction = extract_claims(final_answer, tool_messages, model_id=model_info["id"])
        logger.info("trust_signals: %s", extraction["trust_signals"])

        return {
            "your_question": question,
            "model": model_key,
            "agent_answer": final_answer,
            "facts": extraction["facts"],
            "sources": extraction["sources"],
            "trust_signals": extraction["trust_signals"],
        }
    except Exception as exc:
        logger.exception("ask_agent failed: %s", exc)
        return {"error": f"Agent error: {exc}"}

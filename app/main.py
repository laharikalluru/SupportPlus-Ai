"""
main.py - FastAPI application entry point for SupportPlus AI.

Endpoints:
  POST /ask        – Answer a user query using RAG + Web + Memory + LLM
  GET  /history    – Retrieve a user's interaction history
  GET  /health     – System health check
  POST /reindex    – Force re-index of the FAQ knowledge base (admin)
  DELETE /history  – Delete all stored history for a user (GDPR)
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import (
    APP_TITLE,
    APP_VERSION,
    CORS_ORIGINS,
    RAG_TOP_K,
    LLM_LIGHTWEIGHT_MODE,
    LLM_MEMORY_TOP_K,
)
from app.services.rag_service import get_rag_service
from app.services.memory_service import get_memory_service
from app.services.web_service import get_web_service
from app.services.llm_service import get_llm_service

logger = logging.getLogger("supportplus.main")

# After this many consecutive LLM failures, auto-enable lightweight mode (FAQ + memory only).
_LLM_STRESS_THRESHOLD = 2
_llm_consecutive_failures = 0


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's support question.",
        examples=["How do I reset my password?"],
    )
    user_id: str = Field(
        default="anonymous",
        min_length=1,
        max_length=128,
        description="Unique identifier for the user (enables memory).",
        examples=["user_abc123"],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Number of FAQ chunks to retrieve (overrides server default).",
    )
    web_results: Optional[int] = Field(
        default=1,
        ge=0,
        le=5,
        description="Number of web search results to fetch (server caps at 1 for efficiency).",
    )
    lightweight: Optional[bool] = Field(
        default=None,
        description=(
            "If true, skip web search and shrink the LLM prompt. "
            "If null, the server may auto-enable after repeated LLM errors."
        ),
    )


class AskResponse(BaseModel):
    user_id: str
    query: str
    response: str
    sources: dict
    model_used: str = "unknown"   # "gemini" | "groq" | "fallback" | "unknown"
    latency_ms: float


class HistoryEntry(BaseModel):
    id: int | str
    query: str
    response: str
    created_at: str


class HistoryResponse(BaseModel):
    user_id: str
    total: int
    interactions: list[HistoryEntry]


class HealthResponse(BaseModel):
    status: str
    version: str
    components: dict


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up all services at startup."""
    logger.info("=== SupportPlus AI starting up ===")
    # Initialise singletons eagerly (triggers FAQ indexing if needed)
    get_rag_service()
    get_memory_service()
    get_web_service()
    get_llm_service()
    logger.info("=== All services ready ===")
    yield
    logger.info("=== SupportPlus AI shutting down ===")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "Intelligent AI support system combining FAQ knowledge base, "
        "real-time web data (Firecrawl), user memory, and multi-model "
        "LLM response generation (Gemini primary → Groq fallback → FAQ+Web)."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a support question",
    tags=["Support"],
)
async def ask(request: AskRequest):
    """
    Main RAG pipeline endpoint.

    Flow:
      1. Retrieve top-k relevant FAQ chunks (vector search)
      2. Fetch live web results (Firecrawl or mock), skipped in lightweight mode
      3. Load user memory context (latest interaction only by default)
      4. Generate LLM response via ``safe_generate`` (never raises; no 503 from LLM)
      5. Store interaction in memory
      6. Return structured response with source metadata
    """
    global _llm_consecutive_failures
    start = time.perf_counter()
    req_chars = len(request.query or "")
    logger.info(
        "POST /ask user=%s req_chars=%d query_preview=%r",
        request.user_id,
        req_chars,
        (request.query or "")[:120],
    )

    try:
        rag = get_rag_service()
        mem = get_memory_service()
        web = get_web_service()
        llm = get_llm_service()

        auto_lw = _llm_consecutive_failures >= _LLM_STRESS_THRESHOLD
        use_lightweight = bool(
            LLM_LIGHTWEIGHT_MODE
            or (request.lightweight is True)
            or (request.lightweight is None and auto_lw)
        )

        # ── Step 1: FAQ retrieval (deduped per entry; default RAG_TOP_K, max 10)
        top_k = request.top_k if request.top_k is not None else RAG_TOP_K
        top_k = max(1, min(int(top_k), 10))
        faq_results = rag.retrieve(request.query, top_k=top_k)

        # ── Step 2: Web search (optional) ─────────────────────────────────────
        web_cap = 0 if use_lightweight else min(int(request.web_results or 1), 1)
        if web_cap == 0:
            web_results = []
            if use_lightweight:
                logger.info("Lightweight mode active — web search skipped.")
        else:
            web_results = await web.search(request.query, limit=web_cap)

        # ── Step 3: Memory (latest only) ──────────────────────────────────────
        memory_context = mem.format_memory_context(
            request.user_id, limit=max(1, LLM_MEMORY_TOP_K)
        )

        # ── Step 4: LLM (always safe_generate — never 503 from Gemini) ────────
        pipeline_metrics: dict = {}
        answer = await llm.safe_generate(
            query=request.query,
            faq_results=faq_results,
            web_results=web_results,
            memory_context=memory_context,
            lightweight=use_lightweight,
            metrics=pipeline_metrics,
        )
        # ── Step 5: Persist to memory (track which model was used) ────────
        llm_used = pipeline_metrics.get("llm", "unknown")
        # Only count as a failure when BOTH Gemini AND Groq failed
        if llm_used in ("gemini", "groq"):
            _llm_consecutive_failures = 0
        else:  # "fallback" or "unknown"
            _llm_consecutive_failures += 1
            logger.warning(
                "Both LLMs failed — FAQ+Web fallback served "
                "(consecutive_stress=%d): %s",
                _llm_consecutive_failures,
                pipeline_metrics.get("gemini_error", ""),
            )

        # ── Step 5: Persist to memory ─────────────────────────────────────────
        source_info = llm.rank_sources(faq_results, web_results)
        source_info["pipeline"] = {
            "lightweight": use_lightweight,
            "llm": pipeline_metrics.get("llm", "unknown"),
            "auto_lightweight": auto_lw and request.lightweight is None,
        }
        mem.save_interaction(
            user_id=request.user_id,
            query=request.query,
            response=answer,
            metadata=source_info,
        )

        # ── Step 6: Build response ────────────────────────────────────────────
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "POST /ask done user=%s latency_ms=%.2f confidence=%.4f llm=%s model=%s lw=%s",
            request.user_id,
            elapsed_ms,
            source_info["combined_confidence"],
            pipeline_metrics.get("llm"),
            pipeline_metrics.get("model_used", "?"),
            use_lightweight,
        )

        return AskResponse(
            user_id=request.user_id,
            query=request.query,
            response=answer,
            model_used=pipeline_metrics.get("llm", "unknown"),
            sources={
                "faq": [
                    {
                        "question": r["question"],
                        "section": r["section"],
                        "score": r["score"],
                    }
                    for r in faq_results
                ],
                "web": [
                    {"title": r["title"], "url": r["url"], "score": r["score"]}
                    for r in web_results
                ],
                "confidence": source_info,
            },
            latency_ms=elapsed_ms,
        )

    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.exception(
            "POST /ask unexpected error after %.2f ms (user=%s): %s",
            elapsed_ms,
            request.user_id,
            exc,
        )
        return AskResponse(
            user_id=request.user_id,
            query=request.query,
            response=(
                "We hit an unexpected issue processing your request. "
                "Please try again in a moment. If it keeps happening, contact support."
            ),
            sources={
                "faq": [],
                "web": [],
                "confidence": {
                    "faq_top_score": 0.0,
                    "web_top_score": 0.0,
                    "combined_confidence": 0.0,
                    "faq_chunks": 0,
                    "web_results": 0,
                    "pipeline": {"error": "server_exception", "detail": str(exc)[:200]},
                },
            },
            latency_ms=elapsed_ms,
        )


@app.get(
    "/history",
    response_model=HistoryResponse,
    summary="Get user interaction history",
    tags=["Memory"],
)
def get_history(
    user_id: str = Query(..., description="User ID to fetch history for."),
    limit: int = Query(default=10, ge=1, le=100, description="Max interactions to return."),
):
    """Return the stored conversation history for a given user."""
    mem = get_memory_service()
    interactions = mem.get_history(user_id, limit=limit)
    return HistoryResponse(
        user_id=user_id,
        total=len(interactions),
        interactions=[
            HistoryEntry(
                id=entry.get("id", 0),
                query=entry["query"],
                response=entry["response"],
                created_at=entry["created_at"],
            )
            for entry in interactions
        ],
    )


@app.delete(
    "/history",
    summary="Delete user interaction history (GDPR)",
    tags=["Memory"],
)
def delete_history(
    user_id: str = Query(..., description="User ID whose history should be deleted."),
):
    """Permanently delete all stored interactions for a user."""
    mem = get_memory_service()
    deleted = mem.delete_user_data(user_id)
    return {"user_id": user_id, "deleted_interactions": deleted, "status": "ok"}


@app.post(
    "/reindex",
    summary="Re-index FAQ knowledge base (admin)",
    tags=["Admin"],
)
def reindex():
    """Force a full re-index of the FAQ text file into the vector store."""
    rag = get_rag_service()
    rag.reindex()
    return {"status": "ok", "message": "FAQ knowledge base re-indexed successfully."}


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    tags=["System"],
)
def health():
    """Return the health status of all system components."""
    rag = get_rag_service()
    mem = get_memory_service()
    web = get_web_service()
    llm = get_llm_service()

    components = {
        "rag":    rag.health(),
        "memory": mem.health(),
        "web":    web.health(),
        "llm":    llm.health(),
    }

    # Overall status: ok only if all components are ok
    overall = "ok" if all(c.get("status") == "ok" for c in components.values()) else "degraded"

    return HealthResponse(
        status=overall,
        version=APP_VERSION,
        components=components,
    )


@app.get("/", tags=["System"])
def root():
    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }

import logging
import time
import uuid
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.models import (
    ChatRequest, ChatResponse,
    HealthResponse, IngestResponse, IngestStatusResponse,
    EvaluationResponse, StatsResponse, ClearSessionResponse,
)
from core.config import settings
from core.metrics import METRICS

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# in-memory ingest state — good enough since ingestion is a single background task
_ingest_state = {
    "status": "idle",
    "progress": 0,
    "total": 0,
    "last_completed_at": None,
}

_eval_cache = {"result": None, "timestamp": 0}
EVAL_CACHE_TTL = 3600

_stats = {"queries_served": 0, "total_response_time_ms": 0}


def _ingest_progress_callback(progress: int, total: int):
    _ingest_state["progress"] = progress
    _ingest_state["total"] = total


def _run_ingestion():
    _ingest_state["status"] = "running"
    _ingest_state["progress"] = 0
    _ingest_state["total"] = 0
    start = time.time()
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data.ingest import run_ingest
        run_ingest(progress_callback=_ingest_progress_callback)
        _ingest_state["status"] = "completed"
        _ingest_state["last_completed_at"] = datetime.utcnow().isoformat()

        # rebuild BM25 index so it reflects the new data
        from rag.hybrid_search import rebuild_searcher
        rebuild_searcher()

        elapsed = time.time() - start
        METRICS.ingest_runs.labels("success").inc()
        METRICS.ingest_duration.observe(elapsed)

        # update the knowledge base gauge
        from rag.vectorstore import get_counts_by_source
        counts = get_counts_by_source()
        METRICS.kb_document_count.labels("NVD").set(counts.get("cve", 0))
        METRICS.kb_document_count.labels("OWASP").set(counts.get("owasp", 0))
        METRICS.ingest_docs_added.set(counts.get("total", 0))

        # clear the semantic cache — answers might change with new CVE data
        from rag.semantic_cache import get_cache
        get_cache().invalidate()

        logger.info("Ingestion done in %.1fs — %d total docs", elapsed, counts.get("total", 0))
    except Exception as e:
        METRICS.ingest_runs.labels("failure").inc()
        logger.error("Ingestion failed: %s", e, exc_info=True)
        _ingest_state["status"] = "failed"


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.rate_limit_chat)
async def chat(req: ChatRequest, request: Request):
    from rag.vectorstore import get_document_count
    if get_document_count() == 0:
        METRICS.chat_requests.labels("error").inc()
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not yet populated. Run POST /api/ingest first.",
        )

    session_id = req.session_id or str(uuid.uuid4())
    start = time.time()

    from rag.chain import answer_query
    result = answer_query(
        req.message,
        session_id,
        severity_filter=req.severity_filter,
        source_filter=req.source_filter,
    )

    elapsed_ms = int((time.time() - start) * 1000)
    _stats["queries_served"] += 1
    _stats["total_response_time_ms"] += elapsed_ms

    meta = result["search_metadata"]
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=session_id,
        search_metadata={
            "bm25_hits": meta.get("bm25_hits", 0),
            "semantic_hits": meta.get("semantic_hits", 0),
            "rrf_final": meta.get("rrf_final", 0),
            "filtered_out": meta.get("filtered_out", 0),
        },
        response_time_ms=elapsed_ms,
        verification=result.get("verification"),
    )


@router.post("/chat/stream")
@limiter.limit(settings.rate_limit_chat)
async def chat_stream(req: ChatRequest, request: Request):
    """SSE endpoint — streams LLM tokens as they arrive instead of waiting for the full response.

    Events:
      - {"type": "metadata", "sources": [...], "search_metadata": {...}}
      - {"type": "token", "content": "..."}   (one per LLM token)
      - {"type": "verification", "cited": [...], "verified": [...], "unverified": [...], "ok": bool}
      - {"type": "done", "response_time_ms": N}
      - {"type": "error", "message": "..."}   (only on failure)
    """
    from rag.vectorstore import get_document_count
    if get_document_count() == 0:
        raise HTTPException(status_code=503, detail="Knowledge base empty.")

    session_id = req.session_id or str(uuid.uuid4())

    from rag.chain import stream_answer_query
    return StreamingResponse(
        stream_answer_query(
            req.message,
            session_id,
            severity_filter=req.severity_filter,
            source_filter=req.source_filter,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # tells nginx not to buffer SSE
            "X-Session-ID": session_id,
        },
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    from rag.vectorstore import get_document_count
    try:
        doc_count = get_document_count()
    except Exception:
        doc_count = 0

    from rag.embeddings import _embedding_model
    from rag.reranker import _cross_encoder
    models_ok = _embedding_model is not None and _cross_encoder is not None

    return HealthResponse(
        status="healthy" if models_ok else "degraded",
        chromadb_documents=doc_count,
        models_loaded=models_ok,
        version=settings.app_version,
    )


@router.post("/ingest", response_model=IngestResponse)
@limiter.limit(settings.rate_limit_ingest)
async def ingest(
    request: Request,
    background_tasks: BackgroundTasks,
    x_ingest_key: str = Header(None, alias="X-Ingest-Key"),
):
    if x_ingest_key != settings.ingest_key:
        raise HTTPException(status_code=401, detail="Invalid ingest key.")
    if _ingest_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Ingestion already in progress.")
    background_tasks.add_task(_run_ingestion)
    return IngestResponse(status="started", message="Ingestion running in background...")


@router.get("/ingest/status", response_model=IngestStatusResponse)
async def ingest_status():
    # also include next scheduled refresh time if scheduler is running
    next_run = None
    try:
        from core.scheduler import get_scheduler
        job = get_scheduler().get_job("weekly_cve_refresh")
        if job and job.next_run_time:
            next_run = job.next_run_time.isoformat()
    except Exception:
        pass   # scheduler might not be initialised yet

    return IngestStatusResponse(
        status=_ingest_state["status"],
        progress=_ingest_state["progress"],
        total=_ingest_state["total"],
        last_completed_at=_ingest_state["last_completed_at"],
        next_scheduled_at=next_run,
    )


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate():
    now = time.time()
    if _eval_cache["result"] and (now - _eval_cache["timestamp"]) < EVAL_CACHE_TTL:
        return EvaluationResponse(**_eval_cache["result"])

    from rag.evaluate import run_evaluation
    result = run_evaluation()
    _eval_cache["result"] = result
    _eval_cache["timestamp"] = now

    return EvaluationResponse(
        scores=result.get("scores", {}),
        per_query_results=result.get("per_query_results", []),
        evaluated_at=result.get("evaluated_at", datetime.utcnow().isoformat()),
        num_test_queries=result.get("num_test_queries", 0),
        error=result.get("error"),
    )


@router.get("/stats", response_model=StatsResponse)
async def stats():
    from rag.vectorstore import get_counts_by_source
    from rag.semantic_cache import get_cache
    counts = get_counts_by_source()
    q = _stats["queries_served"]
    avg_rt = _stats["total_response_time_ms"] / q if q > 0 else 0.0
    return StatsResponse(
        total_documents=counts["total"],
        cve_count=counts["cve"],
        owasp_count=counts["owasp"],
        queries_served=q,
        avg_response_time_ms=avg_rt,
        semantic_cache_entries=get_cache().size,
    )


@router.delete("/session/{session_id}", response_model=ClearSessionResponse)
async def clear_session(session_id: str):
    from rag.chain import clear_session as _clear
    _clear(session_id)
    return ClearSessionResponse(cleared=True)

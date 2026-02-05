import logging
import os
import time
import uuid
from datetime import datetime
from fastapi import APIRouter, Header, HTTPException, BackgroundTasks, Request

from api.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    IngestStatusResponse,
    EvaluationResponse,
    StatsResponse,
    ClearSessionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_ingest_state = {
    "status": "idle",
    "progress": 0,
    "total": 0,
    "last_completed_at": None,
}

_eval_cache = {"result": None, "timestamp": 0}
EVAL_CACHE_TTL = 3600

_stats = {
    "queries_served": 0,
    "total_response_time_ms": 0,
}


def _run_ingestion():
    _ingest_state["status"] = "running"
    _ingest_state["progress"] = 0
    _ingest_state["total"] = 0
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data.ingest import run_ingest
        run_ingest(progress_callback=_ingest_progress_callback)
        _ingest_state["status"] = "completed"
        _ingest_state["last_completed_at"] = datetime.utcnow().isoformat()
        from rag.hybrid_search import rebuild_searcher
        rebuild_searcher()
        logger.info("Ingestion completed and BM25 index rebuilt.")
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        _ingest_state["status"] = "failed"


def _ingest_progress_callback(progress: int, total: int):
    _ingest_state["progress"] = progress
    _ingest_state["total"] = total


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    from rag.vectorstore import get_document_count
    if get_document_count() == 0:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not yet populated. Run data ingestion first.",
        )

    if not req.session_id:
        session_id = str(uuid.uuid4())
    else:
        session_id = req.session_id

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
    models_loaded = _embedding_model is not None and _cross_encoder is not None

    return HealthResponse(
        status="healthy",
        chromadb_documents=doc_count,
        models_loaded=models_loaded,
        version="1.0.0",
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    background_tasks: BackgroundTasks,
    x_ingest_key: str = Header(None, alias="X-Ingest-Key"),
):
    expected_key = os.getenv("INGEST_KEY", "securerag-ingest-2024")
    if x_ingest_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid ingest key.")
    if _ingest_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Ingestion already running.")
    background_tasks.add_task(_run_ingestion)
    return IngestResponse(status="started", message="Ingestion running in background...")


@router.get("/ingest/status", response_model=IngestStatusResponse)
async def ingest_status():
    return IngestStatusResponse(
        status=_ingest_state["status"],
        progress=_ingest_state["progress"],
        total=_ingest_state["total"],
        last_completed_at=_ingest_state["last_completed_at"],
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
    counts = get_counts_by_source()
    queries = _stats["queries_served"]
    avg_rt = (
        _stats["total_response_time_ms"] / queries if queries > 0 else 0.0
    )
    return StatsResponse(
        total_documents=counts["total"],
        cve_count=counts["cve"],
        owasp_count=counts["owasp"],
        queries_served=queries,
        avg_response_time_ms=avg_rt,
    )


@router.delete("/session/{session_id}", response_model=ClearSessionResponse)
async def clear_session(session_id: str):
    from rag.chain import clear_session as _clear
    cleared = _clear(session_id)
    return ClearSessionResponse(cleared=True)

import json
import logging
import time
from collections import OrderedDict

import httpx
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from core.config import settings
from core.metrics import METRICS
from rag.hybrid_search import get_searcher
from rag.reranker import get_reranker
from rag.citations import verify_citations
from rag.semantic_cache import get_cache
from rag.embeddings import embed_query

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are SecureRAG, an expert cybersecurity assistant. Answer questions about vulnerabilities, CVEs, and security best practices based ONLY on the provided context.

Rules:
1. If the context doesn't contain enough information, say "I don't have specific information about that in my knowledge base."
2. Always cite your sources: mention CVE IDs or OWASP categories referenced
3. For CVSS scores, always mention the severity level (Critical/High/Medium/Low)
4. Keep answers concise but complete. Use bullet points for lists.
5. Never make up CVE IDs or scores that aren't in the context.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:"""

_session_store: OrderedDict = OrderedDict()

_llm: ChatGroq | None = None
_llm_api_key: str = ""


def get_llm() -> ChatGroq:
    global _llm, _llm_api_key
    current_key = settings.groq_api_key
    if _llm is None or current_key != _llm_api_key:
        if not current_key:
            logger.warning("GROQ_API_KEY is not set — LLM calls will fail")
        _llm = ChatGroq(
            model=settings.groq_model,
            temperature=settings.groq_temperature,
            max_tokens=settings.groq_max_tokens,
            api_key=current_key,
        )
        _llm_api_key = current_key
    return _llm


# tenacity retry for Groq — groq is occasionally flaky, especially at off-peak hours
@retry(
    stop=stop_after_attempt(settings.groq_max_retries),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_llm_with_retry(llm: ChatGroq, messages: list) -> str:
    METRICS.groq_retries.inc()
    response = llm.invoke(messages)
    return response.content


def _classify_groq_error(e: Exception) -> str:
    if isinstance(e, httpx.TimeoutException):
        return "timeout"
    if isinstance(e, httpx.HTTPStatusError):
        if e.response.status_code == 429:
            return "rate_limit"
        if e.response.status_code in (401, 403):
            return "auth"
    return "unknown"


def _apply_filters(docs, severity_filter, source_filter):
    if not severity_filter:
        severity_filter = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    if not source_filter:
        source_filter = "both"

    allowed = {s.upper() for s in severity_filter}
    sf = source_filter.lower()

    out = []
    for text, meta, score in docs:
        src = meta.get("source", "").upper()
        if sf == "nvd" and src != "NVD":
            continue
        if sf == "owasp" and src != "OWASP":
            continue
        if src == "NVD":
            sev = meta.get("severity", "UNKNOWN").upper()
            if sev not in allowed:
                continue
        out.append((text, meta, score))

    return out, len(docs) - len(out)


def _format_sources(docs):
    sources = []
    seen = set()
    for _, meta, score in docs:
        src_type = meta.get("source", "UNKNOWN")
        if src_type == "NVD":
            cve_id = meta.get("cve_id", "UNKNOWN")
            if cve_id not in seen:
                seen.add(cve_id)
                sources.append({
                    "id": cve_id,
                    "source_type": "NVD",
                    "severity": meta.get("severity", "UNKNOWN"),
                    "score": meta.get("cvss_score", 0.0),
                })
        elif src_type == "OWASP":
            owasp_id = meta.get("owasp_id", "UNKNOWN")
            if owasp_id not in seen:
                seen.add(owasp_id)
                sources.append({
                    "id": owasp_id,
                    "source_type": "OWASP",
                    "category": meta.get("category_name", ""),
                })
    return sources


def _format_context(docs) -> str:
    parts = []
    for text, meta, _ in docs:
        src = meta.get("source", "UNKNOWN")
        label = meta.get("cve_id", "CVE") if src == "NVD" else meta.get("owasp_id", "OWASP")
        parts.append(f"[{label}]\n{text}")
    return "\n\n---\n\n".join(parts)


def _format_chat_history(history: list[dict]) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for turn in history[-settings.max_history_turns:]:
        lines.append(f"Human: {turn['human']}")
        lines.append(f"Assistant: {turn['assistant']}")
    return "\n".join(lines)


def _get_session(session_id: str) -> list[dict]:
    if session_id not in _session_store:
        if len(_session_store) >= settings.max_sessions:
            _session_store.popitem(last=False)
        _session_store[session_id] = []
    _session_store.move_to_end(session_id)
    return _session_store[session_id]


def clear_session(session_id: str) -> bool:
    if session_id in _session_store:
        del _session_store[session_id]
        return True
    return False


def answer_query(
    question: str,
    session_id: str,
    severity_filter=None,
    source_filter=None,
) -> dict:
    total_start = time.perf_counter()
    history = _get_session(session_id)

    # semantic cache — only valid for first-turn queries (no session history)
    # multi-turn queries depend on context so we can't safely cache them
    cache = get_cache()
    query_embedding = None
    if not history:
        query_embedding = embed_query(question)
        cached_result = cache.get(query_embedding)
        if cached_result:
            METRICS.cache_hits.labels("semantic").inc()
            # still need to save to history so follow-up questions work
            history.append({"human": question, "assistant": cached_result["answer"]})
            return cached_result
        METRICS.cache_misses.labels("semantic").inc()

    # --- retrieval ---
    t0 = time.perf_counter()
    searcher = get_searcher()
    retrieved, search_meta = searcher.search(question, top_k=settings.retrieval_top_k)
    METRICS.rag_stage_duration.labels("retrieval").observe(time.perf_counter() - t0)
    METRICS.retrieval_count.labels("bm25").observe(search_meta.get("bm25_hits", 0))
    METRICS.retrieval_count.labels("semantic").observe(search_meta.get("semantic_hits", 0))
    METRICS.retrieval_count.labels("rrf").observe(search_meta.get("rrf_final", 0))

    # --- rerank ---
    t0 = time.perf_counter()
    reranker = get_reranker()
    reranked = reranker.rerank(question, retrieved, top_k=settings.rerank_top_k)
    METRICS.rag_stage_duration.labels("rerank").observe(time.perf_counter() - t0)
    METRICS.retrieval_count.labels("reranked").observe(len(reranked))
    for _, _, sc in reranked:
        METRICS.reranker_score.observe(sc)

    # filter after reranking so we don't lose ranking quality
    filtered, n_filtered = _apply_filters(reranked, severity_filter, source_filter)
    search_meta["filtered_out"] = n_filtered

    if not filtered and reranked:
        logger.warning("All results filtered out — falling back to unfiltered set")
        filtered = reranked
        search_meta["filtered_out"] = 0

    context = _format_context(filtered)
    sources = _format_sources(filtered)

    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        context=context,
        chat_history=_format_chat_history(history),
        question=question,
    )

    # --- LLM ---
    t0 = time.perf_counter()
    llm = get_llm()
    try:
        answer = _call_llm_with_retry(llm, [HumanMessage(content=prompt)])
        METRICS.chat_requests.labels("success").inc()
    except Exception as e:
        err_type = _classify_groq_error(e)
        METRICS.groq_errors.labels(err_type).inc()
        METRICS.chat_requests.labels("error").inc()
        logger.error("LLM call failed (%s): %s", err_type, e, exc_info=True)
        answer = "I'm sorry, I couldn't reach the language model. Please check your GROQ_API_KEY and try again."

    METRICS.rag_stage_duration.labels("llm").observe(time.perf_counter() - t0)
    METRICS.rag_stage_duration.labels("total").observe(time.perf_counter() - total_start)

    verification = verify_citations(answer, filtered, question, history)
    if verification["unverified"]:
        METRICS.unverified_citations.inc(len(verification["unverified"]))
        logger.warning("Unverified citations in answer: %s", verification["unverified"])

    result = {
        "answer": answer,
        "sources": sources,
        "search_metadata": search_meta,
        "verification": verification,
    }

    # write to semantic cache — only cache if it was a first-turn query
    # (query_embedding is None when we skipped cache because history existed)
    if query_embedding is not None:
        cache.set(query_embedding, result)
        METRICS.semantic_cache_size.set(cache.size)

    history.append({"human": question, "assistant": answer})
    if len(history) > settings.max_history_turns * 2:
        history.pop(0)

    return result


async def stream_answer_query(
    question: str,
    session_id: str,
    severity_filter=None,
    source_filter=None,
):
    """Async generator that yields SSE-formatted strings.

    The retrieval + reranking pipeline runs first (fast, CPU/local),
    then we stream tokens from the LLM as they arrive from Groq.
    """
    total_start = time.perf_counter()

    # retrieval is synchronous and fast — do it normally
    searcher = get_searcher()
    retrieved, search_meta = searcher.search(question, top_k=settings.retrieval_top_k)

    reranker = get_reranker()
    reranked = reranker.rerank(question, retrieved, top_k=settings.rerank_top_k)

    filtered, n_filtered = _apply_filters(reranked, severity_filter, source_filter)
    search_meta["filtered_out"] = n_filtered
    if not filtered and reranked:
        filtered = reranked
        search_meta["filtered_out"] = 0

    context = _format_context(filtered)
    sources = _format_sources(filtered)
    history = _get_session(session_id)

    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        context=context,
        chat_history=_format_chat_history(history),
        question=question,
    )

    # send sources and metadata immediately so the frontend can render them
    # before the first token arrives
    meta_event = {
        "type": "metadata",
        "sources": sources,
        "search_metadata": search_meta,
    }
    yield f"data: {json.dumps(meta_event)}\n\n"

    # stream tokens
    llm = get_llm()
    full_answer = ""
    try:
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            token = chunk.content
            if token:
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        METRICS.chat_requests.labels("success").inc()
    except Exception as e:
        err_type = _classify_groq_error(e)
        METRICS.groq_errors.labels(err_type).inc()
        METRICS.chat_requests.labels("error").inc()
        logger.error("LLM stream failed (%s): %s", err_type, e, exc_info=True)
        full_answer = "Stream interrupted — please try again."
        yield f"data: {json.dumps({'type': 'error', 'message': full_answer})}\n\n"

    # flag (never rewrite) IDs the context can't support; skipped when the stream failed
    if full_answer and not full_answer.startswith("Stream interrupted"):
        verification = verify_citations(full_answer, filtered, question, history)
        if verification["unverified"]:
            METRICS.unverified_citations.inc(len(verification["unverified"]))
            logger.warning("Unverified citations in streamed answer: %s", verification["unverified"])
        yield f"data: {json.dumps({'type': 'verification', **verification})}\n\n"

    elapsed_ms = int((time.perf_counter() - total_start) * 1000)
    METRICS.rag_stage_duration.labels("total").observe(elapsed_ms / 1000)

    yield f"data: {json.dumps({'type': 'done', 'response_time_ms': elapsed_ms})}\n\n"

    # save to history
    if full_answer:
        history.append({"human": question, "assistant": full_answer})
        if len(history) > settings.max_history_turns * 2:
            history.pop(0)

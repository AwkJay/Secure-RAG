import logging
import os
from collections import OrderedDict
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from rag.hybrid_search import get_searcher
from rag.reranker import get_reranker

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

MAX_HISTORY_TURNS = 6
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "20"))

_session_store: OrderedDict = OrderedDict()

_llm: ChatGroq | None = None
_llm_api_key: str = ""


def get_llm() -> ChatGroq:
    global _llm, _llm_api_key
    current_key = os.getenv("GROQ_API_KEY", "")
    if _llm is None or current_key != _llm_api_key:
        if not current_key:
            logger.warning("GROQ_API_KEY is not set.")
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=1024,
            api_key=current_key,
        )
        _llm_api_key = current_key
        logger.info("Groq LLM client initialised.")
    return _llm


def _apply_filters(
    docs: list[tuple[str, dict, float]],
    severity_filter: list[str] | None,
    source_filter: str | None,
) -> tuple[list[tuple[str, dict, float]], int]:
    """Filter retrieved docs by source type and CVE severity.

    Returns (filtered_docs, num_filtered_out).
    """
    if not severity_filter:
        severity_filter = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    if not source_filter:
        source_filter = "both"

    allowed_severities = {s.upper() for s in severity_filter}
    sf = source_filter.lower()

    result = []
    for text, meta, score in docs:
        source = meta.get("source", "").upper()

        # Source filter
        if sf == "nvd" and source != "NVD":
            continue
        if sf == "owasp" and source != "OWASP":
            continue

        # Severity filter applies only to NVD documents
        if source == "NVD":
            doc_severity = meta.get("severity", "UNKNOWN").upper()
            if doc_severity not in allowed_severities:
                continue

        result.append((text, meta, score))

    filtered_out = len(docs) - len(result)
    return result, filtered_out


def _format_sources(docs: list[tuple[str, dict, float]]) -> list[dict]:
    sources = []
    seen = set()
    for _, meta, score in docs:
        source_type = meta.get("source", "UNKNOWN")
        if source_type == "NVD":
            cve_id = meta.get("cve_id", "UNKNOWN")
            if cve_id not in seen:
                seen.add(cve_id)
                sources.append({
                    "id": cve_id,
                    "source_type": "NVD",
                    "severity": meta.get("severity", "UNKNOWN"),
                    "score": meta.get("cvss_score", 0.0),
                })
        elif source_type == "OWASP":
            owasp_id = meta.get("owasp_id", "UNKNOWN")
            if owasp_id not in seen:
                seen.add(owasp_id)
                sources.append({
                    "id": owasp_id,
                    "source_type": "OWASP",
                    "category": meta.get("category_name", ""),
                })
    return sources


def _format_context(docs: list[tuple[str, dict, float]]) -> str:
    parts = []
    for text, meta, _ in docs:
        source = meta.get("source", "UNKNOWN")
        if source == "NVD":
            label = meta.get("cve_id", "CVE")
        else:
            label = meta.get("owasp_id", "OWASP")
        parts.append(f"[{label}]\n{text}")
    return "\n\n---\n\n".join(parts)


def _format_chat_history(history: list[dict]) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        lines.append(f"Human: {turn['human']}")
        lines.append(f"Assistant: {turn['assistant']}")
    return "\n".join(lines)


def _get_session(session_id: str) -> list[dict]:
    if session_id not in _session_store:
        if len(_session_store) >= MAX_SESSIONS:
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
    severity_filter: list[str] | None = None,
    source_filter: str | None = None,
) -> dict:
    searcher = get_searcher()
    reranker = get_reranker()
    llm = get_llm()

    retrieved, search_metadata = searcher.search(question, top_k=10)
    reranked = reranker.rerank(question, retrieved, top_k=5)

    # Apply filters after reranking so ranking quality is preserved
    filtered, filtered_out = _apply_filters(reranked, severity_filter, source_filter)
    search_metadata["filtered_out"] = filtered_out

    # If filters remove everything, fall back to full reranked set with a notice
    if not filtered and reranked:
        logger.warning("Filters removed all results — returning unfiltered set.")
        filtered = reranked
        search_metadata["filtered_out"] = 0

    context = _format_context(filtered)
    sources = _format_sources(filtered)

    history = _get_session(session_id)
    chat_history_str = _format_chat_history(history)

    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        context=context,
        chat_history=chat_history_str,
        question=question,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        answer = "I'm sorry, I encountered an error connecting to the language model. Please check your GROQ_API_KEY and try again."

    history.append({"human": question, "assistant": answer})
    if len(history) > MAX_HISTORY_TURNS * 2:
        history.pop(0)

    return {
        "answer": answer,
        "sources": sources,
        "search_metadata": search_metadata,
    }

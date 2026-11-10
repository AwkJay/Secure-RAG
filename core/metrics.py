"""Prometheus metrics registry for SecureRAG.

All metrics defined here as module-level singletons.
Import METRICS from this module and call e.g. METRICS.chat_requests.labels("success").inc()
"""
from prometheus_client import Counter, Histogram, Gauge, REGISTRY


# I grouped these by concern rather than alphabetically — easier to scan
class _Metrics:
    def __init__(self):
        # HTTP layer
        self.http_requests = Counter(
            "securerag_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
        )
        self.http_duration = Histogram(
            "securerag_http_request_duration_seconds",
            "HTTP request latency",
            ["method", "endpoint"],
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )
        self.active_requests = Gauge(
            "securerag_active_requests",
            "Requests currently being processed",
        )

        # Chat / RAG pipeline
        self.chat_requests = Counter(
            "securerag_chat_requests_total",
            "Chat endpoint calls by outcome",
            ["status"],  # success | error | rate_limited
        )
        self.rag_stage_duration = Histogram(
            "securerag_rag_pipeline_duration_seconds",
            "Time spent in each RAG stage",
            ["stage"],  # retrieval | rerank | llm | total
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 15.0],
        )
        self.retrieval_count = Histogram(
            "securerag_retrieval_result_count",
            "Number of docs returned per stage",
            ["method"],  # bm25 | semantic | rrf | reranked
            buckets=[0, 1, 5, 10, 15, 20],
        )
        self.reranker_score = Histogram(
            "securerag_reranker_score",
            "Cross-encoder relevance scores",
            buckets=[-10, -5, -2, -1, 0, 1, 2, 5, 10],
        )

        # LLM
        self.llm_tokens = Counter(
            "securerag_llm_tokens_total",
            "Tokens used in LLM calls",
            ["direction"],  # prompt | completion
        )
        self.groq_errors = Counter(
            "securerag_groq_errors_total",
            "Groq API failures by type",
            ["error_type"],  # timeout | rate_limit | auth | unknown
        )
        self.groq_retries = Counter(
            "securerag_groq_retries_total",
            "Number of Groq retry attempts",
        )
        self.unverified_citations = Counter(
            "securerag_unverified_citations_total",
            "CVE/OWASP IDs cited in an answer that were not in the retrieved context or the user's messages",
        )

        # Knowledge base
        self.kb_document_count = Gauge(
            "securerag_chromadb_document_count",
            "Documents in ChromaDB by source",
            ["source"],  # NVD | OWASP
        )
        self.ingest_runs = Counter(
            "securerag_ingest_runs_total",
            "Data ingestion job completions",
            ["status"],  # success | failure
        )
        self.ingest_duration = Histogram(
            "securerag_ingest_duration_seconds",
            "Total wall time for ingestion runs",
            buckets=[30, 60, 120, 300, 600, 900, 1800],
        )
        self.ingest_docs_added = Gauge(
            "securerag_ingest_docs_added",
            "Documents added in the most recent ingest",
        )

        # semantic cache
        self.cache_hits = Counter(
            "securerag_cache_hits_total",
            "Cache hits by cache type",
            ["cache_type"],
        )
        self.cache_misses = Counter(
            "securerag_cache_misses_total",
            "Cache misses by cache type",
            ["cache_type"],
        )
        self.semantic_cache_size = Gauge(
            "securerag_semantic_cache_entries",
            "Number of entries in the in-memory semantic cache",
        )


# module singleton — only one instance ever
METRICS = _Metrics()

from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    severity_filter: list[str] = Field(default_factory=lambda: ["CRITICAL", "HIGH", "MEDIUM", "LOW"])
    source_filter: str = Field(default="both")


class SourceItem(BaseModel):
    id: str
    source_type: str
    severity: Optional[str] = None
    score: Optional[float] = None
    category: Optional[str] = None


class SearchMetadata(BaseModel):
    bm25_hits: int
    semantic_hits: int
    rrf_final: int
    filtered_out: int = 0


class VerificationResult(BaseModel):
    cited: list[str] = Field(default_factory=list)
    verified: list[str] = Field(default_factory=list)
    unverified: list[str] = Field(default_factory=list)
    ok: bool = True


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    session_id: str
    search_metadata: SearchMetadata
    response_time_ms: int
    verification: Optional[VerificationResult] = None


class HealthResponse(BaseModel):
    status: str
    chromadb_documents: int
    models_loaded: bool
    version: str


class IngestResponse(BaseModel):
    status: str
    message: str


class IngestStatusResponse(BaseModel):
    status: str
    progress: int
    total: int
    last_completed_at: Optional[str]
    next_scheduled_at: Optional[str] = None


class EvaluationScores(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    context_precision: float


class EvaluationResponse(BaseModel):
    scores: EvaluationScores
    per_query_results: list[dict]
    evaluated_at: str
    num_test_queries: int
    error: Optional[str] = None


class StatsResponse(BaseModel):
    total_documents: int
    cve_count: int
    owasp_count: int
    queries_served: int
    avg_response_time_ms: float
    semantic_cache_entries: int = 0


class ClearSessionResponse(BaseModel):
    cleared: bool

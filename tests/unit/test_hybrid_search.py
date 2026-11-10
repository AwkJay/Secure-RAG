"""Tests for the hybrid search + RRF logic.

These don't need ChromaDB running — we test the algorithm in isolation.
"""
import pytest
from rag.hybrid_search import HybridSearcher


# ─── RRF Fusion Tests ─────────────────────────────────────────────────────────

def test_rrf_fusion_prefers_consistent_results():
    """A document that ranks well in both BM25 and semantic should win."""
    bm25 = [
        ("doc_a", {"id": "a"}, 9.0),
        ("doc_b", {"id": "b"}, 7.0),
        ("doc_c", {"id": "c"}, 5.0),
    ]
    semantic = [
        ("doc_b", {"id": "b"}, 0.95),   # doc_b ranks well in both
        ("doc_a", {"id": "a"}, 0.80),
        ("doc_d", {"id": "d"}, 0.75),
    ]

    fused = HybridSearcher.rrf_fusion(bm25, semantic)
    ids = [meta["id"] for _, meta, _ in fused]

    # doc_b and doc_a both appear in both lists, so they should beat doc_c (bm25 only)
    # and doc_d (semantic only)
    assert ids.index("doc_b") < ids.index("doc_c")
    assert ids.index("doc_a") < ids.index("doc_d")


def test_rrf_fusion_empty_both():
    result = HybridSearcher.rrf_fusion([], [])
    assert result == []


def test_rrf_fusion_only_bm25():
    bm25 = [("text1", {"id": "x"}, 5.0), ("text2", {"id": "y"}, 3.0)]
    result = HybridSearcher.rrf_fusion(bm25, [])
    assert len(result) == 2
    assert result[0][1]["id"] == "x"   # higher BM25 rank → higher RRF score


def test_rrf_fusion_only_semantic():
    semantic = [("t1", {"id": "p"}, 0.9), ("t2", {"id": "q"}, 0.7)]
    result = HybridSearcher.rrf_fusion([], semantic)
    assert len(result) == 2


def test_rrf_scores_are_positive():
    bm25 = [("a", {}, 1.0)]
    sem = [("b", {}, 0.5)]
    fused = HybridSearcher.rrf_fusion(bm25, sem)
    for _, _, score in fused:
        assert score > 0


# ─── Filter Tests ─────────────────────────────────────────────────────────────

def test_apply_filters_source_nvd_only():
    from rag.chain import _apply_filters
    docs = [
        ("t1", {"source": "NVD", "severity": "CRITICAL", "cve_id": "CVE-001"}, 0.9),
        ("t2", {"source": "OWASP", "owasp_id": "A01"}, 0.8),
    ]
    filtered, n_out = _apply_filters(docs, ["CRITICAL", "HIGH"], "nvd")
    assert n_out == 1
    assert all(m.get("source") == "NVD" for _, m, _ in filtered)


def test_apply_filters_severity():
    from rag.chain import _apply_filters
    docs = [
        ("t1", {"source": "NVD", "severity": "CRITICAL"}, 0.9),
        ("t2", {"source": "NVD", "severity": "LOW"}, 0.5),
    ]
    filtered, n_out = _apply_filters(docs, ["CRITICAL"], "nvd")
    assert n_out == 1
    assert len(filtered) == 1


def test_apply_filters_owasp_always_passes_severity_check():
    """OWASP docs don't have a severity level — they should never be filtered by severity."""
    from rag.chain import _apply_filters
    docs = [("t", {"source": "OWASP", "owasp_id": "A03"}, 0.8)]
    filtered, n_out = _apply_filters(docs, ["CRITICAL"], "both")
    assert n_out == 0
    assert len(filtered) == 1


# ─── Config Tests ─────────────────────────────────────────────────────────────

def test_settings_have_defaults():
    from core.config import Settings
    s = Settings()
    assert s.groq_model == "llama-3.1-8b-instant"
    assert s.groq_temperature == 0.1
    assert s.retrieval_top_k == 10
    assert s.rerank_top_k == 5


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "some-other-model")
    from core.config import Settings
    s = Settings()
    assert s.groq_model == "some-other-model"


# ─── Semantic Cache Tests ─────────────────────────────────────────────────────

import numpy as np

def _make_normed(values: list[float]) -> list[float]:
    """Create an L2-normalized vector (what BGE outputs)."""
    v = np.array(values, dtype=np.float32)
    v = v / np.linalg.norm(v)
    return v.tolist()


def test_semantic_cache_hit_on_similar_query():
    from rag.semantic_cache import SemanticCache
    cache = SemanticCache(threshold=0.90, max_entries=100)

    # two nearly identical embeddings — simulates "SQL CVEs" vs "CVEs about SQL"
    base = [1.0, 0.8, 0.6, 0.3] + [0.1] * 380
    emb1 = _make_normed(base)
    emb2 = _make_normed([x + 0.001 for x in base])   # tiny perturbation

    fake_result = {"answer": "test answer", "sources": [], "search_metadata": {}}
    cache.set(emb1, fake_result)

    result = cache.get(emb2)
    assert result is not None
    assert result["answer"] == "test answer"


def test_semantic_cache_miss_on_different_query():
    from rag.semantic_cache import SemanticCache
    cache = SemanticCache(threshold=0.90, max_entries=100)

    sql_emb = _make_normed([1.0, 0.0] + [0.0] * 382)
    xss_emb = _make_normed([0.0, 1.0] + [0.0] * 382)   # orthogonal = cosine similarity 0.0

    cache.set(sql_emb, {"answer": "sql answer", "sources": [], "search_metadata": {}})

    result = cache.get(xss_emb)
    assert result is None   # should be a miss


def test_semantic_cache_invalidate():
    from rag.semantic_cache import SemanticCache
    cache = SemanticCache(threshold=0.90, max_entries=100)

    emb = _make_normed([1.0] + [0.0] * 383)
    cache.set(emb, {"answer": "test", "sources": [], "search_metadata": {}})
    assert cache.size == 1

    cache.invalidate()
    assert cache.size == 0
    assert cache.get(emb) is None


def test_semantic_cache_lru_eviction():
    from rag.semantic_cache import SemanticCache
    cache = SemanticCache(threshold=0.99, max_entries=3)

    # fill to capacity
    for i in range(3):
        e = _make_normed([float(i + 1)] + [0.001 * i] * 383)
        cache.set(e, {"answer": f"answer {i}", "sources": [], "search_metadata": {}})
    assert cache.size == 3

    # add one more — should evict oldest
    new_e = _make_normed([99.0] + [0.0] * 383)
    cache.set(new_e, {"answer": "new", "sources": [], "search_metadata": {}})
    assert cache.size == 3   # still 3, oldest was evicted

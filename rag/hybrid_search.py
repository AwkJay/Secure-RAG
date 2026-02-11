import logging
from rank_bm25 import BM25Okapi
from rag.vectorstore import get_all_documents, semantic_search

logger = logging.getLogger(__name__)


class HybridSearcher:
    def __init__(self):
        self.corpus_texts: list[str] = []
        self.corpus_metadatas: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self._build_index()

    def _build_index(self) -> None:
        logger.info("Building BM25 index from ChromaDB...")
        all_docs = get_all_documents()
        if not all_docs:
            logger.warning("ChromaDB is empty — BM25 index not built.")
            self.corpus_texts = []
            self.corpus_metadatas = []
            self.bm25 = None
            return

        self.corpus_texts = [doc for doc, _ in all_docs]
        self.corpus_metadatas = [meta for _, meta in all_docs]
        tokenized = [t.lower().split() for t in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built with %d documents.", len(self.corpus_texts))

    def rebuild_index(self) -> None:
        self._build_index()

    def bm25_search(self, query: str, k: int = 20) -> list[tuple[str, dict, float]]:
        if self.bm25 is None or not self.corpus_texts:
            return []
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            (self.corpus_texts[i], self.corpus_metadatas[i], float(scores[i]))
            for i in top_indices
        ]

    def semantic_search(self, query: str, k: int = 20) -> list[tuple[str, dict, float]]:
        return semantic_search(query, k=k)

    @staticmethod
    def rrf_fusion(
        bm25_results: list[tuple[str, dict, float]],
        semantic_results: list[tuple[str, dict, float]],
        k: int = 60,
    ) -> list[tuple[str, dict, float]]:
        scores: dict[str, float] = {}
        doc_map: dict[str, tuple[str, dict]] = {}

        for rank, (text, meta, _) in enumerate(bm25_results):
            key = text[:200]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = (text, meta)

        for rank, (text, meta, _) in enumerate(semantic_results):
            key = text[:200]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = (text, meta)

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [(doc_map[k][0], doc_map[k][1], scores[k]) for k in sorted_keys]

    def search(
        self, query: str, top_k: int = 10
    ) -> tuple[list[tuple[str, dict, float]], dict]:
        bm25_results = self.bm25_search(query, k=20)
        semantic_results = self.semantic_search(query, k=20)
        fused = self.rrf_fusion(bm25_results, semantic_results)
        final = fused[:top_k]
        metadata = {
            "bm25_hits": len(bm25_results),
            "semantic_hits": len(semantic_results),
            "rrf_final": len(final),
        }
        return final, metadata


_searcher: HybridSearcher | None = None


def get_searcher() -> HybridSearcher:
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher()
    return _searcher


def rebuild_searcher() -> None:
    global _searcher
    if _searcher is not None:
        _searcher.rebuild_index()
    else:
        _searcher = HybridSearcher()

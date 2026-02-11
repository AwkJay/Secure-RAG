import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_DOC_TOKENS = 512

_cross_encoder: CrossEncoder | None = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder model: %s", CROSS_ENCODER_MODEL)
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=MAX_DOC_TOKENS)
        logger.info("Cross-encoder loaded.")
    return _cross_encoder


class CrossEncoderReranker:
    def __init__(self):
        self.model = get_cross_encoder()

    def rerank(
        self,
        query: str,
        docs: list[tuple[str, dict, float]],
        top_k: int = 5,
    ) -> list[tuple[str, dict, float]]:
        if not docs:
            return []

        pairs = [(query, text[:2000]) for text, _, _ in docs]
        scores = self.model.predict(pairs)

        scored = sorted(
            zip(scores, docs), key=lambda x: x[0], reverse=True
        )[:top_k]

        return [(text, meta, float(score)) for score, (text, meta, _) in scored]


_reranker: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker

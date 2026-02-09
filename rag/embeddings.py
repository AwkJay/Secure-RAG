import logging
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

_embedding_model = None

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading BAAI/bge-small-en-v1.5 embedding model...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            _embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        logger.info("Embedding model loaded.")
    return _embedding_model


def embed_query(text: str) -> list[float]:
    model = get_embeddings()
    prefixed = BGE_QUERY_PREFIX + text
    return model.embed_query(prefixed)


def embed_documents(texts: list[str]) -> list[list[float]]:
    model = get_embeddings()
    return model.embed_documents(texts)

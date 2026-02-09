import logging
import os
import chromadb
from chromadb.config import Settings
from rag.embeddings import embed_documents, embed_query

logger = logging.getLogger(__name__)

COLLECTION_NAME = "securerag_docs"
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

_client = None
_collection = None


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_documents(docs: list[dict]) -> None:
    collection = get_collection()
    texts = [d["text"] for d in docs]
    metadatas = [d["metadata"] for d in docs]
    ids = [d["id"] for d in docs]

    embeddings = embed_documents(texts)

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    logger.info("Added %d documents to ChromaDB.", len(docs))


def semantic_search(query: str, k: int = 20) -> list[tuple[str, dict, float]]:
    collection = get_collection()
    q_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=min(k, max(collection.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )
    output = []
    if results["documents"]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - dist
            output.append((doc, meta, score))
    return output


def get_all_documents() -> list[tuple[str, dict]]:
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    results = collection.get(include=["documents", "metadatas"])
    return list(zip(results["documents"], results["metadatas"]))


def get_document_count() -> int:
    return get_collection().count()


def get_counts_by_source() -> dict:
    collection = get_collection()
    all_docs = collection.get(include=["metadatas"])
    cve_count = sum(1 for m in all_docs["metadatas"] if m.get("source") == "NVD")
    owasp_count = sum(1 for m in all_docs["metadatas"] if m.get("source") == "OWASP")
    return {"total": collection.count(), "cve": cve_count, "owasp": owasp_count}


def delete_collection() -> None:
    global _collection
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = None

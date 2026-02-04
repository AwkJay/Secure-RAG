import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv(override=True)

# Disable ChromaDB telemetry before any chromadb import
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Silence the noisy chromadb telemetry logger (posthog capture bug in 0.5.x)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting SecureRAG...")

    from rag.embeddings import get_embeddings
    get_embeddings()

    from rag.reranker import get_reranker
    get_reranker()

    from rag.vectorstore import get_document_count
    doc_count = get_document_count()
    if doc_count == 0:
        logger.warning("Knowledge base is empty. Call POST /api/ingest to populate.")
    else:
        logger.info("ChromaDB loaded with %d documents.", doc_count)

    from rag.hybrid_search import get_searcher
    get_searcher()

    logger.info("SecureRAG ready! Listening on 0.0.0.0:8000")
    yield
    logger.info("SecureRAG shutting down.")


app = FastAPI(title="SecureRAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

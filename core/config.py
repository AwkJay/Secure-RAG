"""Central config — one place for all settings instead of os.getenv() everywhere."""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    # app
    app_name: str = "SecureRAG"
    app_version: str = "2.0.0"
    environment: str = "development"
    log_level: str = "INFO"

    # cors — restrict in prod, keep open for local dev
    allowed_origins: list[str] = ["*"]

    # groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 1024
    groq_timeout: int = 30      # seconds before we give up on a request
    groq_max_retries: int = 3

    # chromadb
    chroma_db_path: str = "./chroma_db"

    # rate limiting
    rate_limit_chat: str = "20/minute"
    rate_limit_ingest: str = "5/hour"

    # rag tuning
    max_sessions: int = 20
    max_history_turns: int = 6
    retrieval_top_k: int = 10
    rerank_top_k: int = 5

    # nvd fetch
    nvd_api_key: str = ""
    nvd_cve_cap: int = 2000

    # auth
    ingest_key: str = "securerag-ingest-2024"

    # scheduler — cron for weekly CVE refresh
    cve_refresh_enabled: bool = True
    cve_refresh_cron: str = "0 2 * * 0"   # sunday 2am UTC

    # redis
    redis_url: str = "redis://localhost:6379/0"

    # misc
    prometheus_enabled: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("log_level")
    @classmethod
    def uppercase_log_level(cls, v):
        return v.upper()


# module-level singleton — import this everywhere
settings = Settings()

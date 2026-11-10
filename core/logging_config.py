"""Structured JSON logging setup.

Normal Python logging just dumps plain text which is hard to grep in production.
JSON logs let you filter by request_id, level, etc in Grafana Loki / CloudWatch.
"""
import logging
import sys
from contextvars import ContextVar

from pythonjsonlogger import jsonlogger

# thread-local ish request ID so every log line in a request gets the same id
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Injects the current request_id into every log record."""
    def filter(self, record):
        record.request_id = request_id_var.get("-")
        return True


def configure_logging(log_level: str = "INFO") -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    # don't add duplicate handlers if called twice
    if root_logger.handlers:
        root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    )
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())
    root_logger.addHandler(handler)

    # still silence the noisy chromadb posthog thing
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

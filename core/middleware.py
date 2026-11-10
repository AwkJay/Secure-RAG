"""ASGI middleware for request instrumentation.

Handles two things:
  1. Injects a UUID request_id into every request (shows up in logs + response header)
  2. Records HTTP metrics (latency, status codes) into Prometheus
"""
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from core.metrics import METRICS
from core.logging_config import request_id_var


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # generate a request id for tracing through logs
        req_id = str(uuid.uuid4())[:8]   # short enough to not clutter logs
        token = request_id_var.set(req_id)

        METRICS.active_requests.inc()
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            METRICS.active_requests.dec()
            request_id_var.reset(token)
            raise

        elapsed = time.perf_counter() - start

        # normalise path so /api/session/abc123 doesn't create a million label combos
        path = _normalise_path(request.url.path)

        METRICS.http_requests.labels(
            method=request.method,
            endpoint=path,
            status_code=str(response.status_code),
        ).inc()
        METRICS.http_duration.labels(method=request.method, endpoint=path).observe(elapsed)
        METRICS.active_requests.dec()

        response.headers["X-Request-ID"] = req_id
        request_id_var.reset(token)
        return response


def _normalise_path(path: str) -> str:
    """Replace dynamic path segments so Prometheus doesn't explode with cardinality."""
    # /api/session/{id} -> /api/session/:id
    parts = path.split("/")
    normalised = []
    for p in parts:
        # heuristic: if it looks like a UUID or long hex string, replace it
        if len(p) > 20 or (len(p) == 36 and p.count("-") == 4):
            normalised.append(":id")
        else:
            normalised.append(p)
    return "/".join(normalised)

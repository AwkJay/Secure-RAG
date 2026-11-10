"""APScheduler setup for background tasks.

Right now only one job: weekly CVE refresh. But the scheduler is easy to extend
if we want daily CVSS score updates or monthly OWASP refresh later.
"""
import asyncio
import logging
import time

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


def get_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler()
    return _scheduler


async def _run_weekly_ingest():
    """Runs the full ingest pipeline + rebuilds BM25 index.

    Uses run_in_executor because ingest is synchronous and we don't want to
    block the event loop for 15 minutes.
    """
    from core.metrics import METRICS
    logger.info("Scheduled CVE refresh starting...")
    start = time.time()
    try:
        loop = asyncio.get_event_loop()
        # run the blocking ingest in a thread
        await loop.run_in_executor(None, _blocking_ingest)

        from rag.hybrid_search import rebuild_searcher
        rebuild_searcher()

        # clear semantic cache — CVE data changed, cached answers are stale
        from rag.semantic_cache import get_cache
        get_cache().invalidate()

        elapsed = time.time() - start
        METRICS.ingest_runs.labels("success").inc()
        METRICS.ingest_duration.observe(elapsed)
        logger.info("Scheduled CVE refresh done in %.1fs", elapsed)
    except Exception as e:
        METRICS.ingest_runs.labels("failure").inc()
        logger.error("Scheduled CVE refresh failed: %s", e, exc_info=True)


def _blocking_ingest():
    from data.ingest import run_ingest
    run_ingest(force=True)


def setup_scheduler(cron_expr: str, enabled: bool = True) -> AsyncIOScheduler:
    scheduler = get_scheduler()

    if not enabled:
        logger.info("Scheduled CVE refresh is disabled (CVE_REFRESH_ENABLED=false)")
        return scheduler

    trigger = CronTrigger.from_crontab(cron_expr, timezone="UTC")
    scheduler.add_job(
        _run_weekly_ingest,
        trigger=trigger,
        id="weekly_cve_refresh",
        replace_existing=True,
        misfire_grace_time=3600,   # if the server was down during the scheduled time, run within 1h
    )
    logger.info("CVE refresh scheduled: %s (UTC)", cron_expr)
    return scheduler

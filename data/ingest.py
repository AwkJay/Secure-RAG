#!/usr/bin/env python3
"""
SecureRAG Data Ingestion Script
Fetches CVEs from NVD API and loads OWASP Top 10 data into ChromaDB.
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
OWASP_JSON_PATH = Path(__file__).parent / "owasp_data.json"
CVE_CACHE_PATH = Path(__file__).parent / "cve_cache.json"
RESULTS_PER_PAGE = 2000

# NVD API rate limits: 5 req/30s without key (6s sleep), 50 req/30s with key (0.6s sleep)
_NVD_API_KEY = os.getenv("NVD_API_KEY", "")
RATE_LIMIT_SLEEP = 0.6 if _NVD_API_KEY else 6.0


def _nvd_fmt(dt) -> str:
    """Format a datetime for the NVD 2.0 API. Requires explicit UTC offset."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000")


def _fetch_window(pub_start: str, pub_end: str, headers: dict, cap: int) -> list[dict]:
    """Fetch CVEs for a single date window, paginating as needed up to cap entries."""
    results = []
    start_index = 0
    total_in_window = None

    while total_in_window is None or start_index < total_in_window:
        if len(results) >= cap:
            break
        params = {
            "pubStartDate": pub_start,
            "pubEndDate": pub_end,
            "resultsPerPage": min(RESULTS_PER_PAGE, cap - len(results)),
            "startIndex": start_index,
        }
        try:
            resp = requests.get(NVD_API_BASE, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("NVD API request failed (%s → %s): %s", pub_start[:10], pub_end[:10], e)
            break

        if total_in_window is None:
            total_in_window = data.get("totalResults", 0)
            logger.info("  Window %s → %s: %d CVEs available", pub_start[:10], pub_end[:10], total_in_window)

        chunk = data.get("vulnerabilities", [])
        results.extend(chunk)
        start_index += len(chunk)
        if not chunk:
            break
        time.sleep(RATE_LIMIT_SLEEP)

    return results


def fetch_cves(force: bool = False) -> list[dict]:
    if not force and CVE_CACHE_PATH.exists():
        logger.info("Loading CVEs from cache: %s", CVE_CACHE_PATH)
        with open(CVE_CACHE_PATH) as f:
            return json.load(f)

    logger.info("Fetching CVEs from NVD API (last 365 days in 119-day windows)...")

    # NVD 2.0 enforces a 120-day maximum date range per request.
    # We split the last 365 days into 3 windows of ≤119 days each.
    from datetime import timezone as _tz
    now = datetime.now(_tz.utc)
    WINDOW_DAYS = 119
    TOTAL_DAYS = 365
    CVE_CAP = 2000

    windows = []
    window_end = now
    remaining = TOTAL_DAYS
    while remaining > 0:
        days = min(WINDOW_DAYS, remaining)
        window_start = window_end - timedelta(days=days)
        windows.append((_nvd_fmt(window_start), _nvd_fmt(window_end)))
        window_end = window_start
        remaining -= days

    headers = {}
    nvd_key = os.getenv("NVD_API_KEY", "")
    if nvd_key:
        headers["apiKey"] = nvd_key

    all_cves = []
    per_window_cap = CVE_CAP // len(windows)

    for i, (ws, we) in enumerate(windows):
        logger.info("Fetching window %d/%d: %s → %s", i + 1, len(windows), ws[:10], we[:10])
        chunk = _fetch_window(ws, we, headers, cap=per_window_cap)
        all_cves.extend(chunk)
        logger.info("  Got %d CVEs (total so far: %d)", len(chunk), len(all_cves))
        if len(all_cves) >= CVE_CAP:
            break
        if i < len(windows) - 1:
            time.sleep(RATE_LIMIT_SLEEP)

    all_cves = all_cves[:CVE_CAP]
    logger.info("Caching %d CVEs to %s", len(all_cves), CVE_CACHE_PATH)
    with open(CVE_CACHE_PATH, "w") as f:
        json.dump(all_cves, f)

    return all_cves


def parse_cve(vuln: dict) -> Optional[dict]:
    try:
        cve = vuln.get("cve", {})
        cve_id = cve.get("id", "")
        if not cve_id:
            return None

        descriptions = cve.get("descriptions", [])
        description = next(
            (d["value"] for d in descriptions if d.get("lang") == "en"), ""
        )
        if not description:
            return None

        severity = "UNKNOWN"
        cvss_score = 0.0
        metrics = cve.get("metrics", {})
        if "cvssMetricV31" in metrics and metrics["cvssMetricV31"]:
            cvss_data = metrics["cvssMetricV31"][0].get("cvssData", {})
            severity = cvss_data.get("baseSeverity", "UNKNOWN")
            cvss_score = cvss_data.get("baseScore", 0.0)
        elif "cvssMetricV30" in metrics and metrics["cvssMetricV30"]:
            cvss_data = metrics["cvssMetricV30"][0].get("cvssData", {})
            severity = cvss_data.get("baseSeverity", "UNKNOWN")
            cvss_score = cvss_data.get("baseScore", 0.0)
        elif "cvssMetricV2" in metrics and metrics["cvssMetricV2"]:
            cvss_data = metrics["cvssMetricV2"][0].get("cvssData", {})
            score = cvss_data.get("baseScore", 0.0)
            cvss_score = score
            if score >= 7.0:
                severity = "HIGH"
            elif score >= 4.0:
                severity = "MEDIUM"
            else:
                severity = "LOW"

        published_date = cve.get("published", "")[:10]

        references = [
            r.get("url", "") for r in cve.get("references", [])[:3]
        ]

        configs = cve.get("configurations", [])
        affected_products = "Not specified"
        if configs:
            products = []
            for config in configs[:2]:
                for node in config.get("nodes", [])[:2]:
                    for match in node.get("cpeMatch", [])[:2]:
                        criteria = match.get("criteria", "")
                        parts = criteria.split(":")
                        if len(parts) >= 5:
                            products.append(f"{parts[3]} {parts[4]}")
            if products:
                affected_products = ", ".join(products[:3])

        return {
            "cve_id": cve_id,
            "description": description,
            "severity": severity,
            "cvss_score": cvss_score,
            "published_date": published_date,
            "references": references,
            "affected_products": affected_products,
        }
    except Exception as e:
        logger.warning("Failed to parse CVE: %s", e)
        return None


def cve_to_document(cve: dict) -> dict:
    text = (
        f"CVE ID: {cve['cve_id']}\n"
        f"Severity: {cve['severity']} ({cve['cvss_score']})\n"
        f"Description: {cve['description']}\n"
        f"Affected: {cve['affected_products']}\n"
        f"Published: {cve['published_date']}"
    )
    return {
        "id": cve["cve_id"],
        "text": text,
        "metadata": {
            "source": "NVD",
            "cve_id": cve["cve_id"],
            "severity": cve["severity"],
            "cvss_score": cve["cvss_score"],
            "published_date": cve["published_date"],
        },
    }


def owasp_to_documents(entry: dict) -> list[dict]:
    docs = []
    owasp_id = entry["id"]
    name = entry["name"]

    cwe_str = ", ".join(entry.get("cwe_ids", []))
    chunk1 = (
        f"OWASP ID: {owasp_id}\n"
        f"Category: {name}\n"
        f"Description: {entry['description']}\n"
        f"Related CWEs: {cwe_str}"
    )
    docs.append({
        "id": f"{owasp_id}-desc",
        "text": chunk1,
        "metadata": {
            "source": "OWASP",
            "owasp_id": owasp_id,
            "category_name": name,
            "chunk_type": "description",
        },
    })

    prevention_text = "\n".join(
        f"- {p}" for p in entry.get("how_to_prevent", [])
    )
    chunk2 = (
        f"OWASP ID: {owasp_id} - {name}\n"
        f"How to Prevent:\n{prevention_text}"
    )
    docs.append({
        "id": f"{owasp_id}-prevention",
        "text": chunk2,
        "metadata": {
            "source": "OWASP",
            "owasp_id": owasp_id,
            "category_name": name,
            "chunk_type": "prevention",
        },
    })

    scenarios = entry.get("example_attack_scenarios", [])
    scenarios_text = "\n".join(
        f"Scenario {i+1}: {s}" for i, s in enumerate(scenarios)
    )
    chunk3 = (
        f"OWASP ID: {owasp_id} - {name}\n"
        f"Example Attack Scenarios:\n{scenarios_text}"
    )
    docs.append({
        "id": f"{owasp_id}-scenarios",
        "text": chunk3,
        "metadata": {
            "source": "OWASP",
            "owasp_id": owasp_id,
            "category_name": name,
            "chunk_type": "scenarios",
        },
    })

    return docs


def run_ingest(
    force: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    from rag.vectorstore import add_documents, delete_collection, get_collection

    logger.info("=== SecureRAG Data Ingestion ===")

    raw_cves = fetch_cves(force=force)
    logger.info("Parsing %d raw CVE entries...", len(raw_cves))

    cve_docs = []
    for i, vuln in enumerate(raw_cves):
        cve = parse_cve(vuln)
        if cve:
            cve_docs.append(cve_to_document(cve))
        if (i + 1) % 100 == 0:
            logger.info("Parsed CVEs: %d/%d", i + 1, len(raw_cves))

    logger.info("Parsed %d valid CVE documents.", len(cve_docs))

    with open(OWASP_JSON_PATH) as f:
        owasp_entries = json.load(f)

    owasp_docs = []
    for entry in owasp_entries:
        owasp_docs.extend(owasp_to_documents(entry))
    logger.info("Prepared %d OWASP chunks.", len(owasp_docs))

    all_docs = cve_docs + owasp_docs
    total = len(all_docs)
    logger.info("Total documents to ingest: %d", total)

    delete_collection()
    get_collection()

    BATCH_SIZE = 100
    for i in range(0, total, BATCH_SIZE):
        batch = all_docs[i:i + BATCH_SIZE]
        add_documents(batch)
        ingested = min(i + BATCH_SIZE, total)
        logger.info("Ingested %d/%d documents...", ingested, total)
        if progress_callback:
            progress_callback(ingested, total)

    logger.info("Ingestion complete! %d documents stored in ChromaDB.", total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SecureRAG Data Ingestion")
    parser.add_argument("--force", action="store_true", help="Re-fetch CVEs even if cache exists")
    args = parser.parse_args()
    run_ingest(force=args.force)

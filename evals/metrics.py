"""Ranking metrics + bootstrap helpers. Pure numpy, no model imports (unit-testable)."""
import math

import numpy as np


def dedupe_ranked_ids(docs, k: int | None = None) -> list[str]:
    """Map (text, meta, score) results to a ranked list of unique doc IDs (cve_id / owasp_id).

    OWASP entries are split into 3 chunks that share an `owasp_id`, so we keep the first
    (best-ranked) occurrence only; otherwise one category could occupy several slots.
    """
    seen, out = set(), []
    for _, meta, *_ in docs:
        did = meta.get("cve_id") if meta.get("source") == "NVD" else meta.get("owasp_id")
        did = did or meta.get("cve_id") or meta.get("owasp_id")
        if did and did not in seen:
            seen.add(did)
            out.append(did)
    return out[:k] if k else out


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def reciprocal_rank(ranked: list[str], gold: set[str], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int = 10) -> float:
    """Binary-relevance nDCG@k."""
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def bootstrap_ci(values, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI over per-query values."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_resamples, v.size))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_delta(a, b, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Bootstrap CI for mean(a - b) over the same queries. `significant` = CI excludes 0."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean, lo, hi = bootstrap_ci(d, n_resamples, alpha, seed)
    return {"delta": mean, "ci_low": lo, "ci_high": hi, "significant": bool(lo > 0 or hi < 0)}

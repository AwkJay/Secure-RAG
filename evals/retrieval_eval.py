"""Retrieval ablation: BM25 vs dense vs hybrid (RRF) vs hybrid + cross-encoder rerank.

All configs return a ranked list of unique doc IDs; metrics are binary-relevance Recall@k, MRR@10, nDCG@10.
Note: hybrid_rerank reranks the hybrid top-10, so its Recall@10 equals hybrid's by construction and
its Recall@5 is exactly the production setting (top-5 after rerank).
"""
import statistics
import time
from collections import defaultdict

from evals.metrics import (
    bootstrap_ci, dedupe_ranked_ids, ndcg_at_k, paired_bootstrap_delta, recall_at_k, reciprocal_rank,
)

KS = (1, 3, 5, 10)
CONFIGS = ["bm25", "dense", "hybrid", "hybrid_rerank"]
BASELINE = "dense"


def _run_config(name: str, query: str, searcher, reranker):
    """Returns (ranked unique IDs, latency seconds)."""
    t0 = time.perf_counter()
    if name == "bm25":
        docs = searcher.bm25_search(query, k=20)
    elif name == "dense":
        docs = searcher.semantic_search(query, k=20)
    else:
        docs, _ = searcher.search(query, top_k=10)
        if name == "hybrid_rerank":
            docs = reranker.rerank(query, docs, top_k=10)
    latency = time.perf_counter() - t0
    return dedupe_ranked_ids(docs, k=10), latency


def _per_query_metrics(ranked: list[str], gold: set[str]) -> dict:
    m = {f"recall@{k}": recall_at_k(ranked, gold, k) for k in KS}
    m["mrr@10"] = reciprocal_rank(ranked, gold, 10)
    m["ndcg@10"] = ndcg_at_k(ranked, gold, 10)
    return m


def _summarise(per_query: list[dict], n_resamples: int, seed: int) -> dict:
    out = {}
    for metric in per_query[0]:
        mean, lo, hi = bootstrap_ci([q[metric] for q in per_query], n_resamples, seed=seed)
        out[metric] = {"mean": mean, "ci_low": lo, "ci_high": hi}
    return out


def run_retrieval_eval(dataset: list[dict], n_resamples: int = 1000, seed: int = 0) -> dict:
    from rag.hybrid_search import get_searcher
    from rag.reranker import get_reranker

    searcher, reranker = get_searcher(), get_reranker()

    # warm-up so first-call model/JIT cost doesn't pollute latency
    for name in CONFIGS:
        _run_config(name, "warm up query", searcher, reranker)

    per_cfg: dict[str, list[dict]] = {c: [] for c in CONFIGS}
    latencies: dict[str, list[float]] = {c: [] for c in CONFIGS}
    for row in dataset:
        gold = set(row["gold_ids"])
        for c in CONFIGS:
            ranked, lat = _run_config(c, row["query"], searcher, reranker)
            per_cfg[c].append(_per_query_metrics(ranked, gold))
            latencies[c].append(lat)

    results = {"n_queries": len(dataset), "configs": {}, "by_style": {}, "paired_vs_" + BASELINE: {}, "latency_ms": {}}

    for c in CONFIGS:
        results["configs"][c] = _summarise(per_cfg[c], n_resamples, seed)
        lat = sorted(latencies[c])
        results["latency_ms"][c] = {
            "median": statistics.median(lat) * 1000,
            "p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))] * 1000,
        }

    styles = sorted({r["style"] for r in dataset})
    for style in styles:
        idx = [i for i, r in enumerate(dataset) if r["style"] == style]
        results["by_style"][style] = {
            "n": len(idx),
            **{c: _summarise([per_cfg[c][i] for i in idx], n_resamples, seed) for c in CONFIGS},
        }

    for c in CONFIGS:
        if c == BASELINE:
            continue
        results["paired_vs_" + BASELINE][c] = {
            m: paired_bootstrap_delta(
                [q[m] for q in per_cfg[c]], [q[m] for q in per_cfg[BASELINE]], n_resamples, seed=seed
            )
            for m in ("recall@5", "mrr@10", "ndcg@10")
        }
    # extra paired comparison that answers "does reranking help on top of hybrid?"
    results["paired_rerank_vs_hybrid"] = {
        m: paired_bootstrap_delta(
            [q[m] for q in per_cfg["hybrid_rerank"]], [q[m] for q in per_cfg["hybrid"]], n_resamples, seed=seed
        )
        for m in ("recall@5", "mrr@10", "ndcg@10")
    }
    return results


def _fmt(cell: dict) -> str:
    return f"{cell['mean']:.3f} [{cell['ci_low']:.3f}, {cell['ci_high']:.3f}]"


def to_markdown(res: dict) -> str:
    lines = [f"### Retrieval ablation ({res['n_queries']} queries, 95% bootstrap CIs)", ""]
    cols = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
    lines.append("| Config | " + " | ".join(c.replace("recall", "R") for c in cols) + " | latency p50 / p95 (ms) |")
    lines.append("|---|" + "---|" * (len(cols) + 1))
    for c in CONFIGS:
        lat = res["latency_ms"][c]
        lines.append(f"| {c} | " + " | ".join(_fmt(res["configs"][c][m]) for m in cols)
                     + f" | {lat['median']:.0f} / {lat['p95']:.0f} |")

    lines += ["", "### By query style (MRR@10 / Recall@5)", ""]
    lines.append("| Style (n) | " + " | ".join(CONFIGS) + " |")
    lines.append("|---|" + "---|" * len(CONFIGS))
    for style, d in res["by_style"].items():
        cells = [f"{d[c]['mrr@10']['mean']:.3f} / {d[c]['recall@5']['mean']:.3f}" for c in CONFIGS]
        lines.append(f"| {style} ({d['n']}) | " + " | ".join(cells) + " |")

    lines += ["", f"### Paired deltas vs `{BASELINE}` (mean difference, 95% CI)", ""]
    lines.append("| Config | Metric | Δ | 95% CI | Significant |")
    lines.append("|---|---|---|---|---|")
    for c, ms in res["paired_vs_" + BASELINE].items():
        for m, r in ms.items():
            lines.append(f"| {c} | {m} | {r['delta']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {'yes' if r['significant'] else 'no'} |")
    lines += ["", "### Rerank on top of hybrid (paired)", "", "| Metric | Δ | 95% CI | Significant |", "|---|---|---|---|"]
    for m, r in res["paired_rerank_vs_hybrid"].items():
        lines.append(f"| {m} | {r['delta']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {'yes' if r['significant'] else 'no'} |")
    return "\n".join(lines)

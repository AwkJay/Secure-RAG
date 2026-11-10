## SecureRAG evaluation — 2026-11-10T18:44:01Z
Corpus: 2000 CVEs + 30 OWASP chunks

### Retrieval ablation (192 queries, 95% bootstrap CIs)

| Config | R@1 | R@3 | R@5 | R@10 | mrr@10 | ndcg@10 | latency p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|
| bm25 | 0.760 [0.703, 0.818] | 0.823 [0.771, 0.875] | 0.911 [0.870, 0.948] | 0.932 [0.896, 0.964] | 0.812 [0.764, 0.859] | 0.841 [0.799, 0.883] | 3 / 7 |
| dense | 0.464 [0.391, 0.536] | 0.557 [0.490, 0.630] | 0.589 [0.516, 0.656] | 0.625 [0.557, 0.688] | 0.517 [0.451, 0.583] | 0.543 [0.480, 0.608] | 18 / 22 |
| hybrid | 0.776 [0.719, 0.839] | 0.880 [0.833, 0.922] | 0.906 [0.865, 0.943] | 0.953 [0.922, 0.979] | 0.832 [0.786, 0.879] | 0.861 [0.822, 0.901] | 21 / 31 |
| hybrid_rerank | 0.833 [0.781, 0.886] | 0.901 [0.859, 0.943] | 0.922 [0.885, 0.958] | 0.953 [0.922, 0.979] | 0.873 [0.833, 0.916] | 0.893 [0.856, 0.929] | 352 / 712 |

### By query style (MRR@10 / Recall@5)

| Style (n) | bm25 | dense | hybrid | hybrid_rerank |
|---|---|---|---|---|
| id (60) | 0.936 / 0.967 | 0.044 / 0.067 | 0.927 / 0.933 | 0.958 / 0.967 |
| keyword (58) | 0.736 / 0.879 | 0.702 / 0.810 | 0.774 / 0.879 | 0.801 / 0.897 |
| natural (54) | 0.766 / 0.870 | 0.714 / 0.815 | 0.772 / 0.889 | 0.828 / 0.889 |
| owasp (20) | 0.783 / 0.950 | 0.867 / 0.900 | 0.877 / 0.950 | 0.950 / 0.950 |

### Paired deltas vs `dense` (mean difference, 95% CI)

| Config | Metric | Δ | 95% CI | Significant |
|---|---|---|---|---|
| bm25 | recall@5 | +0.323 | [+0.250, +0.406] | yes |
| bm25 | mrr@10 | +0.295 | [+0.221, +0.373] | yes |
| bm25 | ndcg@10 | +0.298 | [+0.229, +0.373] | yes |
| hybrid | recall@5 | +0.318 | [+0.250, +0.391] | yes |
| hybrid | mrr@10 | +0.315 | [+0.250, +0.388] | yes |
| hybrid | ndcg@10 | +0.318 | [+0.256, +0.388] | yes |
| hybrid_rerank | recall@5 | +0.333 | [+0.266, +0.406] | yes |
| hybrid_rerank | mrr@10 | +0.357 | [+0.293, +0.424] | yes |
| hybrid_rerank | ndcg@10 | +0.350 | [+0.289, +0.417] | yes |

### Rerank on top of hybrid (paired)

| Metric | Δ | 95% CI | Significant |
|---|---|---|---|
| recall@5 | +0.016 | [-0.016, +0.052] | no |
| mrr@10 | +0.042 | [+0.008, +0.077] | yes |
| ndcg@10 | +0.032 | [+0.006, +0.059] | yes |

LLM for generation checks: `openai/gpt-oss-20b`

### Generation checks (n=40 answerable, 10 unanswerable, 0 failed calls)

| Metric | Value |
|---|---|
| Citation precision (verified / cited IDs) | 100.0% |
| Hallucinated-ID rate, answerable queries | 0.0% |
| Hallucinated-ID rate, unanswerable queries | 0.0% |
| Answerable: gold ID cited in answer | 90.0% |
| Answerable: abstained (should be low) | 2.5% |
| Fake CVE ID: abstained (should be high) | 100.0% |
| Off-topic: abstained (should be high) | 100.0% |

# SecureRAG — Interview Knowledge Base

> **Audience: an LLM** that will role-play / coach the author of this project through technical interviews
> (including a hostile "applied scientist" grilling). **Target interview: Amazon Applied Scientist intern** — read §13
> (role-specific prep: resume bullets, ML/IR math, experiment design, AWS mapping, Leadership-Principle stories, hostile ladders, whiteboard code) alongside §8. Everything here was derived by reading the actual
> source in this repo, not from the README. Where the README / CLAUDE.md / INTERVIEW_GUIDE.md disagree
> with the code, **the code wins** (see §12).

---

## 0. Rules for the LLM using this document

1. **Answer in first person as the author** ("I built…", "I chose…") unless asked to coach/critique.
2. **Never invent numbers.** No latency, accuracy, recall, eval-score, cost, or user-count figure exists in
   this document unless tagged `[VERIFIED]`. If asked for one that is tagged `[NOT MEASURED]`, say so
   plainly and describe how you'd measure it (§9). A candidate who says "I haven't benchmarked that, here's
   how I would" beats one who makes up a number and gets caught.
3. **Tags used below**
   - `[VERIFIED]` — confirmed by reading code/config in this repo.
   - `[NOT MEASURED]` — a plausible claim that no one has benchmarked. Do not state as fact.
   - `[AUTHOR?]` — only the author knows (motivation, history). Ask them; see §11.
   - `[UNVERIFIED]` — I believe it from library knowledge but couldn't run it (deps not installed here).
   - `[MEASURED]` — produced by the `evals/` harness on 2026-11-10 (single run, seeded; see §5.12b for exact
     setup and caveats). Quote these with their caveats: synthetic queries, one run, 2,000-CVE corpus. Retrieval
     numbers don't involve the LLM; **generation numbers were produced with `openai/gpt-oss-20b`, not
     Llama-3.1-8B** (see §7 item 18).
4. **Be honest about weaknesses.** §7 lists real defects found in the code. Interviewers probe exactly
   these. Volunteer the trade-off *before* being asked when it's relevant — it signals engineering maturity.
5. **Distinguish "what runs in production" from "what exists in the repo"** (§3). Redis, Prometheus,
   Grafana only exist in the Docker Compose stack; the live Replit deployment is a much smaller thing.
6. Structure good answers as: *decision → alternatives → why this one → what it costs → when I'd revisit.*

---

## 1. Pitches

### 30 seconds
SecureRAG is a cybersecurity question-answering chatbot built on retrieval-augmented generation. It indexes
~2,000 recent CVEs from the NIST NVD API plus the OWASP Top 10 (2021) into ChromaDB. A question goes through
hybrid retrieval — BM25 plus dense vector search fused with Reciprocal Rank Fusion — then a cross-encoder
reranks to the top 5, and Llama 3.1 8B on Groq answers *only* from that context and cites CVE IDs / OWASP
categories. It streams tokens over SSE, has multi-turn memory, severity/source filters, a weekly refresh job,
Prometheus metrics, and a lightweight evaluation harness. Live at rag.awni.in.

### 2 minutes
LLMs hallucinate CVE IDs and CVSS scores and their knowledge is stale, which is a bad failure mode in
security. So instead of fine-tuning, I ground the model in retrieved documents. Retrieval is two-stage:
a cheap high-recall first stage (BM25 for exact identifiers like `CVE-2024-1234`, dense BGE embeddings for
paraphrase/concept queries, merged with RRF because the two score scales are incomparable), then a
precise-but-expensive cross-encoder over just the 10 fused candidates. The top 5 go into a prompt that
forbids answering outside context and requires citations. The backend is FastAPI; the frontend is vanilla JS
with no build step and consumes a POST-based SSE stream. Around the core I added production concerns:
tenacity retries on the LLM, SlowAPI rate limiting, structured JSON logs with request IDs, Prometheus
metrics + Grafana dashboard, an APScheduler weekly NVD refresh, and a Redis semantic cache that keys on
embeddings rather than strings. I also built an eval harness (`evals/`): a 192-query labelled retrieval set with BM25 / dense /
hybrid / hybrid+rerank ablations and bootstrap CIs `[MEASURED]`, plus a citation check that flags CVE/OWASP IDs
not backed by the retrieved context. The older RAGAS-style metrics in `rag/evaluate.py` are lexical proxies and
I treat them as weak (§5.12).

### One-line elevator
"A grounded, citation-first security RAG: hybrid BM25+dense retrieval → RRF → cross-encoder → Llama-3.1-8B on Groq."

---

## 2. Fact sheet

| Area | Value | Source |
|---|---|---|
| Language / runtime | Python 3.11 (Dockerfile + Replit), FastAPI 0.115, Uvicorn 0.30 | `requirements.txt`, `Dockerfile` |
| LLM | `llama-3.1-8b-instant` via Groq, `langchain-groq` 0.2.0, temp **0.1**, max_tokens **1024** | `core/config.py` |
| Embeddings | `BAAI/bge-small-en-v1.5`, 384-dim, CPU, `normalize_embeddings=True` | `rag/embeddings.py` |
| Query prefix | `"Represent this sentence for searching relevant passages: "` (queries only, not documents) | `rag/embeddings.py` |
| Vector DB | ChromaDB 0.5.0 `PersistentClient`, collection `securerag_docs`, `hnsw:space=cosine` | `rag/vectorstore.py` |
| Sparse | `rank-bm25` `BM25Okapi` (defaults k1=1.5, b=0.75), tokenizer = `lower().split()`, whole corpus in RAM | `rag/hybrid_search.py` |
| Fusion | RRF, **k=60**, unweighted; 20 BM25 + 20 dense candidates → top 10 | `rag/hybrid_search.py` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, max_length 512, doc text truncated to 2000 chars, top 5 | `rag/reranker.py` |
| Post-filter | severity + source, applied **after** reranking | `rag/chain.py::_apply_filters` |
| History | in-process `OrderedDict`, LRU cap **20** sessions, last **6** turns injected into prompt | `rag/chain.py` |
| Retry | tenacity, 3 attempts, exp backoff 1–8s, on `httpx.TimeoutException`/`HTTPStatusError` (see §7 caveat) | `rag/chain.py` |
| Rate limit | SlowAPI: chat **20/min/IP**, ingest **5/hour/IP**, in-memory | `core/config.py` |
| Semantic cache | Redis hash per entry; float32 embedding bytes + JSON result; cosine threshold **0.92**, TTL **3600s** (refreshed on hit) | `rag/semantic_cache.py` |
| Scheduler | APScheduler `AsyncIOScheduler`, cron `0 2 * * 0` UTC, `misfire_grace_time=3600` | `core/scheduler.py` |
| Corpus | ~2,000 CVEs (cap 2000) + 10 OWASP entries × 3 chunks = 30 chunks → ~2,030 docs `[AUTHOR? confirm from prod]` | `data/ingest.py` |
| NVD fetch | last 365 days in **119-day** windows → 4 windows (119+119+119+8), per-window cap 500 | `data/ingest.py` |
| NVD pacing | 6.0s sleep without API key, 0.6s with | `data/ingest.py` |
| Ingest batch | 100 docs per `add_documents` call | `data/ingest.py` |
| Endpoints | `POST /api/chat`, `POST /api/chat/stream`, `GET /api/health`, `POST /api/ingest`, `GET /api/ingest/status`, `POST /api/evaluate`, `GET /api/stats`, `DELETE /api/session/{id}`, `GET /metrics` | `api/routes.py`, `main.py` |
| Frontend | Vanilla HTML/CSS/JS (~745 lines JS), `marked` + `Chart.js` from CDN, no build | `frontend/` |
| Observability | Prometheus (`/metrics`), Grafana dashboard, JSON logs + `X-Request-ID` | `core/*`, `monitoring/` |
| Tests | One file `tests/unit/test_hybrid_search.py` (RRF, filters, config, cache) — **cache tests are stale/broken** (§7) | `tests/` |
| App version | `2.0.0` | `core/config.py` |

**Not in this table on purpose (not measured):** end-to-end latency, retrieval recall/MRR, eval scores,
memory footprint, throughput, concurrent-user capacity.

---

## 3. What actually runs where (read before answering any "production" question)

| Capability | Local (`uvicorn`) | Docker Compose | Live: Replit Autoscale (rag.awni.in) |
|---|---|---|---|
| FastAPI + RAG pipeline | yes | yes | yes |
| ChromaDB | local `./chroma_db` | named volume `/data/chroma_db` | **baked into the build image** by running `python data/ingest.py --force` in the build step |
| Redis semantic cache | only if you run Redis | yes (`redis:7.2-alpine`, AOF on) | **almost certainly absent** → cache degrades to a no-op `[AUTHOR? confirm no Redis in prod]` |
| Prometheus + Grafana | no | yes (ports 9090 / 3000) | no |
| Weekly APScheduler refresh | runs | runs | runs per-instance, but writes to each instance's *ephemeral* local disk (§7.9) |
| Session memory | in-process | in-process | in-process, **per instance** — autoscale can route a user's next request to a different instance with no history |

Production run command: `cd securerag && uvicorn main:app --host 0.0.0.0 --port 8080`, health check
`/api/health` (`artifacts/api-server/.replit-artifact/artifact.toml`). The repo is a Replit **pnpm monorepo**
with a leftover TypeScript `api-server` / `mockup-sandbox` — the SecureRAG app is the standalone Python
service in `securerag/`; the Node parts are scaffolding, not part of the product.

**Why ingest at build time:** autoscale instances don't share disk, so ingest-on-click wrote data to one
container that was recycled/unreachable from others. Moving ingest into the *build* phase makes the populated
ChromaDB part of the immutable image. Trade-off: data is as fresh as the last build; build is slow (author
reports ~15–20 min `[AUTHOR?]`); a real fix is an external vector DB (Qdrant/pgvector/Pinecone) shared by instances.

---

## 4. Architecture

### 4.1 Component diagram

```
Browser (vanilla JS)
  │  POST /api/chat/stream  (fetch + ReadableStream, SSE frames)
  ▼
FastAPI  ── MetricsMiddleware (request_id, HTTP histograms) ── CORS ── SlowAPI (20/min/IP)
  │
  ├─ rag/chain.py  stream_answer_query / answer_query
  │     1. session history (OrderedDict LRU)
  │     2. [/chat only, first turn only] SemanticCache.get(embed(q))  → Redis
  │     3. HybridSearcher.search(q)  ─ ThreadPool(2) ─┬─ BM25Okapi.get_scores   → top 20
  │                                                   └─ Chroma.query (cosine)  → top 20
  │          └─ RRF(k=60) → top 10
  │     4. CrossEncoderReranker.rerank → top 5
  │     5. _apply_filters(severity, source)          (post-rerank; falls back to unfiltered if empty)
  │     6. prompt = SYSTEM_RULES + context + history + question   (one HumanMessage)
  │     7. ChatGroq  invoke() w/ tenacity   |   astream() (no retry)
  │     8. history.append; [/chat] cache.set
  ▼
answer + sources[{id,type,severity,score}] + search_metadata{bm25_hits,semantic_hits,rrf_final,filtered_out}

Offline / background:  data/ingest.py  (NVD API + owasp_data.json) → parse → chunk → embed → ChromaDB
                       APScheduler weekly → run_ingest(force=True) → rebuild BM25 → invalidate cache
Observability:         /metrics → Prometheus (15s scrape) → Grafana (provisioned dashboard JSON)
```

### 4.2 Request path, step by step (`POST /api/chat`)

1. SlowAPI decorator enforces `20/minute` per remote IP. `ChatRequest` (Pydantic) validates `message` 1–2000 chars.
2. Route returns **503** if the ChromaDB collection is empty.
3. `answer_query` calls `_get_session()` **first** (before the cache check) so "is this multi-turn?" is knowable
   (`chain.py` — don't reorder).
4. If history is empty → embed query (with BGE prefix) → semantic-cache lookup. Multi-turn queries bypass
   the cache because the answer depends on prior turns.
5. Retrieval: `HybridSearcher.search` runs BM25 and Chroma concurrently in a 2-thread pool (both release the
   GIL in numpy/C++ paths — `[NOT MEASURED]` speedup), fuses with RRF, keeps 10.
6. Rerank: build `(query, doc[:2000])` pairs, `CrossEncoder.predict`, sort by raw logit, keep 5.
7. Filter by severity/source (OWASP chunks have no severity so they're never severity-filtered).
8. Format context as `[CVE-ID]\n<text>` blocks joined by `---`; sources de-duplicated by ID.
9. LLM call (non-stream: `_call_llm_with_retry`; stream: `llm.astream`). On failure returns a canned apology
   string (non-stream) or an SSE `error` event (stream).
10. Append `{human, assistant}` to history; trim; optionally write to semantic cache.

### 4.3 SSE protocol (`/api/chat/stream`)
Events, each `data: {json}\n\n`: `metadata` (sources + search_metadata, sent *before* the first token so the UI
can render citation chips immediately) → N × `token` → `done{response_time_ms}`; `error{message}` on failure.
Headers: `Cache-Control: no-cache`, `X-Accel-Buffering: no` (stop nginx buffering), `X-Session-ID`.
**POST + `fetch().body.getReader()`**, not `EventSource`, because `EventSource` is GET-only and I need a JSON body.

### 4.4 Startup (`main.py::lifespan`)
Eagerly load embedding model → cross-encoder → read Chroma count → build BM25 index → start APScheduler.
Eager loading moves the multi-second model load out of the first user request; Dockerfile additionally
pre-downloads both models at build time (~90 MB) so containers start fast. Static frontend is mounted at `/`
**last** so it acts as catch-all after `/api` and `/metrics`.

### 4.5 Ingest (`data/ingest.py`)
`fetch_cves` (cache in `data/cve_cache.json` unless `--force`) → `parse_cve` (CVSS v3.1 → v3.0 → v2 fallback;
v2 severity derived from score thresholds 7.0/4.0) → `cve_to_document` → OWASP → 3 chunks → `delete_collection()`
→ batches of 100 → `add_documents` (embed + `collection.add`). After: `rebuild_searcher()` +
`cache.invalidate()` + gauge updates. **Full rebuild, not incremental** (§7.9).

### 4.6 Module map
```
main.py                FastAPI app, lifespan, middleware, /metrics, static mount
api/routes.py          endpoints, ingest background task, in-memory ingest/eval/stats state
api/models.py          Pydantic request/response schemas
core/config.py         pydantic-settings Settings singleton (all tunables)
core/metrics.py        Prometheus metric singletons
core/middleware.py     request-id + HTTP metrics (path normalisation to bound label cardinality)
core/logging_config.py JSON logs, request_id via contextvars
core/scheduler.py      APScheduler weekly refresh
rag/embeddings.py      BGE singleton, query prefix
rag/vectorstore.py     Chroma client/collection helpers
rag/hybrid_search.py   BM25 + dense + RRF
rag/reranker.py        cross-encoder
rag/chain.py           orchestration, prompt, sessions, streaming
rag/semantic_cache.py  Redis embedding-keyed cache
rag/evaluate.py        lexical RAGAS-style metrics (legacy, weak)
rag/citations.py       CVE/OWASP ID extraction + citation verification (flag-only)
evals/                 build_dataset.py, retrieval_eval.py, generation_eval.py, metrics.py, run.py, data/retrieval_set.jsonl
data/ingest.py         NVD fetch/parse/chunk/index
frontend/              index.html, style.css, app.js
monitoring/            prometheus.yml, Grafana provisioning + dashboard
```

---

## 5. Component deep dives

Each: **What → Why → Alternatives → Trade-offs/gotchas.**

### 5.1 Why RAG (not fine-tuning, not plain LLM)
- **Why:** CVEs are published continuously; a model's weights are frozen. RAG updates knowledge by re-indexing
  (minutes, ~free) vs. fine-tuning (GPU, data prep, re-run on each change). RAG also yields *citable* evidence,
  which matters for security answers.
- **Fine-tuning's real role:** style/format/behaviour, not fresh facts. Could complement RAG (e.g. fine-tune
  to follow the cite-or-abstain format) but not replace it.
- **Long-context stuffing alternative:** 2k CVEs ≈ far too many tokens per query on an 8B model; cost/latency
  and "lost in the middle" degradation. Retrieval is the filter.
- **Caveat to volunteer:** RAG reduces but doesn't eliminate hallucination; the model can still misread
  context or ignore the rules. Measured at the ID level only: 0 of 50 answers cited a CVE/OWASP ID absent from the context
  (n=40 answerable + 10 unanswerable, `gpt-oss-20b`) `[MEASURED]`; that is a small sample and says nothing about
  whether the *claims* around the IDs are faithful (§5.12b).

### 5.2 Data: NVD + OWASP and chunking
- **NVD API 2.0**, `pubStartDate`/`pubEndDate`. **Hard limit: 120-day range** — exceed it and you get a 404 with
  no message (diagnosed by shrinking the range: 365d → 404, 90d → 200). Fix: 119-day windows.
- **Rate limit:** 5 req / 30s keyless → 6s sleep; keyed 50 req / 30s → 0.6s.
- **Sample, not census.** NVD publishes tens of thousands of CVEs/year; I cap at 2,000 (500/window). So the KB
  is a **sample**; the bot can't claim "no CVE exists for X". The cap is also unrepresentative: windows are
  fetched newest→oldest with the first 500 results of each window, and the final window is only 8 days wide,
  so coverage over time is skewed.
- **One CVE = one document** (descriptions are short, well within the embedder's 512-token limit — chunking
  would only fragment). Text template puts ID, severity+score, description, affected products, date into the
  body so lexical *and* dense retrieval can both see severity/ID; structured fields also go to metadata for filters.
- **OWASP = 3 chunks per category** (description / prevention / scenarios) because the three sections answer
  different question types ("what is X" vs "how do I prevent X" vs "show me an attack") and a single long chunk
  dilutes the embedding. Each chunk repeats the OWASP ID + name for context. `chunk_type` is in metadata.
- **Parsing gotchas:** prefers CVSS v3.1, falls back to v3.0, then v2; missing metrics → severity `UNKNOWN`,
  score 0.0. Only English descriptions. Only first 3 references / first few CPE matches kept.
- **Alternatives:** add CWE/CAPEC/MITRE ATT&CK, vendor advisories, exploit-DB / CISA KEV (which would add
  "is this actively exploited?" — high value for security Q&A). Not done.

### 5.3 Embeddings — `BAAI/bge-small-en-v1.5`
- **What:** 384-dim bi-encoder, ~33M params, 512-token limit, trained for retrieval; vectors L2-normalised so
  dot product = cosine.
- **Why this model:** free, local/CPU (no per-call cost, no data leaves the box), small enough to embed 2k docs
  quickly, strong for its size on retrieval benchmarks (MTEB) — `[NOT MEASURED on this corpus]`. The choice was
  by reputation/benchmarks, I did **not** run a head-to-head on my own queries `[AUTHOR? confirm]`.
- **Asymmetric prefix:** BGE expects the instruction prefix on *queries* but not documents for short-query →
  passage retrieval. Handled in `embed_query()`. Forgetting it silently degrades retrieval.
- **Alternatives:** `bge-base/large` (better quality, slower, bigger), `e5`, `gte`, OpenAI `text-embedding-3`
  (API cost, data egress, vendor lock), domain-tuned security embeddings (none standard; could fine-tune
  on CVE↔question pairs).
- **Gotcha:** changing the embedding model requires re-embedding the whole corpus and dropping the collection
  (dimension / geometry change). Also, the same embedder feeds the semantic-cache keys.

### 5.4 Vector store — ChromaDB
- **What:** embedded persistent store, HNSW index, cosine space. `score = 1 - distance` in `semantic_search`.
- **Why:** zero ops, in-process, free, perfect for ~2k docs and a portfolio deploy.
- **Honest note:** at 2k vectors HNSW is overkill — brute-force numpy would be fast and exact. Chroma was chosen
  for the persistence/metadata API and "this is what a real system would swap in".
- **Alternatives & when I'd switch:** Qdrant/Weaviate/Milvus (filtering, scale, multi-node), **pgvector**
  (if I already run Postgres; transactional + SQL filters + hybrid with `tsvector`), Pinecone (managed),
  FAISS (library, no metadata). Switch trigger: >~100k docs, multi-instance shared state (the autoscale problem),
  or need for server-side metadata pre-filtering at scale.
- **Gotchas:** metric is fixed at collection creation (changing it = delete+recreate). Persistent client on a
  local disk doesn't work across autoscale instances. Chroma 0.5.x telemetry needed an env var set *before*
  import (`ANONYMIZED_TELEMETRY=False` in `main.py`).

### 5.5 BM25 (sparse)
- **What:** in-memory `BM25Okapi` over the *entire* corpus loaded from Chroma at startup
  (`get_all_documents()`), rebuilt via `rebuild_searcher()` after any ingest.
- **Why:** security queries are full of exact tokens — CVE IDs, product names, version strings, acronyms
  (`SSRF`, `Log4Shell`). Dense models blur or mis-embed these; BM25 nails them. And it's dirt cheap.
- **Tokenizer weakness (volunteer this):** `text.lower().split()` — no punctuation stripping, no stemming, no
  stop-words, no handling of `CVE-2024-1234,` / `?`. A user typing `CVE-2024-1234?` produces token
  `cve-2024-1234?` which won't match. Better: regex tokenizer that keeps hyphenated IDs, strip trailing
  punctuation, optional ID-extraction fast path (regex for `CVE-\d{4}-\d+` → direct metadata lookup).
- **Alternatives:** Elasticsearch/OpenSearch/Tantivy/Lucene (persistent, incremental, scalable), SPLADE
  (learned sparse). `rank_bm25` recomputes scores for every doc per query (O(corpus)) — fine at 2k, not at 2M.
- **Scalability limit:** whole corpus in RAM, rebuilt in full on each ingest.

### 5.6 Hybrid fusion — RRF
- **Formula:** `score(d) = Σ_lists 1/(k + rank_list(d) + 1)`, **k=60** (Cormack, Clarke & Büttcher 2009).
- **Why RRF over score fusion:** BM25 scores are unbounded and corpus-dependent; cosine similarity is in [-1,1].
  Min-max/z-score normalisation is brittle (outliers, tiny result sets). RRF uses only ranks → no tuning, robust.
- **Why k=60:** standard default; damps the dominance of rank-1 so agreement across lists beats a single
  list's #1. I did **not** tune k or weight the two retrievers `[NOT MEASURED]`.
  Ablation result (§5.12b): hybrid vs BM25 alone is **not a clear win** here (R@5 0.906 vs 0.911, MRR 0.832 vs 0.812,
  overlapping CIs); both beat dense alone by ~0.3 MRR because dense can't match CVE IDs. Hybrid's edge shows in
  Recall@10 (0.953 vs 0.932) and OWASP-style questions. Don't claim "hybrid beats BM25" without that nuance.
- **Candidates:** 20 from each retriever → fuse → keep 10. (Older docs wrongly say 10 each.)
- **Dedup key:** first 200 chars of text (`text[:200]`). Works because every doc starts with a unique ID header,
  but it's fragile — a cleaner key is the Chroma doc ID (which `get_all_documents` currently drops).
- **Alternatives:** weighted RRF, convex combination of normalised scores (can be better when tuned on labelled
  data), learned fusion / learning-to-rank, ColBERT/late-interaction, single-index hybrid (Qdrant/Weaviate
  native hybrid).
- **Test:** `test_rrf_fusion_prefers_consistent_results` asserts docs ranked well in both lists beat single-list docs.

### 5.7 Cross-encoder reranker
- **What:** `ms-marco-MiniLM-L-6-v2` scores each (query, doc) pair jointly via full cross-attention — more
  accurate than bi-encoder similarity because query and doc tokens interact. O(n) forward passes of the pair,
  so only affordable on a short list (10).
- **Why two-stage:** cheap high-recall → expensive high-precision. Retrieving 10 and reranking to 5 also cuts
  prompt length (cost/latency) and noise (less distractor context for an 8B model).
- **Output is a raw logit**, not a probability — that's why the reranker-score histogram has buckets from -10 to 10.
- **Gotchas to volunteer:**
  - MS MARCO is general web QA; it isn't trained on CVE text. Despite that, on my set reranking the hybrid top-10
    improved MRR@10 0.832 → 0.873 (paired Δ +0.042, 95% CI [+0.008, +0.077]) and nDCG@10 +0.032, but Recall@5 only
    +0.016 (CI includes 0) `[MEASURED]`. Cost: p50 ~350 ms vs ~21 ms for retrieval alone on CPU (~17x).
  - **No score threshold:** the top 5 always go to the LLM even if all are irrelevant. Abstention relies solely
    on the prompt's "say you don't know" rule. A calibrated threshold (e.g. drop docs below a logit cut-off,
    return "no relevant info" if none pass) would be a cheap, high-value improvement.
  - Truncated to 512 tokens / 2000 chars.
- **Alternatives:** bigger cross-encoders (`bge-reranker-base/large`, `ms-marco-MiniLM-L-12`), Cohere Rerank API,
  LLM-as-reranker (listwise; accurate, slow/expensive).
- **Reranking improves ordering, not recall:** if the right doc isn't in the 10 fused candidates, the reranker can't save it.

### 5.8 Filters (severity / source)
- Applied **after** rerank so filters don't distort the ranking stage.
- **Real flaw (volunteer):** this is post-filtering of only 5 items. If the user selects "CRITICAL only" and
  none of the top 5 are critical, `filtered` is empty and the code **falls back to the unfiltered set** —
  silently violating the user's filter (`chain.py` "falling back to unfiltered set"). The right fix is
  **pre-filtering** in retrieval: Chroma `where={"severity": {"$in": [...]}}` for dense, and a masked BM25 score
  array for sparse, so top-k are drawn from the allowed set.
- OWASP chunks have no severity → never removed by the severity filter; `source_filter` is `both|nvd|owasp`.
- Default (no filter) = all four severities; `UNKNOWN`-severity CVEs are dropped when any severity list is applied.

### 5.9 Prompting & the LLM (Groq / Llama 3.1 8B)
- **Prompt:** single `HumanMessage` containing rules + context + chat history + question (no separate system
  role). Rules: answer only from context; fallback phrase if insufficient; cite CVE/OWASP IDs; always give
  severity with CVSS; concise bullets; never invent IDs/scores.
- **Temp 0.1:** near-deterministic for a factual task, avoids the repetition/degeneration of exactly 0.
- **Why Groq + Llama 3.1 8B:** free tier, very low latency (LPU inference) → good UX for streaming demos;
  open-weights 8B is adequate for extractive/summarising answers *given* good context. `[AUTHOR? confirm
  reasons]`.
- **Why 8B is OK here:** the task is mostly grounded extraction/summarisation, not reasoning. It would struggle
  with multi-hop questions ("which of these CVEs affect the same product family and chain into RCE?").
- **Alternatives:** Llama-3.3-70B / Claude / GPT-class (higher faithfulness, citation accuracy, cost/latency),
  local Ollama (privacy, ops burden).
- **Weaknesses (volunteer):**
  - Instructions and untrusted retrieved text share one message → **prompt injection** via CVE descriptions
    (NVD text is third-party-authored) is possible. Mitigations: delimit/escape context, system-role separation,
    output filtering, treat retrieved text as data.
  - No structured output / citation verification: citations are *requested*, not *checked*. A post-hoc check
    ("every CVE ID in the answer must appear in the context") is easy and valuable.
  - History is concatenated as text; no summarisation beyond truncation.
  - LLM calls are text-in/text-out via LangChain; I use LangChain only as a thin client (`ChatGroq`,
    `HumanMessage`) — orchestration is plain Python on purpose (clear, debuggable, no framework magic).
- **LangChain honesty:** it's a thin dependency here (`langchain-groq`, `langchain-community` embeddings
  wrapper). I could drop it for the Groq SDK + sentence-transformers directly.

### 5.10 Multi-turn memory
- In-process `OrderedDict[session_id → list[{human, assistant}]]`; LRU evicts the oldest when 20 sessions exist;
  prompt uses the last 6 turns. Session IDs are client-generated UUIDs. `DELETE /api/session/{id}` clears.
- **Limitations:** lost on restart; not shared across workers/instances; not thread-safe (single event loop, but
  sync code runs on the loop); **retrieval uses only the raw latest question** — no query rewriting. "What about
  its fix?" retrieves poorly because the pronoun has no referent in the retrieval query. Fix: condense
  (history + follow-up → standalone question via a small LLM call) before retrieval.
- Minor: stored list is trimmed at `2 × max_history_turns` entries (12) though only 6 are used in the prompt.
- **Production alternative:** Redis/Postgres keyed by session/user, TTL, summarisation memory.

### 5.11 Streaming (SSE)
- Why: perceived latency. Retrieval+rerank are local and quick, then tokens render as they arrive; citations
  appear first.
- **POST+fetch+ReadableStream** instead of `EventSource` (GET-only). Frontend buffers partial lines, re-renders
  markdown on each token with `marked.parse`.
- **No retry** in the stream path (can't retry mid-stream meaningfully); errors become an `error` event.
- **Gaps (volunteer):** the stream path does **not** use the semantic cache, doesn't update `queries_served`, and
  records fewer Prometheus stage metrics than `/chat`. Since the UI uses `/chat/stream` exclusively, **the semantic
  cache is effectively unused by the UI** (it serves only `/api/chat` API clients). Also, a client disconnect
  mid-stream means the history isn't saved.

### 5.12 Evaluation — legacy lexical harness (`rag/evaluate.py`)
- **Why custom:** RAGAS 0.1.x required `langchain<0.3`; I was on 0.3. Rather than pin back, I reimplemented
  RAGAS-*inspired* metrics.
- **What each metric actually computes (all lexical, stop-word-filtered, tokens >2 chars, *sets*):**
  - **Faithfulness** = |answer tokens ∩ context tokens| / |answer tokens|.
  - **Answer relevancy** = 0.4·Jaccard(q, a) + 0.5·(q-keyword coverage by a) + up to 0.1 length bonus.
  - **Context recall** = |ground-truth tokens ∩ context tokens| / |ground-truth tokens|.
  - **Context precision** = fraction of the 5 chunks with >10% question-token overlap.
- **Test set:** 10 hand-written questions + hand-written ground truths — all OWASP/general; **none CVE-specific**.
  Results cached 1h; `POST /api/evaluate` is unauthenticated and costs 10 LLM calls (§7).
- **Be blunt about validity:** these are *surface-overlap proxies*, not semantic. Faithfulness is high if the
  model copies context words even when it states a wrong number; low for correct paraphrase. Answer relevancy
  with Jaccard is nearly meaningless (answers are long, questions short). n=10, single run, no confidence
  intervals, and the metrics are not validated against human judgement. **Never present the scores as evidence
  of quality** — present the harness as scaffolding. (Scores are not recorded in this repo `[AUTHOR?]`.)
- **What a proper eval looks like** (§9): labelled retrieval set (recall@k, MRR, nDCG), ablations,
  LLM-as-judge or NLI-based faithfulness with claim decomposition, citation-precision, human spot checks,
  bootstrap CIs. The retrieval and citation parts of that list now exist — see §5.12b.

### 5.12b Evaluation — `evals/` harness `[MEASURED]` (run 2026-11-10)
- **How it's built:** `python -m evals.run --build-dataset | --retrieval | --generation | --all`. Corpus at run time:
  2,000 CVEs + 30 OWASP chunks (fresh NVD ingest). Gold labels come from construction, not annotation.
- **Dataset (192 queries):** 60 `id` ("Tell me about CVE-…", templated), 58 `keyword` + 54 `natural` (LLM-written from a
  severity-stratified sample of 60 CVE docs, seed 42; a leak filter dropped 6 that contained the ID or a 6-word verbatim
  run), 20 `owasp` (hand-written; gold = category, any of its 3 chunks counts; results de-duplicated by ID).
- **Metrics:** Recall@1/3/5/10, MRR@10, nDCG@10 (binary relevance), 95% percentile bootstrap CIs (1000 resamples,
  seeded), paired bootstrap for deltas. Latency = CPU, one query at a time, after warm-up.

| Config | R@1 | R@5 | R@10 | MRR@10 [95% CI] | nDCG@10 | p50 / p95 ms |
|---|---|---|---|---|---|---|
| BM25 | 0.760 | 0.911 | 0.932 | 0.812 [0.764, 0.859] | 0.841 | 3 / 7 |
| Dense (BGE-small) | 0.464 | 0.589 | 0.625 | 0.517 [0.451, 0.583] | 0.543 | 18 / 22 |
| Hybrid (RRF) | 0.776 | 0.906 | 0.953 | 0.832 [0.786, 0.879] | 0.861 | 21 / 31 |
| Hybrid + rerank | 0.833 | 0.922 | 0.953 | 0.873 [0.833, 0.916] | 0.893 | 352 / 712 |

- **By style (MRR@10, BM25 / dense / hybrid / hybrid+rerank):** id 0.936 / **0.044** / 0.927 / 0.958 ·
  keyword 0.736 / 0.702 / 0.774 / 0.801 · natural 0.766 / 0.714 / 0.772 / 0.828 · owasp (n=20) 0.783 / 0.867 / 0.877 / 0.950.
- **Findings to state:** (1) dense-only is clearly worst and fails on identifier queries (embeddings don't encode IDs) —
  the concrete justification for hybrid; (2) hybrid ≈ BM25 on Recall@5 on this set; (3) reranking improves ordering
  (MRR/nDCG, significant) but not Recall@5 (CI includes 0), and it reranks the same 10 candidates so Recall@10 is
  identical to hybrid by construction; (4) rerank costs ~17x latency on CPU.
- **Caveats (volunteer them):** queries are synthetic and LLM-written from the doc, so they reuse product names and
  jargon → favours BM25 and likely overstates it vs real users; one run, one sampled 60 CVEs, no tuning of k / candidate
  counts; near-duplicate CVEs can make a "miss" a legitimate alternative; the 20 OWASP questions are too few to rank configs on.
  Production recall@5 = the reranked top-5 (same ordering as the table's `hybrid_rerank`).
- **Generation checks** (`openai/gpt-oss-20b`; 40 answerable = 30 CVE natural + 10 OWASP; 10 unanswerable = 5 fake CVE IDs
  + 5 off-topic; no LLM judge): citation precision 100% (63 IDs cited, all in context); hallucinated-ID rate 0/40 and
  0/10; gold ID cited in 90% of answerable answers; wrongly abstained 1/40; abstained on 5/5 fake IDs and 5/5 off-topic.
  0/40 is a small sample (95% upper bound ≈7%), and **ID existence ≠ claim faithfulness**.
- **Citation guard (`rag/citations.py`):** regex-extracts `CVE-YYYY-NNNN+` and `A01–A10[:2021]`; an ID is allowed if it appears
  in retrieved-doc metadata/text or in the user's messages/history (so "I don't have CVE-2099-0001" isn't flagged). Violations
  are **flagged, not rewritten** (works with streaming; never silently edits the model): `verification` field on `/api/chat`,
  an SSE `verification` event before `done`, a UI warning badge, and counter `securerag_unverified_citations_total`.
  Unit-tested, but I have **not** seen it fire on a real model hallucination (none occurred), so the positive path is tested only synthetically.
- **Bugs the harness surfaced:** (a) gpt-oss emits non-breaking hyphens (`CVE‑2025‑64773`, U+2011); my first regex missed them
  (reported 25% gold-cited instead of 90%) → `extract_ids` now folds Unicode dashes; (b) `/api/chat` crashed on every
  first-turn request (`METRICS.cache_misses` didn't exist) → counters added; (c) `llama-3.1-8b-instant` returned 404
  `model_not_found` (§7 item 18).

### 5.13 Semantic cache (Redis)
- **Idea:** key on meaning: "SQL injection CVEs" ≈ "CVEs about SQL injection". Entry = Redis hash
  `{embedding: float32 bytes (384×4=1536 B), result: JSON}`, prefix `securerag:sem:`, TTL 3600s refreshed on hit.
- **Lookup:** `SCAN` keys → `HGET` each embedding → stack into a matrix → **one matmul** `matrix @ q` (vectors are
  L2-normalised so dot = cosine) → argmax ≥ **0.92** → return cached result.
- **Why 0.92?** A heuristic, **untuned** `[NOT MEASURED]`. Too low → wrong answers served for different-but-
  nearby questions ("is CVE-X exploited?" vs "is CVE-Y exploited?" can be very close in embedding space!);
  too high → rare hits. Tune on labelled paraphrase/non-paraphrase pairs.
- **Bypassed** for multi-turn (answer depends on history) and invalidated after every ingest.
- **Graceful degradation:** every Redis op catches `RedisError` → miss/no-op. App works with no Redis.
- **Known defects (volunteer):**
  1. **Cache key ignores filters.** A result cached for "SQL injection" with `CRITICAL`-only can be returned to
     an unfiltered query (and vice-versa) because only the embedding is compared.
  2. **O(N) Redis round-trips per lookup** (SCAN + one HGET per entry). Fine for hundreds of entries, bad at scale.
     Fix: store vectors in one structure / use RediSearch vector index / Redis Stack, or keep an in-process matrix
     with Redis as persistence.
  3. No max-entries/LRU — only TTL (CLAUDE.md's "500 entries max" describes an older in-memory version).
  4. Cache hits skip retrieval so they don't reflect NVD updates until TTL/invalidate — acceptable because
     ingest invalidates.
  5. Unused by the UI (§5.11); tests for it are stale (§7).
- **Alternatives:** exact-match cache (cheap, low hit rate), GPTCache, Redis Stack vector search, LLM-prompt
  caching at the provider (different thing: caches prefix tokens, not answers).

### 5.14 Reliability: retries, rate limiting, degradation
- **Tenacity:** 3 attempts, `wait_exponential(multiplier=1, min=1, max=8)`, `reraise=True`, logs before sleep.
  Wraps only synchronous `llm.invoke` (not streaming).
- **Caveat `[VERIFIED]` (checked 2026-11-10 against the installed SDK; none of `groq.APITimeoutError`, `RateLimitError`,
  `APIStatusError`, `NotFoundError` subclass the httpx types):** retry/error classification is keyed on `httpx.TimeoutException` /
  `httpx.HTTPStatusError`. `langchain-groq` goes through the **Groq SDK**, which raises its own exception types
  (`groq.APITimeoutError`, `groq.RateLimitError`, …) and has its own internal retry. If those aren't subclasses
  of the httpx types, my tenacity wrapper **never fires** and `_classify_groq_error` returns `"unknown"` for
  everything. Verify with `issubclass(groq.RateLimitError, httpx.HTTPStatusError)`; if false, retry on
  `groq.APIConnectionError, groq.RateLimitError, groq.InternalServerError` (and rely on `Retry-After`). Also
  `groq_timeout=30` exists in config but is **never passed** to `ChatGroq`. Also `groq_retries` counter
  increments on every *attempt* (including the first), so it overstates retries.
- **Fail-soft:** LLM failure → friendly message, not a 500. Redis down → cache miss. Empty KB → 503 with hint.
- **SlowAPI:** in-memory, per-process, keyed by `get_remote_address`. Behind a reverse proxy this may see the
  proxy's IP unless forwarded headers are trusted → everyone could share one bucket `[UNVERIFIED on Replit]`.
  Multi-worker/multi-instance needs a Redis storage backend. Two `Limiter` instances exist (main + routes) — a smell.
- **Ingest protection:** `X-Ingest-Key` header vs `INGEST_KEY` (default `securerag-ingest-2024` — a **default
  secret committed in README/CLAUDE.md**, must be overridden in prod), 5/hour rate limit, 409 if already running.

### 5.15 Concurrency model (volunteer: it's imperfect)
- Routes are `async def` but call **synchronous, CPU-bound** code (`answer_query`, embedding, reranker,
  `llm.invoke`, and `stream_answer_query`'s retrieval) directly → **they block the event loop** while running,
  serialising concurrent users. The correct pattern: plain `def` routes (FastAPI runs them in a threadpool) or
  `await run_in_threadpool(...)` / `asyncio.to_thread(...)` for retrieval+rerank, and use `ainvoke`.
- Inside retrieval I do use a 2-thread `ThreadPoolExecutor` (created per request) to overlap BM25 and Chroma.
- Single uvicorn worker by design (Dockerfile comment): models are loaded per process (~memory × workers), BM25
  index & sessions & rate-limit state are per-process. Horizontal scale = more containers behind an LB, which then
  requires externalising state.
- Ingest runs as a FastAPI `BackgroundTask` (manual) or APScheduler job (`run_in_executor`) — sync, long-running,
  and holds the GIL in places; it does not use a durable job queue (Celery/RQ/Arq) so a restart kills it.

### 5.16 Weekly refresh (APScheduler)
- Cron `0 2 * * 0` UTC; `misfire_grace_time=3600`; runs `run_ingest(force=True)` in an executor, then
  `rebuild_searcher()` and `cache.invalidate()`. Can be disabled via `CVE_REFRESH_ENABLED=false`.
- **Why weekly:** NVD keeps updating; weekly balances freshness vs the 15+ min / rate-limited fetch.
- **Problems (volunteer):** see §7.9 — non-atomic rebuild means KB is empty mid-run; on autoscale each instance
  would run its own copy against ephemeral disk. The refresh also doesn't use the incremental NVD
  `lastModStartDate` API, so it re-downloads everything.

### 5.17 Observability
- **Metrics (`core/metrics.py`):** HTTP count/latency/in-flight; chat outcome; per-stage latency histogram
  (`retrieval|rerank|llm|total`); retrieval counts per stage; reranker score histogram; groq errors/retries; KB
  doc counts by source; ingest runs/duration; semantic cache size.
- **Why these:** latency breakdown per RAG stage tells you where time goes; reranker-score histogram detects
  retrieval drift/quality drops; cache hit/miss + size sizes the cache.
- **Middleware:** per-request 8-char UUID in contextvar → every JSON log line carries `request_id` and the
  response has `X-Request-ID`. Paths are normalised (`/api/session/:id`) to bound Prometheus label cardinality
  (high-cardinality labels blow up Prometheus memory).
- **Gaps:** `llm_tokens` metric is defined but never incremented; `chat_requests{status="rate_limited"}` never
  set; stream path under-instrumented; semantic-cache gauge help text still says "in-memory". No tracing
  (OpenTelemetry), no quality metrics (e.g., online faithfulness sampling), no alerts defined.
- **Stack:** Prometheus scrapes `app:8000/metrics` every 15s (retention 15d); Grafana provisioned from JSON;
  default Grafana creds in compose (`admin/securerag123`) — fine locally, never in prod.

### 5.18 Config
`core/config.py` — `pydantic-settings` `Settings` singleton, `.env` driven, case-insensitive. Tunables:
model, temperature, max tokens, `retrieval_top_k=10`, `rerank_top_k=5`, `max_sessions=20`, `max_history_turns=6`,
rate limits, cron, Redis URL, ingest key, CORS origins. `main.py` calls `load_dotenv(override=True)` because
Replit pre-set an *empty* `GROQ_API_KEY` env var that otherwise beat the `.env` value (a real bug I hit).
Inconsistency: `vectorstore.py` and `ingest.py` still use `os.getenv` directly (not migrated to `settings`).

### 5.19 Frontend (vanilla JS)
- **Why no framework:** single page, small state, zero build step, easy to serve as static files from FastAPI,
  nothing to break at deploy. Trade-off: manual DOM/state management; won't scale to a bigger UI.
- **Features:** chat with streaming + markdown (`marked`), source chips + modal, severity checkboxes / source
  radios, health + KB stats sidebar, ingest button with progress polling, evaluation tab with a Chart.js bar chart (plus a markdown report download)
  and per-query table, "suggested follow-ups" (keyword heuristics, not LLM), mobile sidebar, session reset.
- **Security note (volunteer — it's a security product):** LLM output is rendered with
  `bubbleEl.innerHTML = marked.parse(...)` with **no sanitiser** (no DOMPurify). Because answers derive from
  third-party NVD text, that's an XSS/indirect-prompt-injection → HTML vector. Fix: DOMPurify on the parsed HTML,
  CSP header. Chart.js/marked loaded from CDN without SRI / pinned versions.
- CORS is `*` by default.

### 5.20 Docker / deployment
- **Dockerfile:** `python:3.11-slim`, installs build-essential + curl, pip install, copies code, **pre-downloads
  both models at build** so startup is fast, single uvicorn worker.
- **Compose:** app + Redis (AOF persistence) + Prometheus + Grafana; Chroma on a named volume; `./data` mounted so
  `cve_cache.json` survives; app healthcheck on `/api/health` with 60s start period (model download).
- **Replit Autoscale:** see §3. Gotcha stories: `load_dotenv(override=True)`; deployment 404 "no previewable
  artifacts" (the artifact must be registered and serve `/`); empty KB on deploy (fixed by ingest-at-build).

---

## 6. Decision log

| Decision | Alternatives considered | Why chosen | Cost / what I'd revisit |
|---|---|---|---|
| RAG over fine-tuning | FT, long-context | Fresh data, citations, cheap | Retrieval errors are now my failure mode |
| Hybrid BM25 + dense | dense-only, BM25-only | IDs/jargon (BM25) + paraphrase (dense) | No ablation proving the gain; tokenizer weak |
| RRF (k=60) | weighted sum, learned fusion | Scale-free, no tuning | Unweighted; k untuned |
| Cross-encoder rerank, 10→5 | none; LLM rerank; bigger CE | Precision for little latency | Domain shift; no score threshold |
| ChromaDB embedded | pgvector, Qdrant, FAISS, Pinecone | Free, zero-ops, persistent | Not shareable across instances; overkill index |
| BGE-small | bge-base/large, e5, OpenAI | Local, fast, good MTEB | Not benchmarked on my corpus |
| Groq Llama-3.1-8B | 70B, Claude/GPT, Ollama | Free + very fast streaming | Weaker reasoning/faithfulness; vendor/rate-limit dependency |
| Post-rerank filtering | Pre-filter in retrieval | Simple, keeps ranking pure | Can empty the set → silently unfiltered fallback |
| One CVE = one doc | chunk descriptions | Short docs; chunking fragments | Long CVEs truncated at embed limit (rare) |
| OWASP 3 chunks/category | 1 chunk | Matches question types | More chunks to rank |
| In-memory sessions | Redis/DB | Simplicity | Lost on restart; not shared across instances |
| SSE over POST+fetch | EventSource, WebSocket | Needs JSON body; one-way stream | Manual frame parsing |
| Redis semantic cache | exact cache, none | Latency/cost on repeated questions | Untuned threshold; ignores filters; unused by UI |
| Custom lexical eval | RAGAS, LLM-judge | RAGAS/LangChain 0.3 conflict | Weak validity |
| Ingest at build (prod) | ingest at runtime | Autoscale instances have no shared disk | Stale data; slow build |
| Vanilla JS | React/Vue | No build, tiny | Manual state; XSS hygiene on me |
| Single uvicorn worker | many workers | Models/state per process | Sync code blocks loop → low concurrency |
| Pydantic-settings | scattered `os.getenv` | One typed config | Migration incomplete in 2 files |

---

## 7. Known weaknesses (ranked by "likelihood an interviewer finds it")

Be ready to state each, its impact, and the fix. **These are real, found in the code.**

1. **Evaluation is now a start, not a finish.** Retrieval ablation + ID-level citation check exist (§5.12b), but queries are
   synthetic, one run, no tuning, no claim-level faithfulness, no human labels, generation measured on a different model than
   the app default. The legacy `rag/evaluate.py` remains lexical, n=10, OWASP-only. *Fix:* §9.
2. **Post-filter fallback violates user filters** (§5.8).
3. **Semantic cache ignores filters; UI doesn't hit it; O(N) Redis lookup; untuned 0.92** (§5.13).
4. **Event loop blocking** from sync CPU-bound work inside `async def` routes (§5.15) → poor concurrency under load.
5. **Multi-turn retrieval has no query rewriting** (§5.10).
6. **No relevance threshold / abstention gate** on reranker output (§5.7).
7. **Prompt injection & unsanitised markdown → HTML** (§5.9, §5.19).
8. **Citations: IDs are now checked (flag-only), claims are not.** A fake ID is flagged via `verification`; a real ID with
   a wrong severity/version/fix is not. Flagging doesn't block or correct the answer, and the check only covers IDs.
9. **Ingest is destructive and non-atomic:** `delete_collection()` then re-add means the KB is empty for the
   duration (chat → 503) and a mid-run failure leaves a partial/empty index. Failure inside `_fetch_window` is
   swallowed (`break`) so a partial NVD fetch is cached as if complete. *Fix:* build into a new collection and
   atomically swap (blue/green collection name), upsert incrementally using NVD `lastModStartDate`, validate
   counts before swapping, make fetch failures fatal.
10. **Retry wrapper never fires for Groq SDK exceptions; `groq_timeout` unused** (§5.14) `[VERIFIED]`. All errors are
    classified `"unknown"`, so `groq_errors{error_type}` is uninformative.
11. **Tests:** only one test file; the four `SemanticCache` tests construct
    `SemanticCache(threshold=..., max_entries=...)` and call `.set/.get` without a Redis client — they target an
    earlier in-memory implementation and **will fail** against the current Redis-backed class (`__init__` takes
    `redis_client, threshold, ttl`). `[VERIFIED]` by running pytest on 2026-11-10: 4 cache tests fail, and `test_rrf_fusion_prefers_consistent_results` also fails
    (it indexes `"doc_b"` but the metadata ids are `"b"` — a test bug, not an RRF bug). The existing file has 14 tests (9 pass, 5 fail); the 20 new citation and
    eval-metric tests all pass (29 pass / 5 fail overall). There are no tests for
    chain, ingest parsing, API routes, or the reranker. *Fix:* fakeredis for the cache; mock NVD payloads; API
    tests with `TestClient`; a golden retrieval test set.
12. **Corpus is a sample** (2000 of tens of thousands/yr), time-skewed windows, no CISA-KEV/EPSS/exploit data.
13. **State is per-process** (sessions, rate limits, BM25, eval cache, stats) — breaks under autoscale.
14. **Security defaults:** `INGEST_KEY` default is public; CORS `*`; `/api/evaluate` is unauthenticated, un-rate-limited,
    and triggers 10 LLM calls (cached 1h) — a cheap abuse/cost vector. `DELETE /session/{id}` is unauthenticated.
15. **Observability gaps:** unused `llm_tokens` metric, retries overcounted, stream path under-instrumented, no tracing/alerts.
16. **BM25 tokenizer** naive (§5.5); RRF dedupe key fragile (§5.6).
17. **`queries_served`/avg latency** only count `/api/chat`, so the UI (streaming) leaves them at ~0.
18. **Default model is not guaranteed to exist.** On 2026-11-10 Groq returned `404 model_not_found` for `llama-3.1-8b-instant` on
    my key (the key listed `gpt-oss-20b/120b`, `qwen3.8-27b` and others). The app's `chat` would return its generic "couldn't reach
    the language model" message, and with no startup check or model fallback nothing alerts. Pin/validate the model at startup and
    make it a config you can swap. The generation numbers in §5.12b used `openai/gpt-oss-20b`; re-run with the intended model.
19. **`/api/chat` first-turn crash (fixed in this branch):** `METRICS.cache_hits/cache_misses` were referenced but never defined.
    The UI uses the streaming endpoint, so it went unnoticed.

---

## 8. Grilling question bank (with model answers)

> Answers are first person. Adapt length to the interviewer. `[NOT MEASURED]` means: say so.

### A. Framing & motivation
**Q: Walk me through the project.** → Use the 2-minute pitch (§1).

**Q: Why security? Why RAG here specifically?**
Security is a domain where hallucination is costly (fake CVE IDs, wrong CVSS) and knowledge is volatile (new CVEs
daily). RAG gives fresh, citable answers; fine-tuning bakes in stale facts and can't cite.

**Q: Who's the user and what does "good" look like?**
An engineer/analyst asking "what is X / how do I fix Y / is CVE-Z severe?". Good = correct, cited, abstains
when the KB lacks it. I optimised the pipeline for citation-grounded answers but haven't validated against real users `[NOT MEASURED]`.

**Q: How much of this did you write yourself?** `[AUTHOR? — see §11]` Answer truthfully: describe which parts you
designed/debugged (e.g., NVD 120-day issue, `override=True`, ingest-at-build) and which were scaffolded/assisted,
and show you can explain any line.

### B. Retrieval theory
**Q: Why hybrid? Prove it helps.**
Complementary failure modes: BM25 is exact-term/rare-token strong (CVE IDs, product names) but blind to
paraphrase; dense captures semantics but blurs identifiers. Literature (BEIR, hybrid papers) supports gains, but
*I haven't ablated on this corpus* — I'd run BM25 / dense / hybrid / hybrid+rerank on a labelled set and compare
recall@10 / nDCG@10 / MRR (§9).

**Q: Why RRF and not a weighted sum of normalised scores?**
Incomparable score scales; RRF is rank-based, parameter-light, robust to outliers. A tuned convex combination can
beat RRF when you have labelled data — worth trying once I have an eval set.

**Q: What does k=60 do?** Smooths rank contribution: `1/(60+r)` makes the difference between rank 1 and 2 small
relative to appearing in both lists. Smaller k → top ranks dominate; larger k → flatter. Default from the original
paper; untuned here.

**Q: Why retrieve 20+20 and keep 10, then rerank to 5?**
Recall-first first stage (wide net), precision second stage. 10 candidates bounds cross-encoder cost and still
gives the reranker room to promote a doc from RRF rank 8. Values are heuristics `[NOT MEASURED]`; I'd sweep them (hybrid's Recall@10 of
0.953 vs BM25's 0.932 suggests the candidate pool is adequate, but I haven't swept pool size).

**Q: Bi-encoder vs cross-encoder?**
Bi-encoder embeds query and doc independently (precomputable doc vectors → ANN search, O(1) per doc at query
time) but compresses meaning into one vector. Cross-encoder concatenates query+doc and attends across both —
much better relevance judgement but needs a forward pass per pair, so only for short lists.

**Q: What if the relevant document isn't in your top 10?**
Then nothing downstream can fix it — reranking only reorders. That's why first-stage recall matters; I'd measure
recall@20/@10 of the first stage and tune candidate counts, add query expansion/HyDE, or multi-query retrieval.

**Q: How would exact CVE-ID lookup work?** BM25 matches the `cve-2024-1234` token (the ID is in the doc text).
Dense retrieval alone would be poor. My tokenizer is whitespace-only so trailing punctuation breaks it; I'd add
a regex ID extractor → direct metadata lookup, bypassing ranking.

**Q: Why is BM25 index built in memory? What breaks at 10M docs?**
Memory (whole corpus + index), O(N) scoring per query with `rank_bm25`, full rebuild on every ingest. Move to
Lucene-family (Elasticsearch/OpenSearch/Tantivy) or a vector DB with native sparse/hybrid, incremental indexing.

**Q: ChromaDB uses HNSW — explain it and the trade-offs.**
Layered proximity graph; greedy descent gives ~log N approximate nearest-neighbour search, tunable
`ef_construction`/`M`/`ef_search` trade recall for speed/memory. At 2k docs exact search would be as fast —
HNSW is more than I need.

**Q: Cosine vs dot vs L2?** BGE vectors are normalised so cosine == dot; Chroma configured for cosine and I
convert `distance → 1 - distance`. Changing metric requires recreating the collection.

### C. Embeddings
**Q: Why bge-small? How would you choose objectively?**
Benchmarks + local/CPU + size. Objectively: build a labelled query→doc set from my corpus, compare candidate
models on recall@k/nDCG, plot quality vs latency vs memory, pick the knee. I didn't do that `[AUTHOR? confirm]`.

**Q: Why the query prefix?** BGE is trained asymmetrically: short queries get an instruction prefix, passages don't.
Omitting it lowers retrieval quality. It's in `embed_query` only.

**Q: Would fine-tuning the embedder help?** Possibly: synthetic (question, CVE) pairs generated by an LLM, hard
negatives from BM25, contrastive loss (MultipleNegativesRanking). Needs the eval set first to prove gain.

**Q: Embedding limits for CVE text?** 512 tokens; most CVE descriptions fit. Long ones truncate silently.

### D. LLM & prompting
**Q: How do you reduce hallucination?** (1) Grounding prompt: "ONLY context", explicit fallback phrase, "never invent
IDs/scores"; (2) low temperature 0.1; (3) small, reranked context (5 docs) to reduce distractors; (4) cite IDs.
Not done: citation verification, NLI faithfulness check, abstention threshold. No measured hallucination rate.

**Q: Why an 8B model? Why not 70B/GPT/Claude?** Free + fast on Groq; task is extractive. Trade-off: weaker on
multi-hop/synthesis and citation discipline. I'd evaluate 70B on the same eval set and pick by quality/latency/cost.

**Q: Temperature 0.1 — why not 0?** Near-deterministic for factual QA; exactly 0 can cause repetitive degenerate
outputs on some models; practical difference is small.

**Q: How do you handle context-window limits?** 5 short docs + ≤6 history turns is far below Llama 3.1's context;
history is truncated by turn count not tokens (I'd move to token-budgeted truncation/summarisation).

**Q: What stops prompt injection from an NVD description?** Currently nothing robust: instructions and retrieved text
are in one user message. Mitigations: separate system/user roles, fence context with delimiters and tell the model
it's untrusted data, strip/neutralise instruction-like text, verify output citations, never let the model take
actions (it's read-only, which limits blast radius).

**Q: How do you make citations trustworthy?** Today: prompt-requested + sources list returned from retrieval
metadata (the UI's source chips come from *retrieval*, not from model output — so chips are always real, but the
prose might cite something else). Add: regex all `CVE-\d{4}-\d+` in the answer and assert ⊆ context IDs; else retry/flag.

### E. Evaluation (expect heavy fire here)
**Q: How do you know it works?** I built a labelled retrieval benchmark (192 queries, gold from construction) and ran an
ablation: on this set BM25 R@5 0.911, dense 0.589, hybrid 0.906, hybrid+rerank 0.922, MRR@10 0.812 / 0.517 / 0.832 / 0.873, with
bootstrap CIs `[MEASURED]`. Then be candid: the queries are synthetic and favour lexical match, it's one run, and I haven't
measured claim-level faithfulness or used human labels. Generation: 0/50 answers cited an ID outside the context (small n).

**Q: Explain your metrics. Are they valid?** Separate the two harnesses. `evals/` metrics (Recall@k, MRR, nDCG, bootstrap CIs)
are standard and valid for what they measure, subject to the synthetic-query caveat. For the legacy `rag/evaluate.py`, describe §5.12 and concede they're surface-overlap proxies; faithfulness
can't detect a wrong-number paraphrase; Jaccard relevancy is weak; n=10; no CIs; OWASP-only; not validated against
human labels.

**Q: Why not RAGAS?** Version conflict with LangChain 0.3 at the time. In hindsight I could have isolated it in a
separate venv/service or used a custom LLM-judge — the conflict was a tooling issue, not a reason to degrade
metric quality.

**Q: How would you build a retrieval test set cheaply?** For each sampled CVE doc, have an LLM generate 2–3 questions
that doc answers (synthetic queries), keep the doc ID as the gold label; add hand-written hard cases (ID lookups,
paraphrases, multi-doc). Compute recall@k, MRR, nDCG for BM25 / dense / RRF / +rerank. Beware synthetic-query leakage
(lexical overlap inflates BM25) — mix in paraphrased/human queries.

**Q: How would you measure faithfulness properly?** Decompose answer into atomic claims, verify each against the
context with an NLI model or LLM judge (RAGAS-style); plus citation precision/recall; calibrate the judge against
~100 human labels; report CIs.

**Q: How would you detect regressions in prod?** Log retrieval + rerank scores (already a histogram), sample
queries for offline judging, track "I don't know" rate, thumbs up/down, golden-set CI check on each model/prompt change.

**Q: Statistical rigour?** Run multiple seeds/temperature samples, bootstrap CIs, paired tests when comparing
variants on the same questions; don't claim wins inside the noise.

### F. Systems & scaling
**Q: How does it scale to 1000 concurrent users?** Today: it doesn't — sync CPU work in async routes blocks the loop,
single worker, per-process state. Path: (1) offload retrieval/rerank to threadpool/`to_thread`, use `ainvoke`;
(2) run multiple replicas behind an LB; (3) externalise state (Redis for sessions/rate limits, shared vector DB,
OpenSearch for BM25); (4) batch/GPU-serve the embedder & reranker (or a model server); (5) cache; (6) queue ingest jobs.

**Q: Where's the latency going?** Retrieval stages measured on CPU `[MEASURED]`: BM25 p50 3 ms, dense 18 ms, hybrid 21 ms,
hybrid+rerank 352 ms (p95 712 ms) — so the cross-encoder is ~17x the rest of retrieval. End-to-end with the LLM is
`[NOT MEASURED]` (the streamed path measured ~1.5–2.2 s in a smoke test with `gpt-oss-20b`, one request each, not a benchmark).
Per-stage Prometheus histograms exist for the rest.

**Q: What's your bottleneck and how do you prove it?** Use the stage histogram + profile. Don't guess.

**Q: Why not LangChain chains/agents?** The flow is a fixed pipeline; explicit Python is simpler to debug,
instrument per stage, and stream. LangChain is only a thin client here.

**Q: Why FastAPI?** Async, Pydantic validation, auto OpenAPI/Swagger, StreamingResponse for SSE, dependency
injection. (I should be candid that my handlers don't fully exploit async — §5.15.)

**Q: Why SSE not WebSockets?** One-directional token stream, plain HTTP, auto-reconnect semantics, simpler infra.
WebSockets only if I needed bidirectional/interrupt messaging.

**Q: How would you deploy this properly?** Containerise (done), external vector DB + OpenSearch (or pgvector +
`tsvector`), Redis for sessions/limits/cache, model server for embeddings/rerank, CI/CD with golden-set gate,
secrets manager, autoscaling on queue depth/latency, blue/green index swap on refresh.

**Q: Why did autoscale break your first deploy?** Instances don't share disk; ingest wrote to one instance's local
ChromaDB. Fix: ingest in build → immutable image. Real fix: shared vector store.

**Q: Data freshness strategy?** Weekly full re-ingest (scheduler) + manual endpoint. Better: incremental by NVD
`lastModStartDate`, upserts, delete rejected CVEs, event-driven (NVD feeds), swap indexes atomically.

**Q: What happens during ingest?** Collection is deleted and rebuilt → KB empty until done; `/chat` returns 503;
BM25 index is stale until `rebuild_searcher()`. Known flaw (§7.9).

### G. Reliability & security
**Q: What if Groq is down/rate-limited?** Retry (3×, exp backoff) is configured, but I verified the exception match is
wrong for the Groq SDK, so it never fires `[VERIFIED]` — I hit real `RateLimitError`s and a 404 `model_not_found` while running the
evals and my own eval code needed its own backoff. Fallback is a friendly error message. Better: honour `Retry-After`, circuit
breaker, fallback model/provider, queue.

**Q: What if Redis is down?** Every op catches `RedisError` → treated as miss; app continues, just uncached.

**Q: Threat model?** Assets: API key, ingest endpoint, cost, integrity of KB, users' browsers. Risks: prompt/indirect
injection via NVD text, XSS via unsanitised markdown, abuse of unauthenticated `/evaluate` (cost), default ingest
key, open CORS, rate-limit bypass behind proxy, supply-chain via CDN scripts w/o SRI. Mitigations listed in §5.9/§5.19/§7.

**Q: Is it OK that the ingest key defaults to a public string?** No — it's a demo default; prod must set `INGEST_KEY`.
Better: real auth (JWT/OIDC), disable the endpoint in prod entirely and use the scheduler/CI.

**Q: PII/data privacy?** No user accounts or PII stored; queries go to Groq (third party). Sessions are anonymous UUIDs
in memory. NVD/OWASP data is public. For a sensitive deployment I'd self-host the LLM.

### H. Data & domain
**Q: Why only 2000 CVEs?** NVD rate limits + ingest time + embedding cost on CPU for a demo; it's a sample, so the bot
can't claim exhaustiveness. Scale-up is mostly an ingest-time/infra problem, plus moving BM25/vector storage off
in-process memory.

**Q: How do you handle conflicting/duplicate/updated CVEs?** One doc per `cve_id`; full rebuild each refresh picks up
updates. No explicit dedupe/versioning/rejected-status handling.

**Q: CVSS handling?** Prefer v3.1 → v3.0 → v2. v2 severity inferred by score bands. Missing → `UNKNOWN`. I don't use
EPSS/KEV, so "severity" ≠ "exploitability" — a real limitation for security prioritisation.

### I. Frontend
**Q: Why vanilla JS?** Single page, no build pipeline, served as static files by FastAPI; fewer moving parts.
**Q: How does streaming render work?** `fetch` POST → `response.body.getReader()` → decode chunks → split on `\n\n` →
parse `data:` JSON events → on `metadata` render sources, on `token` append + re-render markdown, on `done` show time.
**Q: Any XSS concern?** Yes — `innerHTML` with `marked` output unsanitised (§5.19). I'd add DOMPurify + CSP + SRI.

### J. Curveballs
**Q: Replace the cross-encoder with the LLM as a reranker?** Listwise LLM rerank can be more accurate and
domain-aware, but adds an extra LLM call (latency/cost), prompt-length limits, and ordering bias. Evaluate
cost/quality vs a stronger cross-encoder first (`bge-reranker`).

**Q: Add HyDE / query expansion / multi-query?** HyDE (generate a hypothetical answer, embed that) can help vague
queries; multi-query + RRF improves recall. Cost: extra LLM call. Test on the eval set before adopting.

**Q: GraphRAG for CVE→CWE→CAPEC→product relationships?** Good fit for relational/multi-hop questions ("what weaknesses
underlie these CVEs?"). I'd start with metadata joins (CWE IDs already partially present in OWASP chunks) before a
full graph.

**Q: How would you do agentic/tool-using version?** Tools: `lookup_cve(id)` (direct NVD), `search_kb`, `get_epss`, `get_kev`.
LLM picks tools; keeps KB for concept questions. Adds latency and non-determinism; needs tracing + evals.

**Q: Compress/optimise the embedder?** ONNX/quantisation (int8), batching, GPU, or a model server — measure quality drop.

**Q: Why not just ask GPT/Claude with browsing?** Reproducibility, controllable sources, cost/latency, private/offline
use, citation guarantees from a curated KB; browsing is slower and less controlled. Different trade-off, not strictly worse.

**Q: Biggest lesson?** (Pick truthfully.) Candidates: undocumented API limits (NVD 120-day), env-var precedence
(`override=True`), deployment-model mismatch (stateful disk on autoscale), and that without an eval set you can't
justify any tuning decision.

**Q: If you had one week, what would you do?** 1) Build the labelled retrieval/generation eval + ablations; 2)
fix pre-filtering + abstention threshold; 3) async/threadpool + shared state; 4) atomic incremental ingest;
5) security hardening (sanitiser, auth, CORS, injection mitigations); 6) fix/extend tests; 7) cache key includes filters.

---

## 9. Improvement roadmap & how I'd evaluate each

| # | Change | Expected effect | How to measure |
|---|---|---|---|
| 1 | ~~Labelled eval set, ablation~~ **done** (§5.12b); next: add human-written/real queries, sweep RRF k & pool size, hard-negative near-duplicate CVEs | Less biased baseline | same metrics on a non-synthetic set |
| 2 | Pre-filtering (Chroma `where`, masked BM25) | Filters honoured | Filter-violation rate → 0; recall within filter |
| 3 | Reranker score threshold + explicit "no answer" path | Fewer confident wrong answers | Abstention precision/recall on unanswerable questions |
| 4 | Query condensation for multi-turn | Better follow-ups | Retrieval recall on multi-turn dialog set |
| 5 | ID-level citation check **done** (flag-only); next: claim-level faithfulness (NLI/LLM judge) | Catch wrong claims, not just fake IDs | Judge-vs-human agreement; claim-level unsupported rate |
| 6 | Tokenizer + CVE-ID fast path | Exact-ID queries always hit | ID-lookup accuracy |
| 7 | Async correctness (threadpool, `ainvoke`) | Throughput under concurrency | Load test (locust/k6): p50/p95, RPS |
| 8 | Atomic, incremental ingest via `lastModStartDate` | No downtime, faster refresh | Refresh duration, 503 count during refresh |
| 9 | Shared state (Redis sessions/limits; pgvector or Qdrant; OpenSearch) | Works on autoscale | Multi-instance integration test |
| 10 | Larger/hybrid models (bge-base, bge-reranker, Llama-70B) | Quality ↑ | Eval deltas vs latency/cost |
| 11 | Enrich corpus (CISA KEV, EPSS, CWE, CAPEC) | Prioritisation answers | Task-specific eval |
| 12 | Security hardening (DOMPurify, CSP, SRI, auth, CORS, injection tests) | Reduce attack surface | Red-team prompts; ZAP scan |
| 13 | Fix tests (fakeredis), add API/ingest tests, CI | Safety net | Coverage + CI gate on golden set |
| 14 | Cache key = (embedding, filters); tune threshold | Correct hits | Precision of cache hits on paraphrase pairs |

---

## 10. Quick reference

**Glossary:** *RAG* retrieve-then-generate. *BM25* TF-IDF-like ranking with term saturation (k1) and length
normalisation (b). *Dense retrieval* nearest-neighbour search on learned embeddings. *RRF* rank-based list fusion.
*Cross-encoder* joint query-doc scorer. *HNSW* graph-based ANN index. *SSE* server→client event stream over HTTP.
*CVE* vulnerability identifier. *CVSS* 0–10 severity score (None/Low/Medium/High/Critical). *NVD* NIST's CVE database.
*OWASP Top 10* top web-app risk categories (2021). *EPSS* exploit-prediction score. *CISA KEV* known-exploited list.

**Metrics for retrieval:** recall@k (gold in top-k), MRR (1/rank of first gold), nDCG (graded, position-discounted).

**Pipeline numbers to remember:** 20+20 → RRF(k=60) → 10 → rerank → 5 → LLM. temp 0.1, max_tokens 1024.
Cache cosine ≥0.92, TTL 1h. 20 req/min chat. 20 sessions × 6 turns. 119-day NVD windows, 500 CVE/window, 2000 cap.

**Commands:**
```bash
cd securerag
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
python data/ingest.py [--force]
pytest tests/ -v            # note: semantic-cache tests are stale
docker-compose up -d        # app:8000 grafana:3000 prometheus:9090 redis:6379
curl localhost:8000/metrics
curl -X POST localhost:8000/api/ingest -H "X-Ingest-Key: $INGEST_KEY"
```

---

## 11. Open items — only the author can answer (fill in, then delete the `[AUTHOR?]` tags)

1. **Authorship/ownership:** The git history shows "Published your App" commits and Replit agent metadata. How much
   did you design vs. have generated/assisted, and which parts can you walk through line by line? (Interviewers
   will probe; the guidance in §8-A depends on your honest answer.)
2. **Why Groq / Llama-3.1-8B / bge-small / Chroma** — were these reasoned choices or defaults? What alternatives did
   you actually try?
3. **Any numbers you actually observed:** typical end-to-end latency, time-to-first-token, build/ingest duration,
   eval-endpoint scores, final doc count in production, any user/traffic numbers. (Retrieval-stage latency and
   retrieval quality are now `[MEASURED]`, §5.12b; production traffic and end-to-end latency are still unknown.)
4. **Is Redis present in the live deployment?** (Affects whether the semantic cache is live.)
5. ~~Did you run any retrieval comparison~~ Now answered by `evals/` (§5.12b). Still useful: 2–3 real queries where hybrid
   or rerank visibly fixed (or broke) an answer, as concrete stories. Also: which model were you actually serving in production
   (`llama-3.1-8b-instant` 404'd on 2026-11-10)?
6. **Which bugs were really yours** beyond those in the guide (NVD 120-day, RAGAS conflict, dotenv override, autoscale
   empty KB, artifact 404)? Add 1–2 extra war stories with the diagnosis steps.
7. ~~Target role~~ **Answered: Amazon Applied Scientist intern** → emphasise §5.12b/§9 (eval rigour), §13. Still `[AUTHOR?]`: your
   real team/timeline/role in the project for the behavioural stories (§13.8), and honest authorship split (§13.2).
8. **Was Redis/Prometheus/Grafana added to show production-readiness** or driven by a real need? Be ready to say which.

---

## 12. Corrections to claims in other docs (do not repeat these)

| Where | Claim | Reality |
|---|---|---|
| README, INTERVIEW_GUIDE | BM25 and vector search each return top **10** | Each returns top **20**; RRF keeps 10 |
| README | "Way better recall than either one alone" | Measured: hybrid beats dense alone by a lot, but is roughly **equal to BM25** on Recall@5 (0.906 vs 0.911); it helps Recall@10 and ranking when reranked (§5.12b) |
| README (earlier) | Eval is RAGAS-style and sufficient | Legacy lexical proxy; real numbers now come from `evals/` |
| Config / docs | `llama-3.1-8b-instant` is the model | Returned 404 `model_not_found` on 2026-11-10 for the author's key; re-check before claiming it |
| README | "Streaming responses" as a future improvement (INTERVIEW_GUIDE) | Already implemented (`/api/chat/stream`) |
| INTERVIEW_GUIDE | Weekly re-ingest listed as future work | Implemented (`core/scheduler.py`), though ineffective on autoscale |
| INTERVIEW_GUIDE | Answer relevancy via cosine similarity | Token-overlap Jaccard + keyword coverage |
| INTERVIEW_GUIDE | "~2–4 s end to end" | Not measured |
| INTERVIEW_GUIDE | Calls `run_rag()` | Function is `answer_query()` / `stream_answer_query()` |
| CLAUDE.md | Semantic cache is in-memory, 500-entry cap | Redis-backed, no entry cap, TTL 1h |
| CLAUDE.md | Test suite includes semantic cache coverage | Cache tests target the old in-memory class and are stale |
| CLAUDE.md | Metrics gauge "in-memory semantic cache" | Redis-backed now (help text stale) |
| Code comment | `ingest.py` "3 windows" | 365/119 → **4** windows (119,119,119,8) |
| Anywhere | The UI uses the semantic cache | UI uses `/chat/stream`, which skips it |

---

## 13. Amazon Applied Scientist (intern) — targeted prep

> Role context (from the author): **Amazon Applied Scientist intern interview; the interviewers will grill me on this
> project because it is on my resume.** Use this section *in addition to* §1–§12. Priority order for an AS intern loop:
> (1) can I explain every ML/IR choice and its math, (2) is my experimental rigour honest, (3) do I own the work and its flaws,
> (4) behavioural stories mapped to Leadership Principles, (5) can I code the small pieces on a whiteboard.
> Interviewers at this level care more about *how you reason and what you'd do next* than about a big number.

### 13.1 Resume bullets I can defend (and ones I must not write)
Defensible (every number is `[MEASURED]`, §5.12b; keep the caveats ready):
- "Built SecureRAG, a retrieval-augmented cybersecurity assistant over 2,000 NVD CVEs + OWASP Top 10: hybrid retrieval
  (BM25 + BGE dense, Reciprocal Rank Fusion) → cross-encoder rerank → Llama 3.1 8B (Groq), streamed via FastAPI SSE."
- "Designed an offline retrieval benchmark (192 labelled queries, 4-way ablation, bootstrap CIs): hybrid+rerank reached
  Recall@5 0.92 / MRR@10 0.87 vs 0.59 / 0.52 for dense-only; reranking added +0.04 MRR (paired bootstrap, 95% CI excludes 0)."
- "Added a citation-verification guard that flags CVE/OWASP IDs not supported by retrieved context; 0 of 50 answers cited an
  unsupported ID in a 50-query check."

**Do not claim** (no evidence exists): "reduced hallucinations by X%" (no before/after baseline), "production-scale/scalable"
(single process, per-process state), "N users/traffic" `[AUTHOR?]`, "hybrid outperforms BM25" (it doesn't clearly, §5.12b),
"improved accuracy" without saying *recall@k on a synthetic benchmark*, or that the guard "prevents" hallucination (it *detects* fake IDs only).

### 13.2 Authorship honesty protocol (read this before the interview)
- Amazon interviewers will pick a random component and say "walk me through this line." **Only claim what you can explain without the
  file open.** The `evals/` harness, `rag/citations.py` and the §5.12b numbers were added late; make sure you can narrate them (13.3).
- If asked whether you used AI coding tools: say yes, plainly, then show you understand and verified the work — e.g. you cross-checked
  that a suspicious 25% "gold cited" figure was a regex bug (non-breaking hyphens), not model behaviour. Honesty + verification is the
  strong answer; bluffing is the fatal one. `[AUTHOR? — decide what you actually wrote vs generated and be consistent]`
- "I don't know, but here's how I'd find out" is acceptable; an invented number is not.

### 13.3 30-second code walkthrough of the eval harness (know this cold)
- `evals/build_dataset.py`: sample 60 CVE docs **stratified by severity** (round-robin, seed 42) → one LLM call per doc writes a
  `natural` and a `keyword` query (told not to copy the ID or any 6-word run) → a **leak filter** drops violations (6 dropped) → add a
  templated `id` query ("Tell me about CVE-…") → add 20 hand-written OWASP questions. Gold = the source doc's `cve_id` / `owasp_id`.
  *Why construction-based gold:* free, scalable labels; *cost:* the queries inherit the doc's vocabulary (favours BM25).
- `evals/retrieval_eval.py`: four configs return a ranked list of **unique IDs** (OWASP's 3 chunks per category are de-duplicated).
  `hybrid_rerank` reranks the hybrid top-10 with `top_k=10` so Recall@10 equals hybrid's by construction. Warm-up before timing.
  Bootstrap: resample queries with replacement 1000×, take the 2.5/97.5 percentiles; paired version resamples the *same* indices for two
  systems and bootstraps the difference.
- `evals/generation_eval.py`: production path without cache/sessions; `verify_citations` on each answer; abstention = fallback phrase.
- `rag/citations.py`: regex for CVE and A01–A10 IDs, Unicode-dash folding, allowed set = retrieved metadata ∪ retrieved text ∪ user text.

### 13.4 ML / IR fundamentals the project invites (with correct answers)
**BM25 — write the formula.** For query terms q_i and document D:
`score = Σ IDF(q_i) · f(q_i,D)·(k1+1) / ( f(q_i,D) + k1·(1 − b + b·|D|/avgdl) )`, `IDF = ln((N − n + 0.5)/(n + 0.5))`.
`rank_bm25.BM25Okapi` defaults: k1=1.5, b=0.75, and negative IDFs are floored with `epsilon × mean IDF` (epsilon=0.25). k1 controls term-frequency
saturation (diminishing returns), b controls length normalisation. I used defaults and a whitespace tokenizer (lower-cased): so
`"cve-2021-44228"` is one token (good for IDs) but `"log4j,"` ≠ `"log4j"` and a trailing `?` or `.` after an ID breaks an exact match `[VERIFIED by reading]`.

**Why does dense retrieval fail on identifiers?** Bi-encoders map text to a semantic vector; `CVE-2021-44228` is split into word-pieces with no
semantic content, so near-identical IDs embed almost the same and exact digits aren't preserved. Measured: dense MRR@10 on `id` queries = 0.044
vs BM25 0.936 `[MEASURED]`. That is the empirical case for hybrid.

**How was the embedding model trained?** (BGE) contrastive learning: pull (query, relevant passage) together, push others away, with in-batch
(and mined hard) negatives. InfoNCE: `L = −log( exp(s(q,d⁺)/τ) / Σ_j exp(s(q,d_j)/τ) )`, τ = temperature. BGE additionally uses large-scale
pre-training before fine-tuning `[UNVERIFIED on exact recipe]`. Query instruction prefix is used because training was asymmetric (short query vs passage).

**Bi-encoder vs cross-encoder (cost and accuracy).** Bi-encoder: encode docs once offline; query = 1 forward pass + ANN search (sub-linear).
Cross-encoder: one forward pass **per (query, doc) pair** with full cross-attention (quadratic in the concatenated length), so only affordable
on a short candidate list — measured ~17× the latency of the rest of retrieval on CPU (352 ms vs 21 ms). More accurate because query and doc
tokens attend to each other; here it improved MRR@10 0.832 → 0.873.

**Cosine vs dot vs L2.** BGE vectors are L2-normalised, so cosine = dot product, and L2² = 2 − 2·cos — all three give the same ranking.

**HNSW in one breath.** Multi-layer proximity graph; search enters at the top sparse layer, greedily moves to the nearest neighbour, descends layers;
parameters M (edges/node), efConstruction, efSearch trade recall vs latency/memory; approximate, so it can miss true neighbours. At 2,030 docs,
brute force would be fine; HNSW is there because Chroma uses it by default.

**RRF.** `score(d) = Σ_lists 1/(k + rank)`; uses ranks only, so no calibration between BM25 scores (unbounded) and cosine ([-1,1]). k damps the head:
with k=60, rank 1 vs 2 differ by 1/61 − 1/62 ≈ 0.00026. Weakness: ignores score *magnitude* (a very confident #1 counts the same as a marginal one).

**Metrics — define and compute.** Recall@k = |relevant ∩ top-k| / |relevant|. MRR = mean(1/rank of first relevant). nDCG@k = DCG/IDCG,
`DCG = Σ rel_i / log2(i+1)`; with a single gold doc, nDCG = 1/log2(rank+1) (rank 2 → 0.631, rank 3 → 0.5). Recall ignores order; MRR only looks at the
first hit; nDCG discounts by position and handles multiple relevant docs. Precision@k isn't informative with one gold doc (max = 1/k).

**Bootstrap CI — what does "[0.764, 0.859]" mean?** Resample the 192 queries with replacement 1000 times, recompute the mean each time, take the 2.5th/97.5th
percentiles. It quantifies *query-sampling* variability for this benchmark, not generalisation to real users, and not model-seed noise.
**Paired bootstrap:** resample the same query indices for both systems and bootstrap the per-query difference; "significant" = CI excludes 0.

**Flaws in my statistics (volunteer these).**
(1) **Non-independence:** the CVE queries come in triples (natural, keyword, id) from the same document, so effective sample size is smaller than 192 and my CIs are
too narrow. Proper fix: *cluster bootstrap* resampling by gold document. `[NOT DONE]` (2) Only 60 CVEs sampled from 2,000, one seed. (3) Many comparisons (4 configs × several metrics)
with no multiplicity correction (Holm/Bonferroni). (4) A CI excluding 0 means "unlikely to be exactly 0," not "practically important"; +0.04 MRR is modest. (5) Synthetic
queries → distribution mismatch with real users. (6) The 20 OWASP questions are far too few to rank configs.

**Choosing the abstention threshold properly.** The reranker output is an uncalibrated logit. Collect answerable and unanswerable queries, plot precision-recall (or ROC) of
"top-1 reranker score ≥ τ", pick τ at a target precision/recall or cost ratio; calibrate with Platt scaling/isotonic regression if I need probabilities. Currently there is **no
threshold**; abstention depends on the prompt (measured: abstained on 10/10 unanswerable with `gpt-oss-20b`, tiny sample).

**LLM-as-judge validity.** Known biases: position bias, verbosity bias, self-preference. Mitigate with swapped order, rubric prompts, a different judge family, and **calibration to human labels**
(report Cohen's κ or Krippendorff's α, and judge–human agreement). I used no LLM judge; my "faithfulness" is only ID-level.

**Why do LLMs hallucinate, and why CVE IDs especially?** Training maximises next-token likelihood, not truth; there's no calibrated "I don't know" signal; rare, arbitrary digit strings like
`CVE-2021-44228` can't be reliably memorised and are easy to confabulate. RAG supplies the facts; the citation check detects IDs that weren't supplied.

**Temperature.** `softmax(z/T)`; T→0 is greedy; T=0.1 is near-deterministic for factual QA; T=0 can still be non-deterministic (batching/floating point). Top-p truncates the tail. I didn't tune either.

**Transformer facts that get asked.** Self-attention is O(n²·d) in sequence length; decoding uses a **KV cache** to avoid recomputing keys/values; Llama 3.1 8B: 32 layers, grouped-query attention
(fewer KV heads → smaller KV cache), 128k context `[UNVERIFIED on exact head counts]`. "Lost in the middle" (Liu et al. 2023): models use information at the start/end of long contexts better than the middle — one reason to keep context short (5 docs) and rerank.

**RAG vs fine-tuning vs long context.** RAG: fresh, citable, cheap to update; FT: changes style/behaviour, bakes in stale facts; long-context: cost/latency and the lost-in-the-middle degradation.
**LoRA:** freeze W, learn a low-rank update ΔW = B·A with rank r ≪ d, training <1% of parameters; QLoRA adds a 4-bit quantised base. If I fine-tuned anything it'd be the *embedder* (synthetic query–CVE pairs with hard negatives) before the generator.

**Distribution shift.** The cross-encoder is trained on MS MARCO (web QA); CVE text is different. Empirically it still helped (+0.04 MRR) but I didn't test a domain-tuned reranker.

### 13.5 Experiment-design questions (expect "how would you test…")
- **"Would a fine-tuned embedding model beat bge-small here?"** Hypothesis, metric (Recall@5/MRR on a *held-out* set of CVEs, split by document so no leakage), baseline (current), training data (LLM-generated queries + mined hard negatives),
  guard against overfitting to synthetic style by evaluating on human-written queries, report paired bootstrap with cluster-by-doc, and latency/memory cost.
- **"Offline metric improved — will users be happier?"** Not necessarily: offline gold is synthetic; need an online A/B (click/thumbs/ "was this cited source helpful", reformulation rate, abstention rate), guardrail metrics (latency, cost, hallucination flags),
  pre-registered metric, adequate sample size (power), and a ramp. Offline–online correlation must be checked, not assumed.
- **"How do you know your test set isn't leaking?"** Gold comes from the same doc that wrote the query (lexical leakage, mitigated not removed by the 6-gram filter); the embedder/reranker may have seen NVD text in pre-training (can't rule out);
  keep a human-written slice and compare gaps.
- **"How big a benchmark do you need?"** Power analysis: the SE of a mean metric ≈ σ/√n; to detect Δ=0.03 MRR at 80% power / α=0.05 you need n ≈ (2.8·σ_d/Δ)² queries for a *paired* comparison, where σ_d is the SD of the per-query difference (e.g. σ_d≈0.3 → n≈780); an unpaired two-group test needs about 2·(2.8·σ/Δ)² per group (σ≈0.35 → ≈2,000). Paired is cheaper because both systems see the same queries. Those queries must be *independent* (mine aren't — see above). Back-of-envelope; state the assumed σ.
- **"What would you do with two more weeks?"** (1) cluster-bootstrap + human-written query slice; (2) sweep RRF k / candidate pool / k-weights; (3) reranker threshold + PR curve for abstention; (4) claim-level faithfulness with an LLM judge calibrated on ~100 human labels; (5) fix the retry wrapper and the filter-fallback bug.
- **"Prove the reranker is worth 17× latency."** Show the quality gain (+0.042 MRR, CI [0.008, 0.077]) against the latency budget; if p95 < budget, keep; otherwise reduce candidates (e.g. top-5), use a smaller/distilled reranker, ONNX/int8, or rerank only when the first-stage score margin is small.

### 13.6 "Why this and not that" cheat table
| Decision | Why | Main alternatives (why not) | When I'd switch |
|---|---|---|---|
| RAG | fresh + citable | fine-tune (stale, no citations), long-context (cost, lost-in-middle) | domain-style tuning on top |
| BM25 + dense | IDs need lexical, concepts need semantic; measured dense MRR 0.044 on IDs | dense only (fails on IDs), BM25 only (measured ≈ hybrid at R@5!) | if real-user queries are keyword-heavy, BM25-only may suffice |
| RRF | rank-based, no tuning | weighted score fusion (needs normalisation + tuning), learned LTR (needs labels) | with labelled data, tune a weighted/LTR fusion |
| Cross-encoder rerank | better ordering (+0.042 MRR) | LLM rerank (costly), none (cheaper, lower MRR) | latency-bound → drop or shrink |
| bge-small-en-v1.5 | small, strong for size, CPU-friendly | bge-base/large, OpenAI/Titan embeddings (cost, lock-in), fine-tuned | if recall@10 on held-out set justifies bigger |
| ChromaDB | zero-ops embedded store | pgvector, Qdrant, OpenSearch (needed for scale/ops) | >~1M docs or multi-instance |
| Llama 3.1 8B on Groq | free + very fast; task is extractive | 70B / GPT / Claude (quality, cost), local | the model I intended 404'd (§7 item 18); re-evaluate on the same set |
| temp 0.1 | stable factual output | 0 (degenerate/also non-deterministic), higher (variance) | never tuned |
| FastAPI + SSE | async, validation, simple streaming | Flask, WebSockets (bidirectional not needed) | – |
| Redis semantic cache | skip repeat LLM calls | exact-match cache, GPTCache, provider prompt caching | filters/multi-turn must be in the key |
| No LangChain agents | fixed pipeline is simpler | LangChain/LlamaIndex (convenience, abstraction leakage) | when tool-use branches appear |

### 13.7 "How would you build this on AWS?" (the Amazon angle)
Frame as *mapping*, not as having done it `[I have not built this on AWS — say so]`:
- **Ingestion:** EventBridge schedule → Lambda/Step Functions/Glue pulls NVD → S3 (raw) → embed in batch (SageMaker Batch Transform or Bedrock embeddings) → write to the index with versioned/aliased indices (blue/green swap — fixes my destructive ingest).
- **Index/retrieval:** Amazon OpenSearch (Service/Serverless) supports BM25 + k-NN vector fields and hybrid queries with score normalisation, i.e. my hybrid in one engine `[UNVERIFIED on current feature details]`; Bedrock Knowledge Bases wraps chunk → embed → retrieve (managed, less control).
- **Embeddings/LLM/rerank:** Bedrock hosts embedding models (e.g. Titan Text Embeddings), LLMs (Claude, Llama, Nova), and a rerank capability `[UNVERIFIED]`; or host my own bge/cross-encoder on SageMaker endpoints for latency/cost control. Guardrails (incl. contextual-grounding checks) overlap with my citation guard `[UNVERIFIED]`.
- **Serving/state:** ECS/Fargate or Lambda behind API Gateway/ALB; sessions + rate limits in DynamoDB/ElastiCache (fixes per-process state); CloudWatch for metrics/logs/alarms (replaces Prometheus/Grafana); WAF + Cognito for auth.
- **Evaluation/ops:** nightly eval job on a golden set (CodeBuild/Step Functions) with a regression gate; log retrieval scores and abstention rate; A/B via traffic split.
- **Trade-off sentence:** managed services cut ops and add scale but reduce control over retrieval details and cost visibility; for an experiment-heavy AS workflow I'd prototype self-hosted, then productionise managed.

### 13.8 Leadership Principles → real stories (STAR skeletons; fill `[AUTHOR?]`)
Only tell what is true for you. These are built from verifiable facts in this project.
- **Dive Deep — NVD 404.** S: ingest returned 404 with no message. T: fix the data fetch. A: shrank the date range (365 d → 404, 90 d → 200) to isolate an undocumented 120-day window limit. R: four 119-day windows; 2,000 CVEs fetched. L: don't trust missing error messages; bisect inputs.
- **Dive Deep / Insist on the Highest Standards — the 25% that wasn't real.** S: first citation check said the gold ID appeared in only 25% of answers despite Recall@5 ≈ 0.9. T: decide if the model or the metric was wrong. A: read the raw answers, found the model used non-breaking hyphens that my regex missed. R: fixed + unit-tested; true figure 90%; the "0% hallucination" before the fix would have been meaningless. L: sanity-check metrics against independent evidence. `[AUTHOR? confirm your involvement]`
- **Are Right A Lot / Earn Trust — reporting that hybrid ≈ BM25.** S: README claimed hybrid gives "way better recall". T: validate. A: built the ablation. R: hybrid ≈ BM25 on Recall@5 (0.906 vs 0.911); dense-only is the weak one; I rewrote the claim and listed caveats. L: change the claim, not the data.
- **Ownership — things nobody asked me to fix.** While verifying I found `/api/chat` crashed on every first-turn request (undefined metric) and that the default Groq model returned 404; fixed the former, documented the latter. L: verify end-to-end, not just unit tests.
- **Learn and Be Curious — RAGAS conflict.** S: RAGAS required `langchain<0.3`. A: wrote own metrics. Later I learned they were lexical proxies and replaced the evidence with standard IR metrics (Recall/MRR/nDCG + bootstrap). L: a workaround can hide a validity problem.
- **Bias for Action / Invent and Simplify.** Flag-only citation guard (works with streaming, no rewrite risk); RRF instead of tuning score fusion; no framework for a fixed pipeline.
- **Customer Obsession.** User = security engineer who needs *citable* answers and an honest "I don't know"; designed for citations, abstention, source chips shown before the answer finishes.
- **Deliver Results.** Live deployment at rag.awni.in `[AUTHOR? verify it works — the Groq model 404'd on 2026-11-10]`; measured benchmark; list concrete outputs, not effort.
- **Frugality.** Free-tier LLM, CPU embedder/reranker, embedded vector DB; and the cost I accepted (per-process state, small corpus).

### 13.9 Hostile follow-up ladders (practise answering each rung)
**Ladder 1 — "Hybrid improves retrieval."** → *Prove it.* Ablation, 192 queries. → *Against what?* BM25/dense/hybrid/+rerank. → *Hybrid vs BM25 is 0.906 vs 0.911?* Right; no clear win at R@5;
benefit is Recall@10 and OWASP-style questions; dense alone is the clear loser. → *Why would hybrid lose to BM25?* My queries share vocabulary with the docs (synthetic); dense adds noise when lexical match is strong; RRF treats both lists equally with k untuned. → *So why ship hybrid?*
Real users paraphrase more than my synthetic set; hybrid is robust across styles (among the first-stage methods it is best on keyword, natural and OWASP and within 0.01 MRR of BM25 on id) and adds ~18 ms over BM25; I'd verify with human queries. → *CI valid?* No — correlated triples; cluster bootstrap needed.
**Ladder 2 — "Your hallucination rate is 0%."** → *n?* 50 (40+10). → *Rule of three?* Upper bound ≈ 3/40 ≈ 7.5% for answerable. → *Which model?* `gpt-oss-20b`, not Llama 3.1 8B (404). → *What does it measure?* Only that cited IDs are in the context, not claim truth.
→ *How would you measure claim faithfulness?* Decompose into atomic claims, NLI/LLM judge against context, calibrate vs ~100 human labels, report κ and CIs.
**Ladder 3 — "How would this scale?"** → today: single process, in-RAM BM25, per-process sessions/limits, sync CPU work in async routes (event-loop blocking), destructive ingest. → fix list (§F, §9): threadpool/`ainvoke`, shared state, OpenSearch/pgvector, blue/green index, autoscale on reranker CPU. → *What breaks first?* The reranker (CPU, ~350 ms p50) and the event loop.
**Ladder 4 — "Why not just use an LLM?"** → stale/hallucinated IDs, no citations, can't verify; measured: retrieval gives the right doc in top-5 92% of the time; the guard catches fabricated IDs. → *Couldn't GPT-4/Claude do this zero-shot?* On well-known CVEs, often; on recent ones (this corpus is 2025–2026) no; and I couldn't audit it.
**Ladder 5 — "Security of the system."** → prompt injection via NVD text (indirect), unsanitised markdown→`innerHTML` (XSS), public default ingest key, unauthenticated `/evaluate` (cost abuse), PII to a third party. → mitigations: DOMPurify/CSP, delimiter + instruction hierarchy, auth, output filtering, red-team prompts.

### 13.10 Whiteboard snippets (be able to write these)
```python
import math, random

def rrf(rank_lists, k=60):                        # rank_lists: list of lists of doc ids (best first)
    s = {}
    for lst in rank_lists:
        for r, d in enumerate(lst, start=1):
            s[d] = s.get(d, 0) + 1 / (k + r)
    return sorted(s, key=s.get, reverse=True)

def mrr(ranked, gold):                            # gold: set
    return next((1 / i for i, d in enumerate(ranked, 1) if d in gold), 0.0)

def ndcg_at_k(ranked, gold, k=10):
    dcg = sum(1 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in gold)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0

def bootstrap_ci(xs, B=1000, alpha=0.05, seed=0):
    rng = random.Random(seed); n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return means[int(B * alpha / 2)], means[int(B * (1 - alpha / 2)) - 1]

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

def softmax_T(logits, T=1.0):
    m = max(logits); ex = [math.exp((z - m) / T) for z in logits]
    s = sum(ex); return [e / s for e in ex]
```
Likely follow-ups: complexity of each (RRF O(Σ|list|·log); bootstrap O(B·n); BM25 scoring O(|Q|·N) here since I score the whole corpus per query), numerical stability (subtract max in softmax), and edge cases (empty gold, ties, duplicate IDs → de-duplicate before ranking, as the harness does).

### 13.11 Rapid-fire numbers (all `[MEASURED]` unless noted)
Corpus 2,000 CVEs + 30 OWASP chunks (2,030 docs) · pipeline 20+20 → RRF(k=60) → 10 → rerank → 5 `[VERIFIED]` · dataset 192 queries (60 id / 58 keyword / 54 natural / 20 owasp) ·
Recall@5: BM25 0.911, dense 0.589, hybrid 0.906, hybrid+rerank 0.922 · MRR@10: 0.812 / 0.517 / 0.832 / 0.873 · nDCG@10: 0.841 / 0.543 / 0.861 / 0.893 ·
dense MRR on id queries 0.044 · rerank paired ΔMRR +0.042 [0.008, 0.077] · latency p50 BM25 3 ms, dense 18, hybrid 21, +rerank 352 (p95 712) ·
generation (`gpt-oss-20b`): 63 IDs cited, all supported; gold cited 90%; wrongly abstained 1/40; fake-ID and off-topic abstention 5/5 and 5/5.

### 13.12 Questions I can ask them (shows research-mindedness)
How does the team decide an offline metric gain is worth shipping? · How do you build evaluation sets when ground truth is expensive? · What are the biggest failure modes you see with retrieval-augmented systems in production?
· How much of an intern's project is experimentation vs engineering? · How do scientists and engineers split ownership on a launch?

### 13.13 Do / don't in the room
Do: state the claim, the evidence, then the caveat in one breath; say "not measured" instead of guessing; draw the pipeline; volunteer the CI/independence flaw before they find it; quantify trade-offs.
Don't: say "production-ready," "scalable," or "reduces hallucinations by…"; defend the 0.92 cache threshold or temperature as tuned (they weren't); claim you built it on AWS; read code from memory you haven't actually read.
**Before the interview:** (1) check rag.awni.in actually answers (the default model 404'd); (2) re-read `evals/` and `rag/citations.py` end to end; (3) rehearse Ladders 1–2 aloud; (4) decide your honest answer to §11 item 1.

# SecureRAG

```
 ____                            ____      _    ____ 
/ ___|  ___  ___ _   _ _ __ ___|  _ \    / \  / ___|
\___ \ / _ \/ __| | | | '__/ _ \ |_) |  / _ \| |  _ 
 ___) |  __/ (__| |_| | | |  __/  _ <  / ___ \ |_| |
|____/ \___|\___|\__,_|_|  \___|_| \_\/_/   \_\____|
```

> A RAG chatbot that actually knows about cybersecurity — not just generic LLM vibes.

Built this because I was tired of asking ChatGPT about CVEs and getting answers that were either hallucinated or two years out of date. SecureRAG pulls from the real NVD database (~2000 recent CVEs) and the OWASP Top 10, runs a proper retrieval pipeline with hybrid search + reranking, and cites its sources every time.

Live demo: [rag.awni.in](https://rag.awni.in/) 
(My OWN custom domain )
---

## What it does

- **Answers security questions with cited sources** — every response includes the CVE ID or OWASP category it pulled from, so you can verify it yourself
- **Hybrid search** — combines keyword matching (BM25) and semantic vector search, then fuses them with Reciprocal Rank Fusion. Way better recall than either one alone
- **Cross-encoder reranking** — after retrieval, a second model rescores the top 10 results and keeps the best 5. Cuts the noise significantly
- **Severity + source filters** — you can tell it to only look at HIGH/CRITICAL CVEs, or only OWASP entries, or both
- **Multi-turn chat** — it keeps track of your conversation context, so follow-up questions actually work
- **Evaluation dashboard** — built a lightweight metrics panel that shows faithfulness and relevancy scores using token-overlap scoring (honest alternative since RAGAS had version conflicts with LangChain 0.3)

---

## The retrieval pipeline

```
Your question
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                           │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │ Chat History │    │          HybridSearcher               │   │
│  │  (session)   │    │                                      │   │
│  └──────┬───────┘    │  ┌─────────────┐  ┌──────────────┐  │   │
│         │            │  │ BM25 (sparse│  │ChromaDB Vec  │  │   │
│         │            │  │  rank-bm25) │  │  (semantic)  │  │   │
│         │            │  └──────┬──────┘  └──────┬───────┘  │   │
│         │            │         └────────┬───────┘           │   │
│         │            │                  ▼                   │   │
│         │            │          RRF Fusion (top 10)         │   │
│         │            └──────────────────┬───────────────────┘   │
│         │                               ▼                       │
│         │            ┌──────────────────────────────────────┐   │
│         │            │     CrossEncoder Re-ranker            │   │
│         │            │  (ms-marco-MiniLM-L-6-v2, top 5)    │   │
│         │            └──────────────────┬───────────────────┘   │
│         └───────────────────────────────┤                       │
│                                         ▼                       │
│                              ┌─────────────────┐                │
│                              │   Groq LLM       │                │
│                              │ llama-3.1-8b     │                │
│                              └────────┬────────┘                │
└───────────────────────────────────────┼─────────────────────────┘
                                        │
                                        ▼
                                Answer + Sources
```

---

## Tech stack

| Component | What I used |
|-----------|-------------|
| Backend | FastAPI 0.115 + Uvicorn |
| RAG framework | LangChain 0.3 |
| Embeddings | BAAI/bge-small-en-v1.5 (via sentence-transformers) |
| Vector store | ChromaDB 0.5, persistent, cosine distance |
| Sparse search | rank-bm25 (BM25Okapi) |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq API — llama-3.1-8b-instant |
| Frontend | Vanilla HTML/CSS/JS, no framework |

---

## Running it locally

### 1. Clone and install

```bash
git clone https://github.com/AwkJay/Secure-RAG
cd securerag
pip install -r requirements.txt
```

### 2. Set up your environment

```bash
cp .env.example .env
# Add your GROQ_API_KEY — get one free at console.groq.com
```

### 3. Ingest the data (takes a few minutes the first time)

```bash
python data/ingest.py
# Re-fetch everything fresh:
python data/ingest.py --force
```

This pulls ~2000 CVEs from the NVD API and loads the OWASP Top 10 into ChromaDB. The NVD free tier rate-limits you to one request per 6 seconds, so grab a coffee.

### 4. Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open [http://localhost:8000](http://localhost:8000)

Swagger docs are at [http://localhost:8000/docs](http://localhost:8000/docs) if you want to poke the API directly.

---

## API endpoints

| Method | Path | What it does |
|--------|------|-------------|
| POST | `/api/chat` | Main chat endpoint — sends query, returns answer + source chunks |
| GET | `/api/health` | Health check, confirms models are loaded |
| POST | `/api/ingest` | Kicks off data ingestion in the background |
| GET | `/api/ingest/status` | Polls ingestion progress |
| POST | `/api/evaluate` | Runs the evaluation suite (cached for 1hr) |
| GET | `/api/stats` | Document counts, query stats |
| DELETE | `/api/session/{id}` | Clears a chat session |

---

## Data sources

- **NVD (National Vulnerability Database)** — nvd.nist.gov — public domain CVE data
- **OWASP Top 10 2021** — owasp.org/Top10 — CC Attribution 3.0

The NVD fetch covers the last 365 days, split into 119-day windows (NVD's API has a 120-day max range per request — learned that the hard way).

---

## Project structure

```
securerag/
├── main.py                # FastAPI entry point, mounts static files
├── requirements.txt
├── .env.example
├── data/
│   ├── ingest.py          # Fetches CVEs from NVD + loads OWASP into ChromaDB
│   └── owasp_data.json    # OWASP Top 10 2021 entries
├── rag/
│   ├── embeddings.py      # BAAI/bge-small-en-v1.5 singleton
│   ├── vectorstore.py     # ChromaDB client
│   ├── hybrid_search.py   # BM25 + semantic search + RRF fusion
│   ├── reranker.py        # CrossEncoder re-ranking
│   ├── chain.py           # Full RAG chain, session management, LLM calls
│   └── evaluate.py        # Token-overlap evaluation metrics
├── api/
│   ├── routes.py          # All endpoint handlers
│   └── models.py          # Pydantic request/response schemas
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js             # ~700 lines of vanilla JS, no build step needed
```

---

## Evaluation

Run from `securerag/` (needs the ingested ChromaDB; dataset build and generation checks also need `GROQ_API_KEY`):

```bash
python -m evals.run --retrieval      # BM25 vs dense vs hybrid vs hybrid+rerank, no LLM calls
python -m evals.run --generation     # citation / abstention checks (calls Groq)
python -m evals.run --build-dataset  # regenerate evals/data/retrieval_set.jsonl
```

Results go to `evals/results/latest.md` (the full per-run JSON is gitignored).

### Retrieval ablation

192 labelled queries against the 2,000 CVE + 30 OWASP-chunk corpus: 60 `id` ("Tell me about CVE-…"), 58 `keyword`, 54 `natural`, 20 `owasp`.
Metrics are binary-relevance Recall@k, MRR@10 and nDCG@10 with 95% bootstrap CIs (1000 resamples, seeded). Latency is CPU, single query at a time, retrieval only.

| Config | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50 / p95 latency |
|---|---|---|---|---|---|---|
| BM25 | 0.760 | 0.911 | 0.932 | 0.812 [0.764, 0.859] | 0.841 | 3 / 7 ms |
| Dense (BGE-small) | 0.464 | 0.589 | 0.625 | 0.517 [0.451, 0.583] | 0.543 | 18 / 22 ms |
| Hybrid (RRF, k=60) | 0.776 | 0.906 | 0.953 | 0.832 [0.786, 0.879] | 0.861 | 21 / 31 ms |
| Hybrid + cross-encoder rerank | **0.833** | **0.922** | 0.953 | **0.873** [0.833, 0.916] | **0.893** | 352 / 712 ms |

What the numbers do and don't say:

- **Dense alone is the weakest**, mostly because of `id` queries (MRR 0.04): embeddings don't encode identifiers. On `natural` and `keyword` queries it is close to BM25 (MRR 0.71 vs 0.77 and 0.70 vs 0.74).
- **Hybrid vs BM25 is not a clear win on this set.** Recall@5 is 0.906 vs 0.911, and MRR differs by 0.02 with overlapping CIs. Hybrid's clearest benefit is Recall@10 and OWASP-style questions, where BM25 is weaker (MRR 0.78 vs 0.88).
- **Reranking helps ranking quality, not recall:** paired MRR@10 +0.042 (95% CI [+0.008, +0.077]) and nDCG@10 +0.032 over hybrid; Recall@5 +0.016 with a CI that includes 0. It reranks the same 10 candidates, so Recall@10 is identical by construction. The cost is about 20x latency on CPU.
- Hybrid, hybrid+rerank and BM25 all beat dense by a wide margin (paired Δ MRR@10 of +0.30 to +0.36, CIs exclude 0).

Caveats, stated plainly:

- **Queries are synthetic.** The CVE `natural` and `keyword` queries were written by an LLM from the CVE description (with the ID and any 6-word verbatim run filtered out). They still reuse product names and jargon from the document, which favours BM25. Real user questions would likely narrow the gap between lexical and dense search. The 20 OWASP questions are hand-written.
- **Queries aren't independent:** each sampled CVE contributes up to three queries (natural, keyword, id), so the bootstrap CIs above are narrower than they should be; a cluster bootstrap by document would fix this and is not done yet.
- Single run, single seeded sample of 60 CVEs; no tuning of RRF `k`, candidate counts or models was done, so none of this is a tuned result.
- Near-duplicate CVEs (same product, same bug class) can make a "miss" a legitimate alternative answer, so recall is slightly understated.
- The corpus is 2,000 recent CVEs; results would shift with a larger or older corpus.

### Generation checks

`rag/citations.py` extracts every CVE / OWASP ID from an answer and flags any that are not in the retrieved context or the user's own messages. The API returns this as `verification` (and as an SSE `verification` event before `done`), the UI shows a warning badge, and `securerag_unverified_citations_total` counts them. Answers are flagged, never rewritten.

Measured on 40 answerable queries (30 CVE `natural` + 10 OWASP) and 10 unanswerable ones (5 nonexistent CVE IDs, 5 off-topic), using `openai/gpt-oss-20b` on Groq (see note below). No LLM judge.

| Metric | Value |
|---|---|
| Citation precision (verified IDs / cited IDs, 63 cited) | 100% |
| Hallucinated-ID rate, answerable | 0 / 40 |
| Hallucinated-ID rate, unanswerable | 0 / 10 |
| Answerable: gold ID appears in the answer | 90% (36 / 40) |
| Answerable: wrongly abstained | 2.5% (1 / 40) |
| Fake CVE ID: abstained | 100% (5 / 5) |
| Off-topic: abstained | 100% (5 / 5) |

- **Model caveat:** `llama-3.1-8b-instant` (the app's default) returned `404 model_not_found` for my Groq key when I ran these, so the generation numbers are for `gpt-oss-20b`, not Llama 3.1 8B. Retrieval numbers don't involve the LLM. Re-run `python -m evals.run --generation` with the intended model before quoting these for it.
- 0 / 40 is a small sample: it means the hallucinated-ID rate is probably low, not that it is zero (the 95% upper bound is roughly 7% at n=40).
- This checks that cited **IDs** exist in the context. It does **not** check that the surrounding claims (severity, affected versions, fixes) are faithful to the source; that needs an LLM or human judge and is future work.
- A bug found by this harness: the model wrote IDs with non-breaking hyphens (`CVE‑2025‑64773`), which a plain regex missed and made the first run report 25% gold-cited instead of 90%. `extract_ids` now normalises Unicode dashes (covered by tests).

---

## Known limitations / things I'd improve

- The ChromaDB store is on disk, so on autoscale deployments the data gets baked in at build time rather than being re-ingested live. Not ideal if you want real-time CVE updates
- No NVD API key means 6-second rate limiting between requests — with a free API key from nvd.nist.gov it drops to 0.6 seconds
- Session history is in-memory, so it resets on server restart

---

*Made with too much caffeine and a genuine frustration with vague security chatbots.*

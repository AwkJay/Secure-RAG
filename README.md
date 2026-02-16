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
| LLM | Groq API — llama-3.1-8b-instant (it's free) |
| Frontend | Vanilla HTML/CSS/JS, no framework |

---

## Running it locally

### 1. Clone and install

```bash
git clone <your-repo>
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

## Known limitations / things I'd improve

- The ChromaDB store is on disk, so on autoscale deployments the data gets baked in at build time rather than being re-ingested live. Not ideal if you want real-time CVE updates
- No NVD API key means 6-second rate limiting between requests — with a free API key from nvd.nist.gov it drops to 0.6 seconds
- Session history is in-memory, so it resets on server restart

---
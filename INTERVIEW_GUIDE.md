# SecureRAG — Complete Interview Prep Guide

> Written as if your senior developer friend is explaining it the night before your interview.  
> Read this once slowly, then skim the bold parts before the interview.

---

## Table of Contents

1. [The Big Picture — What Problem Are We Solving?](#1-the-big-picture)
2. [What is RAG? (And Why Not Just Use ChatGPT?)](#2-what-is-rag)
3. [How We Thought About This Project (The Mental Model)](#3-mental-model)
4. [The Data — Where Does Knowledge Come From?](#4-the-data)
5. [Embeddings — Teaching a Computer to Understand Meaning](#5-embeddings)
6. [Vector Databases — How We Store That Understanding](#6-vector-databases)
7. [BM25 — The Old-School Search That Still Wins](#7-bm25)
8. [Hybrid Search + RRF — Why We Use Both](#8-hybrid-search)
9. [Cross-Encoder Reranking — The Final Quality Check](#9-reranking)
10. [The LLM — What Groq and LLaMA Are Doing](#10-the-llm)
11. [LangChain — The Glue](#11-langchain)
12. [FastAPI — The Backend](#12-fastapi)
13. [The Frontend — Why No Framework](#13-frontend)
14. [Problems I Faced & How I Solved Them](#14-problems)
15. [The Deployment Story](#15-deployment)
16. [How Everything Fits Together (End-to-End Walk)](#16-end-to-end)
17. [Interview Questions & How to Answer Them](#17-interview-prep)

---

## 1. The Big Picture

### What problem does this project solve?

Imagine you're a security engineer and you want to know: *"What are the most critical SQL injection CVEs from the last 6 months?"*

You have two bad options:
- **Google it** → you get blog posts, Stack Overflow, outdated articles. Not precise.
- **Ask ChatGPT** → it gives you a confident-sounding answer that might be completely made up (hallucinated), and it definitely doesn't know about CVEs from last month because its training data has a cutoff.

**SecureRAG solves this** by:
1. Pulling real, fresh CVE data directly from the NVD (National Vulnerability Database) API
2. Storing it in a searchable knowledge base
3. When you ask a question, finding the most relevant real documents first
4. Then giving those documents to the LLM and saying "answer the question using ONLY this information"

This approach is called **RAG — Retrieval-Augmented Generation**.

---

## 2. What is RAG?

### Simple definition

**RAG = Search first, then generate.**

Instead of asking an AI to answer from memory (which can be wrong or outdated), you first search a database for relevant documents, then ask the AI to answer based on those documents.

### A real-world analogy

Think of it like an open-book exam vs a closed-book exam.

- **ChatGPT without RAG** = closed-book exam. The AI answers purely from what it memorized during training. Smart, but can be wrong, and can't know things that happened after training.
- **RAG** = open-book exam. The AI is allowed to look up the relevant pages of the textbook before answering. Much more accurate. The "textbook" is your database.

### The three steps of RAG

```
Step 1: RETRIEVE
User asks: "What is CVE-2024-1234?"
System searches the database and finds the 5 most relevant documents

Step 2: AUGMENT
System says to the LLM: "Here are 5 relevant documents [pastes them].
Now answer the user's question using only this information."

Step 3: GENERATE
LLM reads the documents and writes a coherent answer,
citing which document each fact came from.
```

### Why is RAG better than fine-tuning?

You might ask: "Why not just train the model on security data?"

| Fine-tuning | RAG |
|-------------|-----|
| Expensive — needs GPUs, days of training | Cheap — just update the database |
| Data becomes stale — model has a knowledge cutoff | Fresh — add new CVEs to DB any time |
| Model can still hallucinate | Grounded — answer must come from retrieved docs |
| Hard to update | Easy — just re-ingest new data |

**RAG wins for dynamic, frequently-updated knowledge like CVEs.**

---

## 3. Mental Model — How We Thought About This Project

Before writing a single line of code, here's the thinking process:

### Question 1: "What knowledge do we need?"

We need two things:
- **CVE data** — real vulnerability records. Source: NVD API (free, public, up-to-date)
- **OWASP Top 10** — the 10 most critical web security risk categories with explanations, prevention tips, and attack examples. Source: We hardcoded this as a JSON file since it changes rarely.

### Question 2: "How do users ask questions?"

Users ask things like:
- *"What SQL injection CVEs were published recently?"* → keyword-heavy, specific
- *"How can I prevent injection attacks?"* → conceptual, meaning-based

This told us we need **two types of search**:
- One that's good at keywords (BM25)
- One that understands meaning/concepts (semantic/vector search)

### Question 3: "How do we make sure the answer is good?"

We need:
- **Retrieval quality** → get the right documents first
- **Reranking** → after retrieval, double-check which ones are truly relevant
- **Grounding** → tell the LLM to only use the retrieved docs, not its own memory

### Question 4: "What infrastructure do we need?"

- A database that can do vector similarity search → **ChromaDB**
- A fast LLM API → **Groq** (free tier, very fast)
- A backend that handles HTTP requests → **FastAPI**
- A frontend the user can interact with → vanilla HTML/JS (keep it simple)

---

## 4. The Data

### NVD — National Vulnerability Database

NVD is the US government's official repository of every known software vulnerability. Each entry is a **CVE (Common Vulnerabilities and Exposures)**.

A CVE looks like this:
```json
{
  "id": "CVE-2024-1234",
  "description": "SQL injection vulnerability in ExampleSoft 3.2 allows remote attackers to...",
  "severity": "CRITICAL",
  "cvss_score": 9.8,
  "published": "2024-03-15"
}
```

Every CVE has an ID (`CVE-YEAR-NUMBER`), a description, a severity rating (LOW/MEDIUM/HIGH/CRITICAL), and a CVSS score (0-10 scale).

**How we fetch CVEs:**

We use the NVD 2.0 REST API. We fetch the last 365 days of CVEs, capped at 2000.

```python
# Simplified version of what ingest.py does
response = requests.get(
    "https://services.nvd.nist.gov/rest/json/cves/2.0",
    params={"pubStartDate": "2025-04-11T00:00:00.000", "pubEndDate": "2026-04-11T00:00:00.000"}
)
```

### OWASP Top 10

OWASP (Open Web Application Security Project) publishes the Top 10 list — the 10 most critical categories of web security risks. We hardcoded this as a JSON file because it only changes every few years. For each OWASP entry, we store:
- What the risk is
- How to prevent it
- Example attack scenarios

### How We Chunk the Data

Raw CVEs and OWASP entries are long. We **chunk** them — split them into smaller pieces that are easier to retrieve precisely.

For each CVE, we create **one document**:
```
CVE ID: CVE-2024-1234
Severity: CRITICAL (CVSS: 9.8)
Published: 2024-03-15
Description: SQL injection vulnerability in...
```

For each OWASP entry, we create **three documents** (chunks):
- Chunk 1: Overview + description
- Chunk 2: Prevention techniques
- Chunk 3: Example attack scenarios

**Why three chunks?** Because someone asking "how do I prevent XSS?" needs the prevention chunk, not the attack scenario chunk. Smaller, focused chunks = better retrieval precision.

---

## 5. Embeddings — Teaching a Computer to Understand Meaning

### What is an embedding?

Words and sentences don't naturally mean anything to computers — they just see text. An **embedding** converts text into a list of numbers (a vector) where the numbers capture meaning.

Example:
```
"SQL injection"        → [0.23, -0.87, 0.45, 0.12, ...]  (384 numbers)
"database attack"      → [0.21, -0.84, 0.47, 0.15, ...]  (very similar!)
"chocolate cake recipe" → [-0.65, 0.32, -0.21, 0.88, ...] (very different)
```

The key insight: **semantically similar text produces similar vectors.** "SQL injection" and "database attack" end up close to each other in this 384-dimensional space. "Chocolate cake" ends up far away.

### How we measure similarity

We use **cosine similarity** — measuring the angle between two vectors. If two vectors point in nearly the same direction, they're similar. If they point in opposite directions, they're different.

```
Cosine similarity of 1.0 = identical meaning
Cosine similarity of 0.0 = completely unrelated
Cosine similarity of -1.0 = opposite meaning
```

### The model we use: BAAI/bge-small-en-v1.5

We use a pre-trained embedding model called `BAAI/bge-small-en-v1.5`. 

- **BAAI** = Beijing Academy of Artificial Intelligence (who made it)
- **bge** = BERT Generative Embeddings
- **small** = small version (fast, good for production)
- **en** = English
- **v1.5** = version 1.5

It converts any text into a **384-dimensional vector** (a list of 384 numbers).

**Why this model?** It's free, open-source, runs locally (no API cost), small enough to be fast, and performs very well on retrieval tasks. It consistently ranks high on the MTEB (Massive Text Embedding Benchmark) leaderboard.

```python
# From our embeddings.py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
vector = model.encode("SQL injection vulnerability")
# vector is now a list of 384 floats
```

---

## 6. Vector Databases — How We Store That Understanding

### What is a vector database?

A regular database stores text and lets you search by exact match or keyword. A **vector database** stores embeddings (those lists of numbers) and lets you search by **similarity**.

When you search, your query gets converted to a vector, then the database finds all the stored vectors that are closest to your query vector.

```
Query: "What is cross-site scripting?"
Query vector: [0.34, -0.12, 0.67, ...]

Vector DB finds documents whose vectors are closest:
1. "XSS — Reflected attacks inject scripts..." (similarity: 0.94)
2. "Cross-site scripting allows attackers to..." (similarity: 0.91)
3. "JavaScript injection in web forms..." (similarity: 0.87)
```

### ChromaDB

We use **ChromaDB** — an open-source vector database that runs locally (no external service needed, no cost). It stores documents as both their raw text AND their vector representation.

```python
# Simplified from our vectorstore.py
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("securerag")

# Adding a document
collection.add(
    ids=["CVE-2024-1234"],
    documents=["CVE ID: CVE-2024-1234... description here..."],
    embeddings=[[0.23, -0.87, 0.45, ...]],  # pre-computed
    metadatas=[{"source": "NVD", "severity": "CRITICAL"}]
)

# Searching
results = collection.query(
    query_embeddings=[[0.21, -0.84, 0.47, ...]],  # query vector
    n_results=10
)
```

**Why ChromaDB and not Pinecone/Weaviate?** ChromaDB is:
- Free and runs locally
- No account or API key needed
- Perfect for projects where you want everything self-contained
- Persistent (saves to disk, survives restarts)

---

## 7. BM25 — The Old-School Search That Still Wins

### What is BM25?

**BM25 (Best Match 25)** is a keyword-based ranking algorithm from the 1990s. It's essentially a very smart version of "count how many times the search words appear in the document."

BM25 is what powers Google's early search, Elasticsearch, and most full-text search systems even today.

**The formula cares about:**
1. How often does the keyword appear in the document? (term frequency)
2. How rare is the keyword across all documents? (inverse document frequency)
3. How long is the document? (normalizes so longer docs don't automatically win)

### Example

Query: "SQL injection"

| Document | "SQL" count | "injection" count | BM25 score |
|----------|-------------|-------------------|------------|
| CVE about SQL injection (short) | 3 | 4 | 8.2 |
| CVE about SQL injection (very long) | 3 | 4 | 6.1 (penalized for length) |
| OWASP XSS entry | 0 | 1 | 0.4 |

### Why use BM25 alongside vector search?

BM25 is great at:
- **Exact keyword matches** — if the user types "CVE-2024-1234", BM25 will find it instantly
- **Rare or specific terms** — product names, CVE IDs, version numbers
- **Fast** — no model inference, just math

Vector search is great at:
- **Conceptual similarity** — "database attack" matches "SQL injection" even though the words differ
- **Paraphrasing** — "how to stop XSS" matches "XSS prevention techniques"
- **Handles typos better** — meaning isn't lost by a misspelling

**Neither alone is perfect. Together, they cover each other's weaknesses.**

---

## 8. Hybrid Search + Reciprocal Rank Fusion

### The problem: two ranked lists, one result

After BM25 search and vector search, we have two separate lists of documents, each ranked differently. How do we combine them into one final ranked list?

The naive approach is to average scores — but BM25 scores and cosine similarity scores are on completely different scales. BM25 might give scores of 0–15, cosine similarity gives 0–1. You can't just average them.

### Reciprocal Rank Fusion (RRF)

RRF is elegant. Instead of using the scores, it only uses the **ranks** (positions).

For each document, its RRF score is:
```
RRF_score = 1 / (rank_in_list1 + k) + 1 / (rank_in_list2 + k)
```
where k = 60 (a constant that prevents top ranks from dominating too much).

**Example:**

| Document | BM25 rank | Vector rank | RRF score |
|----------|-----------|-------------|-----------|
| Doc A | 1st | 3rd | 1/(1+60) + 1/(3+60) = 0.0164 + 0.0159 = **0.0323** |
| Doc B | 2nd | 1st | 1/(2+60) + 1/(1+60) = 0.0161 + 0.0164 = **0.0325** |
| Doc C | 1st (BM25 only) | not in top 10 | 1/(1+60) + 0 = **0.0164** |

Doc B wins! It was consistently ranked high across both methods — that consistency is a strong signal it's genuinely relevant.

Doc C ranked #1 in BM25 but wasn't found by semantic search at all — probably a keyword match but not conceptually relevant.

### In our code (hybrid_search.py)

```python
# Simplified
bm25_results = bm25_search(query, top_k=10)     # top 10 from BM25
vector_results = vector_search(query, top_k=10)  # top 10 from ChromaDB

# Combine with RRF
for rank, doc in enumerate(bm25_results):
    scores[doc.id] += 1 / (rank + 60)
for rank, doc in enumerate(vector_results):
    scores[doc.id] += 1 / (rank + 60)

# Sort by combined score, return top 10
final_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
```

---

## 9. Cross-Encoder Reranking — The Final Quality Check

### The problem with the first retrieval

BM25 and vector search are called **bi-encoders** — they encode the query and documents separately and compare them. This is fast but not perfectly accurate. They sometimes retrieve documents that seem relevant by similarity but aren't actually useful for answering the specific question.

### What is a cross-encoder?

A **cross-encoder** takes the query AND a document together and scores them as a pair. It can see the relationship between them.

```
Bi-encoder:  encode(query) vs encode(document) → similarity score
Cross-encoder: encode(query + document together) → relevance score
```

The cross-encoder is more accurate because it reads both at the same time and understands context. But it's slower, so we only use it after the first retrieval narrows things down.

### Our pipeline

```
Query → BM25 + Vector Search → Top 10 documents
                                      ↓
              Cross-encoder rescores all 10 (query, doc) pairs
                                      ↓
                          Final top 5, ranked by true relevance
```

### The model we use: ms-marco-MiniLM-L-6-v2

- **ms-marco** = Microsoft MAchine Reading COmprehension dataset (trained on this)
- **MiniLM** = smaller, efficient transformer
- **L-6** = 6 transformer layers
- **v2** = version 2

This model was specifically trained to score how relevant a document is to a query. It's state-of-the-art for this task.

```python
# From our reranker.py
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Score all 10 (query, document) pairs
pairs = [(query, doc.text) for doc in candidates]
scores = reranker.predict(pairs)  # returns a score for each pair

# Sort by score, keep top 5
reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:5]
```

---

## 10. The LLM — What Groq and LLaMA Are Doing

### What is an LLM?

A **Large Language Model (LLM)** is a neural network trained on massive amounts of text. It learns to predict "given this text, what comes next?" — and from that, it develops the ability to understand and generate human language.

LLMs don't retrieve information — they generate text based on patterns from training. That's why they can hallucinate (confidently state false things). In our RAG pipeline, the LLM's job is only to **read the retrieved documents and write a coherent answer** — not to recall facts from memory.

### Meta's LLaMA 3.1

LLaMA (Large Language Model Meta AI) is an open-source LLM created by Meta (Facebook). We use `llama-3.1-8b-instant`:
- **3.1** = version 3.1
- **8b** = 8 billion parameters (smaller = faster, still very capable)
- **instant** = optimized for speed

### What is Groq?

Groq is a company that makes specialized AI chips (Language Processing Units — LPUs). Their hardware runs LLMs dramatically faster than regular GPUs.

We use **Groq's API** (not to be confused with Grok, the xAI product). Groq hosts open-source models like LLaMA and lets you call them via API. **It's free for moderate usage.**

Why Groq over OpenAI/Anthropic?
- Free tier is generous enough for a project like this
- Fast — responses in under 2 seconds
- We don't need GPT-4 level capabilities for this task; LLaMA 3.1 8B is excellent

### How we call it (chain.py)

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1  # low temperature = more factual, less creative
)

# We pass a prompt like:
prompt = """
You are a cybersecurity expert. Answer the question using ONLY the context below.
Always cite your sources (CVE IDs or OWASP categories).

Context:
{retrieved_documents}

Question: {user_question}
"""
```

**Temperature** is a setting from 0 to 1. Lower = more deterministic and factual. Higher = more creative and varied. For security information, we want 0.1 — stick to the facts.

---

## 11. LangChain — The Glue

### What is LangChain?

LangChain is a Python library that provides building blocks for LLM applications. Think of it as a collection of utilities that handle common patterns in RAG systems.

Without LangChain, you'd have to manually:
- Format prompts
- Call the LLM API
- Parse responses
- Manage conversation history
- Chain multiple steps together

LangChain does all of this with standardized interfaces.

### What we actually use from LangChain

Honestly, we use it primarily for:
1. **ChatGroq** — the Groq LLM connector
2. **Prompt templates** — structured prompts with variables
3. **Message history** — storing conversation history in the right format

We don't use LangChain's retrieval components (we built those ourselves with BM25 + ChromaDB + RRF + CrossEncoder — which gives us much more control).

---

## 12. FastAPI — The Backend

### What is FastAPI?

**FastAPI** is a modern Python web framework for building APIs (Application Programming Interfaces). An API is a program that accepts HTTP requests (GET, POST, etc.) and returns responses.

When the user types a question in the browser and hits send, the browser sends an HTTP POST request to our FastAPI server with the question. FastAPI receives it, runs the RAG pipeline, and sends back the answer.

### Why FastAPI over Flask/Django?

| Feature | FastAPI | Flask | Django |
|---------|---------|-------|--------|
| Speed | Very fast | Moderate | Moderate |
| Async support | Built-in | Limited | Limited |
| Auto documentation | Yes (Swagger UI) | No | No |
| Type checking | Built-in | No | No |
| Learning curve | Low-Medium | Low | High |

FastAPI uses Python type hints to automatically generate API documentation (available at `/docs`) and validate incoming data.

```python
# From our routes.py
from fastapi import FastAPI
from pydantic import BaseModel

class ChatRequest(BaseModel):  # Pydantic validates incoming JSON
    message: str
    session_id: str
    severity_filter: list[str] = []
    source_filter: str = "both"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    answer, sources = await run_rag_pipeline(
        query=request.message,
        session_id=request.session_id,
        severity_filter=request.severity_filter
    )
    return {"answer": answer, "sources": sources}
```

### Pydantic — Data Validation

We use **Pydantic** to define the shape of data going in and out of the API. If someone sends a request missing the `message` field, FastAPI automatically rejects it with a helpful error. No manual validation code needed.

### Async — Why it matters

FastAPI is **async** (asynchronous). When our server is waiting for the Groq API to respond (which takes 1-2 seconds), it doesn't just sit and wait doing nothing. It can handle other incoming requests in the meantime. This is critical for a web app that might have multiple users.

---

## 13. The Frontend — Why No Framework

### The decision

We built the frontend with pure HTML, CSS, and JavaScript — no React, no Vue, no Angular.

**Why?** For a single-page app like a chatbot:
- A framework adds complexity and a build step (npm, webpack, etc.)
- The frontend is ~700 lines of JavaScript — manageable without a framework
- Loading time is faster with no framework overhead
- Easier to debug

This is a valid engineering decision, not a shortcut. Interviewers respect this kind of reasoning.

### How the frontend talks to the backend

The frontend uses the **Fetch API** (built into every browser) to make HTTP requests to our FastAPI backend:

```javascript
// When user hits send
const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        message: userInput,
        session_id: sessionId,
        severity_filter: selectedSeverities,
        source_filter: selectedSource
    })
});
const data = await response.json();
// data.answer is the LLM response
// data.sources is the list of cited documents
```

---

## 14. Problems I Faced & How I Solved Them

This section is gold for interviews. Real problems + real solutions = real engineering.

---

### Problem 1: NVD API — 120-Day Date Range Limit

**What happened:**  
When I tried to fetch CVEs from the last 365 days, the NVD API returned `404 Not Found` with an empty body. No error message. Just silence.

**Why it happened:**  
The NVD 2.0 API silently enforces a **maximum 120-day date range per request**. If you exceed it, you get a 404. There's a note about this in their documentation but it's easy to miss.

**How I diagnosed it:**  
I tested smaller date ranges. 365 days → 404. 90 days → 200 OK with 16,000+ CVEs. That told me the date range was the issue, not the date format.

**How I fixed it:**  
Split the 365-day window into multiple smaller windows of ≤119 days each and fetch them sequentially:

```python
WINDOW_DAYS = 119
TOTAL_DAYS = 365

windows = []
window_end = now
remaining = TOTAL_DAYS
while remaining > 0:
    days = min(WINDOW_DAYS, remaining)
    window_start = window_end - timedelta(days=days)
    windows.append((window_start, window_end))
    window_end = window_start
    remaining -= days

# Fetch each window separately and combine
all_cves = []
for start, end in windows:
    chunk = fetch_window(start, end)
    all_cves.extend(chunk)
```

**Lesson:** APIs have undocumented or poorly-documented constraints. When you get a mysterious 404, binary-search the parameter space to find the breaking point.

---

### Problem 2: RAGAS Library Version Conflict

**What happened:**  
I originally planned to use RAGAS (a popular RAG evaluation library) for scoring faithfulness and answer relevancy. When I tried to install it, pip threw a dependency conflict:

```
ragas 0.1.21 requires langchain<0.3, but langchain==0.3.0 is installed
```

**Why it happened:**  
RAGAS version 0.1.21 was built against an older LangChain API. LangChain 0.3 broke backwards compatibility with libraries that depended on it.

**What I tried:**  
- Upgrading RAGAS → newer RAGAS versions required a completely different setup with async evaluators
- Downgrading LangChain → broke other things that depended on 0.3 features

**What I did instead:**  
Wrote custom token-overlap evaluation metrics. The key metrics for RAG quality can be approximated without the full RAGAS library:

- **Faithfulness** ≈ "how much of the answer's content appears in the retrieved documents?" (token overlap)
- **Answer Relevancy** ≈ "how similar is the answer to the original question?" (cosine similarity)

**Lesson:** Don't let a library version conflict block your project. Understand what the library actually does and implement the core logic yourself if needed. This demonstrates deeper understanding than just installing a package.

---

### Problem 3: GROQ_API_KEY Not Loading in Development

**What happened:**  
I put my GROQ_API_KEY in a `.env` file in the `securerag/` folder, but the code wasn't picking it up. The LLM would fail silently.

**Why it happened:**  
Replit already had an environment variable called `GROQ_API_KEY` set (to an empty string). When Python's `load_dotenv()` runs, it normally won't override variables that are already set in the environment. So the Replit env variable (empty) was winning over the `.env` file (real key).

**How I fixed it:**  
```python
from dotenv import load_dotenv
load_dotenv(override=True)  # the "override=True" flag forces .env to win
```

**Lesson:** Order of precedence for environment variables matters. `dotenv` by default respects already-set env vars (doesn't override). The `override=True` flag reverses that.

---

### Problem 4: Deployed App Had Empty Knowledge Base

**What happened:**  
After deploying to Replit, the app was live but the knowledge base was empty. When users asked questions, it returned "no relevant documents found."

**Why it happened:**  
Replit uses **autoscale deployment** — it spins up multiple container instances of your app. ChromaDB stores data on the local disk of each container. When you clicked "Ingest Data":
- Container A started fetching CVEs and embedding them into its local ChromaDB
- Status checks were sometimes hitting Container B, which had no knowledge of this running task
- Container A's local ChromaDB data was ephemeral (lost when the container cycled)

**What I tried first:**  
Changing the run command to `pip install && run ingest && start server`. But the ingest takes 10-15 minutes and production containers might time out during startup.

**How I actually fixed it:**  
Moved the ingest to the **build phase**, not the run phase:

```toml
# In artifact.toml
[services.production.build]
args = ["sh", "-c", "pip install -r securerag/requirements.txt && cd securerag && python data/ingest.py --force"]
```

**Why this works:**  
- Build phase runs once, creates an immutable image
- That image (with ChromaDB already populated) is deployed to all instances
- Every instance starts pre-loaded with 2000+ CVEs and OWASP data
- No runtime ingest needed

**Tradeoff:**  
Build takes 15-20 minutes (NVD fetch + model download + embedding 2030 documents). Data is from build time, not real-time. For a demo/portfolio project, this is acceptable.

**Lesson:** Understand your deployment model. Autoscale ≠ traditional server. Stateful operations (like writing to a local database) don't work the same way when you have multiple ephemeral instances.

---

### Problem 5: NVD API Rate Limiting

**What happened:**  
Without an API key, NVD limits you to 5 requests per 30 seconds. When we exceeded this, requests would fail.

**How I handled it:**  
Added a `time.sleep(6)` between API requests (6 seconds ≥ 30 seconds / 5 requests). With a free NVD API key, this drops to 0.6 seconds.

```python
NVD_API_KEY = os.getenv("NVD_API_KEY", "")
RATE_LIMIT_SLEEP = 0.6 if NVD_API_KEY else 6.0

# In the fetch loop
time.sleep(RATE_LIMIT_SLEEP)
```

**Lesson:** Always check third-party API rate limits and handle them explicitly. A 6-second sleep is ugly but reliable.

---

### Problem 6: "No previewable artifacts" — Deployment 404

**What happened:**  
After deploying, the live URL showed `404 — This deployment has no previewable artifacts` instead of the app.

**Why it happened:**  
Replit's deployment system uses "artifact mode" — it reads `artifact.toml` config files to know what to serve. Our Python app wasn't registered as an artifact. Replit was deploying the placeholder Node.js API server (which it DID know about), but that wasn't serving anything meaningful at the root URL `/`.

**How I fixed it:**  
Updated the artifact's `artifact.toml` to:
- Serve at path `/` instead of `/api`
- Run `pip install` in the build step instead of `npm build`
- Run `uvicorn main:app` in the production start step instead of `node dist/index.mjs`

**Lesson:** Know your deployment platform's conventions. Replit's artifact system is opinionated — if you don't configure it correctly, it deploys the wrong thing.

---

## 15. The Deployment Story

### Development vs Production

| | Development (Replit workspace) | Production (Replit Autoscale) |
|--|-------------------------------|-------------------------------|
| Server | Uvicorn with `--reload` (auto-restarts on file change) | Uvicorn without `--reload` |
| Port | 8000 (hardcoded) | `$PORT` (injected by Replit) |
| ChromaDB | Populated manually by clicking "Ingest" | Pre-populated during build |
| Secrets | `.env` file + Replit workspace secrets | Replit deployment secrets |
| Instances | 1 | Multiple (autoscale) |

### The `--reload` flag

In development, `uvicorn main:app --reload` watches for file changes and automatically restarts the server. **Never use this in production** — it's slower and wastes resources.

### Environment Variables in Production

Sensitive values (API keys) should never be hardcoded. In Replit:
- Development: set in the Secrets panel or `.env` file
- Production: set in the deployment Secrets panel

The app reads them with `os.getenv("GROQ_API_KEY")`.

---

## 16. End-to-End Walk

Here's exactly what happens when a user types "What are the latest SQL injection CVEs?" and hits send:

```
1. Browser: User types and hits send
   → JavaScript builds a JSON payload:
     {"message": "What are the latest SQL injection CVEs?",
      "session_id": "abc123", "severity_filter": [], "source_filter": "both"}
   → Sends HTTP POST to /api/chat

2. FastAPI (routes.py): Receives the request
   → Validates the payload with Pydantic
   → Calls chain.py's run_rag() function

3. Hybrid Search (hybrid_search.py):
   a. BM25 search over all stored documents
      "SQL" and "injection" are rare in the corpus → documents with these words rank high
      Returns: top 10 documents ranked by BM25 score
   
   b. Vector search in ChromaDB
      "What are the latest SQL injection CVEs?" → embed it → [0.34, -0.12, ...]
      ChromaDB finds the 10 documents whose vectors are closest
      Returns: top 10 documents ranked by cosine similarity
   
   c. RRF Fusion
      Combines both lists using rank positions
      Returns: top 10 documents that scored well across BOTH methods

4. Cross-Encoder Reranker (reranker.py):
   Takes the query + each of the 10 documents as pairs
   ms-marco-MiniLM scores each pair for true relevance
   Returns: top 5 most genuinely relevant documents

5. LLM Generation (chain.py):
   Retrieves session history (previous turns of conversation)
   Formats a prompt:
   "You are a cybersecurity expert. Using ONLY these documents: [top 5 docs]
    Answer: What are the latest SQL injection CVEs?"
   Sends to Groq API (llama-3.1-8b-instant)
   Groq returns an answer in ~1-2 seconds

6. Response:
   FastAPI returns JSON:
   {"answer": "Based on recent CVE data, the most notable SQL injection...",
    "sources": [{"id": "CVE-2025-1234", "severity": "HIGH", ...}]}

7. Browser:
   JavaScript receives the response
   Renders the answer text with a typewriter animation
   Renders the source CVE chips (clickable badges showing CVE IDs)
```

Total time: ~2-4 seconds from question to answer.

---

## 17. Interview Questions & How to Answer Them

### "Walk me through your project."

**Answer template:**  
"SecureRAG is a RAG chatbot for cybersecurity. I built it because regular LLMs hallucinate on security topics and don't have up-to-date CVE data. The pipeline works in three stages: first, hybrid search combining BM25 and vector search with RRF fusion retrieves the top 10 relevant documents from a ChromaDB store of 2000+ CVEs and OWASP Top 10. Second, a cross-encoder reranker rescores those 10 and keeps the best 5. Third, LLaMA 3.1 via Groq reads those 5 documents and generates a grounded answer with source attribution. The backend is FastAPI, the frontend is vanilla JS with a dark VS Code-style theme."

---

### "What is RAG and why did you choose it over fine-tuning?"

**Answer:**  
"RAG stands for Retrieval-Augmented Generation. Instead of relying on an LLM's memorized knowledge, you first retrieve relevant documents from a database and give those to the LLM as context. I chose RAG over fine-tuning because CVE data changes constantly — fine-tuning bakes knowledge into the model, so it becomes stale. With RAG, I just update the database. It's also much cheaper — fine-tuning requires GPU compute. And RAG produces answers grounded in real documents, which reduces hallucination."

---

### "What is the difference between BM25 and semantic search? Why use both?"

**Answer:**  
"BM25 is keyword-based — it scores documents by how often the search terms appear, weighted by how rare those terms are in the corpus. It's great for exact matches: if someone searches for 'CVE-2024-1234', BM25 finds it instantly. Semantic search converts text to embeddings — numerical representations where similar meanings produce similar vectors. It's great for conceptual queries: 'database attack' finds 'SQL injection' even though the words are different. The problem is neither is perfect alone. BM25 misses paraphrases; semantic search misses rare specific terms. Combining them with Reciprocal Rank Fusion gives better recall than either alone."

---

### "Explain RRF."

**Answer:**  
"Reciprocal Rank Fusion is a technique for merging multiple ranked lists. The key insight is that you can't directly average scores from BM25 and vector search because they're on different scales. RRF ignores the scores and only uses the ranks. A document's RRF score is the sum of 1/(rank + k) for each list it appears in, where k=60 is a smoothing constant. A document that ranks 2nd in BM25 and 2nd in vector search will score higher than a document that ranks 1st in only one list. Consistent performance across both methods is a strong relevance signal."

---

### "Why use a cross-encoder if you already have hybrid search?"

**Answer:**  
"Hybrid search is fast but approximate. BM25 and vector search encode the query and documents separately — they never 'see' each other directly. A cross-encoder processes the query and document together, so it understands the relationship between them. It's more accurate but much slower. The solution is to use it only on a small set: hybrid search narrows 2000 documents down to 10, then the cross-encoder precisely reranks those 10. This is the standard two-stage retrieval architecture — fast approximate retrieval followed by slow precise reranking."

---

### "What problems did you face?"

**Answer:**  
"A few significant ones. The NVD API silently returns 404 if your date range exceeds 120 days — I had to split the 365-day fetch into 4 windows of 119 days each. I discovered this by testing smaller and smaller date ranges until I found where it broke. Second, RAGAS (the evaluation library) had a version conflict with LangChain 0.3, so I wrote custom token-overlap evaluation metrics instead. Third, deploying to autoscale infrastructure meant multiple container instances that don't share disk — my ChromaDB data written by one instance wasn't visible to others. I fixed this by moving ingestion to the build phase, so all instances start with the same pre-populated database."

---

### "What is ChromaDB and why not use Pinecone?"

**Answer:**  
"ChromaDB is an open-source vector database that runs locally. It stores document embeddings on disk and supports similarity search. I chose it because it's free, runs in-process with no external service dependency, and is well-suited for projects where you control the full stack. Pinecone is a managed cloud vector database — great for production systems that need to scale to millions of documents, but it requires an account, API key, and has costs. For 2000-3000 documents, ChromaDB is the pragmatic choice."

---

### "What is an embedding? What model did you use?"

**Answer:**  
"An embedding is a numerical representation of text in a high-dimensional vector space, where semantically similar texts have similar vectors. We measure similarity using cosine similarity — the angle between two vectors. I used BAAI/bge-small-en-v1.5, which converts text to 384-dimensional vectors. I chose it because it's open-source, runs locally without API costs, small enough to be fast (hence 'small'), and consistently performs well on retrieval benchmarks. It was trained specifically for retrieval tasks, not just general language modeling."

---

### "Why FastAPI over Flask?"

**Answer:**  
"FastAPI has built-in async support, which is important because our pipeline makes external API calls (Groq) that would block the server. It also auto-generates Swagger documentation from type hints with no extra code, and uses Pydantic for automatic request validation. For a project with an API, these features save significant development time compared to Flask where you'd have to add these manually."

---

### "What would you improve if you had more time?"

**Answer (shows self-awareness):**  
"A few things. First, the data is baked in at build time, so it gets stale. I'd set up a scheduled job to re-ingest CVEs weekly and use a cloud vector database (like Pinecone or Weaviate) so the data persists independently of deployment. Second, session history is in-memory, so it resets on restart — I'd move that to Redis or a database. Third, I'd add streaming responses so the answer types out in real-time rather than appearing all at once. And I'd add a proper NVD API key to reduce ingestion time from 15 minutes to under 2 minutes."

---

## Quick Reference — Terms to Know Cold

| Term | One-line definition |
|------|---------------------|
| **RAG** | Retrieve relevant documents first, then give them to an LLM to answer from |
| **Embedding** | Text converted to a list of numbers where similar texts have similar numbers |
| **Vector database** | Database that stores and searches by vector similarity |
| **BM25** | Keyword-based ranking: scores documents by how often and how rarely the query terms appear |
| **Cosine similarity** | Measure of angle between two vectors; 1.0 = identical, 0 = unrelated |
| **RRF** | Merge multiple ranked lists using positions, not scores |
| **Cross-encoder** | Model that scores a (query, document) pair together — more accurate than separate encoding |
| **Bi-encoder** | Model that encodes query and document separately (BM25, vector search) — faster but less precise |
| **ChromaDB** | Open-source local vector database |
| **BAAI/bge-small-en-v1.5** | Embedding model producing 384-dim vectors, good for retrieval |
| **ms-marco-MiniLM-L-6-v2** | Cross-encoder model trained on MS MARCO reading comprehension |
| **Groq** | Company with fast AI inference hardware; hosts LLaMA for free |
| **LLaMA 3.1 8B** | Open-source LLM by Meta, 8 billion parameters |
| **FastAPI** | Modern Python web framework with built-in async, validation, and auto-docs |
| **Pydantic** | Python library for data validation using type hints |
| **CVE** | Common Vulnerabilities and Exposures — a standardized identifier for a known security flaw |
| **NVD** | National Vulnerability Database — the US government's CVE repository |
| **OWASP Top 10** | The 10 most critical web security risks, published by OWASP |
| **CVSS score** | Common Vulnerability Scoring System — 0-10 scale for CVE severity |
| **Temperature (LLM)** | Controls randomness of output; 0 = deterministic/factual, 1 = creative/varied |
| **Autoscale deployment** | Multiple instances that scale with traffic; no shared state between instances |

---

*Good luck with the interview. You built this thing — you know it. Trust that.*

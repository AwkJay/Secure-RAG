"""Build the labelled retrieval set (evals/data/retrieval_set.jsonl).

Gold labels come from construction, not annotation:
  - CVE queries: a doc is sampled, an LLM writes queries *about* it, so gold = that doc's cve_id.
  - OWASP queries: hand-written, gold = the intended category (any of its 3 chunks counts).

Styles: natural (conversational), keyword (short terms), id (templated "Tell me about CVE-..."), owasp.
Caveat (also in README): LLM-written queries tend to reuse the doc's vocabulary, which favours BM25.
"""
import json
import logging
import random
import re
import time
from pathlib import Path

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATASET_PATH = DATA_DIR / "retrieval_set.jsonl"
CACHE_PATH = Path(__file__).parent / "results" / "gen_cache.json"

SEED = 42
N_CVE_DOCS = 60

# (query, gold owasp_id). First 8 are relabelled from rag/evaluate.py::TEST_QUESTIONS; the rest are hand-written.
OWASP_QUESTIONS = [
    ("What are the main prevention methods for SQL injection according to OWASP?", "A03:2021"),
    ("How can broken access control vulnerabilities be prevented?", "A01:2021"),
    ("What is SSRF and why is it in the OWASP Top 10?", "A10:2021"),
    ("What does OWASP say about cryptographic failures?", "A02:2021"),
    ("How should authentication failures be handled in web applications?", "A07:2021"),
    ("What is insecure design and how does it differ from implementation flaws?", "A04:2021"),
    ("What are vulnerable and outdated components and why are they dangerous?", "A06:2021"),
    ("What security logging practices does OWASP recommend?", "A09:2021"),
    ("How do I prevent security misconfiguration in a deployed application?", "A05:2021"),
    ("What are software and data integrity failures, like unsigned updates or insecure deserialization?", "A08:2021"),
    ("Attack scenario where an attacker changes an account number in the URL to view someone else's account", "A01:2021"),
    ("Example attack scenario where a server fetches a URL controlled by the attacker to reach cloud metadata", "A10:2021"),
    ("Why is storing passwords with unsalted MD5 hashes a problem?", "A02:2021"),
    ("What is a credential stuffing attack scenario and how do I stop it?", "A07:2021"),
    ("Default accounts, verbose error messages and unnecessary features left enabled", "A05:2021"),
    ("Auto-update functionality that downloads updates without verifying a signature", "A08:2021"),
    ("Missing threat modeling and secure design patterns in the development lifecycle", "A04:2021"),
    ("Using unsupported libraries and not tracking component versions", "A06:2021"),
    ("Insufficient logging and monitoring means breaches go undetected for months", "A09:2021"),
    ("Untrusted user data concatenated into an interpreter query, including LDAP and OS commands", "A03:2021"),
]

PROMPT = """You write search queries for a vulnerability knowledge base.
Given this CVE description, write two queries a security engineer might type WITHOUT knowing the CVE ID.
- "natural": a conversational question, at most 25 words.
- "keyword": 3 to 6 keywords or a short phrase.
Do NOT include the CVE ID. Do NOT copy any run of 6 or more consecutive words from the description.
Respond with JSON only: {{"natural": "...", "keyword": "..."}}

Description:
{description}"""


def _description(text: str) -> str:
    m = re.search(r"Description:\s*(.*?)\nAffected:", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def _leaks(query: str, cve_id: str, description: str, n: int = 6) -> bool:
    if cve_id.lower() in query.lower():
        return True
    q_tokens = re.findall(r"\w+", query.lower())
    d_tokens = re.findall(r"\w+", description.lower())
    d_ngrams = {tuple(d_tokens[i:i + n]) for i in range(len(d_tokens) - n + 1)}
    return any(tuple(q_tokens[i:i + n]) in d_ngrams for i in range(len(q_tokens) - n + 1))


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1))


def _generate_queries(llm, description: str, retries: int = 5) -> dict | None:
    for attempt in range(retries):
        try:
            raw = llm.invoke([HumanMessage(content=PROMPT.format(description=description[:1500]))]).content
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(m.group(0))
            if isinstance(data.get("natural"), str) and isinstance(data.get("keyword"), str):
                return data
        except Exception as e:  # rate limit, bad JSON, network
            wait = 2 ** attempt * 2
            logger.warning("query generation failed (%s); retry in %ss", type(e).__name__, wait)
            time.sleep(wait)
    return None


def _sample_cve_docs(n: int, seed: int) -> list[tuple[str, dict]]:
    from rag.vectorstore import get_collection

    res = get_collection().get(where={"source": "NVD"}, include=["documents", "metadatas"])
    by_sev: dict[str, list] = {}
    for doc, meta in zip(res["documents"], res["metadatas"]):
        by_sev.setdefault(meta.get("severity", "UNKNOWN"), []).append((doc, meta))

    rng = random.Random(seed)
    sevs = sorted(by_sev)
    for s in sevs:
        by_sev[s].sort(key=lambda x: x[1]["cve_id"])  # deterministic base order before shuffling
        rng.shuffle(by_sev[s])

    picked, i = [], 0
    while len(picked) < n and any(by_sev.values()):  # round-robin across severities
        s = sevs[i % len(sevs)]
        if by_sev[s]:
            picked.append(by_sev[s].pop())
        i += 1
    return picked


def build(n_docs: int = N_CVE_DOCS, seed: int = SEED, sleep_s: float = 2.5) -> list[dict]:
    from rag.chain import get_llm

    llm = get_llm()
    cache = _load_cache()
    rows: list[dict] = []
    dropped = 0

    for doc, meta in _sample_cve_docs(n_docs, seed):
        cve_id = meta["cve_id"]
        desc = _description(doc)
        if cve_id not in cache:
            gen = _generate_queries(llm, desc)
            if gen is None:
                logger.warning("skipping %s: generation failed", cve_id)
                continue
            cache[cve_id] = gen
            _save_cache(cache)
            time.sleep(sleep_s)
        gen = cache[cve_id]

        for style in ("natural", "keyword"):
            q = gen[style].strip()
            if not q or _leaks(q, cve_id, desc):
                dropped += 1
                continue
            rows.append({"query": q, "style": style, "gold_ids": [cve_id], "source": "NVD", "severity": meta.get("severity")})
        rows.append({"query": f"Tell me about {cve_id}", "style": "id", "gold_ids": [cve_id], "source": "NVD", "severity": meta.get("severity")})

    for q, gold in OWASP_QUESTIONS:
        rows.append({"query": q, "style": "owasp", "gold_ids": [gold], "source": "OWASP", "severity": None})

    # dedupe on query text, then assign stable ids
    seen, out = set(), []
    for r in rows:
        key = r["query"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    for i, r in enumerate(out):
        r["qid"] = f"q{i:03d}"
        # put qid first for readability
    out = [{"qid": r["qid"], **{k: v for k, v in r.items() if k != "qid"}} for r in out]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.write_text("\n".join(json.dumps(r) for r in out) + "\n")

    counts: dict[str, int] = {}
    for r in out:
        counts[r["style"]] = counts.get(r["style"], 0) + 1
    print(f"Wrote {len(out)} queries to {DATASET_PATH} ({counts}); dropped {dropped} leaky queries")
    rng = random.Random(seed)
    print("\nSample queries for eyeballing:")
    for r in rng.sample(out, min(10, len(out))):
        print(f"  [{r['style']:7}] {r['query']}  -> {r['gold_ids']}")
    return out


def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} missing — run `python -m evals.run --build-dataset` first")
    return [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]

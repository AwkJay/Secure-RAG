"""Small generation check: hallucinated-ID rate, citation precision, abstention rate.

Runs the production path (hybrid -> rerank top-5 -> prompt -> Groq) without the semantic cache or
session store, then applies rag.citations.verify_citations. No LLM judge: faithfulness of prose
claims is explicitly NOT measured here.
"""
import logging
import random
import time

from langchain_core.messages import HumanMessage

from core.config import settings
from rag.citations import verify_citations

logger = logging.getLogger(__name__)

N_ANSWERABLE_CVE = 30

# nonexistent CVE IDs (year/number far outside what NVD has issued) and clearly off-topic prompts
UNANSWERABLE = [
    ("Tell me about CVE-2099-0001", "fake_id"),
    ("Explain CVE-2088-77777 and its CVSS score", "fake_id"),
    ("What is the severity of CVE-1999-99999?", "fake_id"),
    ("Summarise CVE-2077-12345", "fake_id"),
    ("Does CVE-2066-424242 affect OpenSSL?", "fake_id"),
    ("What's a good recipe for banana bread?", "off_topic"),
    ("Who won the 2018 FIFA World Cup?", "off_topic"),
    ("Write me a poem about cats", "off_topic"),
    ("What is the capital of Australia?", "off_topic"),
    ("How do I train for a marathon?", "off_topic"),
]

ABSTAIN_MARKERS = ("don't have specific information", "do not have specific information", "don't have information",
                   "not in my knowledge base", "cannot find", "no information")


def _abstained(answer: str) -> bool:
    a = answer.lower().replace("’", "'")
    return any(m in a for m in ABSTAIN_MARKERS)


def _answer(question: str, searcher, reranker, llm, retries: int = 5) -> tuple[str, list]:
    from rag.chain import SYSTEM_PROMPT_TEMPLATE, _format_context

    retrieved, _ = searcher.search(question, top_k=settings.retrieval_top_k)
    docs = reranker.rerank(question, retrieved, top_k=settings.rerank_top_k)
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        context=_format_context(docs), chat_history="No previous conversation.", question=question
    )
    for attempt in range(retries):
        try:
            return llm.invoke([HumanMessage(content=prompt)]).content, docs
        except Exception as e:
            wait = 2 ** attempt * 2
            logger.warning("LLM call failed (%s); retry in %ss", type(e).__name__, wait)
            time.sleep(wait)
    return "", docs


def run_generation_eval(dataset: list[dict], seed: int = 42, sleep_s: float = 2.5) -> dict:
    from rag.chain import get_llm
    from rag.hybrid_search import get_searcher
    from rag.reranker import get_reranker

    searcher, reranker, llm = get_searcher(), get_reranker(), get_llm()
    rng = random.Random(seed)

    cve_nat = [r for r in dataset if r["source"] == "NVD" and r["style"] == "natural"]
    owasp = [r for r in dataset if r["source"] == "OWASP"]
    answerable = rng.sample(cve_nat, min(N_ANSWERABLE_CVE, len(cve_nat))) + owasp[:10]

    records = []
    for row in answerable:
        ans, docs = _answer(row["query"], searcher, reranker, llm)
        v = verify_citations(ans, docs, row["query"])
        records.append({"kind": "answerable", "query": row["query"], "gold": row["gold_ids"], "answer": ans,
                        "abstained": _abstained(ans), "gold_cited": bool(set(row["gold_ids"]) & set(v["cited"])), **v})
        time.sleep(sleep_s)

    for q, kind in UNANSWERABLE:
        ans, docs = _answer(q, searcher, reranker, llm)
        v = verify_citations(ans, docs, q)
        records.append({"kind": kind, "query": q, "gold": [], "answer": ans,
                        "abstained": _abstained(ans), "gold_cited": False, **v})
        time.sleep(sleep_s)

    valid = [r for r in records if r["answer"]]  # drop calls that failed all retries
    ans_recs = [r for r in valid if r["kind"] == "answerable"]
    total_cited = sum(len(r["cited"]) for r in valid)
    total_verified = sum(len(r["verified"]) for r in valid)

    def rate(rs, f):
        return (sum(1 for r in rs if f(r)) / len(rs)) if rs else None

    summary = {
        "n_answerable": len(ans_recs),
        "n_unanswerable": len(valid) - len(ans_recs),
        "n_failed_calls": len(records) - len(valid),
        "citation_precision": (total_verified / total_cited) if total_cited else None,
        "hallucinated_id_rate_answerable": rate(ans_recs, lambda r: not r["ok"]),
        "hallucinated_id_rate_unanswerable": rate([r for r in valid if r["kind"] != "answerable"], lambda r: not r["ok"]),
        "gold_cited_rate_answerable": rate(ans_recs, lambda r: r["gold_cited"]),
        "abstention_rate_answerable": rate(ans_recs, lambda r: r["abstained"]),
        "abstention_rate_fake_id": rate([r for r in valid if r["kind"] == "fake_id"], lambda r: r["abstained"]),
        "abstention_rate_off_topic": rate([r for r in valid if r["kind"] == "off_topic"], lambda r: r["abstained"]),
    }
    return {"summary": summary, "records": records}


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def to_markdown(res: dict) -> str:
    s = res["summary"]
    rows = [
        ("Citation precision (verified / cited IDs)", _pct(s["citation_precision"])),
        ("Hallucinated-ID rate, answerable queries", _pct(s["hallucinated_id_rate_answerable"])),
        ("Hallucinated-ID rate, unanswerable queries", _pct(s["hallucinated_id_rate_unanswerable"])),
        ("Answerable: gold ID cited in answer", _pct(s["gold_cited_rate_answerable"])),
        ("Answerable: abstained (should be low)", _pct(s["abstention_rate_answerable"])),
        ("Fake CVE ID: abstained (should be high)", _pct(s["abstention_rate_fake_id"])),
        ("Off-topic: abstained (should be high)", _pct(s["abstention_rate_off_topic"])),
    ]
    out = [f"### Generation checks (n={s['n_answerable']} answerable, {s['n_unanswerable']} unanswerable, "
           f"{s['n_failed_calls']} failed calls)", "", "| Metric | Value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(out)

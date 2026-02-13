"""
SecureRAG Custom Evaluation — RAGAS-inspired metrics computed without the RAGAS library.
Computes: Faithfulness, Answer Relevancy, Context Recall, Context Precision
using embedding similarity and token overlap methods.
"""
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

TEST_QUESTIONS = [
    "What is CVSS score and how is it calculated?",
    "What are the main prevention methods for SQL injection according to OWASP?",
    "What are the most critical vulnerability categories in OWASP Top 10 2021?",
    "How can broken access control vulnerabilities be prevented?",
    "What is SSRF and why is it in the OWASP Top 10?",
    "What does OWASP say about cryptographic failures?",
    "How should authentication failures be handled in web applications?",
    "What is insecure design and how does it differ from implementation flaws?",
    "What are vulnerable and outdated components and why are they dangerous?",
    "What security logging practices does OWASP recommend?",
]

TEST_GROUND_TRUTHS = [
    "CVSS is a framework for rating the severity of security vulnerabilities on a scale of 0 to 10. It considers factors like attack vector, complexity, privileges required, and impact on confidentiality, integrity, and availability.",
    "OWASP recommends using parameterized queries, stored procedures, input validation, escaping special characters, and applying least privilege for database accounts to prevent SQL injection.",
    "The most critical OWASP Top 10 2021 categories are Broken Access Control (A01), Cryptographic Failures (A02), Injection (A03), Insecure Design (A04), and Security Misconfiguration (A05).",
    "Broken access control can be prevented by implementing deny-by-default access, role-based access control, logging access control failures, and rate-limiting API and controller access.",
    "SSRF allows attackers to make the server send requests to unintended locations, bypassing firewalls. It is in the OWASP Top 10 because it can expose internal services and cloud metadata.",
    "Cryptographic failures involve weak or misconfigured encryption leading to sensitive data exposure. Prevention includes strong algorithms, proper key management, and avoiding deprecated functions.",
    "Authentication failures should be addressed by implementing multi-factor authentication, limiting failed login attempts, using secure session management, and not exposing sensitive data in errors.",
    "Insecure design refers to missing security controls in design phase, unlike implementation flaws which are code bugs. It requires threat modeling and secure design patterns from the start.",
    "Vulnerable and outdated components are libraries and frameworks without current security patches. They are dangerous because attackers can exploit known vulnerabilities to compromise systems.",
    "OWASP recommends logging authentication events, access control failures, input validation failures, and ensuring logs are in formats consumable by log management solutions.",
]


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'on',
                  'at', 'by', 'for', 'with', 'from', 'and', 'or', 'but', 'not', 'it',
                  'its', 'this', 'that', 'these', 'those', 'as', 'if', 'so', 'than',
                  'such', 'what', 'how', 'when', 'where', 'which', 'who', 'why'}
    return {t for t in tokens if t not in stop_words and len(t) > 2}


def compute_faithfulness(answer: str, contexts: list[str]) -> float:
    if not answer or not contexts:
        return 0.0
    full_context = " ".join(contexts)
    answer_tokens = _tokenize(answer)
    context_tokens = _tokenize(full_context)
    if not answer_tokens:
        return 0.0
    overlap = len(answer_tokens & context_tokens)
    return min(1.0, overlap / max(len(answer_tokens), 1))


def compute_answer_relevancy(question: str, answer: str) -> float:
    if not question or not answer:
        return 0.0
    q_tokens = _tokenize(question)
    a_tokens = _tokenize(answer)
    if not q_tokens or not a_tokens:
        return 0.0
    intersection = q_tokens & a_tokens
    union = q_tokens | a_tokens
    jaccard = len(intersection) / max(len(union), 1)
    keyword_coverage = len(intersection) / max(len(q_tokens), 1)
    length_bonus = min(1.0, len(answer.split()) / 50.0) * 0.1
    return min(1.0, (jaccard * 0.4 + keyword_coverage * 0.5 + length_bonus))


def compute_context_recall(ground_truth: str, contexts: list[str]) -> float:
    if not ground_truth or not contexts:
        return 0.0
    full_context = " ".join(contexts)
    gt_tokens = _tokenize(ground_truth)
    ctx_tokens = _tokenize(full_context)
    if not gt_tokens:
        return 0.0
    recall = len(gt_tokens & ctx_tokens) / max(len(gt_tokens), 1)
    return min(1.0, recall)


def compute_context_precision(question: str, contexts: list[str]) -> float:
    if not question or not contexts:
        return 0.0
    q_tokens = _tokenize(question)
    relevant = 0
    for ctx in contexts:
        ctx_tokens = _tokenize(ctx)
        if not ctx_tokens:
            continue
        overlap = len(q_tokens & ctx_tokens) / max(len(q_tokens), 1)
        if overlap > 0.1:
            relevant += 1
    return min(1.0, relevant / max(len(contexts), 1))


def run_evaluation() -> dict:
    from rag.hybrid_search import get_searcher
    from rag.reranker import get_reranker
    from rag.chain import get_llm, _format_context
    from langchain_core.messages import HumanMessage
    from rag.chain import SYSTEM_PROMPT_TEMPLATE

    logger.info("Running evaluation on %d test queries...", len(TEST_QUESTIONS))

    searcher = get_searcher()
    reranker = get_reranker()
    llm = get_llm()

    per_query = []
    agg = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_recall": [],
        "context_precision": [],
    }

    for i, (q, gt) in enumerate(zip(TEST_QUESTIONS, TEST_GROUND_TRUTHS)):
        try:
            retrieved, _ = searcher.search(q, top_k=10)
            reranked = reranker.rerank(q, retrieved, top_k=5)
            context_chunks = [text for text, _, _ in reranked]
            context_str = _format_context(reranked)

            prompt = SYSTEM_PROMPT_TEMPLATE.format(
                context=context_str,
                chat_history="No previous conversation.",
                question=q,
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content

            f = compute_faithfulness(answer, context_chunks)
            ar = compute_answer_relevancy(q, answer)
            cr = compute_context_recall(gt, context_chunks)
            cp = compute_context_precision(q, context_chunks)

            agg["faithfulness"].append(f)
            agg["answer_relevancy"].append(ar)
            agg["context_recall"].append(cr)
            agg["context_precision"].append(cp)

            per_query.append({
                "query": q,
                "faithfulness": round(f, 3),
                "answer_relevancy": round(ar, 3),
                "context_recall": round(cr, 3),
                "context_precision": round(cp, 3),
            })
            logger.info("Evaluated query %d/%d: f=%.2f ar=%.2f cr=%.2f cp=%.2f",
                        i + 1, len(TEST_QUESTIONS), f, ar, cr, cp)

        except Exception as e:
            logger.error("Error on query %d: %s", i + 1, e)
            per_query.append({
                "query": q,
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_recall": 0.0,
                "context_precision": 0.0,
            })
            agg["faithfulness"].append(0.0)
            agg["answer_relevancy"].append(0.0)
            agg["context_recall"].append(0.0)
            agg["context_precision"].append(0.0)

    def safe_avg(lst):
        return round(sum(lst) / max(len(lst), 1), 3) if lst else 0.0

    scores = {
        "faithfulness": safe_avg(agg["faithfulness"]),
        "answer_relevancy": safe_avg(agg["answer_relevancy"]),
        "context_recall": safe_avg(agg["context_recall"]),
        "context_precision": safe_avg(agg["context_precision"]),
    }

    return {
        "scores": scores,
        "per_query_results": per_query,
        "evaluated_at": datetime.utcnow().isoformat(),
        "num_test_queries": len(TEST_QUESTIONS),
    }

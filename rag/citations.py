"""Citation verification: flag CVE / OWASP IDs in an answer that the context can't support.

Pure Python on purpose (no model, Redis or Chroma imports) so it is trivially unit-testable
and can be reused by the offline eval harness.

An ID is *allowed* if it appears in any of:
  - metadata (`cve_id` / `owasp_id`) of the retrieved docs
  - the text of the retrieved docs
  - the user's question or earlier turns (so "I don't have CVE-2099-0001 in my KB" isn't flagged)
Anything else the model cites is reported as `unverified`. We flag, we never rewrite the answer.
"""
import re

_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)
# A01..A10, optionally followed by ":2021". Negative lookarounds avoid matching inside words/hex like "CA01".
_OWASP_RE = re.compile(r"(?<![A-Za-z0-9])A(0[1-9]|10)(?::2021)?(?![A-Za-z0-9])", re.IGNORECASE)


# LLMs sometimes emit typographic hyphens (U+2010 hyphen, U+2011 non-breaking hyphen, en/em dash, minus sign)
# inside IDs, e.g. "CVE\u20112025\u201164773". Fold them to ASCII "-" so the regex still sees the ID.
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")


def _norm_owasp(num: str) -> str:
    return f"A{num}:2021"


def extract_ids(text: str) -> set[str]:
    """Return normalised CVE and OWASP IDs found in `text` (CVE upper-cased, OWASP as 'A03:2021')."""
    if not text:
        return set()
    text = text.translate(_DASHES)
    ids = {m.group(0).upper() for m in _CVE_RE.finditer(text)}
    ids |= {_norm_owasp(m.group(1)) for m in _OWASP_RE.finditer(text)}
    return ids


def _normalise_one(raw: str) -> set[str]:
    return extract_ids(raw or "")


def allowed_ids(docs, question: str = "", history: list[dict] | None = None) -> set[str]:
    """IDs the answer is permitted to cite. `docs` is a list of (text, metadata, score)."""
    allowed: set[str] = set()
    for text, meta, *_ in docs:
        allowed |= extract_ids(text)
        allowed |= _normalise_one(meta.get("cve_id", ""))
        allowed |= _normalise_one(meta.get("owasp_id", ""))
    allowed |= extract_ids(question)
    for turn in history or []:
        allowed |= extract_ids(turn.get("human", ""))
        allowed |= extract_ids(turn.get("assistant", ""))
    return allowed


def verify_citations(answer: str, docs, question: str = "", history: list[dict] | None = None) -> dict:
    """Check every ID cited in `answer` against what retrieval/user supplied.

    Returns {cited, verified, unverified, ok}; lists are sorted for stable output.
    """
    cited = extract_ids(answer)
    allowed = allowed_ids(docs, question, history)
    verified = cited & allowed
    unverified = cited - allowed
    return {
        "cited": sorted(cited),
        "verified": sorted(verified),
        "unverified": sorted(unverified),
        "ok": not unverified,
    }

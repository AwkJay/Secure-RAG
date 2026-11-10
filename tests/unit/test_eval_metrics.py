import math

from evals.metrics import (
    bootstrap_ci, dedupe_ranked_ids, ndcg_at_k, paired_bootstrap_delta, recall_at_k, reciprocal_rank,
)


def test_dedupe_collapses_owasp_chunks():
    docs = [
        ("a", {"source": "OWASP", "owasp_id": "A03:2021"}, 1.0),
        ("b", {"source": "OWASP", "owasp_id": "A03:2021"}, 0.9),
        ("c", {"source": "NVD", "cve_id": "CVE-2020-0001"}, 0.8),
    ]
    assert dedupe_ranked_ids(docs) == ["A03:2021", "CVE-2020-0001"]


def test_recall_and_rr():
    ranked = ["x", "y", "z"]
    assert recall_at_k(ranked, {"y"}, 1) == 0.0
    assert recall_at_k(ranked, {"y"}, 2) == 1.0
    assert reciprocal_rank(ranked, {"y"}) == 0.5
    assert reciprocal_rank(ranked, {"q"}) == 0.0


def test_ndcg_single_gold():
    assert ndcg_at_k(["g", "a"], {"g"}) == 1.0
    assert math.isclose(ndcg_at_k(["a", "g"], {"g"}), 1 / math.log2(3))
    assert ndcg_at_k(["a", "b"], {"g"}) == 0.0


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    vals = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
    m, lo, hi = bootstrap_ci(vals)
    assert lo <= m <= hi
    assert bootstrap_ci(vals) == (m, lo, hi)


def test_paired_delta_significance():
    a = [1.0] * 30
    b = [0.0] * 30
    r = paired_bootstrap_delta(a, b)
    assert r["delta"] == 1.0 and r["significant"]
    assert not paired_bootstrap_delta(a, a)["significant"]

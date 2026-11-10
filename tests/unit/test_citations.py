from rag.citations import allowed_ids, extract_ids, verify_citations


def _doc(text="", **meta):
    return (text, meta, 0.0)


def test_extract_cve_case_insensitive():
    assert extract_ids("see cve-2021-44228 and CVE-2023-12345") == {"CVE-2021-44228", "CVE-2023-12345"}


def test_extract_owasp_normalises_forms():
    assert extract_ids("A03 and A03:2021 and a10") == {"A03:2021", "A10:2021"}


def test_extract_ignores_lookalikes():
    assert extract_ids("CA01 data A11 A00 BA03x") == set()


def test_extract_empty():
    assert extract_ids("") == set()
    assert extract_ids(None) == set()


def test_valid_citation_passes():
    docs = [_doc("Log4Shell RCE", cve_id="CVE-2021-44228", source="NVD")]
    r = verify_citations("It is CVE-2021-44228, critical.", docs)
    assert r["ok"] and r["verified"] == ["CVE-2021-44228"] and r["unverified"] == []


def test_hallucinated_id_flagged():
    docs = [_doc("Log4Shell RCE", cve_id="CVE-2021-44228")]
    r = verify_citations("See CVE-2021-44228 and CVE-2020-99999.", docs)
    assert not r["ok"]
    assert r["unverified"] == ["CVE-2020-99999"]
    assert r["verified"] == ["CVE-2021-44228"]


def test_owasp_metadata_matches_short_form():
    docs = [_doc("Injection", owasp_id="A03:2021", source="OWASP")]
    r = verify_citations("This is A03 Injection.", docs)
    assert r["ok"]


def test_id_echoed_from_question_allowed():
    r = verify_citations("I don't have CVE-2099-0001 in my knowledge base.", [], question="Tell me about CVE-2099-0001")
    assert r["ok"]


def test_id_only_in_history_allowed():
    hist = [{"human": "What about CVE-2019-0708?", "assistant": "BlueKeep."}]
    r = verify_citations("Following up on CVE-2019-0708.", [], question="more?", history=hist)
    assert r["ok"]


def test_id_in_context_text_but_not_metadata_allowed():
    docs = [_doc("Related to CVE-2018-1111 as well", cve_id="CVE-2018-0001")]
    assert verify_citations("CVE-2018-1111", docs)["ok"]


def test_no_ids_ok():
    r = verify_citations("I don't have specific information about that.", [])
    assert r["ok"] and r["cited"] == []


def test_empty_answer():
    assert verify_citations("", [])["ok"]


def test_allowed_ids_unions_sources():
    docs = [_doc("x", cve_id="CVE-2020-0001", owasp_id="A01:2021")]
    assert allowed_ids(docs, "CVE-2020-0002") == {"CVE-2020-0001", "A01:2021", "CVE-2020-0002"}


def test_extract_handles_unicode_hyphens():
    # non-breaking hyphen (U+2011), en dash (U+2013), minus sign (U+2212)
    text = "CVE\u20112025\u201164773, CVE\u20132021\u201344228 and CVE\u22122020\u22120001"
    assert extract_ids(text) == {"CVE-2025-64773", "CVE-2021-44228", "CVE-2020-0001"}


def test_hallucinated_id_with_unicode_hyphen_still_flagged():
    r = verify_citations("See CVE\u20112020\u201199999.", [_doc("x", cve_id="CVE-2021-44228")])
    assert not r["ok"] and r["unverified"] == ["CVE-2020-99999"]

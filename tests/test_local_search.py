import pytest

from toolforge_rl.tools.local_search import LocalDocument, LocalSearch


DOCS = [
    LocalDocument("d0", "Paris is the capital of France.", "France"),
    LocalDocument("d1", "Berlin is the capital of Germany.", "Germany"),
    LocalDocument("d2", "The Seine flows through Paris.", "Seine"),
    LocalDocument("d3", "Tokyo is the capital of Japan.", "Japan"),
]


@pytest.mark.parametrize(
    ("query", "expected_id"),
    [
        ("capital France", "d0"),
        ("Germany Berlin", "d1"),
        ("Seine river", "d2"),
        ("capital Japan", "d3"),
        ("Paris France", "d0"),
        ("Tokyo Japan", "d3"),
        ("flows through Paris", "d2"),
        ("Berlin Germany", "d1"),
    ],
)
def test_search_ranks_relevant_document_first(query, expected_id):
    output = LocalSearch(DOCS).search(query, top_k=1)
    assert f"id={expected_id}" in output


def test_search_is_deterministic():
    search = LocalSearch(DOCS)
    assert search.search("capital") == search.search("capital")


def test_search_tie_breaks_by_input_order():
    docs = [LocalDocument("first", "same token"), LocalDocument("second", "same token")]
    assert "[1] id=first" in LocalSearch(docs).search("same", top_k=2)


def test_search_top_k_default_is_three():
    assert LocalSearch(DOCS).search("capital").count("id=") == 3


def test_search_top_k_one():
    assert LocalSearch(DOCS).search("capital", top_k=1).count("id=") == 1


def test_search_top_k_is_capped_at_five():
    docs = [LocalDocument(str(i), f"token {i}") for i in range(8)]
    assert LocalSearch(docs).search("token", top_k=99).count("id=") == 5


def test_search_top_k_zero_becomes_one():
    assert LocalSearch(DOCS).search("capital", top_k=0).count("id=") == 1


def test_search_empty_query_is_error():
    assert LocalSearch(DOCS).search("  ").startswith("SEARCH_ERROR")


def test_search_empty_documents_has_closed_fallback():
    assert "No local evidence" in LocalSearch([]).search("anything")


def test_search_returns_title():
    assert "title=France" in LocalSearch(DOCS).search("France", top_k=1)


def test_search_returns_snippet():
    assert "capital of France" in LocalSearch(DOCS).search("France", top_k=1)


def test_search_output_is_bounded():
    docs = [LocalDocument("long", "x " * 1000)]
    assert len(LocalSearch(docs, max_chars=80).search("x")) <= 100


def test_search_never_mutates_documents():
    docs = list(DOCS)
    LocalSearch(docs).search("Paris")
    assert docs == DOCS


def test_search_handles_unicode_query():
    docs = [LocalDocument("cn", "北京 是 中国 的 首都", "中国")]
    assert "id=cn" in LocalSearch(docs).search("北京首都")


def test_search_case_insensitive():
    assert "id=d0" in LocalSearch(DOCS).search("FRANCE", top_k=1)


def test_search_stable_across_instances():
    assert LocalSearch(DOCS).search("Paris") == LocalSearch(DOCS).search("Paris")

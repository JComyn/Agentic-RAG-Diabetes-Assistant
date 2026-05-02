import pytest

from src.retrieval.query_understanding import split_into_subqueries, detect_ambiguity


def test_split_into_subqueries_compound_query():
    query = "What are the symptoms and treatments for diabetes?"
    parts = split_into_subqueries(query)
    assert isinstance(parts, list)
    assert len(parts) > 1


def test_detect_ambiguity_single_word():
    assert detect_ambiguity("effects") is True

import pytest

from src.retrieval.query_understanding import split_into_subqueries, detect_ambiguity


def test_split_into_subqueries_compound_query():
    query = "What are the symptoms and treatments for diabetes?"
    parts = split_into_subqueries(query)
    assert isinstance(parts, list)
    assert len(parts) > 1


def test_detect_ambiguity_single_word():
    assert detect_ambiguity("effects") is True


def test_split_mixed_case_and():
    query = "Causes And Treatments for asthma"
    parts = split_into_subqueries(query)
    assert isinstance(parts, list)
    assert len(parts) == 2
    assert parts[0].lower().startswith("causes")
    assert "treatments" in parts[1].lower()


def test_detect_ambiguity_non_ambiguous():
    # multi-word query shouldn't be considered ambiguous even if it contains an ambiguous token
    assert detect_ambiguity("diabetes symptoms") is False


def test_integration_split_and_ambiguity():
    query = "effects and treatments"
    parts = split_into_subqueries(query)
    ambiguous = detect_ambiguity(query)
    assert isinstance(parts, list)
    assert isinstance(ambiguous, bool)
    assert len(parts) > 1
    # combined query is not single-word ambiguous
    assert ambiguous is False
